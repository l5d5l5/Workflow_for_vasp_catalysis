"""
ml_prediction_service.py
───────────────────────────────────────────────
基于 FairChem UMA 的 ML 结构预测服务

功能：
  - 接收 pymatgen Structure 或 XYZ 字符串
  - 使用 UMA 模型进行结构弛豫（FIRE + FrechetCellFilter）
  - 返回：总能量、最大力、每原子力列表、优化后结构（多格式字符串）
  - 支持保存轨迹文件到本地

不包含：
  - FastAPI 路由定义
  - MP 查询逻辑（见 mp_query_service.py）
───────────────────────────────────────────────
"""

import io
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE, LBFGS
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ──────────────────────────────────────────────
# 日志
# ──────────────────────────────────────────────
logger = logging.getLogger("MLPredictionService")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────
# 支持的模型与任务
# ──────────────────────────────────────────────
SUPPORTED_MODELS = {
    "uma-s-1p2": "最新小模型，速度最快，综合性能最佳（推荐）",
    "uma-s-1p1": "早期小模型",
    "uma-m-1p1": "中等模型，精度最高，但速度较慢",
}

SUPPORTED_TASKS = {
    "omat": "无机晶体材料（Bulk 结构推荐使用）",
    "oc20": "催化表面",
    "oc22": "氧化物催化（仅 1p2）",
    "oc25": "电催化（仅 1p2）",
    "omol": "分子 / 聚合物",
    "odac": "MOF",
    "omc":  "分子晶体",
}

SUPPORTED_OPTIMIZERS = {
    "FIRE":  "FIRE 算法，适合晶体弛豫（默认）",
    "LBFGS": "L-BFGS 算法，适合分子和表面",
}

# ──────────────────────────────────────────────
# 懒加载：predictor 单例（避免重复加载模型权重）
# ──────────────────────────────────────────────
_predictor_cache: Dict[str, Any] = {}

def _get_predictor(model_name: str, device: str) -> Any:
    """
    懒加载 FairChem predictor 单例。
    相同 model_name + device 只加载一次，后续复用。
    """
    cache_key = f"{model_name}::{device}"
    if cache_key not in _predictor_cache:
        logger.info("加载模型 %s（device=%s）…", model_name, device)
        t0 = time.time()
        # 延迟导入，避免未安装 fairchem 时模块级报错
        from fairchem.core import pretrained_mlip
        _predictor_cache[cache_key] = pretrained_mlip.get_predict_unit(
            model_name, device=device
        )
        logger.info("模型加载完成，耗时 %.1fs", time.time() - t0)
    return _predictor_cache[cache_key]

# ──────────────────────────────────────────────
# 结构转换工具
# ──────────────────────────────────────────────

def xyz_str_to_ase(xyz_str: str) -> Atoms:
    """
    XYZ 字符串 → ASE Atoms 对象。
    使用临时文件中转（ase.io.read 需要文件句柄）。
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xyz", delete=False, encoding="utf-8"
    ) as f:
        f.write(xyz_str)
        tmp_path = f.name
    try:
        atoms = ase_read(tmp_path, format="xyz")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return atoms


def pymatgen_to_ase(struct: Structure) -> Atoms:
    """pymatgen Structure → ASE Atoms（保留周期性边界条件）。"""
    return AseAtomsAdaptor.get_atoms(struct)


def ase_to_pymatgen(atoms: Atoms) -> Structure:
    """ASE Atoms → pymatgen Structure。"""
    return AseAtomsAdaptor.get_structure(atoms)


def ase_to_xyz_str(atoms: Atoms, comment: str = "") -> str:
    """ASE Atoms → XYZ 字符串（内存操作）。"""
    buf = io.StringIO()
    ase_write(buf, atoms, format="xyz", comment=comment)
    return buf.getvalue()


def pymatgen_to_cif_str(struct: Structure) -> str:
    """pymatgen Structure → CIF 字符串。"""
    try:
        return str(CifWriter(struct))
    except Exception as e:
        logger.warning("CIF 转换失败：%s", e)
        return ""


def pymatgen_to_poscar_str(struct: Structure, comment: str = "") -> str:
    """pymatgen Structure → POSCAR 字符串。"""
    try:
        return Poscar(
            struct,
            comment=comment or struct.composition.reduced_formula
        ).get_str()
    except Exception as e:
        logger.warning("POSCAR 转换失败：%s", e)
        return ""


def get_conventional(struct: Structure) -> Structure:
    """转换为传统标准结构，失败时返回原结构。"""
    try:
        return SpacegroupAnalyzer(
            struct, symprec=1e-3
        ).get_conventional_standard_structure()
    except Exception:
        return struct

# ──────────────────────────────────────────────
# 弛豫过程回调：收集每步数据
# ──────────────────────────────────────────────

class _RelaxLogger:
    """
    附加到优化器的回调，记录每步的能量和最大力。
    供前端展示收敛曲线使用。
    """
    def __init__(self, atoms: Atoms):
        self._atoms = atoms
        self.steps:    List[int]   = []
        self.energies: List[float] = []
        self.fmax:     List[float] = []

    def __call__(self):
        step = len(self.steps)
        try:
            e    = float(self._atoms.get_potential_energy())
            fmax = float(np.max(np.linalg.norm(self._atoms.get_forces(), axis=1)))
        except Exception:
            return
        self.steps.append(step)
        self.energies.append(round(e, 6))
        self.fmax.append(round(fmax, 6))
        logger.debug("  step=%d  E=%.4f eV  Fmax=%.4f eV/Å", step, e, fmax)

# ──────────────────────────────────────────────
# 核心预测函数
# ──────────────────────────────────────────────

def run_ml_prediction(
    structure_input: Union[str, Structure, Atoms],
    model_name:  str  = "uma-s-1p2",
    task_name:   str  = "omat",
    optimizer:   str  = "FIRE",
    fmax:        float = 0.05,
    max_steps:   int   = 200,
    relax_cell:  bool  = True,
    device:      str   = "cuda",
    save_dir:    Optional[str] = None,
    job_id:      Optional[str] = None,
) -> Dict[str, Any]:
    """
    ★ 对外主接口 ★

    对输入结构执行 ML 弛豫，返回能量、力和优化后结构。

    参数：
      structure_input : XYZ字符串 | pymatgen Structure | ASE Atoms
      model_name      : UMA 模型名称（默认 uma-s-1p2）
      task_name       : 任务类型（Bulk晶体用 omat，默认）
      optimizer       : "FIRE"（晶体推荐）| "LBFGS"
      fmax            : 收敛标准，最大力（eV/Å），默认 0.05
      max_steps       : 最大弛豫步数，默认 200
      relax_cell      : True = 同时弛豫晶胞形状和体积（使用 FrechetCellFilter）
      device          : "cuda" | "cpu"
      save_dir        : 若指定，将结构文件和轨迹保存到此目录
      job_id          : 任务标识符，用于文件命名（默认自动生成）

    返回 dict（可直接序列化为 JSON）：
      status          : "converged" | "not_converged" | "error"
      message         : 状态描述
      model_name      : 使用的模型
      task_name       : 使用的任务
      # ── 能量与力 ──
      energy_eV       : 总能量（eV）
      energy_per_atom : 每原子能量（eV/atom）
      fmax_eV_Ang     : 最大原子力（eV/Å）
      forces          : 每原子力列表 [[fx,fy,fz], ...]（eV/Å）
      # ── 收敛过程 ──
      n_steps         : 实际弛豫步数
      convergence_curve: [{"step":0,"energy":...,"fmax":...}, ...]
      # ── 优化后结构（字符串，前端直接使用）──
      xyz             : XYZ 字符串（3Dmol 渲染）
      cif             : CIF 字符串（下载）
      poscar          : POSCAR 字符串（下载）
      # ── 晶格信息 ──
      lattice         : {a,b,c,alpha,beta,gamma,volume}
      formula         : 化学式
      n_atoms         : 原子数
      # ── 本地文件路径（save_dir 不为 None 时）──
      saved_files     : {"xyz":..., "cif":..., "poscar":..., "traj":...}
    """
    t_start = time.time()

    # ── 参数校验 ──
    if model_name not in SUPPORTED_MODELS:
        return _error_result(f"不支持的模型：{model_name}，可选：{list(SUPPORTED_MODELS)}")
    if task_name not in SUPPORTED_TASKS:
        return _error_result(f"不支持的任务：{task_name}，可选：{list(SUPPORTED_TASKS)}")
    if optimizer not in SUPPORTED_OPTIMIZERS:
        return _error_result(f"不支持的优化器：{optimizer}，可选：{list(SUPPORTED_OPTIMIZERS)}")

    job_id = job_id or f"ml_{int(time.time())}"

    try:
        # ── Step 1：统一转换为 ASE Atoms ──
        atoms = _to_ase_atoms(structure_input)
        logger.info("[%s] 输入结构：%d 原子，化学式 %s",
                    job_id, len(atoms), atoms.get_chemical_formula())

        # ── Step 2：加载模型，设置计算器 ──
        from fairchem.core import FAIRChemCalculator
        predictor = _get_predictor(model_name, device)
        atoms.calc = FAIRChemCalculator(predictor, task_name=task_name)

        # ── Step 3：设置优化器（晶体弛豫使用 FrechetCellFilter）──
        relax_logger = _RelaxLogger(atoms)

        if relax_cell:
            # FrechetCellFilter 同时弛豫原子位置和晶胞
            filtered = FrechetCellFilter(atoms)
            opt = _build_optimizer(optimizer, filtered)
        else:
            opt = _build_optimizer(optimizer, atoms)

        # 附加回调，记录每步数据
        opt.attach(relax_logger, interval=1)

        # ── Step 4：设置轨迹文件（可选）──
        traj_path = None
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            traj_path = str(Path(save_dir) / f"{job_id}.traj")
            traj = Trajectory(traj_path, "w", atoms)
            opt.attach(traj.write, interval=1)

        # ── Step 5：执行弛豫 ──
        logger.info("[%s] 开始弛豫（optimizer=%s, fmax=%.3f, max_steps=%d, relax_cell=%s）",
                    job_id, optimizer, fmax, max_steps, relax_cell)
        converged = opt.run(fmax=fmax, steps=max_steps)

        # ── Step 6：提取结果 ──
        energy      = float(atoms.get_potential_energy())
        forces_arr  = atoms.get_forces()                         # shape (N, 3)
        fmax_val    = float(np.max(np.linalg.norm(forces_arr, axis=1)))
        n_steps     = len(relax_logger.steps)

        logger.info("[%s] 弛豫完成：E=%.4f eV  Fmax=%.4f eV/Å  steps=%d  converged=%s",
                    job_id, energy, fmax_val, n_steps, converged)

        # ── Step 7：转换优化后结构 ──
        pmg_struct   = ase_to_pymatgen(atoms)
        conv_struct  = get_conventional(pmg_struct)
        lat          = conv_struct.lattice
        formula      = conv_struct.composition.reduced_formula
        comment      = f"{formula} | {job_id} | E={energy:.4f}eV"

        xyz_str    = ase_to_xyz_str(atoms, comment=comment)
        cif_str    = pymatgen_to_cif_str(conv_struct)
        poscar_str = pymatgen_to_poscar_str(conv_struct, comment=comment)

        # ── Step 8：保存文件到本地（可选）──
        saved_files: Dict[str, str] = {}
        if save_dir:
            base = Path(save_dir)
            # XYZ
            p = base / f"{job_id}.xyz"
            p.write_text(xyz_str, encoding="utf-8")
            saved_files["xyz"] = str(p)
            # CIF
            p = base / f"{job_id}.cif"
            p.write_text(cif_str, encoding="utf-8")
            saved_files["cif"] = str(p)
            # POSCAR
            p = base / f"{job_id}.POSCAR"
            p.write_text(poscar_str, encoding="utf-8")
            saved_files["poscar"] = str(p)
            # 轨迹
            if traj_path:
                saved_files["traj"] = traj_path
            logger.info("[%s] 文件已保存至 %s", job_id, save_dir)

        # ── 构建返回结果 ──
        return {
            "status":  "converged" if converged else "not_converged",
            "message": "结构弛豫收敛" if converged else f"未在 {max_steps} 步内收敛，返回当前最优结构",
            "model_name": model_name,
            "task_name":  task_name,
            "job_id":     job_id,
            "elapsed_s":  round(time.time() - t_start, 2),
            # ── 能量与力 ──
            "energy_eV":        round(energy, 6),
            "energy_per_atom":  round(energy / len(atoms), 6),
            "fmax_eV_Ang":      round(fmax_val, 6),
            "forces": [
                [round(float(f), 6) for f in row]
                for row in forces_arr.tolist()
            ],
            # ── 收敛过程 ──
            "n_steps": n_steps,
            "convergence_curve": [
                {"step": s, "energy": e, "fmax": f}
                for s, e, f in zip(
                    relax_logger.steps,
                    relax_logger.energies,
                    relax_logger.fmax,
                )
            ],
            # ── 优化后结构字符串 ──
            "xyz":    xyz_str,
            "cif":    cif_str,
            "poscar": poscar_str,
            # ── 晶格信息 ──
            "lattice": {
                "a":     round(lat.a,     4),
                "b":     round(lat.b,     4),
                "c":     round(lat.c,     4),
                "alpha": round(lat.alpha, 2),
                "beta":  round(lat.beta,  2),
                "gamma": round(lat.gamma, 2),
                "volume": round(lat.volume, 4),
            },
            "formula": formula,
            "n_atoms": len(atoms),
            # ── 本地文件路径 ──
            "saved_files": saved_files,
            # ── 后端计算用（路由序列化时排除）──
            "_structure": conv_struct,
        }

    except Exception as exc:
        logger.exception("[%s] ML预测失败：%s", job_id, exc)
        return _error_result(str(exc), job_id=job_id, elapsed=time.time() - t_start)

# ──────────────────────────────────────────────
# 内部辅助函数
# ──────────────────────────────────────────────

def _to_ase_atoms(inp: Union[str, Structure, Atoms]) -> Atoms:
    """统一将各种输入格式转换为 ASE Atoms。"""
    if isinstance(inp, Atoms):
        return inp.copy()
    if isinstance(inp, Structure):
        return pymatgen_to_ase(inp)
    if isinstance(inp, str):
        return xyz_str_to_ase(inp)
    raise TypeError(f"不支持的输入类型：{type(inp)}")


def _build_optimizer(name: str, target: Any) -> Any:
    """根据名称构建 ASE 优化器。"""
    if name == "FIRE":
        return FIRE(target, logfile=None)
    if name == "LBFGS":
        return LBFGS(target, logfile=None)
    raise ValueError(f"未知优化器：{name}")


def _error_result(
    message: str,
    job_id: str = "",
    elapsed: float = 0.0,
) -> Dict[str, Any]:
    """构建统一的错误返回结构。"""
    return {
        "status":  "error",
        "message": message,
        "job_id":  job_id,
        "elapsed_s": round(elapsed, 2),
        "energy_eV": None,
        "energy_per_atom": None,
        "fmax_eV_Ang": None,
        "forces": [],
        "n_steps": 0,
        "convergence_curve": [],
        "xyz": "",
        "cif": "",
        "poscar": "",
        "lattice": {},
        "formula": "",
        "n_atoms": 0,
        "saved_files": {},
        "_structure": None,
    }