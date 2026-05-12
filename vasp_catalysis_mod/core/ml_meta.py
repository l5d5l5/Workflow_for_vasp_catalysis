"""
ml_meta.py
───────────────────────────────────────────────
基于 FAIRChem / OCPCalculator 的结构优化与预测服务。

模型路径读取优先级：
1. API 传入 model_name
2. 环境变量 FAIRCHEM_MODEL_PATH
3. 环境变量 ML_MODEL_DIR / ML_DEFAULT_MODEL
4. ML_MODEL_DIR 下的内置推荐模型列表

.env 示例：
ML_MODEL_DIR=E:/workflow/MLIP
ML_DEFAULT_MODEL=esen_30m_oam.pt
ML_DEVICE=cuda

推荐模型优先级：
  1. esen_30m_oam.pt
  2. eqV2_153M_omat_mp_salex.pt
  3. eqV2_86M_omat_mp_salex.pt
  4. eqV2_31M_omat_mp_salex.pt
───────────────────────────────────────────────
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("MLPredictionService")

if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)

logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


# ============================================================
# Constants
# ============================================================

_METAL_ELEMENTS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
}

_COORDINATION_ELEMENTS = {"O", "N"}

_DEFAULT_MLIP_DIR = Path("/data/home/luodh/MLIP")

_MODEL_PRIORITY = [
    "esen_30m_oam.pt",
    "eqV2_153M_omat_mp_salex.pt",
    "eqV2_86M_omat_mp_salex.pt",
    "eqV2_31M_omat_mp_salex.pt",
]


# ============================================================
# Env / Model Path Helpers
# ============================================================

def _clean_env_value(value: Optional[str]) -> str:
    """
    清理 .env 中可能存在的引号与空格。
    """
    if not value:
        return ""

    return value.strip().strip('"').strip("'")


def get_ml_model_dir() -> Path:
    """
    从环境变量读取 ML 模型目录。

    优先级：
    1. ML_MODEL_DIR
    2. 默认集群路径 /data/home/luodh/MLIP
    """
    raw = _clean_env_value(os.getenv("ML_MODEL_DIR"))

    if raw:
        model_dir = Path(raw).expanduser()
    else:
        model_dir = _DEFAULT_MLIP_DIR

    model_dir = model_dir.resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"ML_MODEL_DIR 不存在：{model_dir}")

    if not model_dir.is_dir():
        raise NotADirectoryError(f"ML_MODEL_DIR 不是目录：{model_dir}")

    return model_dir


def get_default_model_name() -> str:
    """
    从环境变量读取默认模型名。
    """
    model_name = _clean_env_value(os.getenv("ML_DEFAULT_MODEL"))

    if model_name:
        return model_name

    return "esen_30m_oam.pt"


def _candidate_model_paths(model_name: Optional[str] = None) -> List[Path]:
    """
    生成候选模型路径列表。

    优先级：
    1. API 传入 model_name
    2. FAIRCHEM_MODEL_PATH
    3. ML_MODEL_DIR / ML_DEFAULT_MODEL
    4. ML_MODEL_DIR 下的内置推荐模型列表
    """
    candidates: List[Path] = []

    model_dir = get_ml_model_dir()

    # 1. API 传入模型名或完整路径
    if model_name:
        name = _clean_env_value(model_name)
        p = Path(name).expanduser()

        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(model_dir / name)

    # 2. 兼容旧变量 FAIRCHEM_MODEL_PATH
    fairchem_model_path = _clean_env_value(os.getenv("FAIRCHEM_MODEL_PATH"))

    if fairchem_model_path:
        candidates.append(Path(fairchem_model_path).expanduser())

    # 3. 默认模型
    default_model = get_default_model_name()
    candidates.append(model_dir / default_model)

    # 4. 备用模型
    for name in _MODEL_PRIORITY:
        p = model_dir / name

        if p not in candidates:
            candidates.append(p)

    return candidates


def resolve_ml_model_path(model_name: Optional[str] = None) -> Path:
    """
    解析最终模型文件路径。

    model_name 支持：
    - None：使用 ML_DEFAULT_MODEL
    - 文件名：esen_30m_oam.pt
    - 绝对路径：E:/workflow/MLIP/esen_30m_oam.pt
    """
    candidates = _candidate_model_paths(model_name)

    for p in candidates:
        p = p.expanduser().resolve()

        if not p.exists():
            continue

        if not p.is_file():
            continue

        # 防止误读空文件或错误文件
        if p.stat().st_size < 1024 * 1024:
            logger.warning("跳过疑似无效模型文件：%s，文件过小。", p)
            continue

        logger.info(
            "找到 ML 模型：%s（%.1f MB）",
            p,
            p.stat().st_size / 1024**2,
        )

        return p

    checked = "\n".join(f"  {x}" for x in candidates)

    raise FileNotFoundError(
        "未找到可用的 FAIRChem 模型文件。\n"
        f"已检查路径：\n{checked}"
    )


def find_local_model(model_name: Optional[str] = None) -> Optional[str]:
    """
    兼容旧代码的模型查找函数。
    """
    try:
        return str(resolve_ml_model_path(model_name))
    except Exception as exc:
        logger.warning("模型查找失败：%s", exc)
        return None


def list_available_models() -> Dict[str, Any]:
    """
    返回当前 ML_MODEL_DIR 下可用模型列表。

    注意：
    - 不返回完整服务器路径
    - 只返回模型文件名和默认模型状态
    """
    model_dir = get_ml_model_dir()
    default_model = get_default_model_name()
    default_model_path = model_dir / default_model

    models = sorted(
        [
            p.name
            for p in model_dir.glob("*.pt")
            if p.is_file()
        ]
    )

    return {
        "model_dir_configured": True,
        "model_dir_name": model_dir.name,
        "default_model": default_model,
        "default_model_exists": default_model_path.exists(),
        "models": models,
        "count": len(models),
    }


# ============================================================
# Calculator Lazy Cache
# ============================================================

_calculator_cache: Dict[str, Any] = {}


def _get_calculator(
    device: str,
    model_name: Optional[str] = None,
) -> Any:
    """
    加载 FAIRChem OCPCalculator。

    缓存 key 包含：
    - device
    - model_path

    这样切换模型时不会误用旧模型。
    """
    from fairchem.core import OCPCalculator

    model_path = resolve_ml_model_path(model_name)
    cache_key = f"{device}::{model_path}"

    if cache_key in _calculator_cache:
        return _calculator_cache[cache_key]

    logger.info("=" * 70)
    logger.info("加载 ML 模型：%s", model_path)
    logger.info("device=%s | cpu=%s", device, device == "cpu")
    logger.info("=" * 70)

    t0 = time.time()

    calc = OCPCalculator(
        checkpoint_path=str(model_path),
        cpu=(device == "cpu"),
    )

    logger.info("✅ 模型加载完成，耗时 %.1fs", time.time() - t0)

    _calculator_cache[cache_key] = calc

    return calc


# ============================================================
# Structure Type Detection
# ============================================================

def _detect_vacuum_axis(atoms: Atoms) -> Optional[int]:
    """
    用笛卡尔投影检测真空层，适用于斜晶胞。
    """
    MIN_VACUUM_ABS = 5.0
    MIN_VACUUM_RATIO = 0.20

    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    cart_pos = atoms.get_positions()

    for axis in range(3):
        cell_vec = cell[axis]
        cell_len = np.linalg.norm(cell_vec)

        if cell_len < 1e-6:
            continue

        if not pbc[axis]:
            return axis

        unit_vec = cell_vec / cell_len
        proj = cart_pos @ unit_vec
        vacuum_abs = cell_len - (proj.max() - proj.min())
        vacuum_frac = vacuum_abs / cell_len

        if vacuum_abs > MIN_VACUUM_ABS and vacuum_frac > MIN_VACUUM_RATIO:
            logger.info(
                "[detect_vacuum] axis=%d vacuum=%.2f Å (%.0f%%)",
                axis,
                vacuum_abs,
                vacuum_frac * 100,
            )
            return axis

    return None


def _is_mof(atoms: Atoms) -> bool:
    """
    简单判断是否可能是 MOF。
    """
    symbols = atoms.get_chemical_symbols()
    elem_set = set(symbols)

    metal_atoms = [s for s in symbols if s in _METAL_ELEMENTS]
    carbon_atoms = [s for s in symbols if s == "C"]

    if not (
        metal_atoms
        and carbon_atoms
        and "H" in elem_set
        and elem_set & _COORDINATION_ELEMENTS
    ):
        return False

    return (
        len(metal_atoms) / len(symbols) < 0.25
        and len(carbon_atoms) / len(metal_atoms) > 2.0
    )


def detect_struct_type(atoms: Atoms) -> str:
    """
    判断结构类型。
    """
    pbc = atoms.get_pbc()
    elem_set = set(atoms.get_chemical_symbols())

    if not any(pbc):
        return "molecule"

    if _detect_vacuum_axis(atoms) is not None:
        return "surface"

    if _is_mof(atoms):
        return "mof"

    has_metal = bool(elem_set & _METAL_ELEMENTS)
    has_organic = bool(elem_set & {"C", "H", "N", "O", "S", "P"})

    if has_organic and not has_metal:
        return "molecular_crystal"

    return "bulk"


# ============================================================
# Structure Conversion Utilities
# ============================================================

def _to_ase_atoms(inp: Union[str, Structure, Atoms]) -> Atoms:
    """
    将输入结构转换为 ASE Atoms。

    支持：
    - ASE Atoms
    - pymatgen Structure
    - CIF 字符串
    - XYZ 字符串
    """
    if isinstance(inp, Atoms):
        return inp.copy()

    if isinstance(inp, Structure):
        return AseAtomsAdaptor.get_atoms(inp)

    if isinstance(inp, str):
        text = inp.strip()

        # 优先尝试 CIF
        if text.startswith("data_") or "_cell_length_a" in text:
            try:
                struct = Structure.from_str(text, fmt="cif")
                return AseAtomsAdaptor.get_atoms(struct)
            except Exception:
                pass

        # 再尝试 XYZ
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xyz",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(inp)
            tmp = f.name

        try:
            return ase_read(tmp, format="xyz")
        finally:
            Path(tmp).unlink(missing_ok=True)

    raise TypeError(f"不支持的输入类型：{type(inp)}")


def _ase_to_xyz_str(atoms: Atoms, comment: str = "") -> str:
    """
    ASE Atoms 转 XYZ 字符串。
    """
    buf = io.StringIO()
    ase_write(buf, atoms, format="xyz", comment=comment)
    return buf.getvalue()


def _try_get_conventional(struct: Structure) -> Structure:
    """
    尝试转 conventional cell，失败则返回原结构。
    """
    try:
        return SpacegroupAnalyzer(
            struct,
            symprec=1e-3,
        ).get_conventional_standard_structure()
    except Exception:
        return struct


def _atoms_to_cif_str(atoms: Atoms) -> str:
    """
    ASE Atoms → CIF 字符串。
    """
    try:
        pmg_struct = AseAtomsAdaptor.get_structure(atoms)
        conv_struct = _try_get_conventional(pmg_struct)
        return str(CifWriter(conv_struct))
    except Exception as exc:
        logger.warning("CIF 转换失败：%s", exc)
        return ""


def _atoms_to_poscar_str(atoms: Atoms) -> str:
    """
    ASE Atoms → POSCAR 字符串。
    """
    try:
        pmg_struct = AseAtomsAdaptor.get_structure(atoms)
        conv_struct = _try_get_conventional(pmg_struct)
        poscar = Poscar(conv_struct)

        if hasattr(poscar, "get_string"):
            return poscar.get_string()

        return poscar.get_str()

    except Exception as exc:
        logger.warning("POSCAR 转换失败：%s", exc)
        return ""


def _get_lattice_dict(atoms: Atoms) -> Dict[str, float]:
    """
    提取晶格参数。
    """
    try:
        pmg_struct = AseAtomsAdaptor.get_structure(atoms)
        conv_struct = _try_get_conventional(pmg_struct)
        lat = conv_struct.lattice

        return {
            "a": round(float(lat.a), 4),
            "b": round(float(lat.b), 4),
            "c": round(float(lat.c), 4),
            "alpha": round(float(lat.alpha), 2),
            "beta": round(float(lat.beta), 2),
            "gamma": round(float(lat.gamma), 2),
            "volume": round(float(lat.volume), 4),
        }

    except Exception:
        return {}


def _write(path: Path, content: str) -> None:
    """
    写文件辅助。
    """
    try:
        path.write_text(content, encoding="utf-8")
        logger.info("已保存 → %s", path)
    except Exception as exc:
        logger.warning("写文件失败 %s：%s", path, exc)


# ============================================================
# Core ML Function
# ============================================================

def run_ml_prediction(
    structure_input: Union[str, Structure, Atoms],
    fmax: float = 0.05,
    max_steps: int = 200,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    job_id: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    执行 ML 预测 / 结构优化。

    参数：
    - structure_input：pymatgen Structure / ASE Atoms / CIF 字符串 / XYZ 字符串
    - fmax：优化收敛阈值
    - max_steps：最大优化步数。若 <= 0，则只做单点能量/力计算
    - device：cuda / cpu
    - save_dir：可选保存目录
    - job_id：任务 ID
    - model_name：模型文件名或绝对路径

    返回：
    - JSON 安全 dict
    - 周期结构优先返回 CIF
    """
    t_start = time.time()
    job_id = job_id or f"ml_{int(time.time())}"

    try:
        model_path = resolve_ml_model_path(model_name)

        atoms = _to_ase_atoms(structure_input)

        logger.info(
            "[%s] 输入：%d 原子，化学式 %s",
            job_id,
            len(atoms),
            atoms.get_chemical_formula(),
        )

        struct_type = detect_struct_type(atoms)
        use_cell_filter = struct_type in ("bulk", "mof", "molecular_crystal")
        is_periodic = any(atoms.get_pbc())

        if max_steps <= 0:
            optimizer_name = "single_point"
        else:
            optimizer_name = "FIRE" if use_cell_filter else "LBFGS"

        logger.info(
            "[%s] 结构类型：%s | 优化器：%s | cell_filter：%s | model=%s",
            job_id,
            struct_type,
            optimizer_name,
            use_cell_filter,
            model_path.name,
        )

        atoms.calc = _get_calculator(device, model_name=model_name)

        step_counter = [0]

        def _count():
            step_counter[0] += 1

        # ====================================================
        # Run optimization or single point
        # ====================================================

        if max_steps > 0:
            target = FrechetCellFilter(atoms) if use_cell_filter else atoms

            opt = (
                FIRE(target, logfile=None)
                if optimizer_name == "FIRE"
                else LBFGS(target, logfile=None)
            )

            opt.attach(_count, interval=1)

            traj = None

            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                traj = Trajectory(str(Path(save_dir) / f"{job_id}.traj"), "w", atoms)
                opt.attach(traj.write, interval=1)

            logger.info(
                "[%s] 开始弛豫：fmax=%.3f, max_steps=%d",
                job_id,
                fmax,
                max_steps,
            )

            converged = bool(opt.run(fmax=fmax, steps=max_steps))

            if traj is not None:
                traj.close()

        else:
            logger.info("[%s] 执行单点能量/力计算。", job_id)
            converged = True

        # ====================================================
        # Energy and forces
        # ====================================================

        energy = float(atoms.get_potential_energy())
        forces_arr = atoms.get_forces()
        force_mags = np.linalg.norm(forces_arr, axis=1)

        fmax_val = float(force_mags.max())
        fmean_val = float(force_mags.mean())

        symbols = atoms.get_chemical_symbols()
        n_atoms = len(atoms)

        logger.info(
            "[%s] 完成：E=%.6f eV | Fmax=%.6f | Fmean=%.6f eV/Å | "
            "steps=%d | converged=%s",
            job_id,
            energy,
            fmax_val,
            fmean_val,
            step_counter[0],
            converged,
        )

        # ====================================================
        # Visualization structure
        # ====================================================

        if is_periodic:
            cif_str = _atoms_to_cif_str(atoms)

            if cif_str:
                vis_format = "cif"
                vis_content = cif_str
            else:
                vis_format = "xyz"
                vis_content = _ase_to_xyz_str(atoms)

            lattice = _get_lattice_dict(atoms)
            formula = atoms.get_chemical_formula()

        else:
            vis_format = "xyz"
            vis_content = _ase_to_xyz_str(atoms)
            lattice = {}
            formula = atoms.get_chemical_formula()

        # ====================================================
        # Optional save
        # ====================================================

        if save_dir:
            base = Path(save_dir)
            base.mkdir(parents=True, exist_ok=True)

            _write(base / f"{job_id}.{vis_format}", vis_content)

            if is_periodic:
                poscar_str = _atoms_to_poscar_str(atoms)

                if poscar_str:
                    _write(base / f"{job_id}.POSCAR", poscar_str)

        # ====================================================
        # Return
        # ====================================================

        optimized_cif = _atoms_to_cif_str(atoms)

        if not optimized_cif:
            raise RuntimeError("优化后的结构无法转换为 CIF 格式。")

        total_energy_eV = round(float(energy), 6)

        # 这里按照前端原本显示的 F_max 口径返回
        total_force_eV_Ang = round(float(fmax_val), 6)

        return {
            "status": "converged" if converged else "not_converged",

            # 前端需要的 3 个核心字段
            "total_energy_eV": total_energy_eV,
            "total_force_eV_Ang": total_force_eV_Ang,
            "optimized_cif": optimized_cif,
        }

    except Exception as exc:
        logger.exception("[%s] ML 任务失败：%s", job_id, exc)

        return {
            "status": "error",
            "message": str(exc),

            # 保持字段存在，方便 main.py 判断和前端兜底
            "total_energy_eV": None,
            "total_force_eV_Ang": None,
            "optimized_cif": "",
        }


# ============================================================
# Service Class for FastAPI
# ============================================================

class MLPredictionService:
    """
    给 FastAPI 路由使用的 ML 服务封装类。

    兼容 main.py 中的调用方式：
    - service.predict(structure=..., model_name=..., fmax=..., steps=...)
    - service.optimize(structure=..., model_name=..., fmax=..., steps=...)
    - service.relax(structure=..., model_name=..., fmax=..., steps=...)
    - service.run(structure=..., model_name=..., fmax=..., steps=...)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        save_dir: Optional[str] = None,
    ):
        self.model_name = model_name or get_default_model_name()
        self.device = device or _clean_env_value(os.getenv("ML_DEVICE")) or "cuda"
        self.save_dir = save_dir or _clean_env_value(os.getenv("ML_SAVE_DIR")) or None

        logger.info(
            "MLPredictionService 初始化：model_name=%s | device=%s | save_dir=%s",
            self.model_name,
            self.device,
            self.save_dir,
        )

    def optimize(
        self,
        structure: Structure,
        model_name: Optional[str] = None,
        fmax: float = 0.05,
        steps: int = 300,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        结构优化。
        """
        return run_ml_prediction(
            structure_input=structure,
            fmax=fmax,
            max_steps=steps,
            device=kwargs.get("device", self.device),
            save_dir=kwargs.get("save_dir", self.save_dir),
            job_id=kwargs.get("job_id"),
            model_name=model_name or self.model_name,
        )

    def relax(
        self,
        structure: Structure,
        model_name: Optional[str] = None,
        fmax: float = 0.05,
        steps: int = 300,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        结构弛豫。行为等同 optimize。
        """
        return self.optimize(
            structure=structure,
            model_name=model_name,
            fmax=fmax,
            steps=steps,
            **kwargs,
        )

    def predict(
        self,
        structure: Structure,
        model_name: Optional[str] = None,
        fmax: float = 0.05,
        steps: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        单点预测。

        注意：
        - predict 固定只做单点能量/力计算
        - 即使 main.py 传入 steps，也不会执行结构优化
        - 真正需要优化请使用 task=optimize 或 task=relax
        """
        return run_ml_prediction(
            structure_input=structure,
            fmax=fmax,
            max_steps=0,
            device=kwargs.get("device", self.device),
            save_dir=kwargs.get("save_dir", self.save_dir),
            job_id=kwargs.get("job_id"),
            model_name=model_name or self.model_name,
        )

    def run(
        self,
        structure: Structure,
        model_name: Optional[str] = None,
        fmax: float = 0.05,
        steps: int = 300,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        兼容 run 方法，默认执行 optimize。
        """
        return self.optimize(
            structure=structure,
            model_name=model_name,
            fmax=fmax,
            steps=steps,
            **kwargs,
        )
