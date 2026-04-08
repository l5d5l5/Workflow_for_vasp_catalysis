"""
/ml 路由 — 基于 FAIRChem OCPCalculator 的结构优化预测

路由列表：
  POST /ml/predict          前端「ML预测」按钮 → 结构优化 + 返回可视化数据
  GET  /ml/model/info       查询当前加载的模型信息
  GET  /ml/model/available  列出集群上所有可用模型文件
"""
import logging
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..deps import get_current_user
from ...core.ml_meta import run_ml_prediction, find_local_model, _MODEL_CANDIDATES

logger = logging.getLogger("catalyst_workbench.ml")
router = APIRouter()


# ─────────────────────────────────────────────────────
# 请求模型
# ─────────────────────────────────────────────────────

class MLPredictRequest(BaseModel):
    structure_str: str  = Field(
        ...,
        description="结构字符串，支持 POSCAR / XYZ / CIF 格式"
    )
    fmt: str = Field(
        "poscar",
        description="输入格式：poscar | xyz | cif"
    )
    fmax: float = Field(
        0.05,
        ge=0.001, le=1.0,
        description="力收敛阈值（eV/Å），默认 0.05"
    )
    max_steps: int = Field(
        200,
        ge=1, le=500,
        description="最大优化步数，默认 200"
    )
    device: str = Field(
        "cuda",
        description="计算设备：cuda | cpu"
    )


# ─────────────────────────────────────────────────────
# 响应模型
# ─────────────────────────────────────────────────────

class EnergyResult(BaseModel):
    total_eV:    float
    per_atom_eV: float

class ForcePerAtom(BaseModel):
    element: str
    fx:   float
    fy:   float
    fz:   float
    fmag: float

class ForcesResult(BaseModel):
    fmax_eV_Ang:  float
    mean_eV_Ang:  float
    per_atom:     List[ForcePerAtom]

class StructureResult(BaseModel):
    format:  str    # "cif" | "xyz"  → 直接传给 3Dmol.js viewer.addModel()
    content: str    # 结构文件字符串
    formula: str
    n_atoms: int
    lattice: Dict[str, float]   # a/b/c/alpha/beta/gamma/volume，分子时为 {}

class RelaxationInfo(BaseModel):
    converged:      bool
    n_steps:        int
    optimizer:      str
    fmax_threshold: float

class MLPredictResponse(BaseModel):
    # ── 状态 ──────────────────────────────────────────
    status:      str            # "converged" | "not_converged" | "error"
    job_id:      str
    elapsed_s:   float
    struct_type: Optional[str]  # "bulk" | "surface" | "molecule" | "mof" | ...

    # ── 前端 quickOptOut 展示字符串 ───────────────────
    # 格式：E = -x.xxx eV | E/atom = -x.xxx eV | Fmax = x.xxx eV/Å | 收敛 ✓
    display: str

    # ── 数值结果 ──────────────────────────────────────
    energy:    Optional[EnergyResult]
    forces:    Optional[ForcesResult]

    # ── 优化后结构（前端直接用于更新可视化）─────────────
    structure: Optional[StructureResult]

    # ── 优化元信息 ────────────────────────────────────
    relaxation: Optional[RelaxationInfo]

    # ── 错误信息（status="error" 时有值）─────────────
    message: Optional[str] = None


# ─────────────────────────────────────────────────────
# 内部辅助
# ─────────────────────────────────────────────────────

def _convert_input(structure_str: str, fmt: str) -> str:
    """
    将非 XYZ 格式转换为 XYZ 字符串，供 ml_meta._to_ase_atoms() 使用。
    ml_meta 内部通过临时文件 + ase_read 解析，
    这里直接把原始字符串透传即可，格式由 fmt 标识。
    """
    # ml_meta._to_ase_atoms 支持 str 输入（按 xyz 解析）
    # 对于 poscar/cif，需要先用 pymatgen 转换为 ASE Atoms
    import os, tempfile
    from ase.io import read as ase_read
    from pymatgen.io.ase import AseAtomsAdaptor

    suffix_map = {
        "poscar": ".vasp", "vasp": ".vasp",
        "xyz":    ".xyz",  "cif":  ".cif",
    }
    suffix = suffix_map.get(fmt.lower(), ".vasp")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(structure_str)
        tmp_path = tmp.name
    try:
        atoms = ase_read(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"结构解析失败: {e}")
    finally:
        os.unlink(tmp_path)
    return atoms


def _build_display(result: dict) -> str:
    """
    从 run_ml_prediction 返回的 dict 构建前端 quickOptOut 展示字符串。
    对应前端：document.getElementById('quickOptOut').textContent = display
    """
    if result["status"] == "error":
        return f"❌ 预测失败：{result.get('message', '未知错误')}"

    energy    = result["energy"]
    forces    = result["forces"]
    relaxation = result["relaxation"]

    converged_str = "收敛 ✓" if relaxation["converged"] else f"未收敛（{relaxation['n_steps']}步）"

    return (
        f"E = {energy['total_eV']:.3f} eV  |  "
        f"E/atom = {energy['per_atom_eV']:.3f} eV  |  "
        f"Fmax = {forces['fmax_eV_Ang']:.3f} eV/Å  |  "
        f"{converged_str}"
    )


def _build_response(result: dict) -> MLPredictResponse:
    """将 run_ml_prediction 的原始 dict 转换为 Pydantic 响应模型。"""
    display = _build_display(result)

    # ── error 情况 ────────────────────────────────────
    if result["status"] == "error":
        return MLPredictResponse(
            status      = "error",
            job_id      = result["job_id"],
            elapsed_s   = result["elapsed_s"],
            struct_type = None,
            display     = display,
            energy      = None,
            forces      = None,
            structure   = None,
            relaxation  = None,
            message     = result.get("message"),
        )

    # ── 正常情况 ──────────────────────────────────────
    raw_forces = result["forces"]
    raw_struct = result["structure"]
    raw_relax  = result["relaxation"]

    return MLPredictResponse(
        status      = result["status"],
        job_id      = result["job_id"],
        elapsed_s   = result["elapsed_s"],
        struct_type = result["struct_type"],
        display     = display,

        energy = EnergyResult(
            total_eV    = result["energy"]["total_eV"],
            per_atom_eV = result["energy"]["per_atom_eV"],
        ),

        forces = ForcesResult(
            fmax_eV_Ang = raw_forces["fmax_eV_Ang"],
            mean_eV_Ang = raw_forces["mean_eV_Ang"],
            per_atom    = [ForcePerAtom(**f) for f in raw_forces["per_atom"]],
        ),

        structure = StructureResult(
            format  = raw_struct["format"],
            content = raw_struct["content"],
            formula = raw_struct["formula"],
            n_atoms = raw_struct["n_atoms"],
            lattice = raw_struct["lattice"],
        ),

        relaxation = RelaxationInfo(
            converged      = raw_relax["converged"],
            n_steps        = raw_relax["n_steps"],
            optimizer      = raw_relax["optimizer"],
            fmax_threshold = raw_relax["fmax_threshold"],
        ),
    )


# ─────────────────────────────────────────────────────
# 路由
# ─────────────────────────────────────────────────────

@router.post(
    "/predict",
    summary="ML 结构优化预测（前端「ML预测」按钮）",
    response_model=MLPredictResponse,
)
def predict(
    req:      MLPredictRequest,
    username: str = Depends(get_current_user),
) -> MLPredictResponse:
    """
    接收当前结构 → FAIRChem OCPCalculator 弛豫 → 返回优化结果。

    前端对接说明：
      1. display     → document.getElementById('quickOptOut').textContent
      2. structure.format + structure.content → viewer.addModel(content, format)
      3. structure.lattice → 更新 a/b/c/α/β/γ/V 信息格
      4. structure.formula / n_atoms → 更新 Formula / Atoms 信息格
    """
    logger.info(
        "用户 %s 发起 ML 预测，fmt=%s fmax=%.3f max_steps=%d device=%s",
        username, req.fmt, req.fmax, req.max_steps, req.device,
    )

    # 解析输入结构 → ASE Atoms
    atoms = _convert_input(req.structure_str, req.fmt)

    # 调用核心预测函数
    result = run_ml_prediction(
        structure_input = atoms,
        fmax            = req.fmax,
        max_steps       = req.max_steps,
        device          = req.device,
        job_id          = f"{username}_{int(__import__('time').time())}",
    )

    return _build_response(result)


@router.get(
    "/model/info",
    summary="查询当前加载的模型信息",
)
def model_info():
    """
    返回当前集群上找到的模型文件路径及大小。
    前端可用于在 UI 上展示「当前使用模型：esen_30m_oam」。
    """
    from pathlib import Path
    model_path = find_local_model()
    if not model_path:
        return {
            "available": False,
            "model_path": None,
            "model_name": None,
            "size_mb": None,
            "message": "未找到可用模型文件，请检查集群路径 /data/home/luodh/MLIP/",
        }
    p = Path(model_path)
    return {
        "available":  True,
        "model_path": model_path,
        "model_name": p.name,
        "size_mb":    round(p.stat().st_size / 1024 ** 2, 1),
        "message":    "模型就绪",
    }


@router.get(
    "/model/available",
    summary="列出集群上所有可用模型文件",
)
def list_available_models():
    """
    扫描所有候选路径，返回存在的模型文件列表。
    用于前端「选择模型」下拉框（如后续支持多模型切换）。
    """
    from pathlib import Path
    models = []
    for path_str in _MODEL_CANDIDATES:
        if not path_str:
            continue
        p = Path(path_str)
        models.append({
            "path":      path_str,
            "name":      p.name,
            "exists":    p.exists(),
            "size_mb":   round(p.stat().st_size / 1024 ** 2, 1) if p.exists() else None,
            "is_active": p.exists() and p.stat().st_size > 1024 * 1024 * 10,
        })
    return {"models": models}