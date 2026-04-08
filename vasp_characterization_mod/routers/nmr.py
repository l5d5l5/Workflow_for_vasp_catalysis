import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from parsers.nmr_parser import get_nmr_data

router = APIRouter(prefix="/nmr", tags=["NMR 核磁共振"])


# ─────────────────────────────────────────────────────────────
#  响应数据模型 —— 化学屏蔽 (CS)
# ─────────────────────────────────────────────────────────────

class MarylandEntry(BaseModel):
    iso_shift: float
    span:      float
    skew:      float

class HaeberlenValues(BaseModel):
    sigma_iso:       float
    delta_sigma_iso: float
    zeta:            float
    eta:             float

class MehringValues(BaseModel):
    sigma_iso: float
    sigma_11:  float
    sigma_22:  float
    sigma_33:  float

class MarylandValues(BaseModel):
    sigma_iso: float
    omega:     float
    kappa:     float

class CSIon(BaseModel):
    ion:              int
    valence_and_core: MarylandEntry
    # 修复 P6：张量分析字段提供默认值，避免缺失时 422
    valence_only:     MarylandEntry       = Field(default_factory=lambda: MarylandEntry(iso_shift=0.0, span=0.0, skew=0.0))
    haeberlen:        HaeberlenValues     = Field(default_factory=lambda: HaeberlenValues(sigma_iso=0.0, delta_sigma_iso=0.0, zeta=0.0, eta=0.0))
    mehring:          MehringValues       = Field(default_factory=lambda: MehringValues(sigma_iso=0.0, sigma_11=0.0, sigma_22=0.0, sigma_33=0.0))
    maryland:         MarylandValues      = Field(default_factory=lambda: MarylandValues(sigma_iso=0.0, omega=0.0, kappa=0.0))
    raw_tensor:       list[list[float]]   = Field(default_factory=list)

class ChemicalShieldingResult(BaseModel):
    n_ions:            int
    core_contribution: dict[str, float]
    # 修复 P8：g0_contribution 允许空列表
    g0_contribution:   list[list[float]]  = Field(default_factory=list)
    ions:              list[CSIon]


# ─────────────────────────────────────────────────────────────
#  响应数据模型 —— 电场梯度 (EFG)
# ─────────────────────────────────────────────────────────────

class PrincipalAxis(BaseModel):
    V_xx:      float
    V_yy:      float
    V_zz:      float
    asymmetry: float

class EFGIon(BaseModel):
    ion:                       int
    cq_mhz:                    float
    eta:                       float
    nuclear_quadrupole_moment: float
    # 修复 P7：principal_axis 提供默认值
    principal_axis:            PrincipalAxis = Field(
        default_factory=lambda: PrincipalAxis(V_xx=0.0, V_yy=0.0, V_zz=0.0, asymmetry=0.0)
    )
    raw_tensor:                list[list[float]] = Field(default_factory=list)

class EFGResult(BaseModel):
    n_ions: int
    ions:   list[EFGIon]


# ─────────────────────────────────────────────────────────────
#  顶层响应模型
# ─────────────────────────────────────────────────────────────

class NMRResponse(BaseModel):
    source:             str
    n_ions:             int
    has_cs:             bool
    has_efg:            bool
    chemical_shielding: ChemicalShieldingResult | None = None
    efg:                EFGResult               | None = None


# ─────────────────────────────────────────────────────────────
#  辅助：统一错误处理
# ─────────────────────────────────────────────────────────────

def _handle_errors(e: Exception) -> HTTPException:
    if isinstance(e, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(e))
    if isinstance(e, ValueError):
        return HTTPException(status_code=422, detail=str(e))
    return HTTPException(status_code=500, detail=f"解析失败: {e}")


# ─────────────────────────────────────────────────────────────
#  路由：上传 OUTCAR 解析 NMR
# ─────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=NMRResponse,
    summary="上传 OUTCAR 解析 NMR 数据",
    description=(
        "自动检测 OUTCAR 中的 NMR 类型：\n"
        "- **化学屏蔽 (CS)**：需 INCAR 设置 `LCHIMAG = .TRUE.`\n"
        "- **电场梯度 (EFG)**：需含四极核元素"
    ),
)
async def parse_nmr_from_upload(
    file: UploadFile = File(..., description="VASP OUTCAR 文件"),
):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "OUTCAR"
        filepath.write_bytes(await file.read())
        try:
            return get_nmr_data(filepath)
        except Exception as e:
            raise _handle_errors(e)


# ─────────────────────────────────────────────────────────────
#  路由：按任务 ID 读取 NMR
# ─────────────────────────────────────────────────────────────

@router.get(
    "/task/{task_id}",
    response_model=NMRResponse,
    summary="按任务 ID 解析 NMR 数据",
)
async def parse_nmr_from_task(task_id: str):
    task_dir = Path(os.environ.get("TASK_DIR", "/data/tasks")) / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail=f"任务目录不存在: '{task_dir}'")
    try:
        return get_nmr_data(task_dir)
    except Exception as e:
        raise _handle_errors(e)
