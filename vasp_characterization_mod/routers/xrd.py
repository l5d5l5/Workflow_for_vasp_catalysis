import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from parsers.xrd_parser import get_xrd_data, WAVELENGTHS

router = APIRouter(prefix="/xrd", tags=["XRD 衍射图谱"])


# ─────────────────────────────────────────────────────────────
#  响应数据模型
# ─────────────────────────────────────────────────────────────

class HKLFamily(BaseModel):
    hkl:          str
    multiplicity: int

class XRDPeak(BaseModel):
    theta:        float
    intensity:    float
    hkl:          str
    d_hkl:        float
    hkl_families: list[HKLFamily]

class MaxPeak(BaseModel):
    theta:     float | None
    hkl:       str   | None
    intensity: float | None

class XRDResponse(BaseModel):
    radiation:       str
    wavelength:      float
    formula:         str
    spacegroup:      str
    two_theta_range: list[float]
    total_peaks:     int
    major_peaks:     int
    max_peak:        MaxPeak
    peaks:           list[XRDPeak]
    source:          str


# ─────────────────────────────────────────────────────────────
#  辅助：统一错误处理
# ─────────────────────────────────────────────────────────────

def _handle_errors(e: Exception) -> HTTPException:
    if isinstance(e, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(e))
    if isinstance(e, ValueError):
        return HTTPException(status_code=422, detail=str(e))
    return HTTPException(status_code=500, detail=f"计算失败: {e}")


# ─────────────────────────────────────────────────────────────
#  辅助：辐射源说明（用于文档）
# ─────────────────────────────────────────────────────────────

_RADIATION_DOC = (
    "辐射源名称或波长数值（Å）。\n"
    "支持模糊输入，例如 'cuka'、'Cu'、'MoKa1'，\n"
    "或直接输入波长数值，例如 '1.54184'。\n"
    f"内置名称: {', '.join(WAVELENGTHS.keys())}"
)


# ─────────────────────────────────────────────────────────────
#  路由：上传结构文件计算 XRD
# ─────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=XRDResponse,
    summary="上传结构文件计算 XRD 图谱",
)
async def calculate_xrd_from_upload(
    file:          UploadFile = File(..., description="POSCAR 或 CIF 结构文件"),
    radiation:     str        = Query("CuKa", description=_RADIATION_DOC),
    two_theta_min: float      = Query(10.0,   ge=0,  lt=180, description="衍射角下限（度）"),
    two_theta_max: float      = Query(90.0,   gt=0,  le=180, description="衍射角上限（度）"),
    symprec:       float      = Query(0.1,    ge=0,          description="对称性精度，0 表示不精化"),
):
    suffix = Path(file.filename or "POSCAR").suffix or ""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / f"structure{suffix}"
        filepath.write_bytes(await file.read())
        try:
            return get_xrd_data(
                filepath,
                radiation=radiation,
                two_theta_min=two_theta_min,
                two_theta_max=two_theta_max,
                symprec=symprec,
            )
        except Exception as e:
            raise _handle_errors(e)


# ─────────────────────────────────────────────────────────────
#  路由：按任务 ID 读取服务器已有结构
# ─────────────────────────────────────────────────────────────

@router.get(
    "/task/{task_id}",
    response_model=XRDResponse,
    summary="按任务 ID 计算 XRD 图谱",
)
async def calculate_xrd_from_task(
    task_id:       str,
    radiation:     str   = Query("CuKa", description=_RADIATION_DOC),
    two_theta_min: float = Query(10.0,   ge=0,  lt=180, description="衍射角下限（度）"),
    two_theta_max: float = Query(90.0,   gt=0,  le=180, description="衍射角上限（度）"),
    symprec:       float = Query(0.1,    ge=0,          description="对称性精度，0 表示不精化"),
):
    task_dir = Path(os.environ.get("TASK_DIR", "/data/tasks")) / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail=f"任务目录不存在: '{task_dir}'")
    try:
        return get_xrd_data(
            task_dir,
            radiation=radiation,
            two_theta_min=two_theta_min,
            two_theta_max=two_theta_max,
            symprec=symprec,
        )
    except Exception as e:
        raise _handle_errors(e)


# ─────────────────────────────────────────────────────────────
#  路由：查询所有可用辐射源（供前端下拉列表使用）
# ─────────────────────────────────────────────────────────────

@router.get(
    "/radiations",
    summary="获取所有可用辐射源",
    description="返回 pymatgen 内置的全部辐射源名称及对应波长，供前端选项列表使用。",
)
async def list_radiations():
    return {
        "radiations": [
            {"name": k, "wavelength": round(v, 6)}
            for k, v in WAVELENGTHS.items()
        ]
    }
