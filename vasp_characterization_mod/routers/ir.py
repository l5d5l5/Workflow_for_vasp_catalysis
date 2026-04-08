import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..parsers.ir_parser import get_ir_data

router = APIRouter(prefix="/ir", tags=["IR 红外光谱"])


# ── 响应模型 ──────────────────────────────────────────────────

class IRPeak(BaseModel):
    mode:         int
    frequency:    float
    intensity:    float
    is_imaginary: bool

class IRResponse(BaseModel):
    total_modes:   int
    active_peaks:  int
    max_frequency: float
    peaks:         list[IRPeak]
    source:        str


# ── 路由 ──────────────────────────────────────────────────────

@router.post("/upload", response_model=IRResponse, summary="上传 OUTCAR 解析 IR 光谱")
async def parse_ir_from_upload(
    file: UploadFile = File(..., description="VASP OUTCAR 文件")
):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "OUTCAR"
        filepath.write_bytes(await file.read())
        try:
            return get_ir_data(filepath)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"解析失败: {e}")


@router.get("/task/{task_id}", response_model=IRResponse, summary="按任务 ID 读取 IR 光谱")
async def parse_ir_from_task(task_id: str):
    task_dir = Path(os.environ.get("TASK_DIR", "/data/tasks")) / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail=f"任务目录不存在: '{task_dir}'")
    try:
        return get_ir_data(task_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败: {e}")
