# main.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from Analysis import VaspAnalysisDispatcher


app = FastAPI(title="VASP Analysis API", version="2.2.0")


# ============================================================
# CORS
# ============================================================
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env.strip():
    ALLOWED_ORIGINS = [x.strip() for x in allowed_origins_env.split(",") if x.strip()]
else:
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "null",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ============================================================
# Request Models
# ============================================================
class BaseInputRequest(BaseModel):
    workDir: Optional[str] = None
    inputType: Optional[Literal["path"]] = "path"
    inputValue: Optional[str] = None
    save_data: bool = False

    @model_validator(mode="after")
    def validate_path_input(self):
        if not self.workDir and not self.inputValue:
            raise ValueError("workDir or inputValue is required")
        return self


class StructureRequest(BaseInputRequest):
    pass


class DosCurveRequest(BaseModel):
    mode: Literal["element", "site"] = "element"
    element: Optional[str] = None
    site: Optional[int] = None
    orbital: str = "d"

    @model_validator(mode="after")
    def validate_curve(self):
        if self.mode == "element" and not self.element:
            raise ValueError("element is required when mode='element'")
        if self.mode == "site" and (self.site is None or self.site < 1):
            raise ValueError("site must be >= 1 when mode='site'")
        return self


class DosRequest(BaseInputRequest):
    curves: List[DosCurveRequest] = Field(default_factory=list, max_length=20)
    erange: List[float] = Field(default_factory=lambda: [-10.0, 5.0], min_length=2, max_length=2)
    show_tdos: bool = False

    @model_validator(mode="after")
    def validate_erange(self):
        if self.erange[0] >= self.erange[1]:
            raise ValueError("erange[0] must be less than erange[1]")
        return self


class RelaxRequest(BaseInputRequest):
    get_site_mag: bool = False


# ---------- COHP 新拆分：summary ----------
class CohpSummaryRequest(BaseInputRequest):
    n_top_bonds: int = Field(default=20, ge=1, le=500)
    # 注意：这里将 "bond_index" 改为了 "index"，与后端 CohpAnalysis 保持一致
    filter_type: Literal["none", "element_pair", "index"] = "none"
    # element_pair: ["Fe","O"] 或 ("Fe","O")
    # index: ["1","2"] 或 [1,2]
    filter_value: Optional[Union[List[Union[str, int]], Tuple[Union[str, int], Union[str, int]]]] = None


# ---------- COHP 新拆分：curves ----------
class CohpCurvesRequest(BaseInputRequest):
    bond_labels: Optional[List[str]] = None
    erange: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    include_orbitals: bool = False


# ---------- COHP 新拆分：export ----------
class CohpExportRequest(BaseInputRequest):
    export_type: Literal["single"] = "single"
    bond_label: str
    erange: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    include_orbitals: bool = True


# ============================================================
# Security
# ============================================================
def _get_allowed_base_dirs() -> List[Path]:
    raw = os.getenv("ALLOWED_BASE_DIRS", "").strip()
    bases: List[Path] = []

    if raw:
        for item in raw.split(","):
            p = Path(item.strip()).expanduser().resolve()
            if p.exists() and p.is_dir():
                bases.append(p)

    if not bases:
        bases = [Path.cwd().resolve()]

    return bases


ALLOWED_BASE_DIRS = _get_allowed_base_dirs()


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_work_dir(req: BaseInputRequest) -> Path:
    raw_path = (req.workDir or req.inputValue or "").strip()
    if not raw_path:
        raise ValueError("workDir is empty")

    p = Path(raw_path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")

    if not any(_is_subpath(p, base) for base in ALLOWED_BASE_DIRS):
        allowed = ", ".join(str(x) for x in ALLOWED_BASE_DIRS)
        raise PermissionError(f"Access denied: {p}. Allowed base dirs: {allowed}")

    return p


# ============================================================
# Dispatcher Wrapper
# ============================================================
def _error_payload(message: str, code: int) -> Dict[str, Any]:
    return {"success": False, "code": code, "message": message, "data": {}}


def _parse_dispatcher_json(json_str: str) -> Dict[str, Any]:
    try:
        obj = json.loads(json_str)
        if isinstance(obj, dict) and {"success", "code", "message", "data"}.issubset(obj.keys()):
            return obj
        return {"success": True, "code": 200, "message": "Success", "data": obj if isinstance(obj, dict) else {"raw": obj}}
    except Exception:
        return _error_payload("Dispatcher returned invalid JSON", 500)


def run_task(task_type: str, req: BaseInputRequest, **kwargs) -> Dict[str, Any]:
    try:
        work_dir = resolve_work_dir(req)
        dispatch_kwargs = dict(kwargs)
        dispatch_kwargs["save_data"] = req.save_data

        json_resp = VaspAnalysisDispatcher.dispatch(
            task_type=task_type,
            work_dir=work_dir,
            **dispatch_kwargs,
        )
        return _parse_dispatcher_json(json_resp)

    except FileNotFoundError as e:
        return _error_payload(str(e), 404)
    except PermissionError as e:
        return _error_payload(str(e), 403)
    except ValueError as e:
        return _error_payload(str(e), 400)
    except Exception as e:
        return _error_payload(f"Internal server error: {e}", 500)


# ============================================================
# Routes
# ============================================================
@app.get("/healthz")
async def healthz():
    return {
        "success": True,
        "code": 200,
        "message": "ok",
        "data": {
            "service": "VASP Analysis API",
            "dispatcher": "VaspAnalysisDispatcher",
            "allowedBaseDirs": [str(p) for p in ALLOWED_BASE_DIRS],
        },
    }


@app.post("/api/vasp/structure")
async def structure_info(req: StructureRequest):
    return run_task("structure_info", req)


@app.post("/api/vasp/dos")
async def dos(req: DosRequest):
    return run_task(
        "dos",
        req,
        curves=[c.model_dump() for c in req.curves],
        erange=req.erange,
        show_tdos=req.show_tdos,
    )


@app.post("/api/vasp/relax")
async def relax(req: RelaxRequest):
    return run_task("relax", req, get_site_mag=req.get_site_mag)


# ============================================================
# Routes (COHP 部分)
# ============================================================

@app.post("/api/vasp/cohp/summary")
async def cohp_summary(req: CohpSummaryRequest):
    kwargs: Dict[str, Any] = {
        "n_top_bonds": req.n_top_bonds,
        # 如果前端传 "none"，则转为 None 传给底层分析器
        "filter_type": None if req.filter_type == "none" else req.filter_type,
    }
    if req.filter_value is not None:
        kwargs["filter_value"] = req.filter_value
        
    return run_task("cohp_summary", req, **kwargs)


@app.post("/api/vasp/cohp/curves")
async def cohp_curves(req: CohpCurvesRequest):
    kwargs: Dict[str, Any] = {"include_orbitals": req.include_orbitals}
    if req.bond_labels:
        kwargs["bond_labels"] = req.bond_labels
    if req.erange is not None:
        kwargs["erange"] = req.erange
        
    return run_task("cohp_curves", req, **kwargs)


@app.post("/api/vasp/cohp/export")
async def cohp_export(req: CohpExportRequest):
    kwargs: Dict[str, Any] = {
        "export_type": req.export_type,
        "bond_label": req.bond_label,
        "include_orbitals": req.include_orbitals,
        # 强制指定 export_format 为 csv，使底层返回 df.to_dict(orient="list") 供前端拼接
        "export_format": "csv" 
    }
    if req.erange is not None:
        kwargs["erange"] = req.erange
        
    return run_task("cohp_export", req, **kwargs)
