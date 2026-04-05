# main.py
from __future__ import annotations

import json
import os
import logging
import secrets
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv
# 引入限流库
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from Analysis import VaspAnalysisDispatcher

# ============================================================
# Logging Setup (安全日志记录)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vasp_api")

app = FastAPI(title="VASP Analysis API", version="2.2.0")

# ============================================================
# Rate Limiting Setup (防刷限流)
# ============================================================
# 根据客户端 IP 地址进行限流
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================
# Security: Authentication (API 密钥认证)
# ============================================================
# 从环境变量获取密码，如果没有设置，则使用默认值（强烈建议在生产环境设置环境变量！）
# 启动方式示例: API_SECRET_KEY="MySuperSecretPassword123!" uvicorn main:app --host 0.0.0.0
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "default_secret_password_please_change")

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    验证前端传来的 Bearer Token (密码)
    """
    # 使用 secrets.compare_digest 防止时序攻击
    if not secrets.compare_digest(credentials.credentials, API_SECRET_KEY):
        logger.warning("Failed authentication attempt.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key or Password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# ============================================================
# CORS (修复了 "null" 漏洞)
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
        # ⚠️ 已经移除了危险的 "null"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"], # 确保允许 Authorization 头
)

# ============================================================
# Request Models (保持不变)
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

class StructureRequest(BaseInputRequest): pass

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

class CohpSummaryRequest(BaseInputRequest):
    n_top_bonds: int = Field(default=20, ge=1, le=500)
    filter_type: Literal["none", "element_pair", "index"] = "none"
    filter_value: Optional[Union[List[Union[str, int]], Tuple[Union[str, int], Union[str, int]]]] = None

class CohpCurvesRequest(BaseInputRequest):
    bond_labels: Optional[List[str]] = None
    erange: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    include_orbitals: bool = False

class CohpExportRequest(BaseInputRequest):
    export_type: Literal["single"] = "single"
    bond_label: str
    erange: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    include_orbitals: bool = True

# ============================================================
# Security: Path Resolution (保持不变)
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
        logger.warning(f"Path traversal attempt blocked: {p}")
        raise PermissionError(f"Access denied. Path outside allowed directories.")

    return p

# ============================================================
# Dispatcher Wrapper (修复了异常信息泄露)
# ============================================================
def _error_payload(message: str, code: int) -> Dict[str, Any]:
    return {"success": False, "code": code, "message": message, "data": {}}

def _parse_dispatcher_json(json_str: str) -> Dict[str, Any]:
    try:
        obj = json.loads(json_str)
        if isinstance(obj, dict) and {"success", "code", "message", "data"}.issubset(obj.keys()):
            return obj
        return {"success": True, "code": 200, "message": "Success", "data": obj if isinstance(obj, dict) else {"raw": obj}}
    except Exception as e:
        logger.error(f"Failed to parse dispatcher JSON: {e}", exc_info=True)
        return _error_payload("Internal server error: Invalid response format.", 500)

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
        return _error_payload("Directory or file not found.", 404) # 模糊化具体路径
    except PermissionError as e:
        return _error_payload(str(e), 403)
    except ValueError as e:
        return _error_payload(str(e), 400)
    except Exception as e:
        # ⚠️ 关键修复：记录真实错误到日志，给前端返回模糊错误，防止泄露服务器信息
        logger.error(f"Task '{task_type}' failed with unexpected error: {e}", exc_info=True)
        return _error_payload("An internal server error occurred during analysis.", 500)

# ============================================================
# Routes
# ============================================================
@app.get("/healthz")
@limiter.limit("60/minute") # 健康检查接口，限制每分钟 60 次
async def healthz(request: Request):
    """健康检查接口无需密码，用于监控系统检查存活状态"""
    return {
        "success": True,
        "code": 200,
        "message": "ok",
        "data": {"service": "VASP Analysis API API is running securely."}
    }

# 👇 注意：所有业务接口都添加了 dependencies=[Depends(verify_api_key)] 进行密码保护
# 👇 并且添加了 @limiter.limit 防止恶意并发请求

@app.post("/api/vasp/structure", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def structure_info(request: Request, req: StructureRequest):
    return run_task("structure_info", req)

@app.post("/api/vasp/dos", dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute") # DOS 解析较慢，限制更严格
async def dos(request: Request, req: DosRequest):
    return run_task(
        "dos",
        req,
        curves=[c.model_dump() for c in req.curves],
        erange=req.erange,
        show_tdos=req.show_tdos,
    )

@app.post("/api/vasp/relax", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def relax(request: Request, req: RelaxRequest):
    return run_task("relax", req, get_site_mag=req.get_site_mag)

@app.post("/api/vasp/cohp/summary", dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def cohp_summary(request: Request, req: CohpSummaryRequest):
    kwargs: Dict[str, Any] = {
        "n_top_bonds": req.n_top_bonds,
        "filter_type": None if req.filter_type == "none" else req.filter_type,
    }
    if req.filter_value is not None:
        kwargs["filter_value"] = req.filter_value
    return run_task("cohp_summary", req, **kwargs)

@app.post("/api/vasp/cohp/curves", dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def cohp_curves(request: Request, req: CohpCurvesRequest):
    kwargs: Dict[str, Any] = {"include_orbitals": req.include_orbitals}
    if req.bond_labels:
        kwargs["bond_labels"] = req.bond_labels
    if req.erange is not None:
        kwargs["erange"] = req.erange
    return run_task("cohp_curves", req, **kwargs)

@app.post("/api/vasp/cohp/export", dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def cohp_export(request: Request, req: CohpExportRequest):
    kwargs: Dict[str, Any] = {
        "export_type": req.export_type,
        "bond_label": req.bond_label,
        "include_orbitals": req.include_orbitals,
        "export_format": "csv" 
    }
    if req.erange is not None:
        kwargs["erange"] = req.erange
    return run_task("cohp_export", req, **kwargs)
