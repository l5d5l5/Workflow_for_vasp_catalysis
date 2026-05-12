"""
main.py
───────────────────────────────────────────────
Secure Crystal Structure Backend API

稳定版策略：
1. 不大改业务逻辑，只调整限流、鉴权、异常处理、FRP 客户端识别。
2. FRP TCP 转发下，FastAPI 通常看不到同事真实公网 IP。
3. 限流优先按 Bearer Token，其次按 request.client.host。
4. 业务接口使用 Bearer Token 鉴权。
5. /healthz、/whoami、/docs、/openapi.json 不需要鉴权。
6. 所有业务接口统一返回：
   {
       "code": 1,
       "msg": "success",
       "time": "1775095470",
       "data": {}
   }
───────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import time
import logging
import secrets
import hashlib
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    Request,
    UploadFile,
    File,
    Form,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


# ============================================================
# Load .env
# ============================================================

load_dotenv()


# ============================================================
# Logging Setup
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("crystal_structure_api")


# ============================================================
# Optional numpy support for JSON serialization
# ============================================================

try:
    import numpy as np
except Exception:
    np = None


# ============================================================
# Internal Service Imports
# ============================================================

try:
    from utils.structure_utils import parse_supercell_matrix
except Exception:
    try:
        from .utils.structure_utils import parse_supercell_matrix
    except Exception:
        parse_supercell_matrix = None


try:
    from search import MPQueryService
except Exception:
    try:
        from .search import MPQueryService
    except Exception:
        MPQueryService = None


try:
    from ml_meta import MLPredictionService, list_available_models
except Exception:
    try:
        from .ml_meta import MLPredictionService, list_available_models
    except Exception:
        MLPredictionService = None
        list_available_models = None


try:
    from structure_modify import StructureModify
except Exception:
    try:
        from .structure_modify import StructureModify
    except Exception:
        StructureModify = None


try:
    from structure_io import (
        structure_to_cif,
        build_cif_payload,
        build_export_payload,
    )
except Exception:
    try:
        from .structure_io import (
            structure_to_cif,
            build_cif_payload,
            build_export_payload,
        )
    except Exception:
        structure_to_cif = None
        build_cif_payload = None
        build_export_payload = None


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="Crystal Structure Backend API",
    description="晶体结构平台后端 API。",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ============================================================
# Rate Limiting Setup
# ============================================================

def get_rate_limit_key(request: Request) -> str:
    """
    FRP TCP 转发适配版限流 key。

    优先级：
    1. Authorization Bearer Token 的 hash；
    2. slowapi 默认 remote address；
    3. request.client.host；
    4. unknown。

    说明：
    - FRP TCP 转发通常不会传递同事真实公网 IP；
    - 后端看到的 client_host 大概率是 node1/frpc 侧地址；
    - 所以业务接口不要只按 IP 限流。
    """
    auth_header = request.headers.get("authorization", "")

    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if token:
            token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
            return f"token:{token_hash}"

    remote_addr = get_remote_address(request)
    if remote_addr:
        return f"ip:{remote_addr}"

    if request.client and request.client.host:
        return f"client:{request.client.host}"

    return "unknown"


limiter = Limiter(key_func=get_rate_limit_key)
app.state.limiter = limiter

RATE_LIMIT_HEALTH = os.getenv("RATE_LIMIT_HEALTH", "1000/minute")
RATE_LIMIT_SEARCH = os.getenv("RATE_LIMIT_SEARCH", "1000/minute")
RATE_LIMIT_STRUCTURE = os.getenv("RATE_LIMIT_STRUCTURE", "1000/minute")
RATE_LIMIT_ML = os.getenv("RATE_LIMIT_ML", "1000/minute")


# ============================================================
# Security: Authentication
# ============================================================

REQUIRE_API_AUTH = os.getenv("REQUIRE_API_AUTH", "true").strip().lower() == "true"

API_SECRET_KEY = os.getenv(
    "API_SECRET_KEY",
    "default_secret_password_please_change",
).strip()

security = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """
    Bearer Token 鉴权。

    与先前稳定版本保持一致：
    - 使用 secrets.compare_digest；
    - 鉴权失败时抛 HTTPException；
    - /docs、/healthz、/whoami 不绑定该依赖，不会被拦截。
    """
    if not REQUIRE_API_AUTH:
        return "auth_disabled"

    if credentials is None or not credentials.credentials:
        logger.warning("Authentication failed: missing token.")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key or Bearer Token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(credentials.credentials, API_SECRET_KEY):
        logger.warning("Authentication failed: invalid token.")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key or Bearer Token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials


# ============================================================
# CORS
# ============================================================

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()

if allowed_origins_env:
    ALLOWED_ORIGINS = [
        x.strip()
        for x in allowed_origins_env.split(",")
        if x.strip()
    ]
else:
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ============================================================
# Request Logging Middleware
# ============================================================

@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    """
    轻量请求日志，用于排查 FRP 转发后是否进入 FastAPI。

    如果看到 REQ 但没有 RESP，说明请求进入应用后卡住。
    如果 REQ 都没有，说明请求没有进入 FastAPI。
    """
    start = time.time()
    client_host = request.client.host if request.client else "-"

    logger.info(
        "REQ method=%s path=%s client=%s",
        request.method,
        request.url.path,
        client_host,
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed_ms = int((time.time() - start) * 1000)

        logger.error(
            "REQ_FAILED method=%s path=%s client=%s elapsed_ms=%s error=%s",
            request.method,
            request.url.path,
            client_host,
            elapsed_ms,
            exc,
            exc_info=True,
        )
        raise

    elapsed_ms = int((time.time() - start) * 1000)

    logger.info(
        "RESP method=%s path=%s client=%s status=%s elapsed_ms=%s",
        request.method,
        request.url.path,
        client_host,
        response.status_code,
        elapsed_ms,
    )

    return response


# ============================================================
# Unified Response
# ============================================================

class BizCode(IntEnum):
    SUCCESS = 1
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    VALIDATION_ERROR = 422
    RATE_LIMITED = 429
    SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503


def now_ts() -> str:
    return str(int(time.time()))


def make_response(
    code: int,
    msg: str,
    data: Any = None,
) -> Dict[str, Any]:
    return {
        "code": int(code),
        "msg": msg,
        "time": now_ts(),
        "data": data if data is not None else {},
    }


def ok(
    msg: str = "success",
    data: Any = None,
) -> Dict[str, Any]:
    return make_response(BizCode.SUCCESS, msg, data)


def fail(
    code: int,
    msg: str,
    data: Any = None,
) -> Dict[str, Any]:
    return make_response(code, msg, data)


class APIBizError(Exception):
    def __init__(
        self,
        code: int,
        msg: str,
        error_type: str = "BUSINESS_ERROR",
        status_code: int = 400,
        data: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.msg = msg
        self.error_type = error_type
        self.status_code = status_code
        self.data = data or {}
        super().__init__(msg)

    def to_response(self) -> Dict[str, Any]:
        payload = {
            "error_type": self.error_type,
            **self.data,
        }
        return fail(self.code, self.msg, payload)


# ============================================================
# Global Exception Handlers
# ============================================================

@app.exception_handler(APIBizError)
async def api_biz_error_handler(request: Request, exc: APIBizError):
    logger.warning(
        "Business error: path=%s code=%s msg=%s type=%s",
        request.url.path,
        exc.code,
        exc.msg,
        exc.error_type,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_response(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        "HTTP error: path=%s status=%s detail=%s",
        request.url.path,
        exc.status_code,
        exc.detail,
    )

    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        code = BizCode.UNAUTHORIZED
        msg = "认证失败，请检查 API Token。"
        error_type = "AUTH_FAILED"
    elif exc.status_code == status.HTTP_403_FORBIDDEN:
        code = BizCode.FORBIDDEN
        msg = "没有权限访问该接口。"
        error_type = "FORBIDDEN"
    elif exc.status_code == status.HTTP_404_NOT_FOUND:
        code = BizCode.NOT_FOUND
        msg = "请求的资源不存在。"
        error_type = "NOT_FOUND"
    elif exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
        code = BizCode.VALIDATION_ERROR
        msg = "请求参数校验失败。"
        error_type = "VALIDATION_ERROR"
    elif exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        code = BizCode.RATE_LIMITED
        msg = "请求过于频繁，请稍后再试。"
        error_type = "RATE_LIMITED"
    elif exc.status_code >= 500:
        code = BizCode.SERVER_ERROR
        msg = "服务器内部错误，请稍后重试。"
        error_type = "SERVER_ERROR"
    else:
        code = BizCode.BAD_REQUEST
        msg = str(exc.detail) if exc.detail else "请求参数错误。"
        error_type = "HTTP_ERROR"

    return JSONResponse(
        status_code=exc.status_code,
        content=fail(
            code,
            msg,
            {
                "error_type": error_type,
            },
        ),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        "Request validation failed: path=%s errors=%s",
        request.url.path,
        exc.errors(),
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=fail(
            BizCode.VALIDATION_ERROR,
            "请求参数校验失败，请检查必填字段和字段类型。",
            {
                "error_type": "REQUEST_VALIDATION_ERROR",
                "errors": exc.errors(),
            },
        ),
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(
        "Rate limit exceeded: path=%s client=%s key=%s",
        request.url.path,
        request.client.host if request.client else "-",
        get_rate_limit_key(request),
    )

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=fail(
            BizCode.RATE_LIMITED,
            "请求过于频繁，请稍后再试。",
            {
                "error_type": "RATE_LIMIT_EXCEEDED",
            },
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled server error: path=%s error=%s",
        request.url.path,
        exc,
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=fail(
            BizCode.SERVER_ERROR,
            "服务器内部错误，请稍后重试。",
            {
                "error_type": "INTERNAL_SERVER_ERROR",
            },
        ),
    )


# ============================================================
# Config Helpers
# ============================================================

def _get_max_upload_bytes() -> int:
    raw = os.getenv("MAX_CIF_UPLOAD_MB", "20").strip()

    try:
        mb = float(raw)
    except Exception:
        mb = 20.0

    return int(mb * 1024 * 1024)


MAX_UPLOAD_BYTES = _get_max_upload_bytes()


# ============================================================
# JSON Safe Helper
# ============================================================

def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, Structure):
        return {
            "cif": _structure_to_cif(obj),
        }

    if isinstance(obj, dict):
        return {
            str(k): _json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        return [
            _json_safe(x)
            for x in obj
        ]

    return str(obj)


# ============================================================
# Structure Utilities
# ============================================================

async def _read_upload_text(file: UploadFile) -> str:
    content = await file.read()

    if not content:
        raise APIBizError(
            code=BizCode.BAD_REQUEST,
            msg="上传的 CIF 文件为空。",
            error_type="EMPTY_FILE",
            status_code=status.HTTP_400_BAD_REQUEST,
            data={
                "filename": file.filename,
            },
        )

    if len(content) > MAX_UPLOAD_BYTES:
        raise APIBizError(
            code=BizCode.BAD_REQUEST,
            msg=f"上传文件过大，最大允许 {MAX_UPLOAD_BYTES} bytes。",
            error_type="FILE_TOO_LARGE",
            status_code=status.HTTP_400_BAD_REQUEST,
            data={
                "filename": file.filename,
                "max_bytes": MAX_UPLOAD_BYTES,
                "actual_bytes": len(content),
            },
        )

    for encoding in ["utf-8", "latin-1", "gbk"]:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise APIBizError(
        code=BizCode.BAD_REQUEST,
        msg="无法解码上传文件，请确认 CIF 文件编码。",
        error_type="FILE_DECODE_ERROR",
        status_code=status.HTTP_400_BAD_REQUEST,
        data={
            "filename": file.filename,
        },
    )


async def _read_cif_structure(file: UploadFile) -> Structure:
    text = await _read_upload_text(file)

    try:
        return Structure.from_str(text, fmt="cif")
    except Exception as exc:
        logger.warning("Failed to parse CIF file '%s': %s", file.filename, exc)

        raise APIBizError(
            code=BizCode.BAD_REQUEST,
            msg="CIF 文件解析失败，请检查文件格式。",
            error_type="CIF_PARSE_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            data={
                "filename": file.filename,
            },
        )


def _structure_to_cif(structure: Structure) -> str:
    try:
        if structure_to_cif is not None:
            return structure_to_cif(structure)

        return str(CifWriter(structure))

    except Exception as exc:
        logger.error("Failed to convert Structure to CIF: %s", exc, exc_info=True)

        raise APIBizError(
            code=BizCode.SERVER_ERROR,
            msg="结构转换为 CIF 失败。",
            error_type="CIF_CONVERT_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _build_cif_payload(structure: Structure) -> Dict[str, Any]:
    try:
        if build_cif_payload is not None:
            payload = build_cif_payload(structure)

            if isinstance(payload, dict) and payload.get("cif"):
                return {
                    "cif": payload.get("cif"),
                }

        return {
            "cif": _structure_to_cif(structure),
        }

    except APIBizError:
        raise
    except Exception as exc:
        logger.error("Failed to build CIF payload: %s", exc, exc_info=True)

        raise APIBizError(
            code=BizCode.SERVER_ERROR,
            msg="结构转换为 CIF 失败。",
            error_type="CIF_CONVERT_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _parse_matrix(matrix: str) -> Union[List[int], List[List[float]]]:
    matrix = matrix.strip()

    if not matrix:
        raise APIBizError(
            code=BizCode.BAD_REQUEST,
            msg="矩阵参数不能为空。",
            error_type="EMPTY_MATRIX",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if parse_supercell_matrix is not None:
        try:
            parsed = parse_supercell_matrix(matrix)
            return _json_safe(parsed)
        except Exception:
            pass

    if matrix.startswith("["):
        try:
            value = json.loads(matrix)

            if (
                isinstance(value, list)
                and len(value) == 3
                and all(isinstance(row, list) and len(row) == 3 for row in value)
            ):
                return value

            raise ValueError("matrix is not 3x3")

        except Exception:
            raise APIBizError(
                code=BizCode.BAD_REQUEST,
                msg='转换矩阵格式错误，请传入 3x3 JSON 矩阵，例如 [[1,0,0],[0,1,0],[0,0,1]]。',
                error_type="INVALID_TRANSFORM_MATRIX",
                status_code=status.HTTP_400_BAD_REQUEST,
                data={
                    "input": matrix,
                },
            )

    cleaned = (
        matrix.lower()
        .replace("×", "x")
        .replace("*", "x")
        .replace(",", " ")
        .replace("x", " ")
    )

    parts = [p for p in cleaned.split() if p]

    if len(parts) == 3:
        try:
            nums = [int(p) for p in parts]

            if any(n <= 0 for n in nums):
                raise ValueError("non-positive factor")

            return nums

        except Exception:
            raise APIBizError(
                code=BizCode.BAD_REQUEST,
                msg='超胞矩阵格式错误，请使用正整数格式，例如 "2x2x1"。',
                error_type="INVALID_SUPERCELL_MATRIX",
                status_code=status.HTTP_400_BAD_REQUEST,
                data={
                    "input": matrix,
                },
            )

    raise APIBizError(
        code=BizCode.BAD_REQUEST,
        msg='矩阵格式错误，支持 "2x2x1" 或 3x3 JSON 矩阵。',
        error_type="INVALID_MATRIX_FORMAT",
        status_code=status.HTTP_400_BAD_REQUEST,
        data={
            "input": matrix,
        },
    )


def _get_structure_modifier(structure: Structure) -> Any:
    if StructureModify is None:
        raise APIBizError(
            code=BizCode.SERVICE_UNAVAILABLE,
            msg="结构修改服务不可用，请检查 structure_modify.py 配置。",
            error_type="STRUCTURE_MODIFY_SERVICE_NOT_AVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        return StructureModify(structure)
    except Exception as exc:
        logger.error("Failed to initialize StructureModify: %s", exc, exc_info=True)

        raise APIBizError(
            code=BizCode.SERVER_ERROR,
            msg="结构修改器初始化失败。",
            error_type="STRUCTURE_MODIFIER_INIT_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ============================================================
# Materials Project Service Wrapper
# ============================================================

_MP_SERVICE_INSTANCE = None
_MP_SERVICE_INIT_ERROR = None


def _get_mp_service() -> Any:
    global _MP_SERVICE_INSTANCE
    global _MP_SERVICE_INIT_ERROR

    if _MP_SERVICE_INSTANCE is not None:
        return _MP_SERVICE_INSTANCE

    if MPQueryService is None:
        raise APIBizError(
            code=BizCode.SERVICE_UNAVAILABLE,
            msg="结构库查询服务不可用，请检查后端 MPQueryService 配置。",
            error_type="MP_SERVICE_NOT_AVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        try:
            _MP_SERVICE_INSTANCE = MPQueryService(
                only_stable=False,
                max_results=10,
            )
        except TypeError:
            mp_api_key = os.getenv("MP_API_KEY", "").strip()

            if not mp_api_key:
                raise RuntimeError("MP_API_KEY is missing.")

            _MP_SERVICE_INSTANCE = MPQueryService(
                api_key=mp_api_key,
                only_stable=False,
                max_results=10,
            )

        _MP_SERVICE_INIT_ERROR = None
        logger.info("MPQueryService initialized successfully.")

        return _MP_SERVICE_INSTANCE

    except Exception as exc:
        _MP_SERVICE_INIT_ERROR = str(exc)
        logger.error("Failed to initialize MPQueryService: %s", exc, exc_info=True)

        raise APIBizError(
            code=BizCode.SERVICE_UNAVAILABLE,
            msg="结构库查询服务初始化失败。",
            error_type="MP_SERVICE_INIT_FAILED",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            data={
                "detail": _MP_SERVICE_INIT_ERROR,
            },
        )


def _call_mp_search(service: Any, query: str, limit: int) -> Any:
    method = None
    method_name = None

    for name in ["search", "query", "query_structures", "search_structures"]:
        if hasattr(service, name):
            method = getattr(service, name)
            method_name = name
            break

    if method is None:
        raise APIBizError(
            code=BizCode.SERVICE_UNAVAILABLE,
            msg="结构库查询方法不可用，请检查 MPQueryService。",
            error_type="MP_SEARCH_METHOD_NOT_FOUND",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        if method_name == "search":
            return method(query=query, limit=limit)

        try:
            return method(raw_input=query, limit=limit)
        except TypeError:
            return method(query, limit)

    except Exception as exc:
        logger.error("Materials Project search failed: %s", exc, exc_info=True)

        raise APIBizError(
            code=BizCode.SERVICE_UNAVAILABLE,
            msg="结构库查询失败，请稍后重试。",
            error_type="MP_SEARCH_FAILED",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            data={
                "detail": str(exc),
            },
        )


def _extract_structure_from_mp_item(item: Any) -> Optional[Structure]:
    if isinstance(item, dict):
        structure = item.get("structure")
    else:
        structure = getattr(item, "structure", None)

    if isinstance(structure, Structure):
        return structure

    return None


def _extract_mp_field(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)

    return getattr(item, key, default)


# ============================================================
# ML Service Wrapper
# ============================================================

def _get_ml_service(model_name: Optional[str] = None) -> Any:
    if MLPredictionService is None:
        raise APIBizError(
            code=BizCode.SERVICE_UNAVAILABLE,
            msg="ML 预测服务不可用，请检查后端 MLPredictionService 配置。",
            error_type="ML_SERVICE_NOT_AVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        return MLPredictionService(model_name=model_name)
    except TypeError:
        try:
            return MLPredictionService()
        except Exception as exc:
            logger.error("Failed to initialize MLPredictionService: %s", exc, exc_info=True)

            raise APIBizError(
                code=BizCode.SERVICE_UNAVAILABLE,
                msg="ML 预测服务初始化失败。",
                error_type="ML_SERVICE_INIT_FAILED",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
    except Exception as exc:
        logger.error("Failed to initialize MLPredictionService: %s", exc, exc_info=True)

        raise APIBizError(
            code=BizCode.SERVICE_UNAVAILABLE,
            msg="ML 预测服务初始化失败。",
            error_type="ML_SERVICE_INIT_FAILED",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


def _normalize_ml_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        output = dict(result)

        for key in [
            "structure",
            "optimized_structure",
            "relaxed_structure",
            "final_structure",
        ]:
            value = output.get(key)

            if isinstance(value, Structure):
                output[f"{key}_cif"] = _structure_to_cif(value)
                output.pop(key, None)

        return _json_safe(output)

    if isinstance(result, Structure):
        return {
            "optimized_structure_cif": _structure_to_cif(result),
        }

    return {
        "raw_result": _json_safe(result),
    }


def _run_ml_prediction(
    structure: Structure,
    task: str,
    model_name: Optional[str],
    fmax: float,
    steps: int,
) -> Dict[str, Any]:
    service = _get_ml_service(model_name=model_name)

    if task in {"optimize", "relax"}:
        method_candidates = ["optimize", "relax", "run", "predict"]
    else:
        method_candidates = ["predict", "run"]

    last_error: Optional[Exception] = None

    for method_name in method_candidates:
        method = getattr(service, method_name, None)

        if method is None:
            continue

        try:
            result = method(
                structure=structure,
                model_name=model_name,
                fmax=fmax,
                steps=steps,
            )
            return _normalize_ml_result(result)

        except TypeError:
            try:
                result = method(structure)
                return _normalize_ml_result(result)
            except Exception as exc:
                last_error = exc

        except Exception as exc:
            last_error = exc

    logger.error("Failed to run ML task. Last error: %s", last_error)

    raise APIBizError(
        code=BizCode.SERVICE_UNAVAILABLE,
        msg="ML 任务执行失败，请检查模型配置或输入结构。",
        error_type="ML_TASK_FAILED",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


# ============================================================
# Routes: Basic
# ============================================================

@app.get("/")
@limiter.limit(RATE_LIMIT_HEALTH)
async def root(request: Request):
    return ok(
        "Crystal Structure Backend API is running.",
        {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "healthz": "/healthz",
            "whoami": "/whoami",
            "format_policy": "CIF in, CIF out",
            "auth_required": REQUIRE_API_AUTH,
            "response_format": {
                "code": 1,
                "msg": "success",
                "time": now_ts(),
                "data": {},
            },
        },
    )


@app.get("/healthz")
@limiter.limit(RATE_LIMIT_HEALTH)
async def healthz(request: Request):
    return ok(
        "ok",
        {
            "service": "Crystal Structure Backend API",
            "status": "running",
            "format_policy": "CIF in, CIF out",
            "auth_required": REQUIRE_API_AUTH,
        },
    )


@app.get("/whoami")
@limiter.limit(RATE_LIMIT_HEALTH)
async def whoami(request: Request):
    """
    FRP / 反向代理诊断接口。

    用来确认后端 FastAPI 实际看到的客户端地址。
    FRP TCP 转发下，这里通常不是同事公网 IP。
    """
    return ok(
        "ok",
        {
            "client_host": request.client.host if request.client else None,
            "client_port": request.client.port if request.client else None,
            "x_forwarded_for": request.headers.get("x-forwarded-for"),
            "x_real_ip": request.headers.get("x-real-ip"),
            "authorization_present": bool(request.headers.get("authorization")),
            "user_agent": request.headers.get("user-agent"),
            "rate_limit_key": get_rate_limit_key(request),
        },
    )


# ============================================================
# Routes: Materials Project Search
# ============================================================

@app.get(
    "/api/structures/search",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT_SEARCH)
async def search_structures(
    request: Request,
    query: str = Query(..., description="查询关键词，例如 Pt、TiO2、mp-149"),
    limit: int = Query(10, ge=1, le=10, description="最多返回数量，最大为 10"),
):
    service = _get_mp_service()
    safe_limit = max(1, min(int(limit), 10))

    raw_results = _call_mp_search(
        service,
        query=query,
        limit=safe_limit,
    )

    if isinstance(raw_results, dict) and "items" in raw_results:
        items = raw_results.get("items") or []
        clean_items = []

        for item in items[:safe_limit]:
            if not isinstance(item, dict):
                continue

            clean_items.append(
                {
                    "formula": item.get("formula") or item.get("formula_pretty"),
                    "material_id": item.get("material_id"),
                    "band_gap": item.get("band_gap"),
                    "cif": item.get("cif"),
                }
            )

        return ok(
            "结构库查询完成。",
            {
                "query": query,
                "limit": safe_limit,
                "count": len(clean_items),
                "items": clean_items,
            },
        )

    if isinstance(raw_results, dict):
        items = (
            raw_results.get("results")
            or raw_results.get("data")
            or []
        )
    else:
        items = raw_results

    results = []

    for item in list(items)[:safe_limit]:
        try:
            if isinstance(item, dict) and item.get("cif"):
                results.append(
                    {
                        "formula": item.get("formula") or item.get("formula_pretty"),
                        "material_id": item.get("material_id"),
                        "band_gap": item.get("band_gap"),
                        "cif": item.get("cif"),
                    }
                )
                continue

            structure = _extract_structure_from_mp_item(item)

            if structure is None:
                continue

            material_id = str(_extract_mp_field(item, "material_id", ""))
            formula = (
                _extract_mp_field(item, "formula", None)
                or _extract_mp_field(item, "formula_pretty", None)
                or structure.composition.reduced_formula
            )
            band_gap = _extract_mp_field(item, "band_gap", None)

            results.append(
                {
                    "formula": formula,
                    "material_id": material_id,
                    "band_gap": band_gap,
                    "cif": _structure_to_cif(structure),
                }
            )

        except Exception as exc:
            logger.warning("Skipped one MP search result: %s", exc)

    return ok(
        "结构库查询完成。",
        {
            "query": query,
            "limit": safe_limit,
            "count": len(results),
            "items": results,
        },
    )


# ============================================================
# Routes: Structure
# ============================================================

@app.post(
    "/api/structures/validate",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT_STRUCTURE)
async def validate_cif_structure(
    request: Request,
    file: UploadFile = File(..., description="CIF 结构文件"),
):
    structure = await _read_cif_structure(file)

    return ok(
        "CIF 结构解析成功。",
        {
            "cif": _structure_to_cif(structure),
        },
    )


@app.post(
    "/api/structures/supercell",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT_STRUCTURE)
async def build_supercell(
    request: Request,
    file: UploadFile = File(..., description="CIF 结构文件"),
    matrix: str = Form("2x2x1", description='超胞矩阵，例如 "2x2x1"'),
):
    structure = await _read_cif_structure(file)

    try:
        parsed_matrix = _parse_matrix(matrix)
        modifier = _get_structure_modifier(structure)
        new_structure = modifier.make_supercell(parsed_matrix).get_structure()

    except APIBizError:
        raise
    except Exception as exc:
        logger.warning("Failed to build supercell: %s", exc, exc_info=True)

        raise APIBizError(
            code=BizCode.BAD_REQUEST,
            msg="超胞构建失败，请检查 CIF 结构和矩阵参数。",
            error_type="SUPERCELL_BUILD_FAILED",
            status_code=status.HTTP_400_BAD_REQUEST,
            data={
                "input_matrix": matrix,
            },
        )

    return ok(
        "超胞构建成功。",
        {
            "cif": _structure_to_cif(new_structure),
        },
    )


@app.post(
    "/api/structures/transform",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT_STRUCTURE)
async def transform_structure(
    request: Request,
    file: UploadFile = File(..., description="CIF 结构文件"),
    matrix: str = Form(
        ...,
        description='3x3 转换矩阵 JSON 字符串，例如 [[1,0,0],[0,1,0],[0,0,1]]',
    ),
):
    structure = await _read_cif_structure(file)

    try:
        parsed_matrix = _parse_matrix(matrix)

        if not (
            isinstance(parsed_matrix, list)
            and len(parsed_matrix) == 3
            and all(isinstance(row, list) and len(row) == 3 for row in parsed_matrix)
        ):
            raise APIBizError(
                code=BizCode.BAD_REQUEST,
                msg="转换矩阵必须是 3x3 矩阵。",
                error_type="TRANSFORM_MATRIX_NOT_3X3",
                status_code=status.HTTP_400_BAD_REQUEST,
                data={
                    "input_matrix": matrix,
                },
            )

        modifier = _get_structure_modifier(structure)
        new_structure = modifier.make_supercell(parsed_matrix).get_structure()

    except APIBizError:
        raise
    except Exception as exc:
        logger.warning("Failed to apply transform matrix: %s", exc, exc_info=True)

        raise APIBizError(
            code=BizCode.BAD_REQUEST,
            msg="转换矩阵应用失败，请检查 CIF 结构和矩阵参数。",
            error_type="TRANSFORM_FAILED",
            status_code=status.HTTP_400_BAD_REQUEST,
            data={
                "input_matrix": matrix,
            },
        )

    return ok(
        "转换矩阵应用成功。",
        {
            "cif": _structure_to_cif(new_structure),
        },
    )


@app.post(
    "/api/structures/export",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT_STRUCTURE)
async def export_structure(
    request: Request,
    file: UploadFile = File(..., description="CIF 结构文件"),
    fmt: Literal["cif", "poscar", "xyz"] = Form(
        "cif",
        description="导出格式：cif / poscar / xyz",
    ),
    filename: str = Form(
        "structure",
        description="导出的文件名前缀，不需要扩展名",
    ),
):
    if build_export_payload is None:
        raise APIBizError(
            code=BizCode.SERVICE_UNAVAILABLE,
            msg="结构导出服务不可用，请检查 structure_io.py 配置。",
            error_type="STRUCTURE_IO_SERVICE_NOT_AVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    structure = await _read_cif_structure(file)

    try:
        raw_export_data = build_export_payload(
            structure=structure,
            fmt=fmt,
            filename=filename,
        )

        export_data = {
            "format": raw_export_data.get("format"),
            "filename": raw_export_data.get("filename"),
            "content": raw_export_data.get("content"),
            "mime_type": raw_export_data.get("mime_type"),
        }

    except ValueError as exc:
        raise APIBizError(
            code=BizCode.BAD_REQUEST,
            msg="不支持的导出格式，目前支持 cif、poscar、xyz。",
            error_type="UNSUPPORTED_EXPORT_FORMAT",
            status_code=status.HTTP_400_BAD_REQUEST,
            data={
                "detail": str(exc),
                "supported_formats": ["cif", "poscar", "xyz"],
            },
        )

    except APIBizError:
        raise
    except Exception as exc:
        logger.error("Failed to export structure: %s", exc, exc_info=True)

        raise APIBizError(
            code=BizCode.SERVER_ERROR,
            msg="结构文件导出失败。",
            error_type="STRUCTURE_EXPORT_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return ok(
        "结构文件导出成功。",
        export_data,
    )


# ============================================================
# Routes: ML
# ============================================================

@app.get(
    "/api/ml/models",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT_STRUCTURE)
async def ml_models(request: Request):
    if list_available_models is None:
        raise APIBizError(
            code=BizCode.SERVICE_UNAVAILABLE,
            msg="ML 模型列表服务不可用，请检查 ml_meta.py 配置。",
            error_type="ML_MODEL_LIST_NOT_AVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        info = list_available_models()
    except Exception as exc:
        logger.error("Failed to list ML models: %s", exc, exc_info=True)

        raise APIBizError(
            code=BizCode.SERVICE_UNAVAILABLE,
            msg="读取 ML 模型目录失败，请检查 ML_MODEL_DIR 和模型文件。",
            error_type="ML_MODEL_DIR_READ_FAILED",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return ok(
        "ML 模型列表读取成功。",
        info,
    )


@app.post(
    "/api/ml/predict",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT_ML)
async def ml_predict(
    request: Request,
    file: UploadFile = File(..., description="CIF 结构文件"),
    task: Literal["predict", "optimize", "relax"] = Form(
        "optimize",
        description="ML 任务类型",
    ),
    model_name: Optional[str] = Form(
        None,
        description="模型名称，例如 esen_30m_oam.pt",
    ),
    fmax: float = Form(
        0.05,
        description="结构优化收敛阈值，单位 eV/Å",
    ),
    steps: int = Form(
        300,
        ge=0,
        le=5000,
        description="最大优化步数。predict 任务可传 0 表示只做单点能量/力计算。",
    ),
):
    structure = await _read_cif_structure(file)

    result = _run_ml_prediction(
        structure=structure,
        task=task,
        model_name=model_name,
        fmax=fmax,
        steps=steps,
    )

    if result.get("status") == "error":
        raise APIBizError(
            code=BizCode.SERVICE_UNAVAILABLE,
            msg="ML 任务执行失败，请检查模型配置或输入结构。",
            error_type="ML_TASK_FAILED",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            data={
                "detail": result.get("message", "unknown error"),
            },
        )

    total_energy_eV = result.get("total_energy_eV")
    total_force_eV_Ang = result.get("total_force_eV_Ang")
    optimized_cif = (
        result.get("optimized_cif")
        or result.get("optimized_structure_cif")
        or result.get("relaxed_structure_cif")
        or result.get("final_structure_cif")
    )

    if total_energy_eV is None or total_force_eV_Ang is None or not optimized_cif:
        raise APIBizError(
            code=BizCode.SERVICE_UNAVAILABLE,
            msg="ML 任务返回结果不完整。",
            error_type="ML_RESULT_INCOMPLETE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            data={
                "required_fields": [
                    "total_energy_eV",
                    "total_force_eV_Ang",
                    "optimized_cif",
                ],
            },
        )

    return ok(
        "ML 任务执行成功。",
        {
            "total_energy_eV": total_energy_eV,
            "total_force_eV_Ang": total_force_eV_Ang,
            "optimized_cif": optimized_cif,
        },
    )
