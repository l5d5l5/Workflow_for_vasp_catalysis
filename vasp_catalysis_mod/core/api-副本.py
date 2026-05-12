"""
main.py
───────────────────────────────────────────────
Secure Crystal Structure Backend API

统一约定：
1. 前端上传结构：CIF 文件
2. 后端返回结构：CIF 字符串
3. 所有业务接口统一返回：
   {
       "code": 1,
       "msg": "success",
       "time": "1775095470",
       "data": {}
   }
4. 业务接口使用 Bearer Token 鉴权
5. 使用 .env 管理密钥、CORS、限流、上传大小等配置
6. 使用 slowapi 做基础限流

已支持：
- 健康检查
- Materials Project 结构库查询
- CIF 校验与标准化
- 超胞构建
- 转换矩阵应用
- CIF 导出
- ML 预测/结构优化
───────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import time
import logging
import secrets
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
# 根据你的项目结构，必要时修改导入路径
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
    description="""
    晶体结构平台后端 API。

    当前统一格式：
    - 前端上传 CIF
    - 后端解析 CIF
    - 后端返回 CIF
    - 业务接口统一返回 code/msg/time/data
    - 业务接口使用 Bearer Token 鉴权

    已支持：
    - Materials Project 结构库查询
    - CIF 校验与标准化
    - 超胞构建
    - 转换矩阵
    - CIF 导出
    - ML 预测/结构优化
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ============================================================
# Rate Limiting Setup
# ============================================================

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

RATE_LIMIT_HEALTH = os.getenv("RATE_LIMIT_HEALTH", "60/minute")
RATE_LIMIT_SEARCH = os.getenv("RATE_LIMIT_SEARCH", "20/minute")
RATE_LIMIT_STRUCTURE = os.getenv("RATE_LIMIT_STRUCTURE", "30/minute")
RATE_LIMIT_ML = os.getenv("RATE_LIMIT_ML", "5/minute")


# ============================================================
# Security: Authentication
# ============================================================

REQUIRE_API_AUTH = os.getenv("REQUIRE_API_AUTH", "true").strip().lower() == "true"
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "").strip()

if REQUIRE_API_AUTH and not API_SECRET_KEY:
    raise RuntimeError(
        "API_SECRET_KEY is required when REQUIRE_API_AUTH=true. "
        "Please set API_SECRET_KEY in your .env file."
    )

# auto_error=False：方便我们自己返回统一格式的鉴权错误
security = HTTPBearer(auto_error=False)


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
    """
    返回当前 Unix 时间戳字符串。
    """
    return str(int(time.time()))


def make_response(
    code: int,
    msg: str,
    data: Any = None,
) -> Dict[str, Any]:
    """
    统一 API 返回格式。

    成功示例：
    {
        "code": 1,
        "msg": "success",
        "time": "1775095470",
        "data": {}
    }

    失败示例：
    {
        "code": 400,
        "msg": "CIF 文件解析失败，请检查文件格式。",
        "time": "1775095470",
        "data": {
            "error_type": "CIF_PARSE_ERROR"
        }
    }
    """
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
    """
    业务异常。

    用途：
    - 保证错误返回格式统一
    - 支持不同错误展示不同 msg
    - 后端日志记录真实错误，前端只接收安全信息
    """

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
# Authentication Helper
# ============================================================

def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """
    验证前端传来的 Bearer Token。

    请求头示例：
    Authorization: Bearer your_secret_key
    """
    if not REQUIRE_API_AUTH:
        return "auth_disabled"

    if credentials is None or not credentials.credentials:
        logger.warning("Authentication failed: missing token.")

        raise APIBizError(
            code=BizCode.UNAUTHORIZED,
            msg="认证失败，请在请求头中提供 API Token。",
            error_type="AUTH_TOKEN_MISSING",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    token = credentials.credentials

    if not secrets.compare_digest(token, API_SECRET_KEY):
        logger.warning("Authentication failed: invalid token.")

        raise APIBizError(
            code=BizCode.UNAUTHORIZED,
            msg="认证失败，请检查 API Token。",
            error_type="AUTH_FAILED",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    return token


# ============================================================
# Global Exception Handlers
# ============================================================

@app.exception_handler(APIBizError)
async def api_biz_error_handler(request: Request, exc: APIBizError):
    """
    业务异常统一返回。
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_response(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    FastAPI HTTPException 统一返回。
    """
    status_code_value = exc.status_code

    if status_code_value == 401:
        code = BizCode.UNAUTHORIZED
        msg = "认证失败，请检查 API Token。"
        error_type = "AUTH_FAILED"
    elif status_code_value == 403:
        code = BizCode.FORBIDDEN
        msg = "没有权限访问该接口。"
        error_type = "FORBIDDEN"
    elif status_code_value == 404:
        code = BizCode.NOT_FOUND
        msg = "请求的资源不存在。"
        error_type = "NOT_FOUND"
    elif status_code_value == 422:
        code = BizCode.VALIDATION_ERROR
        msg = "请求参数校验失败。"
        error_type = "VALIDATION_ERROR"
    elif status_code_value == 429:
        code = BizCode.RATE_LIMITED
        msg = "请求过于频繁，请稍后再试。"
        error_type = "RATE_LIMITED"
    elif status_code_value >= 500:
        code = BizCode.SERVER_ERROR
        msg = "服务器内部错误，请稍后重试。"
        error_type = "SERVER_ERROR"
    else:
        code = BizCode.BAD_REQUEST
        msg = str(exc.detail) if exc.detail else "请求参数错误。"
        error_type = "BAD_REQUEST"

    return JSONResponse(
        status_code=status_code_value,
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
    """
    Pydantic / FastAPI 参数校验错误统一返回。
    """
    logger.warning("Request validation failed: %s", exc)

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
    """
    slowapi 限流异常统一返回。
    """
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
    """
    未捕获异常统一返回。

    注意：
    - 真实错误写入日志
    - 前端返回模糊错误
    - 避免泄露服务器路径、模型路径、堆栈等敏感信息
    """
    logger.error("Unhandled server error: %s", exc, exc_info=True)

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
    """
    从环境变量读取最大上传文件大小。
    """
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
    """
    将常见不可 JSON 序列化对象转成安全对象。

    处理：
    - numpy 标量
    - numpy 数组
    - pathlib.Path
    - pymatgen Structure
    - dict/list/tuple/set
    """
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
    """
    读取上传的 CIF 文件内容。
    """
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
    """
    从上传的 CIF 文件读取 pymatgen Structure。
    """
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
    """
    pymatgen Structure 转 CIF 字符串。
    """
    try:
        return str(CifWriter(structure))
    except Exception as exc:
        logger.error("Failed to convert Structure to CIF: %s", exc, exc_info=True)

        raise APIBizError(
            code=BizCode.SERVER_ERROR,
            msg="结构转换为 CIF 失败。",
            error_type="CIF_CONVERT_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _structure_summary(structure: Structure) -> Dict[str, Any]:
    """
    结构摘要信息。
    """
    return {
        "formula": structure.composition.reduced_formula,
        "num_sites": len(structure),
        "lattice": {
            "a": float(structure.lattice.a),
            "b": float(structure.lattice.b),
            "c": float(structure.lattice.c),
            "alpha": float(structure.lattice.alpha),
            "beta": float(structure.lattice.beta),
            "gamma": float(structure.lattice.gamma),
            "volume": float(structure.lattice.volume),
        },
    }


def _build_cif_payload(structure: Structure) -> Dict[str, Any]:
    """
    统一构建返回前端的 CIF payload。
    """
    return {
        "cif": _structure_to_cif(structure),
        **_structure_summary(structure),
    }


def _parse_matrix(matrix: str) -> Union[List[int], List[List[float]]]:
    """
    解析超胞矩阵或转换矩阵。

    支持：
    - 2x2x1
    - 2 2 1
    - 2,2,1
    - [[2,0,0],[0,2,0],[0,0,1]]
    """
    matrix = matrix.strip()

    if not matrix:
        raise APIBizError(
            code=BizCode.BAD_REQUEST,
            msg="矩阵参数不能为空。",
            error_type="EMPTY_MATRIX",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    # 优先使用项目已有工具函数
    if parse_supercell_matrix is not None:
        try:
            parsed = parse_supercell_matrix(matrix)
            return _json_safe(parsed)
        except Exception:
            pass

    # JSON 3x3 矩阵
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

    # 解析 2x2x1 / 2 2 1 / 2,2,1
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


# ============================================================
# Materials Project Service Wrapper
# ============================================================

_MP_SERVICE_INSTANCE = None
_MP_SERVICE_INIT_ERROR = None


def _get_mp_service() -> Any:
    """
    初始化 Materials Project 查询服务。

    使用单例，避免每次请求都重新初始化。
    """
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
        # 优先使用新版 search.py：
        # MPQueryService 会自动从 .env 读取 MP_API_KEY
        try:
            _MP_SERVICE_INSTANCE = MPQueryService(
                only_stable=False,
                max_results=10,
            )
        except TypeError:
            # 兼容旧版 search.py：可能必须显式传 api_key
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
    """
    兼容 search.py 中可能存在的查询方法名。

    优先调用：
    - search(query=..., limit=...)
    - query(raw_input, limit=...)
    """
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
        # 新版 search.py 推荐方法：
        # search(query="VO2", limit=1)
        if method_name == "search":
            return method(query=query, limit=limit)

        # query() 旧版本可能参数名是 raw_input，不一定接受 query=
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
    """
    从 MP 查询结果中提取 Structure。
    """
    if isinstance(item, dict):
        structure = item.get("structure")
    else:
        structure = getattr(item, "structure", None)

    if isinstance(structure, Structure):
        return structure

    return None


def _extract_mp_field(item: Any, key: str, default: Any = None) -> Any:
    """
    从 MP 查询结果中提取字段。
    """
    if isinstance(item, dict):
        return item.get(key, default)

    return getattr(item, key, default)


# ============================================================
# ML Service Wrapper
# ============================================================
def _get_ml_service(model_name: Optional[str] = None) -> Any:
    """
    初始化 ML 服务。
    """
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
    """
    统一 ML 服务返回结果。

    如果返回中包含 Structure，则转为 CIF。
    """
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
                output[f"{key}_summary"] = _structure_summary(value)
                output.pop(key, None)

        return _json_safe(output)

    if isinstance(result, Structure):
        return {
            "optimized_structure_cif": _structure_to_cif(result),
            "optimized_structure_summary": _structure_summary(result),
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
    """
    调用 ML 服务。

    兼容方法名：
    - optimize
    - relax
    - predict
    - run
    """
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

    logger.error("Failed to run ML task. Last error: %s", last_error, exc_info=True)

    raise APIBizError(
        code=BizCode.SERVICE_UNAVAILABLE,
        msg="ML 任务执行失败，请检查模型配置或输入结构。",
        error_type="ML_TASK_FAILED",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


# ============================================================
# Routes
# ============================================================

@app.get("/")
@limiter.limit(RATE_LIMIT_HEALTH)
async def root(request: Request):
    """
    根路径信息。
    """
    return ok(
        "Crystal Structure Backend API is running.",
        {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "healthz": "/healthz",
            "format_policy": "CIF in, CIF out",
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
    """
    健康检查接口。

    无需密码，方便部署平台或监控系统检查服务状态。
    """
    return ok(
        "ok",
        {
            "service": "Crystal Structure Backend API",
            "status": "running",
            "format_policy": "CIF in, CIF out",
            "auth_required": REQUIRE_API_AUTH,
        },
    )


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
    """
    Materials Project 结构库查询。

    返回字段：
    - formula
    - material_id
    - band_gap
    - cif

    最多返回 10 个结构。
    """
    service = _get_mp_service()

    # 后端再次强制限制，避免绕过 Swagger 传大 limit
    safe_limit = max(1, min(int(limit), 10))

    raw_results = _call_mp_search(
        service,
        query=query,
        limit=safe_limit,
    )

    # 情况 1：新版 search.py 返回标准 dict：
    # {
    #   "query": "...",
    #   "limit": 10,
    #   "count": 1,
    #   "items": [...]
    # }
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

    # 情况 2：兼容旧版 search.py，返回 list，且 item 里可能有 Structure
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
            # 优先兼容已经有 cif 的 dict
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

            # 兼容旧逻辑：从 Structure 转 CIF
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


@app.post(
    "/api/structures/validate",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT_STRUCTURE)
async def validate_cif_structure(
    request: Request,
    file: UploadFile = File(..., description="CIF 结构文件"),
):
    """
    CIF 结构校验与标准化。

    用途：
    - 检查前端上传的 CIF 是否有效
    - 重新输出标准 CIF 字符串
    - 返回结构基本信息
    """
    structure = await _read_cif_structure(file)
    payload = _build_cif_payload(structure)

    return ok(
        "CIF 结构解析成功。",
        {
            "input_filename": file.filename,
            **payload,
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
    """
    构建超胞。

    matrix 支持：
    - 2x2x1
    - 2 2 1
    - 2,2,1
    - [[2,0,0],[0,2,0],[0,0,1]]
    """
    structure = await _read_cif_structure(file)

    try:
        parsed_matrix = _parse_matrix(matrix)

        new_structure = structure.copy()
        new_structure.make_supercell(parsed_matrix)

    except APIBizError:
        raise
    except Exception as exc:
        logger.warning("Failed to build supercell: %s", exc)

        raise APIBizError(
            code=BizCode.BAD_REQUEST,
            msg="超胞构建失败，请检查 CIF 结构和矩阵参数。",
            error_type="SUPERCELL_BUILD_FAILED",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    payload = _build_cif_payload(new_structure)

    return ok(
        "超胞构建成功。",
        {
            "input_filename": file.filename,
            "matrix": _json_safe(parsed_matrix),
            "input_num_sites": len(structure),
            "output_num_sites": len(new_structure),
            "input_formula": structure.composition.reduced_formula,
            "output_formula": new_structure.composition.reduced_formula,
            **payload,
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
    """
    应用 3x3 转换矩阵。

    底层调用：
    pymatgen Structure.make_supercell(matrix)
    """
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
            )

        new_structure = structure.copy()
        new_structure.make_supercell(parsed_matrix)

    except APIBizError:
        raise
    except Exception as exc:
        logger.warning("Failed to apply transform matrix: %s", exc)

        raise APIBizError(
            code=BizCode.BAD_REQUEST,
            msg="转换矩阵应用失败，请检查 CIF 结构和矩阵参数。",
            error_type="TRANSFORM_FAILED",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    payload = _build_cif_payload(new_structure)

    return ok(
        "转换矩阵应用成功。",
        {
            "input_filename": file.filename,
            "matrix": _json_safe(parsed_matrix),
            "input_num_sites": len(structure),
            "output_num_sites": len(new_structure),
            **payload,
        },
    )


@app.post(
    "/api/structures/export",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT_STRUCTURE)
async def export_cif_structure(
    request: Request,
    file: UploadFile = File(..., description="CIF 结构文件"),
    filename: str = Form("structure.cif", description="导出的文件名"),
):
    """
    导出 CIF。

    为了保持接口返回格式统一，这里返回 JSON：
    {
        "code": 1,
        "msg": "CIF 文件导出成功。",
        "time": "...",
        "data": {
            "filename": "structure.cif",
            "cif": "..."
        }
    }

    前端根据 data.cif 自行生成下载文件。
    """
    structure = await _read_cif_structure(file)
    cif_text = _structure_to_cif(structure)

    safe_filename = Path(filename).name.strip() or "structure.cif"

    if not safe_filename.lower().endswith(".cif"):
        safe_filename = f"{safe_filename}.cif"

    return ok(
        "CIF 文件导出成功。",
        {
            "filename": safe_filename,
            "cif": cif_text,
            **_structure_summary(structure),
        },
    )


@app.get(
    "/api/ml/models",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT_STRUCTURE)
async def ml_models(request: Request):
    """
    查看当前后端可识别的 ML 模型文件。

    注意：
    - 不向前端暴露完整服务器路径
    - 只返回模型名、默认模型是否存在、模型数量
    """
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
    """
    ML 预测或结构优化。

    成功时只返回前端需要的 3 个字段：
    - total_energy_eV
    - total_force_eV_Ang
    - optimized_cif
    """
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
    optimized_cif = result.get("optimized_cif")

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

