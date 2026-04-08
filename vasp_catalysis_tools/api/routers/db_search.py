"""
/db 路由 — Materials Project 数据库检索
基于 MPQueryService，API Key 服务端统一配置，前端无需传入。

路由列表：
  GET  /db/search?q=...          前端搜索框 → 返回结构列表（含xyz/cif/poscar）
  GET  /db/entry/{material_id}   点击条目   → 返回单条完整结构
  POST /db/cache/clear           清空服务端查询缓存（管理用）
"""
import os
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...core.search import MPQueryService

logger = logging.getLogger("catalyst_workbench.db")
router = APIRouter()

# ─────────────────────────────────────────────────────
# 服务端单例：API Key 从环境变量读取，前端不感知
# 生产部署：export MP_API_KEY="your_key_here"
# ─────────────────────────────────────────────────────
_MP_API_KEY = os.environ.get("MP_API_KEY", "n7GbikHqL3SlQCH3diuPBBlMNqRcTnFR")

def _get_service() -> MPQueryService:
    """
    获取 MPQueryService 单例。
    延迟初始化：首次调用时才检查 API Key，
    避免服务启动时因未配置 Key 而崩溃。
    """
    if not _MP_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "MP_API_KEY 未配置。"
                "请联系管理员在服务端设置环境变量 MP_API_KEY。"
            ),
        )
    # 模块级单例，避免每次请求重新创建
    if not hasattr(_get_service, "_instance"):
        _get_service._instance = MPQueryService(
            api_key=_MP_API_KEY,
            only_stable=True,
            max_results=20,
            cache_ttl=300,
        )
    return _get_service._instance


# ─────────────────────────────────────────────────────
# 响应模型
# ─────────────────────────────────────────────────────

class StructureEntry(BaseModel):
    """
    单条结构的完整响应体。
    对应前端 DB 列表条目 + 点击后加载到预览器的所有数据。
    """
    material_id:    str             # 如 "mp-126"
    formula:        str             # 如 "Pt"
    space_group:    str             # 如 "Fm-3m"
    crystal_system: str             # 如 "cubic"
    # 晶格参数
    a:     float
    b:     float
    c:     float
    alpha: float
    beta:  float
    gamma: float
    # 结构文件字符串（前端直接使用）
    xyz:    str                     # 3Dmol.js 渲染用
    cif:    str                     # 下载 CIF 用
    poscar: str                     # 下载 POSCAR 用 / 传给修改接口用


def _to_entry(item: dict) -> StructureEntry:
    """
    将 MPQueryService.query() 返回的 dict 转为响应模型。
    注意：_structure 是 pymatgen 对象，必须在此处排除，不能进入序列化。
    """
    return StructureEntry(
        material_id    = item["material_id"],
        formula        = item["formula"],
        space_group    = item["space_group"],
        crystal_system = item["crystal_system"],
        a              = item["a"],
        b              = item["b"],
        c              = item["c"],
        alpha          = item["alpha"],
        beta           = item["beta"],
        gamma          = item["gamma"],
        xyz            = item["xyz"],
        cif            = item["cif"],
        poscar         = item["poscar"],
    )


# ─────────────────────────────────────────────────────
# 路由
# ─────────────────────────────────────────────────────

@router.get(
    "/search",
    summary="搜索 Materials Project 结构库",
    response_model=List[StructureEntry],
)
def search(
    q: str = Query(
        ...,
        description=(
            "搜索关键词，支持以下格式：\n"
            "- material_id：mp-126\n"
            "- 化学式：Fe2O3、Pt\n"
            "- 化学体系（连字符）：Fe-O、Pt-Ni\n"
            "- 元素列表（逗号/空格）：Fe, O\n"
        ),
        min_length=1,
    ),
) -> List[StructureEntry]:
    """
    前端右侧结构库搜索框对应接口。

    - 输入内容由 MPQueryService 内部的 detect_query_mode() 自动识别模式
    - 结果按原子数升序排列（MPQueryService 内部已处理）
    - 带 TTL 缓存（默认 5 分钟），相同查询不重复请求 MP
    - 返回结果包含 xyz（3Dmol渲染）、cif、poscar（下载/传给修改接口）
    """
    svc = _get_service()
    try:
        results = svc.query(q)
    except ValueError as e:
        # detect_query_mode 抛出的输入校验错误
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("MP 查询失败: %s", e, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Materials Project 查询失败: {e}",
        )

    if not results:
        # 返回空列表而非 404，前端展示"无结果"
        return []

    return [_to_entry(item) for item in results]


@router.get(
    "/entry/{material_id}",
    summary="按 material_id 获取单条结构详情",
    response_model=StructureEntry,
)
def get_entry(material_id: str) -> StructureEntry:
    """
    前端点击结构库列表条目时调用。
    直接复用 query() 的缓存，不额外发起 MP 请求。

    返回完整结构数据，前端用 xyz 渲染预览，
    poscar 字符串可直接传给 /bulk/modify/* 接口。
    """
    svc = _get_service()
    try:
        results = svc.query(material_id)
    except Exception as e:
        logger.error("MP 查询失败: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Materials Project 查询失败: {e}")

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"未在 Materials Project 中找到 {material_id}",
        )

    return _to_entry(results[0])


@router.post(
    "/cache/clear",
    summary="清空 MP 查询缓存（管理员用）",
)
def clear_cache():
    """手动清空服务端 TTL 缓存，用于调试或强制刷新数据。"""
    svc = _get_service()
    svc.clear_cache()
    return {"message": "MP 查询缓存已清空"}