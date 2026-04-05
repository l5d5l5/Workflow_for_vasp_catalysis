"""
/db 路由 — Materials Project 数据库检索
使用 mp-api（新版 MPRester）
返回字段：mp_id / reduced_formula / spacegroup / structure_str(poscar)
"""
import os
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger("catalyst_workbench.db")
router = APIRouter()

# MP API Key 从环境变量读取
# 生产环境：export MP_API_KEY="your_key_here"
MP_API_KEY = os.environ.get("MP_API_KEY", "")


# ── 响应模型 ──────────────────────────────────────────

class MPEntry(BaseModel):
    mp_id:           str
    formula:         str   # 完整化学式，如 "Pt6"
    reduced_formula: str   # 最简化学式，如 "Pt"
    spacegroup:      str   # 空间群符号，如 "Fm-3m"
    spacegroup_num:  int   # 空间群编号，如 225
    nsites:          int   # 原子数
    structure_str:   str   # POSCAR 字符串
    energy_per_atom: Optional[float] = None  # DFT 能量（eV/atom）


# ── 辅助函数 ──────────────────────────────────────────

def _check_api_key():
    if not MP_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="MP_API_KEY 未配置，请设置环境变量 MP_API_KEY"
        )

def _structure_to_poscar(structure) -> str:
    """pymatgen Structure → POSCAR 字符串"""
    from pymatgen.io.vasp import Poscar
    return str(Poscar(structure))

def _build_entry(doc) -> MPEntry:
    """将 MP 文档转换为统一响应格式"""
    sg = doc.symmetry
    return MPEntry(
        mp_id           = doc.material_id,
        formula         = doc.formula_pretty,
        reduced_formula = doc.composition.reduced_formula,
        spacegroup      = sg.symbol if sg else "Unknown",
        spacegroup_num  = sg.number if sg else 0,
        nsites          = doc.nsites,
        structure_str   = _structure_to_poscar(doc.structure),
        energy_per_atom = round(doc.energy_per_atom, 4) if doc.energy_per_atom else None,
    )


# ── 路由 ─────────────────────────────────────────────

@router.get("/search", summary="按化学式/元素/ID 检索 Materials Project")
def search(
    q:        str           = Query(...,  description="搜索关键词：mp-id / 化学式 / 元素符号"),
    nelements: Optional[int] = Query(None, description="元素种数过滤，如 1 表示单质"),
    limit:    int           = Query(20,   ge=1, le=100, description="最大返回条数"),
) -> List[MPEntry]:
    """
    前端搜索框对应接口。
    - 输入 'mp-123'   → 按 mp_id 精确查询
    - 输入 'Pt'       → 按元素检索（单元素）
    - 输入 'Pt,Ni'    → 按元素组合检索
    - 输入 'Pt3Ni'    → 按化学式检索
    """
    _check_api_key()

    try:
        from mp_api.client import MPRester
    except ImportError:
        raise HTTPException(503, "mp-api 未安装，请运行: pip install mp-api")

    q = q.strip()

    try:
        with MPRester(MP_API_KEY) as mpr:

            # ── 情况1：mp-id 精确查询 ─────────────────
            if q.lower().startswith("mp-"):
                docs = mpr.materials.summary.search(
                    material_ids=[q],
                    fields=["material_id","formula_pretty","composition",
                            "symmetry","nsites","structure","energy_per_atom"]
                )
                return [_build_entry(d) for d in docs]

            # ── 情况2：元素列表（逗号分隔，如 "Pt,Ni"）──
            if "," in q or (q.isalpha() and q[0].isupper()):
                elements = [e.strip() for e in q.split(",")]
                kwargs = dict(
                    elements=elements,
                    fields=["material_id","formula_pretty","composition",
                            "symmetry","nsites","structure","energy_per_atom"]
                )
                if nelements is not None:
                    kwargs["nelements"] = nelements
                docs = mpr.materials.summary.search(**kwargs)
                # 按原子数升序排列，优先返回小结构
                docs = sorted(docs, key=lambda d: d.nsites)[:limit]
                return [_build_entry(d) for d in docs]

            # ── 情况3：化学式（如 "Pt3Ni"）────────────
            docs = mpr.materials.summary.search(
                formula=[q],
                fields=["material_id","formula_pretty","composition",
                        "symmetry","nsites","structure","energy_per_atom"]
            )
            docs = sorted(docs, key=lambda d: d.nsites)[:limit]
            return [_build_entry(d) for d in docs]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MP 检索失败: {e}", exc_info=True)
        raise HTTPException(502, f"Materials Project 查询失败: {e}")


@router.get("/entry/{mp_id}", summary="按 mp-id 获取单条结构详情")
def get_entry(mp_id: str) -> MPEntry:
    """前端点击结构库条目时调用，获取完整 POSCAR"""
    _check_api_key()
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise HTTPException(503, "mp-api 未安装")

    try:
        with MPRester(MP_API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                material_ids=[mp_id],
                fields=["material_id","formula_pretty","composition",
                        "symmetry","nsites","structure","energy_per_atom"]
            )
        if not docs:
            raise HTTPException(404, f"未找到 {mp_id}")
        return _build_entry(docs[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Materials Project 查询失败: {e}")