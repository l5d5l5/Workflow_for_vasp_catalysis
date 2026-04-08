"""
/store 路由 — 用户结构本地存储

存储策略：
  - 结构文件（POSCAR）保存在 /home/{username}/catalyst_workbench/
  - 元数据存储在 SQLite（data/workbench.db）
  - struct_id 格式：MY-{username}-{YYYYmmddHHMMSSffffff[:18]}

与前端对接：
  - 「保留设置」按钮 → POST /store/save
  - 「我的」tab 列表 → GET  /store/list
  - 点击「我的」条目 → GET  /store/{struct_id}
  - 删除结构         → DELETE /store/{struct_id}
  - 重命名           → PATCH  /store/{struct_id}/rename

响应字段与 /db/search 的 StructureEntry 保持一致，
前端可用同一套渲染逻辑处理「数据库」和「我的」两个 tab。
"""
import os
import sqlite3
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..deps import get_current_user, get_db, get_user_dir
from ...core.search import (
    structure_to_xyz,
    structure_to_cif,
    structure_to_poscar,
    get_conventional,
)

logger = logging.getLogger("catalyst_workbench.store")
router = APIRouter()


# ─────────────────────────────────────────────────────
# 请求 / 响应模型
# ─────────────────────────────────────────────────────

class SaveRequest(BaseModel):
    structure_str: str           = Field(..., description="POSCAR 字符串（来自修改接口或 MP）")
    fmt:           str           = Field("poscar", description="输入格式: poscar|xyz|cif")
    name:          Optional[str] = Field(None, description="用户自定义名称，默认用最简化学式")
    source:        str           = Field(
        "upload",
        description="来源标记: upload（上传）| mp（来自MP）| modified（修改后）"
    )


class StructureMeta(BaseModel):
    """
    列表接口返回的轻量元数据（不含结构字符串，减少传输量）。
    对应前端「我的」tab 列表条目。
    """
    struct_id:       str
    name:            str
    formula:         str
    reduced_formula: str
    space_group:     str
    source:          str
    created_at:      str
    updated_at:      str


class StructureDetail(StructureMeta):
    """
    点击条目后返回的完整数据（含结构字符串）。
    字段与 /db/search 的 StructureEntry 对齐，前端渲染逻辑可复用。
    """
    # 晶格参数
    a:     float
    b:     float
    c:     float
    alpha: float
    beta:  float
    gamma: float
    # 结构文件（前端渲染 & 下载）
    xyz:    str
    cif:    str
    poscar: str


# ─────────────────────────────────────────────────────
# 内部辅助函数
# ─────────────────────────────────────────────────────

def _load_structure(structure_str: str, fmt: str):
    """结构字符串 → pymatgen Structure"""
    from ...core.utils.structure_utils import load_structure
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
        return load_structure(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"结构解析失败: {e}")
    finally:
        os.unlink(tmp_path)


def _get_spacegroup(structure) -> str:
    """获取空间群符号，失败时返回 Unknown"""
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        return SpacegroupAnalyzer(structure, symprec=1e-3).get_space_group_symbol()
    except Exception:
        return "Unknown"


def _make_struct_id(username: str) -> str:
    """生成唯一结构 ID：MY-{user}-{timestamp}"""
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")[:18]
    return f"MY-{username}-{ts}"


def _row_to_meta(row: sqlite3.Row) -> StructureMeta:
    return StructureMeta(
        struct_id       = row["struct_id"],
        name            = row["name"],
        formula         = row["formula"],
        reduced_formula = row["reduced_formula"],
        space_group     = row["spacegroup"],
        source          = row["source"],
        created_at      = row["created_at"],
        updated_at      = row["updated_at"],
    )


def _build_detail(row: sqlite3.Row, structure) -> StructureDetail:
    """
    从数据库行 + pymatgen Structure 构建完整响应。
    结构转换复用 mp_query_service 中的工具函数，保持格式一致。
    """
    conv    = get_conventional(structure)
    lat     = conv.lattice
    comment = f"{row['reduced_formula']} | {row['struct_id']}"

    return StructureDetail(
        # 元数据
        struct_id       = row["struct_id"],
        name            = row["name"],
        formula         = row["formula"],
        reduced_formula = row["reduced_formula"],
        space_group     = row["spacegroup"],
        source          = row["source"],
        created_at      = row["created_at"],
        updated_at      = row["updated_at"],
        # 晶格参数
        a     = round(lat.a,     4),
        b     = round(lat.b,     4),
        c     = round(lat.c,     4),
        alpha = round(lat.alpha, 2),
        beta  = round(lat.beta,  2),
        gamma = round(lat.gamma, 2),
        # 结构文件
        xyz    = structure_to_xyz(conv, comment),
        cif    = structure_to_cif(conv),
        poscar = structure_to_poscar(conv, comment),
    )


# ─────────────────────────────────────────────────────
# 路由
# ─────────────────────────────────────────────────────

@router.post(
    "/save",
    summary="保存结构到用户文件夹（前端「保留设置」按钮）",
    response_model=StructureMeta,
)
def save_structure(
    req:      SaveRequest,
    username: str                = Depends(get_current_user),
    db:       sqlite3.Connection = Depends(get_db),
) -> StructureMeta:
    """
    保存流程：
      1. 解析结构字符串 → pymatgen Structure
      2. 提取化学式、空间群
      3. 将 POSCAR 写入 /home/{user}/catalyst_workbench/{struct_id}.vasp
      4. 在 SQLite 中插入元数据记录
    """
    structure = _load_structure(req.structure_str, req.fmt)
    conv      = get_conventional(structure)

    composition     = conv.composition
    formula         = composition.formula.replace(" ", "")
    reduced_formula = composition.reduced_formula
    space_group     = _get_spacegroup(conv)
    struct_id       = _make_struct_id(username)
    display_name    = req.name or reduced_formula
    now             = datetime.now().isoformat(timespec="seconds")

    # ── 写入文件 ──────────────────────────────────
    user_dir  = get_user_dir(username)
    file_path = user_dir / f"{struct_id}.vasp"

    from pymatgen.io.vasp import Poscar
    Poscar(conv, comment=f"{reduced_formula} | {struct_id}").write_file(str(file_path))

    # ── 写入 SQLite ───────────────────────────────
    try:
        db.execute(
            """
            INSERT INTO user_structures
              (username, struct_id, name, formula, reduced_formula,
               spacegroup, file_path, fmt, source, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                username, struct_id, display_name,
                formula, reduced_formula, space_group,
                str(file_path), "poscar", req.source,
                now, now,
            ),
        )
        db.commit()
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=f"结构 ID 冲突: {e}")

    logger.info("用户 %s 保存结构 %s → %s", username, struct_id, file_path)

    return StructureMeta(
        struct_id       = struct_id,
        name            = display_name,
        formula         = formula,
        reduced_formula = reduced_formula,
        space_group     = space_group,
        source          = req.source,
        created_at      = now,
        updated_at      = now,
    )


@router.get(
    "/list",
    summary="获取用户已保存的结构列表（前端「我的」tab）",
    response_model=List[StructureMeta],
)
def list_structures(
    q:        Optional[str] = Query(None, description="按名称/化学式模糊搜索"),
    limit:    int           = Query(50, ge=1, le=200),
    offset:   int           = Query(0,  ge=0),
    username: str                = Depends(get_current_user),
    db:       sqlite3.Connection = Depends(get_db),
) -> List[StructureMeta]:
    """
    前端「我的」tab 列表数据源。
    支持与「数据库」tab 相同的搜索框联动（传入 q 参数）。
    """
    if q:
        rows = db.execute(
            """
            SELECT * FROM user_structures
            WHERE username = ?
              AND (name LIKE ? OR formula LIKE ? OR reduced_formula LIKE ?)
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (username, f"%{q}%", f"%{q}%", f"%{q}%", limit, offset),
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT * FROM user_structures
            WHERE username = ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (username, limit, offset),
        ).fetchall()

    return [_row_to_meta(r) for r in rows]


@router.get(
    "/{struct_id}",
    summary="获取单条结构完整数据（前端点击「我的」条目）",
    response_model=StructureDetail,
)
def get_structure(
    struct_id:  str,
    username:   str                = Depends(get_current_user),
    db:         sqlite3.Connection = Depends(get_db),
) -> StructureDetail:
    """
    返回结构的完整数据，包含 xyz/cif/poscar 字符串。
    - xyz   → 3Dmol.js 渲染
    - poscar → 传给 /bulk/modify/* 接口进行修改
    - cif/poscar → 下载
    """
    row = db.execute(
        "SELECT * FROM user_structures WHERE struct_id = ? AND username = ?",
        (struct_id, username),
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"未找到结构 {struct_id}")

    file_path = Path(row["file_path"])
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"结构文件已丢失，请重新保存。路径: {file_path}",
        )

    from ...core.utils.structure_utils import load_structure
    try:
        structure = load_structure(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"结构文件读取失败: {e}")

    return _build_detail(row, structure)


@router.delete(
    "/{struct_id}",
    summary="删除用户结构",
)
def delete_structure(
    struct_id: str,
    username:  str                = Depends(get_current_user),
    db:        sqlite3.Connection = Depends(get_db),
):
    """同时删除磁盘文件和数据库记录。"""
    row = db.execute(
        "SELECT * FROM user_structures WHERE struct_id = ? AND username = ?",
        (struct_id, username),
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"未找到结构 {struct_id}")

    # 删除磁盘文件
    file_path = Path(row["file_path"])
    if file_path.exists():
        file_path.unlink()
        logger.info("已删除文件: %s", file_path)

    # 删除数据库记录
    db.execute("DELETE FROM user_structures WHERE struct_id = ?", (struct_id,))
    db.commit()

    return {"deleted": struct_id}


@router.patch(
    "/{struct_id}/rename",
    summary="重命名用户结构",
)
def rename_structure(
    struct_id: str,
    name:      str                = Query(..., description="新名称"),
    username:  str                = Depends(get_current_user),
    db:        sqlite3.Connection = Depends(get_db),
):
    now = datetime.now().isoformat(timespec="seconds")
    cur = db.execute(
        """
        UPDATE user_structures
        SET name = ?, updated_at = ?
        WHERE struct_id = ? AND username = ?
        """,
        (name, now, struct_id, username),
    )
    db.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"未找到结构 {struct_id}")
    return {"struct_id": struct_id, "name": name, "updated_at": now}