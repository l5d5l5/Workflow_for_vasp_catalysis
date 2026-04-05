"""
/store 路由 — 用户结构本地存储
- 结构文件保存在 /home/{username}/catalyst_workbench/
- 元数据存储在 SQLite（data/workbench.db）
- struct_id 格式：MY-{username}-{timestamp}
"""
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ..deps import get_current_user, get_db, get_user_dir

logger = logging.getLogger("catalyst_workbench.store")
router = APIRouter()


# ── 请求 / 响应模型 ───────────────────────────────────

class SaveRequest(BaseModel):
    structure_str:   str            = Field(..., description="POSCAR 字符串")
    fmt:             str            = Field("poscar")
    name:            Optional[str]  = Field(None,  description="用户自定义名称")
    source:          str            = Field("upload", description="upload|mp|modified")

class StructureMeta(BaseModel):
    struct_id:       str
    name:            Optional[str]
    formula:         str
    reduced_formula: str
    spacegroup:      str
    source:          str
    created_at:      str
    updated_at:      str

class StructureDetail(StructureMeta):
    structure_str:   str
    fmt:             str


# ── 辅助函数 ──────────────────────────────────────────

def _get_spacegroup(structure) -> str:
    """获取空间群符号，失败时返回 Unknown"""
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(structure)
        return sga.get_space_group_symbol()
    except Exception:
        return "Unknown"

def _parse_structure(structure_str: str, fmt: str):
    """结构字符串 → pymatgen Structure"""
    import os, tempfile
    from ...core.utils.structure_utils import load_structure
    suffix_map = {"poscar":".vasp","vasp":".vasp","xyz":".xyz","cif":".cif"}
    suffix = suffix_map.get(fmt.lower(), ".vasp")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(structure_str)
        tmp_path = tmp.name
    try:
        return load_structure(tmp_path)
    except Exception as e:
        raise HTTPException(422, f"结构解析失败: {e}")
    finally:
        os.unlink(tmp_path)

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
        spacegroup      = row["spacegroup"],
        source          = row["source"],
        created_at      = row["created_at"],
        updated_at      = row["updated_at"],
    )


# ── 路由 ─────────────────────────────────────────────

@router.post("/save", summary="保存结构到用户文件夹", response_model=StructureMeta)
def save_structure(
    req:      SaveRequest,
    username: str                = Depends(get_current_user),
    db:       sqlite3.Connection = Depends(get_db),
) -> StructureMeta:
    """
    前端「保留设置」按钮对应接口。
    1. 解析结构，提取化学式 / 空间群
    2. 将 POSCAR 文件写入 /home/{user}/catalyst_workbench/
    3. 在 SQLite 中插入元数据记录
    """
    structure = _parse_structure(req.structure_str, req.fmt)
    composition = structure.composition
    formula         = composition.formula.replace(" ", "")
    reduced_formula = composition.reduced_formula
    spacegroup      = _get_spacegroup(structure)
    struct_id       = _make_struct_id(username)
    now             = datetime.now().isoformat(timespec="seconds")

    # 写入文件
    user_dir  = get_user_dir(username)
    file_path = user_dir / f"{struct_id}.vasp"
    from pymatgen.io.vasp import Poscar
    Poscar(structure).write_file(str(file_path))

    # 写入 SQLite
    try:
        db.execute("""
            INSERT INTO user_structures
              (username, struct_id, name, formula, reduced_formula,
               spacegroup, file_path, fmt, source, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            username, struct_id,
            req.name or reduced_formula,
            formula, reduced_formula, spacegroup,
            str(file_path), "poscar", req.source, now, now
        ))
        db.commit()
    except sqlite3.IntegrityError as e:
        raise HTTPException(409, f"结构 ID 冲突: {e}")

    return StructureMeta(
        struct_id=struct_id, name=req.name or reduced_formula,
        formula=formula, reduced_formula=reduced_formula,
        spacegroup=spacegroup, source=req.source,
        created_at=now, updated_at=now,
    )


@router.get("/list", summary="获取用户已保存的结构列表", response_model=List[StructureMeta])
def list_structures(
    q:        Optional[str] = Query(None, description="按化学式/名称模糊搜索"),
    limit:    int           = Query(50, ge=1, le=200),
    offset:   int           = Query(0,  ge=0),
    username: str                = Depends(get_current_user),
    db:       sqlite3.Connection = Depends(get_db),
) -> List[StructureMeta]:
    """前端「我的」tab 对应接口"""
    if q:
        rows = db.execute("""
            SELECT * FROM user_structures
            WHERE username=?
              AND (name LIKE ? OR formula LIKE ? OR reduced_formula LIKE ?)
            ORDER BY updated_at DESC LIMIT ? OFFSET ?
        """, (username, f"%{q}%", f"%{q}%", f"%{q}%", limit, offset)).fetchall()
    else:
        rows = db.execute("""
            SELECT * FROM user_structures
            WHERE username=?
            ORDER BY updated_at DESC LIMIT ? OFFSET ?
        """, (username, limit, offset)).fetchall()
    return [_row_to_meta(r) for r in rows]


@router.get("/{struct_id}", summary="获取单条结构详情（含结构字符串）",
            response_model=StructureDetail)
def get_structure(
    struct_id: str,
    output_fmt: str           = Query("poscar"),
    username:   str                = Depends(get_current_user),
    db:         sqlite3.Connection = Depends(get_db),
) -> StructureDetail:
    """前端点击「我的」列表条目时调用"""
    row = db.execute("""
        SELECT * FROM user_structures WHERE struct_id=? AND username=?
    """, (struct_id, username)).fetchone()

    if not row:
        raise HTTPException(404, f"未找到结构 {struct_id}")

    file_path = Path(row["file_path"])
    if not file_path.exists():
        raise HTTPException(404, f"结构文件已丢失: {file_path}")

    from ...core.structure_modify import StructureModify
    from ...core.utils.structure_utils import load_structure
    structure = load_structure(file_path)
    modifier  = StructureModify(structure)

    return StructureDetail(
        **_row_to_meta(row).dict(),
        structure_str = modifier.to_string(output_fmt),
        fmt           = output_fmt,
    )


@router.delete("/{struct_id}", summary="删除用户结构")
def delete_structure(
    struct_id: str,
    username:  str                = Depends(get_current_user),
    db:        sqlite3.Connection = Depends(get_db),
):
    """删除结构文件及数据库记录"""
    row = db.execute("""
        SELECT * FROM user_structures WHERE struct_id=? AND username=?
    """, (struct_id, username)).fetchone()

    if not row:
        raise HTTPException(404, f"未找到结构 {struct_id}")

    # 删除文件
    file_path = Path(row["file_path"])
    if file_path.exists():
        file_path.unlink()

    # 删除数据库记录
    db.execute("DELETE FROM user_structures WHERE struct_id=?", (struct_id,))
    db.commit()

    return {"deleted": struct_id}


@router.patch("/{struct_id}/rename", summary="重命名用户结构")
def rename_structure(
    struct_id: str,
    name:      str                = Query(..., description="新名称"),
    username:  str                = Depends(get_current_user),
    db:        sqlite3.Connection = Depends(get_db),
):
    now = datetime.now().isoformat(timespec="seconds")
    cur = db.execute("""
        UPDATE user_structures SET name=?, updated_at=?
        WHERE struct_id=? AND username=?
    """, (name, now, struct_id, username))
    db.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, f"未找到结构 {struct_id}")
    return {"struct_id": struct_id, "name": name}