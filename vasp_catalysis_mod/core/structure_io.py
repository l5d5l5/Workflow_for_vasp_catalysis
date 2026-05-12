"""
structure_io.py
───────────────────────────────────────────────
结构文件导出工具层。

职责：
- pymatgen Structure -> CIF
- pymatgen Structure -> POSCAR
- pymatgen Structure -> XYZ
- 生成前端下载 payload

注意：
- 不负责结构修改
- 不返回结构摘要信息
- 不写服务器磁盘，默认返回文本内容给前端下载
───────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar


def structure_to_cif(structure: Structure) -> str:
    """
    pymatgen Structure 转 CIF 字符串。
    """
    return str(CifWriter(structure))


def structure_to_poscar(
    structure: Structure,
    comment: Optional[str] = None,
) -> str:
    """
    pymatgen Structure 转 POSCAR 字符串。
    """
    return Poscar(
        structure,
        comment=comment or "structure",
    ).get_str()


def structure_to_xyz(
    structure: Structure,
    comment: Optional[str] = None,
    precision: int = 8,
) -> str:
    """
    pymatgen Structure 转 XYZ 字符串。

    注意：
    XYZ 不保留晶胞和周期性信息。
    """
    fmt = f"{{:.{precision}f}}"

    lines = [
        str(len(structure)),
        comment or "structure",
    ]

    for site in structure.sites:
        symbol = site.specie.symbol
        x, y, z = site.coords

        lines.append(
            f"{symbol} {fmt.format(x)} {fmt.format(y)} {fmt.format(z)}"
        )

    return "\n".join(lines)


def build_cif_payload(structure: Structure) -> Dict[str, Any]:
    """
    构建 CIF 返回 payload。

    不返回 formula / num_sites / lattice 等摘要信息。
    """
    return {
        "cif": structure_to_cif(structure),
    }


def sanitize_filename_stem(filename: str, default: str = "structure") -> str:
    """
    清理文件名前缀，防止路径注入。
    """
    stem = Path(filename or default).stem.strip()

    if not stem:
        stem = default

    return stem


def build_export_payload(
    structure: Structure,
    fmt: str = "cif",
    filename: str = "structure",
) -> Dict[str, Any]:
    """
    根据格式生成前端下载 payload。

    返回：
    {
        "format": "cif",
        "filename": "structure.cif",
        "content": "...",
        "mime_type": "chemical/x-cif"
    }

    不返回结构摘要信息。
    """
    fmt = fmt.strip().lower()
    stem = sanitize_filename_stem(filename)

    if fmt == "cif":
        content = structure_to_cif(structure)
        output_filename = f"{stem}.cif"
        mime_type = "chemical/x-cif"

    elif fmt == "poscar":
        content = structure_to_poscar(structure, comment=stem)
        output_filename = f"{stem}.POSCAR"
        mime_type = "text/plain"

    elif fmt == "xyz":
        content = structure_to_xyz(structure, comment=stem)
        output_filename = f"{stem}.xyz"
        mime_type = "chemical/x-xyz"

    else:
        raise ValueError(f"Unsupported export format: {fmt}")

    return {
        "format": fmt,
        "filename": output_filename,
        "content": content,
        "mime_type": mime_type,
    }
