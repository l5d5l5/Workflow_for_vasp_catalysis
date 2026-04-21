# -*- coding: utf-8 -*-
"""
Utility functions for structure loading, VASP format parsing, functional
inference, and atom-index selection used throughout the flow package.

flow 包使用的工具函数，涵盖结构读取、VASP 格式解析、泛函推断
以及原子索引选取等功能。
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pymatgen.core import Composition, Structure
from pymatgen.io.vasp.inputs import Poscar

# Module-level logger; consumers can configure its level and handlers.
# 模块级 logger；调用方可自行配置其级别和处理器。
logger = logging.getLogger(__name__)


def load_structure(struct_source: Union[str, Path, Structure]) -> Structure:
    """Load a structure from a file path, directory, or existing Structure object.

    When a directory is supplied the function searches for structure files in the
    following priority order: CONTCAR > POSCAR > POSCAR.vasp > *.vasp > *.cif.
    If multiple .vasp files are found the first one is used and a warning is
    emitted.  Multiple .cif files raise FileNotFoundError.

    从文件路径、目录或已有 Structure 对象加载结构。

    当传入目录时，按以下优先级搜索结构文件：
    CONTCAR > POSCAR > POSCAR.vasp > *.vasp > *.cif。
    若找到多个 .vasp 文件，使用第一个并发出警告；
    找到多个 .cif 文件则抛出 FileNotFoundError。

    Args:
        struct_source: File path, directory path, or a pymatgen Structure instance.
                       文件路径、目录路径或 pymatgen Structure 实例。

    Returns:
        A pymatgen Structure object.
        pymatgen Structure 对象。

    Raises:
        FileNotFoundError: When no valid structure file can be found.
                           找不到有效结构文件时抛出。
    """
    # Return immediately if already a Structure object.
    # 若已是 Structure 对象则直接返回。
    if isinstance(struct_source, Structure):
        return struct_source

    # Resolve to an absolute Path.
    # 解析为绝对路径。
    p = Path(struct_source).expanduser().resolve()

    # Direct file: delegate to pymatgen.
    # 直接文件：委托给 pymatgen。
    if p.is_file():
        return Structure.from_file(p)

    if p.is_dir():
        # Try canonical VASP output/input filenames in priority order.
        # 按优先级依次尝试标准 VASP 输出/输入文件名。
        for fname in ["CONTCAR", "POSCAR", "POSCAR.vasp"]:
            fp = p / fname
            if fp.exists():
                return Structure.from_file(fp)

        # Fall back to any single .vasp file in the directory.
        # 退而查找目录内唯一的 .vasp 文件。
        vasp_files = list(p.glob("*.vasp"))
        if len(vasp_files) == 1:
            return Structure.from_file(vasp_files[0])
        elif len(vasp_files) > 1:
            # Warn and use the first file when ambiguous.
            # 存在歧义时发出警告并使用第一个文件。
            logger.warning("Multiple .vasp files found in %s; using %s", p, vasp_files[0])
            return Structure.from_file(vasp_files[0])

        # Fall back to a single .cif file.
        # 退而查找唯一的 .cif 文件。
        cif_files = list(p.glob("*.cif"))
        if len(cif_files) == 1:
            return Structure.from_file(cif_files[0])
        elif len(cif_files) > 1:
            raise FileNotFoundError(f"Multiple CIF files found in {p}. Please specify one explicitly.")

    raise FileNotFoundError(f"No valid structure file found in: {p}")


def _parse_vasp_compressed_list(vasp_format: Union[str, Sequence[Any]]) -> Optional[List[float]]:
    """Parse a VASP-style compressed entry string such as "2*1.0 3*0.0 1.5".

    Each token is either a plain float ("1.5") or a repeat expression ("3*0.0").
    Returns None when any token cannot be parsed.

    解析 VASP 风格的压缩列表字符串，例如 "2*1.0 3*0.0 1.5"。

    每个 token 可以是普通浮点数（"1.5"）或重复表达式（"3*0.0"）。
    任意 token 解析失败时返回 None。

    Args:
        vasp_format: A VASP compressed string, or a list/tuple of tokens.
                     VASP 压缩字符串，或 token 组成的列表/元组。

    Returns:
        Expanded list of floats, or None on parse failure.
        展开后的浮点数列表，解析失败时返回 None。
    """

    def expand_token(tok: Any) -> Optional[List[float]]:
        """Expand a single token into a list of floats.
        扩展单个 token 为浮点数列表。
        """
        if tok is None:
            return None
        # Numeric types are accepted directly.
        # 直接接受数值类型。
        if isinstance(tok, (int, float)):
            return [float(tok)]
        s = str(tok).strip()
        if not s:
            return None
        if "*" in s:
            # Format: "N*value" — repeat value N times.
            # 格式："N*value" —— 将 value 重复 N 次。
            parts = s.split("*")
            if len(parts) != 2:
                return None
            n_str, v_str = parts[0].strip(), parts[1].strip()
            try:
                n = int(float(n_str))
                v = float(v_str)
            except ValueError:
                return None
            if n < 0:
                return None
            return [v] * n
        else:
            # Plain float token.
            # 普通浮点数 token。
            try:
                return [float(s)]
            except ValueError:
                return None

    # Split a string into whitespace-delimited tokens.
    # 将字符串按空白符分割为 token。
    if isinstance(vasp_format, str):
        tokens = vasp_format.split()
    elif isinstance(vasp_format, (list, tuple)):
        tokens = list(vasp_format)
    else:
        return None

    values: List[float] = []
    for t in tokens:
        expanded = expand_token(t)
        if expanded is None:
            # Log and abort on the first unparseable token.
            # 遇到第一个无法解析的 token 时记录日志并中止。
            logger.warning("Failed to parse VASP compressed token: %r", t)
            return None
        values.extend(expanded)
    return values


def get_vasp_species_order(structure: Structure) -> List[str]:
    """Return the element order as it would appear in a POSCAR file for the structure.

    返回结构在 POSCAR 文件中的元素顺序列表。

    Args:
        structure: A pymatgen Structure object.
                   pymatgen Structure 对象。

    Returns:
        Ordered list of element symbols without duplicates, preserving site order.
        按位点顺序排列的不重复元素符号列表。
    """
    # dict.fromkeys preserves insertion order while deduplicating.
    # dict.fromkeys 在去重的同时保留插入顺序。
    return list(dict.fromkeys(site.species_string for site in structure))


def convert_vasp_format_to_pymatgen_dict(structure: Structure, key: str, vasp_format: Union[str, Sequence[Any]]):
    """Convert a VASP-style compressed list to a pymatgen per-species dict for supported INCAR keys.

    Handles two input sizes: one value per atom (atom-level) or one value per
    unique species.  When values differ between atoms of the same species, the
    conversion is aborted and None is returned.

    将 VASP 风格的压缩列表转换为 pymatgen 按物种的字典，适用于受支持的 INCAR 键。

    支持两种输入长度：每个原子一个值（原子级），或每个独立物种一个值。
    同一物种的原子间值不一致时，转换中止并返回 None。

    Args:
        structure: A pymatgen Structure object.
                   pymatgen Structure 对象。
        key: The INCAR tag name (e.g. "MAGMOM", "LDAUU").
             INCAR 标签名称（如 "MAGMOM"、"LDAUU"）。
        vasp_format: VASP compressed value string or sequence.
                     VASP 压缩值字符串或序列。

    Returns:
        Dict of the form {key: {species: value}} for supported keys, or None.
        受支持键返回 {key: {物种: 值}} 形式的字典，否则返回 None。
    """
    # Expand the VASP compressed list.
    # 展开 VASP 压缩列表。
    values = _parse_vasp_compressed_list(vasp_format)
    if not values:
        return None

    num_total_atoms = len(structure)
    species_order = get_vasp_species_order(structure)
    num_unique_species = len(species_order)

    species_map: Dict[str, float] = {}
    if len(values) == num_total_atoms:
        # Atom-level values: verify consistency within each species.
        # 原子级值：验证同一物种内部的一致性。
        for site, v in zip(structure, values):
            sp = site.species_string
            if sp not in species_map:
                species_map[sp] = v
            elif abs(species_map[sp] - v) > 1e-12:
                # Differing values within the same species cannot be reduced to a species dict.
                # 同一物种内值不同，无法归约为按物种的字典。
                logger.warning(
                    "%s: atom-level values differ for species '%s'. Cannot convert to species dict.",
                    key, sp
                )
                return None
    elif len(values) == num_unique_species:
        # Species-level values: map directly by position.
        # 物种级值：按位置直接映射。
        species_map = dict(zip(species_order, values))
    else:
        # Value count matches neither total atoms nor unique species.
        # 值的数量既不匹配总原子数也不匹配独立物种数。
        warnings.warn(
            f"{key}: value count ({len(values)}) matches neither total atoms ({num_total_atoms}) "
            f"nor unique species ({num_unique_species})."
        )
        return None

    # Only wrap in a keyed dict for INCAR tags that use per-species dicts.
    # 仅对使用按物种字典的 INCAR 标签进行包装。
    if key in {"MAGMOM", "LDAUU", "LDAUJ", "LDAUL"}:
        return {key: species_map}
    return None


def infer_functional_from_incar(incar: Dict[str, Any]) -> str:
    """Infer the DFT functional from an INCAR parameter dictionary.

    Returns "BEEF" if BEEF-vdW-specific keys are present, otherwise "PBE".
    Falls back to a minimal heuristic when the package context is unavailable.

    从 INCAR 参数字典推断 DFT 泛函。

    若存在 BEEF-vdW 专属键则返回 "BEEF"，否则返回 "PBE"。
    当包上下文不可用时退回到最小启发式判断。

    Args:
        incar: Dictionary of INCAR key-value pairs.
               INCAR 键值对字典。

    Returns:
        "BEEF" or "PBE" as a string.
        字符串 "BEEF" 或 "PBE"。
    """
    # Prefer package-relative import; fall back to a minimal heuristic if the package context is missing.
    # 优先使用包内相对导入；若包上下文缺失则退回最小启发式判断。
    try:
        from .constants import _BEEF_INCAR
        beef_keys = set(_BEEF_INCAR.keys())
    except Exception:
        beef_keys = {"LUSE_VDW", "Zab_vdW"}

    # 1. 检查特定的 BEEF 标志键（排除 GGA 以避免误判）
    # 1. Check for BEEF-specific flag keys (exclude GGA to avoid false positives).
    if any(k in incar for k in beef_keys if k != "GGA"):
        return "BEEF"

    # 2. 安全地检查 GGA 键的值
    # 2. Safely inspect the GGA key value.
    gga_val = str(incar.get("GGA", "")).upper()
    if "BF" in gga_val:
        return "BEEF"

    return "PBE"

def formula_to_counts(formula: str) -> Dict[str, int]:
    """Convert a chemical formula string to an element-count dictionary.

    将化学式字符串转换为元素计数字典。

    Args:
        formula: Chemical formula string (e.g. "H2O", "CO2").
                 化学式字符串（如 "H2O"、"CO2"）。

    Returns:
        Dict mapping element symbol to integer atom count.
        元素符号到整数原子数的字典。

    Raises:
        ValueError: If any stoichiometric coefficient is non-integer or the
                    formula is empty/invalid.
                    若任意化学计量系数为非整数或化学式为空/无效时抛出。
    """
    comp = Composition(formula)
    counts: Dict[str, int] = {}
    for el, amt in comp.get_el_amt_dict().items():
        # Ensure all amounts are integer stoichiometry.
        # 确保所有数量均为整数化学计量数。
        if abs(amt - round(amt)) > 1e-8:
            raise ValueError(f"adsorbate_formula must be integer stoichiometry, got: {formula}")
        counts[str(el)] = int(round(amt))
    if not counts:
        raise ValueError(f"adsorbate_formula is empty/invalid: {formula}")
    return counts


def structure_element_counts(structure: Structure) -> Dict[str, int]:
    """Return a dict of element symbol to integer atom count for a structure.

    返回结构中元素符号到整数原子数的字典。

    Args:
        structure: A pymatgen Structure object.
                   pymatgen Structure 对象。

    Returns:
        Dict mapping element symbol to integer count.
        元素符号到整数计数的字典。
    """
    return {str(el): int(amt) for el, amt in structure.composition.get_el_amt_dict().items()}


def pick_adsorbate_indices_by_formula_strict(
    structure: Structure, adsorbate_formula: str, prefer: str = "tail"
) -> List[int]:
    """Choose the site indices in a structure that correspond to an adsorbate formula (strict mode).

    Raises ValueError when the structure lacks the required atoms or when the
    adsorbate elements are ambiguous (appear in non-adsorbate sites too).

    在结构中选取与吸附质化学式对应的位点索引（严格模式）。

    若结构缺少所需原子，或吸附质元素存在歧义（同时出现在非吸附质位点），
    则抛出 ValueError。

    Args:
        structure: A pymatgen Structure object.
                   pymatgen Structure 对象。
        adsorbate_formula: Chemical formula of the adsorbate (e.g. "CO", "OH").
                           吸附质化学式（如 "CO"、"OH"）。
        prefer: Which end of each element's site list to pick from: "tail" (default,
                typically adsorbate atoms added last) or "head".
                从每种元素位点列表的哪一端选取："tail"（默认，通常吸附质原子最后添加）
                或 "head"。

    Returns:
        Sorted list of integer site indices corresponding to the adsorbate.
        对应吸附质的整数位点索引的有序列表。

    Raises:
        ValueError: If required atoms are missing or the formula is ambiguous.
                    若缺少所需原子或化学式存在歧义时抛出。
    """
    # 直接利用 pymatgen 底层能力
    # Use pymatgen's Composition to parse the adsorbate formula.
    # 使用 pymatgen 的 Composition 解析吸附质化学式。
    need = {str(k): int(v) for k, v in Composition(adsorbate_formula).get_el_amt_dict().items()}
    have = {str(k): int(v) for k, v in structure.composition.get_el_amt_dict().items()}

    # Check for missing elements.
    # 检查是否缺少元素。
    missing = {el: n for el, n in need.items() if have.get(el, 0) < n}
    if missing:
        raise ValueError(f"Structure lacks atoms for {adsorbate_formula}: {missing}")

    # Detect ambiguous elements that appear more than needed (may be in substrate).
    # 检测出现次数多于需要量的歧义元素（可能存在于基底中）。
    ambiguous = {el: (have.get(el, 0), need[el]) for el in need if have.get(el, 0) > need[el]}
    if ambiguous:
        raise ValueError(
            f"Formula ambiguous. Elements appear elsewhere: {ambiguous}. Provide vibrate_indices explicitly."
        )

    # Build a mapping from element symbol to list of site indices.
    # 构建元素符号到位点索引列表的映射。
    elem_to_indices: Dict[str, List[int]] = {}
    for i, site in enumerate(structure):
        elem_to_indices.setdefault(site.species_string, []).append(i)

    # Select indices from the head or tail of each element's pool.
    # 从每种元素池的头部或尾部选取索引。
    chosen: List[int] = []
    for el, n in need.items():
        pool = elem_to_indices[el]
        chosen.extend(pool[:n] if prefer == "head" else pool[-n:])
    return sorted(chosen)


def detect_adsorbate_indices(structure: Structure, z_cutoff: float = 2.0) -> List[int]:
    """Heuristically identify adsorbate atoms as those above the largest z-coordinate gap.

    Returns an empty list when no vacuum gap larger than z_cutoff is found.

    启发式地将最大 z 坐标间隙以上的原子识别为吸附质。

    若未找到大于 z_cutoff 的真空间隙则返回空列表。

    Args:
        structure: A pymatgen Structure object.
                   pymatgen Structure 对象。
        z_cutoff: Minimum gap size in Angstroms to be considered a vacuum layer (default 2.0).
                  被视为真空层的最小间隙尺寸（单位 Å，默认 2.0）。

    Returns:
        List of site indices above the largest z gap.
        最大 z 间隙以上位点索引的列表。
    """
    if len(structure) < 2:
        return []

    # 提取 Z 坐标并排序 (Cartesian coordinates)
    # Extract and sort Cartesian z-coordinates along with their site indices.
    # 提取并排序笛卡尔 z 坐标及其位点索引。
    sites_z = sorted([(i, site.coords[2]) for i, site in enumerate(structure)], key=lambda x: x[1])

    max_gap = 0.0
    gap_index = -1

    # Identify the largest gap between consecutive z-positions.
    # 找出相邻 z 位置之间的最大间隙。
    for i in range(len(sites_z) - 1):
        gap = sites_z[i + 1][1] - sites_z[i][1]
        if gap > max_gap:
            max_gap = gap
            gap_index = i

    # Return empty list when gap is below threshold.
    # 间隙低于阈值时返回空列表。
    if max_gap < z_cutoff or gap_index == -1:
        logger.info("No vacuum gap > %.2f Å found. Assuming no adsorbate.", z_cutoff)
        return []

    # All sites above the gap are considered adsorbate atoms.
    # 间隙以上的所有位点均视为吸附质原子。
    return [x[0] for x in sites_z[gap_index + 1 :]]
