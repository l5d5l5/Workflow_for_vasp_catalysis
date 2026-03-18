# -*- coding: utf-8 -*-
"""工具函数：结构读取、VASP格式解析、功能推断等。"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pymatgen.core import Composition, Structure
from pymatgen.io.vasp.inputs import Poscar

logger = logging.getLogger(__name__)


def load_structure(struct_source: Union[str, Path, Structure]) -> Structure:
    """Load a structure from a file/dir/Structure object.
    
    Priority for directories: CONTCAR > POSCAR > POSCAR.vasp > *.vasp > *.cif
    """
    if isinstance(struct_source, Structure):
        return struct_source

    p = Path(struct_source).expanduser().resolve()

    if p.is_file():
        return Structure.from_file(p)
    
    if p.is_dir():
        for fname in ["CONTCAR", "POSCAR", "POSCAR.vasp"]:
            fp = p / fname
            if fp.exists():
                return Structure.from_file(fp)
            
        vasp_files = list(p.glob("*.vasp"))
        if len(vasp_files) == 1:
            return Structure.from_file(vasp_files[0])
        elif len(vasp_files) > 1:
            logger.warning("Multiple .vasp files found in %s; using %s", p, vasp_files[0])
            return Structure.from_file(vasp_files[0])
        
        cif_files = list(p.glob("*.cif"))
        if len(cif_files) == 1:
            return Structure.from_file(cif_files[0])
        elif len(cif_files) > 1:
            raise FileNotFoundError(f"Multiple CIF files found in {p}. Please specify one explicitly.")

    raise FileNotFoundError(f"No valid structure file found in: {p}")    


def _parse_vasp_compressed_list(vasp_format: Union[str, Sequence[Any]]) -> Optional[List[float]]:
    """
    Parse VASP-style compressed entries like "2*1.0 3*0.0 1.5".
    Returns a list of floats or None when unable to parse.
    """

    def expand_token(tok: Any) -> Optional[List[float]]:
        if tok is None:
            return None
        if isinstance(tok, (int, float)):
            return [float(tok)]
        s = str(tok).strip()
        if not s:
            return None
        if "*" in s:
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
            try:
                return [float(s)]
            except ValueError:
                return None

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
            logger.warning("Failed to parse VASP compressed token: %r", t)
            return None
        values.extend(expanded)
    return values


def get_vasp_species_order(structure: Structure) -> List[str]:
    """Get POSCAR element order for a structure."""
    return list(dict.fromkeys(site.species_string for site in structure))


def convert_vasp_format_to_pymatgen_dict(structure: Structure, key: str, vasp_format: Union[str, Sequence[Any]]):
    """Convert VASP-style compressed list to pymatgen dict format."""
    values = _parse_vasp_compressed_list(vasp_format)
    if not values:
        return None

    num_total_atoms = len(structure)
    species_order = get_vasp_species_order(structure)
    num_unique_species = len(species_order)

    species_map: Dict[str, float] = {}
    if len(values) == num_total_atoms:
        for site, v in zip(structure, values):
            sp = site.species_string
            if sp not in species_map:
                species_map[sp] = v
            elif abs(species_map[sp] - v) > 1e-12:
                logger.warning(
                    "%s: atom-level values differ for species '%s'. Cannot convert to species dict.",
                    key, sp
                )
                return None
    elif len(values) == num_unique_species:
        species_map = dict(zip(species_order, values))
    else:
        warnings.warn(
            f"{key}: value count ({len(values)}) matches neither total atoms ({num_total_atoms}) "
            f"nor unique species ({num_unique_species})."
        )
        return None

    if key in {"MAGMOM", "LDAUU", "LDAUJ", "LDAUL"}:
        return {key: species_map}
    return None


def infer_functional_from_incar(incar: Dict[str, Any]) -> str:
    """Infer functional (PBE/BEEF) from INCAR safely."""
    # Prefer package-relative import; fall back to a minimal heuristic if the package context is missing.
    try:
        from .constants import _BEEF_INCAR
        beef_keys = set(_BEEF_INCAR.keys())
    except Exception:
        beef_keys = {"LUSE_VDW", "Zab_vdW"}

    # 1. 检查特定的 BEEF 标志键（排除 GGA 以避免误判）
    if any(k in incar for k in beef_keys if k != "GGA"):
        return "BEEF"

    # 2. 安全地检查 GGA 键的值
    gga_val = str(incar.get("GGA", "")).upper()
    if "BF" in gga_val:
        return "BEEF"

    return "PBE"

def formula_to_counts(formula: str) -> Dict[str, int]:
    comp = Composition(formula)
    counts: Dict[str, int] = {}
    for el, amt in comp.get_el_amt_dict().items():
        if abs(amt - round(amt)) > 1e-8:
            raise ValueError(f"adsorbate_formula must be integer stoichiometry, got: {formula}")
        counts[str(el)] = int(round(amt))
    if not counts:
        raise ValueError(f"adsorbate_formula is empty/invalid: {formula}")
    return counts


def structure_element_counts(structure: Structure) -> Dict[str, int]:
    return {str(el): int(amt) for el, amt in structure.composition.get_el_amt_dict().items()}


def pick_adsorbate_indices_by_formula_strict(
    structure: Structure, adsorbate_formula: str, prefer: str = "tail"
) -> List[int]:
    """Choose atomic indices matching an adsorbate formula (strict mode)."""
    # 直接利用 pymatgen 底层能力
    need = {str(k): int(v) for k, v in Composition(adsorbate_formula).get_el_amt_dict().items()}
    have = {str(k): int(v) for k, v in structure.composition.get_el_amt_dict().items()}

    missing = {el: n for el, n in need.items() if have.get(el, 0) < n}
    if missing:
        raise ValueError(f"Structure lacks atoms for {adsorbate_formula}: {missing}")

    ambiguous = {el: (have.get(el, 0), need[el]) for el in need if have.get(el, 0) > need[el]}
    if ambiguous:
        raise ValueError(
            f"Formula ambiguous. Elements appear elsewhere: {ambiguous}. Provide vibrate_indices explicitly."
        )

    elem_to_indices: Dict[str, List[int]] = {}
    for i, site in enumerate(structure):
        elem_to_indices.setdefault(site.species_string, []).append(i)

    chosen: List[int] = []
    for el, n in need.items():
        pool = elem_to_indices[el]
        chosen.extend(pool[:n] if prefer == "head" else pool[-n:])
    return sorted(chosen)


def detect_adsorbate_indices(structure: Structure, z_cutoff: float = 2.0) -> List[int]:
    """Heuristic: identify adsorbate atoms above the largest gap along z."""
    if len(structure) < 2:
        return []

    # 提取 Z 坐标并排序 (Cartesian coordinates)
    sites_z = sorted([(i, site.coords[2]) for i, site in enumerate(structure)], key=lambda x: x[1])

    max_gap = 0.0
    gap_index = -1
    
    for i in range(len(sites_z) - 1):
        gap = sites_z[i + 1][1] - sites_z[i][1]
        if gap > max_gap:
            max_gap = gap
            gap_index = i

    if max_gap < z_cutoff or gap_index == -1:
        logger.info("No vacuum gap > %.2f Å found. Assuming no adsorbate.", z_cutoff)
        return []

    return [x[0] for x in sites_z[gap_index + 1 :]]
