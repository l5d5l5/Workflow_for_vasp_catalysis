"""工具函数：结构读取、VASP格式解析、功能推断等。"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pymatgen.core import Composition, Structure
from pymatgen.io.vasp.inputs import Poscar

logger = logging.getLogger(__name__)


def load_structure(struct_source: Union[str, Path, Structure]) -> Structure:
    """Load a structure from a file/dir/Structure object.

    Supported inputs:
    - Structure instance
    - Directory: searches for CONTCAR/POSCAR/POSCAR.vasp > *.vasp > *.cif
    - File path
    """
    if isinstance(struct_source, Structure):
        return struct_source

    p = Path(struct_source).expanduser()

    if p.is_dir():
        candidates: List[Path] = []
        for fname in ["CONTCAR", "POSCAR", "POSCAR.vasp"]:
            fp = p / fname
            if fp.exists():
                candidates.append(fp)
        if not candidates:
            vasp_like = sorted(p.glob("*.vasp"))
            if vasp_like:
                candidates.extend(vasp_like)
        if not candidates:
            cif_files = sorted(p.glob("*.cif"))
            if len(cif_files) == 1:
                candidates.append(cif_files[0])
            elif len(cif_files) > 1:
                raise FileNotFoundError(
                    f"Multiple CIF files found in {p}. Please specify one explicitly: "
                    + ", ".join(str(x.name) for x in cif_files)
                )

        if not candidates:
            raise FileNotFoundError(f"No structure file found in folder: {p}")

        if len(candidates) > 1:
            logger.warning(
                "Multiple structure candidates found in %s; using the first one: %s", p, candidates[0]
            )

        # Read the first candidate structure file with pymatgen
        selected = candidates[0]
        try:
            if selected.suffix.lower() in {".cif"}:
                return Structure.from_file(str(selected))
            # For VASP-format files (POSCAR/CONTCAR/*.vasp)
            return Poscar.from_file(str(selected)).structure
        except Exception as e:
            raise RuntimeError(f"Failed to load structure from {selected}: {e}")


def _parse_vasp_compressed_list(vasp_format: Union[str, Sequence[Any]]) -> Optional[List[float]]:
    """Parse VASP-style compressed entries like "2*1.0 3*0.0 1.5".

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
    try:
        return Poscar(structure).site_symbols
    except Exception:
        seen: set = set()
        order: List[str] = []
        for site in structure:
            sp = site.species_string
            if sp not in seen:
                seen.add(sp)
                order.append(sp)
        return order


def convert_vasp_format_to_pymatgen_dict(
    structure: Structure, key: str, vasp_format: Union[str, Sequence[Any]]
) -> Optional[Dict[str, Dict[str, float]]]:
    """Convert VASP-style compressed list to pymatgen dict format.

    Supports:
      - MAGMOM: {"MAGMOM": {"Fe": 2.0, "O": 0.6}}
      - LDAU*: similarly.

    Supports both atom-level (len==num_atoms) and species-level (len==num_species) lists.
    """
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
            v = float(v)
            if sp not in species_map:
                species_map[sp] = v
            else:
                if abs(species_map[sp] - v) > 1e-12:
                    logger.warning(
                        "%s: atom-level values contain different entries for the same species '%s'. "
                        "Cannot safely convert to species dict; leaving as-is.",
                        key,
                        sp,
                    )
                    return None

    elif len(values) == num_unique_species:
        species_map = dict(zip(species_order, [float(x) for x in values]))

    else:
        warnings.warn(
            f"{key}: value count ({len(values)}) does not match total atoms ({num_total_atoms}) "
            f"or unique species ({num_unique_species}). Cannot convert."
        )
        return None

    if key == "MAGMOM":
        return {"MAGMOM": species_map}
    if key in ["LDAUU", "LDAUJ", "LDAUL"]:
        return {key: species_map}
    return None


def infer_functional_from_incar(incar: Dict[str, Any]) -> str:
    """Infer functional (PBE/BEEF) from INCAR.

    The heuristics are intentionally conservative to avoid misclassifying regular PBE jobs
    that happen to contain keys used by BEEF (e.g., GGA).
    """
    from .constants import _BEEF_INCAR

    # If any explicit BEEF-related key is present (except GGA), treat it as BEEF.
    for key in _BEEF_INCAR:
        if key in incar and key != "GGA":
            return "BEEF"

    # If GGA is present, check whether it explicitly indicates BEEF/BF.
    gga_val = incar.get("GGA")
    if isinstance(gga_val, str) and "BF" in gga_val.upper():
        return "BEEF"

    # If any value contains 'BEEF', treat as BEEF (covers some nonstandard cases).
    for v in incar.values():
        if isinstance(v, str) and "BEEF" in v.upper():
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
    counts: Dict[str, int] = {}
    for site in structure:
        el = site.species_string
        counts[el] = counts.get(el, 0) + 1
    return counts


def pick_adsorbate_indices_by_formula_strict(
    structure: Structure, adsorbate_formula: str, prefer: str = "tail"
) -> List[int]:
    """Choose atomic indices matching an adsorbate formula (strict mode)."""
    need = formula_to_counts(adsorbate_formula)
    have = structure_element_counts(structure)

    missing = {el: n for el, n in need.items() if have.get(el, 0) < n}
    if missing:
        raise ValueError(
            f"Structure does not contain enough atoms for {adsorbate_formula}: {missing}"
        )

    ambiguous = {el: (have.get(el, 0), need[el]) for el in need if have.get(el, 0) > need[el]}
    if ambiguous:
        raise ValueError(
            "adsorbate_formula is ambiguous because these elements also appear elsewhere in the structure: "
            f"{ambiguous}. Please provide vibrate_indices explicitly."
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
        return [0]

    sites_z = sorted([(i, site.coords[2]) for i, site in enumerate(structure)], key=lambda x: x[1])

    max_gap = 0.0
    gap_index = 0
    for i in range(len(sites_z) - 1):
        gap = sites_z[i + 1][1] - sites_z[i][1]
        if gap > max_gap:
            max_gap = gap
            gap_index = i

    adsorbate_indices = [x[0] for x in sites_z[gap_index + 1 :]]
    return adsorbate_indices
