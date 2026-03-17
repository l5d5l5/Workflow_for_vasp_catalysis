"""Modularized VASP input generation package (ECAT-style)."""

from .constants import _BEEF_INCAR
from .input_sets import (
    BulkRelaxSetEcat,
    FreqSetEcat,
    LobsterSetEcat,
    MPStaticSetEcat,
    NEBSetEcat,
    SlabSetEcat,
)
from .maker import VaspInputMaker
from .script import ScriptRenderer
from .utils import (
    convert_vasp_format_to_pymatgen_dict,
    detect_adsorbate_indices,
    formula_to_counts,
    get_vasp_species_order,
    infer_functional_from_incar,
    load_structure,
    pick_adsorbate_indices_by_formula_strict,
    structure_element_counts,
)
from .kpoints import build_kpoints_by_lengths

__all__ = [
    "_BEEF_INCAR",
    "BulkRelaxSetEcat",
    "FreqSetEcat",
    "LobsterSetEcat",
    "MPStaticSetEcat",
    "NEBSetEcat",
    "SlabSetEcat",
    "VaspInputMaker",
    "ScriptRenderer",
    "convert_vasp_format_to_pymatgen_dict",
    "detect_adsorbate_indices",
    "formula_to_counts",
    "get_vasp_species_order",
    "infer_functional_from_incar",
    "load_structure",
    "pick_adsorbate_indices_by_formula_strict",
    "structure_element_counts",
    "build_kpoints_by_lengths",
]
