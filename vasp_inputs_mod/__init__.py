"""Modularized VASP input generation package."""

from .constants import (
    _BEEF_INCAR,
    DEFAULT_INCAR_BULK,
    DEFAULT_INCAR_SLAB,
    DEFAULT_INCAR_STATIC,
    DEFAULT_INCAR_NEB,
    DEFAULT_INCAR_DIMER,
    DEFAULT_INCAR_FREQ,
    DEFAULT_INCAR_NBO,
    DEFAULT_INCAR_LOBSTER,
    DEFAULT_NBO_CONFIG_PARAMS,
    MODULE_DIR,
)
from .input_sets import (
    BulkRelaxSetEcat,
    FreqSetEcat,
    LobsterSetEcat,
    MPStaticSetEcat,
    NEBSetEcat,
    SlabSetEcat,
    NBOSetEcat,
    DimerSetEcat,
    VaspInputSetEcat,
)
from .maker import VaspInputMaker
from .script import Script
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
    "NBOSetEcat",
    "VaspInputSetEcat",
    "DEFAULT_INCAR_BULK",
    "DEFAULT_INCAR_SLAB",
    "DEFAULT_INCAR_STATIC",
    "DEFAULT_INCAR_NEB",
    "DEFAULT_INCAR_DIMER",
    "DEFAULT_INCAR_FREQ",
    "DEFAULT_INCAR_NBO",
    "DEFAULT_INCAR_LOBSTER",
    "DEFAULT_NBO_CONFIG_PARAMS",
    "MODULE_DIR",
    "DimerSetEcat",
    "SlabSetEcat",
    "VaspInputMaker",
    "Script",
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
