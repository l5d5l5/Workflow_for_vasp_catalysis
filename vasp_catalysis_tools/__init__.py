"""
VASP Catalysis Tools
-------------------
A modular package for structure modification and catalysis simulation with VASP.

This package provides tools for:
- Converting bulk structures to slabs
- Finding and modifying adsorption sites
- Manipulating crystal structures
"""

__version__ = "1.0.0"

from .core.bulk_to_slab import BulkToSlabGenerator
from .core.adsorption import AdsorptionModify
from .core.structure_modify import StructureModify

__all__ = [
    'BulkToSlabGenerator',
    'AdsorptionModify',
    'StructureModify'
]