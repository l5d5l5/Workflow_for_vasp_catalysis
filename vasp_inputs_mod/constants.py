# -*- coding: utf-8 -*-
"""常量与默认值"""
from pathlib import Path

_BEEF_INCAR = {
    "GGA": "BF",
    "LUSE_VDW": True,
    "AGGAC": 0.0000,
    "LASPH": True,
    "Zab_VDW": -1.8867,
}

DEFAULT_INCAR_BULK = {
    "EDIFFG": -0.02,
    "EDIFF": 1e-6,
    "POTIM": 0.20,
    "ENCUT": 520,
    "IBRION": 2,
    "LORBIT": 10,
    "NSW": 500,
    "ISIF": 3,
    "LREAL": "Auto",
}

DEFAULT_INCAR_SLAB = {
    "EDIFFG": -0.02,
    "ENCUT": 420,
    "POTIM": 0.20,
    "EDIFF": 1e-4,
    "IBRION": 2,
    "ISIF": 2,
    "NSW": 500,
    "LREAL": "Auto",
}

DEFAULT_INCAR_STATIC = {
    "EDIFF": 1e-6,
    "NELM": 200,
    "IBRION": -1,
    "NEDOS": 3001,
    "LCHARG": True,
    "LAECHG": True,
    "LELF": True,
    "NSW": 0,
    "ISIF": 0,
    "LORBIT": 11,
    "ISMEAR": -5,
    "SIGMA": 0.05,
}

DEFAULT_INCAR_FREQ = {
    "IBRION": 5,
    "POTIM": 0.015,
    "NFREE": 2,
    "NSW": 1,
    "EDIFF": 1e-7,
    "NELM": 200,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LREAL": False,
    "ALGO": "Fast",
    "LCHARG": False,
    "LWAVE": False,
    "LORBIT": 11,
}

DEFAULT_INCAR_LOBSTER = {
    "NELM": 150,
    "NCORE": 6,
    "IBRION": -1,
    "EDIFF": 1e-6,
    "LORBIT": 11,
    "NSW": 0,
    "LCHARG": True,
    "LWAVE": True,
}

DEFAULT_INCAR_NEB = {
    "EDIFF": 1e-5,
    "NELM": 150,
    "POTIM": 0.02,
    "ICHAIN": 0,
    "SPRING": -5.0,
    "IBRION": 3,
    "LCLIMB": True,
    "LREAL": "Auto",
}

DEFAULT_INCAR_DIMER = {
    "ICHAIN": 2,
    "IOPT": 2,
    "IBRION": 3,
    "POTIM": 0.0,
    "EDIFF": 1e-7,
    "DdR": 0.005,
    "DRotMax": 3,
    "DFNMax": 1.0,
    "DFMin": 0.01,
    "NSW": 1000,
    "LREAL": "Auto",
}

DEFAULT_INCAR_NBO = {
    "NSW": 0,          
    "IBRION": -1,      
    "LNBO": True,      
    "LWAVE": True,     
    "LCHARG": True,    
}

DEFAULT_NBO_CONFIG_PARAMS = {
    "occ_1c": "1.60",
    "occ_2c": "1.85",
    "print_cube": "F",
    "density": "F",
    "vis_start": "0",
    "vis_end": "-1",
    "mesh_x": "0", "mesh_y": "0", "mesh_z": "0",
    "box_x": "1", "box_y": "1", "box_z": "1",
    "origin_fact": "0.00"
}

NBO_CONFIG_TEMPLATE = """#####NBO search parameters####
  {occ_1c}   #Occupancy cutoff for one-center NBOs
  {occ_2c}   #Occupancy cutoff for two-center NBOs
#####Visualization output control parameters####
     {print_cube}   #Control over printing of .cube files for visualization.
     {density}   #density - Whether density (T) or wavefunctions (F) are visualized.
  {vis_start} {vis_end}   #vis_start vis_end - Start and end of NBOs to print .cube files for
{mesh_x} {mesh_y} {mesh_z}   #mesh - Number of points along each lattice vectors to use in .cube files
{box_x} {box_y} {box_z}   #box_int - Number of unit cell to use for .cube file. See READ_ME.txt for guidance
  {origin_fact}   #origin_fact - Shift of the origin for .cube file. See READ_ME.txt for guidance
"""

MODULE_DIR = Path(__file__).resolve().parent
NBO_BASIS_PATH = MODULE_DIR / "basis" / "basis-ANO-RCC-MB.txt"