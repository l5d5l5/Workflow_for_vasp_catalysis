# -*- coding: utf-8 -*-
"""常量与默认值"""

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