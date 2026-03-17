"""常量与默认值"""

from __future__ import annotations

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
