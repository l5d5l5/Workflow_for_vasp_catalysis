# -*- coding: utf-8 -*-
"""
flow.calc_type — CalcType enum, single source of truth.
=========================================================
Both modules now import CalcType from here instead of from each other.
"""

from enum import Enum


class CalcType(Enum):
    """标准化计算类型枚举 - 用户唯一需要选择的参数"""
    # === 结构优化类 ===
    BULK_RELAX = "bulk_relax"
    SLAB_RELAX = "slab_relax"

    # === 电子结构计算类 ===
    STATIC_SP = "static_sp"
    DOS_SP = "static_dos"
    CHG_SP = "static_charge_density"
    ELF_SP = "static_elf"

    # === 过渡态搜索类 ===
    NEB = "neb"
    DIMER = "dimer"

    # === 频率计算类 ===
    FREQ = "freq"
    FREQ_IR = "freq_ir"

    # === 性质分析类 ===
    LOBSTER = "lobster"
    NMR_CS = "nmr_cs"
    NMR_EFG = "nmr_efg"
    NBO = "nbo"

    # === 分子动力学类 ===
    MD_NVT = "md_nvt"
    MD_NPT = "md_npt"