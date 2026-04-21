# -*- coding: utf-8 -*-
"""
INCAR defaults, functional patches, and shared constants for the flow package.
Provides per-functional INCAR patch dictionaries, per-calculation-type default
INCAR parameter sets, NBO configuration defaults, and filesystem path constants.

为 flow 包提供的 INCAR 默认值、泛函补丁和共享常量。
包含各泛函的 INCAR 补丁字典、各计算类型的默认 INCAR 参数集、
NBO 配置默认值以及文件系统路径常量。
"""

from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Functional-specific INCAR patches
# Applied on top of any base INCAR when the matching functional is used.
# input_sets.py reads these via BEEF_INCAR_SETTINGS; new code uses
# FUNCTIONAL_INCAR_PATCHES[functional_upper] for extensible lookup.
# ---------------------------------------------------------------------------
# 各泛函专属 INCAR 补丁
# 使用对应泛函时，这些设置会叠加覆盖基础 INCAR。
# input_sets.py 通过 BEEF_INCAR_SETTINGS 访问；新代码使用
# FUNCTIONAL_INCAR_PATCHES[functional_upper] 进行可扩展查找。
# ---------------------------------------------------------------------------

FUNCTIONAL_INCAR_PATCHES: Dict[str, Dict[str, Any]] = {
    # BEEF-vdW functional settings.
    # BEEF-vdW 泛函设置。
    "BEEF": {
        "GGA": "BF",
        "LUSE_VDW": True,
        "AGGAC": 0.0000,
        "LASPH": True,
        "Zab_VDW": -1.8867,
    },
    # BEEF-vdW + VTST (transition-state) functional settings.
    # BEEF-vdW + VTST（过渡态）泛函设置。
    "BEEFVTST": {
        "GGA": "BF",
        "LUSE_VDW": True,
        "AGGAC": 0.0000,
        "LASPH": True,
        "Zab_VDW": -1.8867,
    },
    # SCAN meta-GGA functional settings.
    # SCAN meta-GGA 泛函设置。
    "SCAN": {
        "METAGGA": "SCAN",
        "LASPH": True,
        "ADDGRID": True,
    },
    # HSE06 hybrid functional settings.
    # HSE06 杂化泛函设置。
    "HSE": {
        "LHFCALC": True,
        "AEXX": 0.25,
        "HFSCREEN": 0.2,
        "LASPH": True,
    },
    # PBE0 hybrid functional settings.
    # PBE0 杂化泛函设置。
    "PBE0": {
        "LHFCALC": True,
        "AEXX": 0.25,
        "LASPH": True,
    },
}

# Kept for input_sets.py backward compatibility.
# 保留此别名以维持 input_sets.py 的向后兼容性。
BEEF_INCAR_SETTINGS: Dict[str, Any] = FUNCTIONAL_INCAR_PATCHES["BEEF"]


# ---------------------------------------------------------------------------
# Default INCAR parameters per calculation type
# ---------------------------------------------------------------------------
# 各计算类型的默认 INCAR 参数
# ---------------------------------------------------------------------------

# Default INCAR settings for bulk structure relaxation.
# 体相结构弛豫的默认 INCAR 设置。
DEFAULT_INCAR_BULK: Dict[str, Any] = {
    "EDIFFG": -0.02,    # Force convergence criterion (eV/Å). / 力收敛判据（eV/Å）。
    "EDIFF": 1e-6,      # Electronic convergence criterion (eV). / 电子步收敛判据（eV）。
    "POTIM": 0.20,      # Ionic step size. / 离子步步长。
    "ENCUT": 520,       # Plane-wave energy cutoff (eV). / 平面波截断能（eV）。
    "IBRION": 2,        # Use conjugate gradient algorithm. / 使用共轭梯度算法。
    "LORBIT": 10,       # Output projected DOS. / 输出投影态密度。
    "NSW": 500,         # Maximum number of ionic steps. / 最大离子步数。
    "ISIF": 3,          # Relax ions and cell shape/volume. / 弛豫离子及晶胞形状/体积。
    "LREAL": "Auto",    # Real-space projection (automatic). / 实空间投影（自动）。
}

# Default INCAR settings for slab structure relaxation.
# 表面板结构弛豫的默认 INCAR 设置。
DEFAULT_INCAR_SLAB: Dict[str, Any] = {
    "EDIFFG": -0.02,    # Force convergence criterion (eV/Å). / 力收敛判据（eV/Å）。
    "ENCUT": 420,       # Plane-wave energy cutoff (eV). / 平面波截断能（eV）。
    "POTIM": 0.20,      # Ionic step size. / 离子步步长。
    "EDIFF": 1e-4,      # Electronic convergence criterion (eV). / 电子步收敛判据（eV）。
    "IBRION": 2,        # Use conjugate gradient algorithm. / 使用共轭梯度算法。
    "ISIF": 2,          # Relax ions only, fix cell. / 仅弛豫离子，固定晶胞。
    "NSW": 500,         # Maximum number of ionic steps. / 最大离子步数。
    "LREAL": "Auto",    # Real-space projection (automatic). / 实空间投影（自动）。
}

# Default INCAR settings for static (single-point) calculations.
# 静态（单点）计算的默认 INCAR 设置。
DEFAULT_INCAR_STATIC: Dict[str, Any] = {
    "EDIFF":  1e-6,     # Electronic convergence criterion (eV). / 电子步收敛判据（eV）。
    "NELM":   200,      # Maximum electronic SCF steps. / 最大电子自洽步数。
    "IBRION": -1,       # No ionic relaxation. / 不进行离子弛豫。
    "NEDOS":  3001,     # Number of DOS energy grid points. / DOS 能量网格点数。
    "NSW":    0,        # No ionic steps. / 离子步数为零。
    "ISIF":   0,        # No stress tensor calculation. / 不计算应力张量。
    "ISMEAR": -5,       # Tetrahedron smearing (good for insulators). / 四面体展宽（适用于绝缘体）。
    "SIGMA":  0.05,     # Smearing width (eV). / 展宽宽度（eV）。
    "LWAVE":  False,    # Do not write WAVECAR. / 不输出 WAVECAR。
}

# ── Static 子类型增量（仅包含各自独有的 key）────────────────────────────────
# Incremental INCAR settings for static sub-types (keys unique to each sub-type).
# 静态计算子类型的增量 INCAR 设置（仅包含各子类型独有的键）。

# static_sp: pure single-point, no extra output files.
# static_sp：纯单点，不输出额外文件。
INCAR_DELTA_STATIC_SP: Dict[str, Any] = {
    "LCHARG": False,    # Do not write CHGCAR. / 不输出 CHGCAR。
}

# static_dos: requires projected DOS output.
# static_dos：需要输出投影态密度。
INCAR_DELTA_STATIC_DOS: Dict[str, Any] = {
    "LORBIT": 11,       # Write lm-decomposed projected DOS. / 输出 lm 分解的投影 DOS。
    "LCHARG": True,     # Write CHGCAR. / 输出 CHGCAR。
}

# static_charge: write full charge density (AECCAR0/AECCAR2).
# static_charge：输出全电荷密度（AECCAR0/AECCAR2）。
INCAR_DELTA_STATIC_CHG: Dict[str, Any] = {
    "LCHARG": True,     # Write CHGCAR. / 输出 CHGCAR。
    "LAECHG": True,     # Write AECCAR0 and AECCAR2. / 输出 AECCAR0 和 AECCAR2。
}

# static_elf: write electron localisation function.
# static_elf：输出电子局域函数。
INCAR_DELTA_STATIC_ELF: Dict[str, Any] = {
    "LCHARG": True,     # Write CHGCAR. / 输出 CHGCAR。
    "LELF":   True,     # Compute and write ELF. / 计算并输出 ELF。
    "LWAVE":  False,    # Do not write WAVECAR. / 不输出 WAVECAR。
}

# Default INCAR settings for vibrational frequency calculations.
# 振动频率计算的默认 INCAR 设置。
DEFAULT_INCAR_FREQ: Dict[str, Any] = {
    "IBRION": 5,        # Finite-difference Hessian. / 有限差分 Hessian 矩阵。
    "POTIM": 0.015,     # Displacement magnitude (Å). / 位移幅度（Å）。
    "NFREE": 2,         # Number of displacements per atom. / 每个原子的位移次数。
    "NSW": 1,           # Single ionic step for IBRION=5. / IBRION=5 需要 NSW=1。
    "EDIFF": 1e-7,      # Tight electronic convergence for accurate forces. / 严格电子收敛以确保力的精度。
    "NELM": 200,        # Maximum electronic SCF steps. / 最大电子自洽步数。
    "ISMEAR": 0,        # Gaussian smearing. / 高斯展宽。
    "SIGMA": 0.05,      # Smearing width (eV). / 展宽宽度（eV）。
    "LREAL": False,     # Reciprocal-space projection for accuracy. / 倒空间投影以提高精度。
    "ALGO": "Fast",     # RMM-DIIS algorithm. / RMM-DIIS 算法。
    "LCHARG": False,    # Do not write CHGCAR. / 不输出 CHGCAR。
    "LWAVE": False,     # Do not write WAVECAR. / 不输出 WAVECAR。
    "LORBIT": 11,       # Write lm-decomposed projected DOS. / 输出 lm 分解的投影 DOS。
}

# Default INCAR settings for LOBSTER chemical bonding analysis.
# LOBSTER 化学键分析的默认 INCAR 设置。
DEFAULT_INCAR_LOBSTER: Dict[str, Any] = {
    "NELM": 150,        # Maximum electronic SCF steps. / 最大电子自洽步数。
    "NCORE": 6,         # Number of cores per band. / 每个能带的核数。
    "IBRION": -1,       # No ionic relaxation. / 不进行离子弛豫。
    "EDIFF": 1e-6,      # Electronic convergence criterion (eV). / 电子步收敛判据（eV）。
    "LORBIT": 11,       # Write lm-decomposed projected DOS. / 输出 lm 分解的投影 DOS。
    "NSW": 0,           # No ionic steps. / 离子步数为零。
    "LCHARG": True,     # Write CHGCAR (required by LOBSTER). / 输出 CHGCAR（LOBSTER 需要）。
    "LWAVE": True,      # Write WAVECAR (required by LOBSTER). / 输出 WAVECAR（LOBSTER 需要）。
}

# Default INCAR settings for NEB (Nudged Elastic Band) calculations.
# NEB（弹性带法）计算的默认 INCAR 设置。
DEFAULT_INCAR_NEB: Dict[str, Any] = {
    "EDIFF": 1e-5,      # Electronic convergence criterion (eV). / 电子步收敛判据（eV）。
    "NELM": 150,        # Maximum electronic SCF steps. / 最大电子自洽步数。
    "POTIM": 0.02,      # Step size for optimizer. / 优化器步长。
    "ICHAIN": 0,        # NEB method selector (0 = NEB). / NEB 方法选择（0 = NEB）。
    "SPRING": -5.0,     # Spring constant between images (eV/Å²). / 像间弹簧常数（eV/Å²）。
    "IBRION": 3,        # Use quick-min optimizer. / 使用 quick-min 优化器。
    "LCLIMB": True,     # Climbing-image NEB enabled. / 启用爬坡像 NEB。
    "LREAL": "Auto",    # Real-space projection (automatic). / 实空间投影（自动）。
}

# Default INCAR settings for dimer method (saddle-point search) calculations.
# 二聚体方法（鞍点搜索）计算的默认 INCAR 设置。
DEFAULT_INCAR_DIMER: Dict[str, Any] = {
    "ICHAIN": 2,        # Dimer method selector. / 二聚体方法选择。
    "IOPT": 2,          # Conjugate gradient optimizer for dimer. / 二聚体共轭梯度优化器。
    "IBRION": 3,        # Use optimizer controlled by IOPT. / 使用 IOPT 控制的优化器。
    "POTIM": 0.0,       # Step size managed by IOPT (set to 0). / 步长由 IOPT 管理（设为 0）。
    "EDIFF": 1e-7,      # Tight electronic convergence. / 严格电子收敛。
    "DdR": 0.005,       # Dimer separation (Å). / 二聚体间距（Å）。
    "DRotMax": 3,       # Maximum dimer rotation steps per ionic step. / 每个离子步的最大二聚体旋转步数。
    "DFNMax": 1.0,      # Max force component for rotation. / 旋转时的最大力分量。
    "DFMin": 0.01,      # Convergence criterion for rotation. / 旋转收敛判据。
    "NSW": 1000,        # Maximum number of ionic steps. / 最大离子步数。
    "LREAL": "Auto",    # Real-space projection (automatic). / 实空间投影（自动）。
}

# Default INCAR settings for NBO (Natural Bond Orbital) analysis.
# NBO（自然键轨道）分析的默认 INCAR 设置。
DEFAULT_INCAR_NBO: Dict[str, Any] = {
    "NSW": 0,           # No ionic steps. / 离子步数为零。
    "IBRION": -1,       # No ionic relaxation. / 不进行离子弛豫。
    "LNBO": True,       # Enable NBO analysis output. / 启用 NBO 分析输出。
    "LWAVE": True,      # Write WAVECAR (required by NBO). / 输出 WAVECAR（NBO 需要）。
    "LCHARG": True,     # Write CHGCAR. / 输出 CHGCAR。
}

# Default INCAR settings for NMR chemical shift calculations.
# NMR 化学位移计算的默认 INCAR 设置。
DEFAULT_INCAR_NMR_CS: Dict[str, Any] = {
    "NSW": 0,               # No ionic steps. / 离子步数为零。
    "ISMEAR": -5,           # Tetrahedron smearing. / 四面体展宽。
    "LCHARG": False,        # Do not write CHGCAR. / 不输出 CHGCAR。
    "LORBIT": 11,           # Write lm-decomposed projected DOS. / 输出 lm 分解的投影 DOS。
    "LREAL": False,         # Reciprocal-space projection. / 倒空间投影。
    "LCHIMAG": True,        # Enable chemical shift calculation. / 启用化学位移计算。
    "EDIFF": -1e-10,        # Negative value: stop when |dE|<|EDIFF|. / 负值：当 |dE|<|EDIFF| 时停止。
    "ISYM": 0,              # Disable symmetry for NMR. / 为 NMR 禁用对称性。
    "LNMR_SYM_RED": True,   # NMR-specific symmetry reduction. / NMR 专用对称性约化。
    "NELMIN": 10,           # Minimum electronic SCF steps. / 最小电子自洽步数。
    "NLSPLINE": True,       # Use spline interpolation for PAW. / 对 PAW 使用样条插值。
    "PREC": "ACCURATE",     # High precision mode. / 高精度模式。
    "SIGMA": 0.01,          # Smearing width (eV). / 展宽宽度（eV）。
}

# Default INCAR settings for NMR electric field gradient (EFG) calculations.
# NMR 电场梯度（EFG）计算的默认 INCAR 设置。
DEFAULT_INCAR_NMR_EFG: Dict[str, Any] = {
    "NSW": 0,           # No ionic steps. / 离子步数为零。
    "ISMEAR": -5,       # Tetrahedron smearing. / 四面体展宽。
    "LCHARG": False,    # Do not write CHGCAR. / 不输出 CHGCAR。
    "LORBIT": 11,       # Write lm-decomposed projected DOS. / 输出 lm 分解的投影 DOS。
    "LREAL": False,     # Reciprocal-space projection. / 倒空间投影。
    "ALGO": "FAST",     # RMM-DIIS algorithm. / RMM-DIIS 算法。
    "EDIFF": -1e-10,    # Negative value: stop when |dE|<|EDIFF|. / 负值：当 |dE|<|EDIFF| 时停止。
    "ISYM": 0,          # Disable symmetry for NMR. / 为 NMR 禁用对称性。
    "LEFG": True,       # Enable EFG calculation. / 启用 EFG 计算。
    "NELMIN": 10,       # Minimum electronic SCF steps. / 最小电子自洽步数。
    "PREC": "ACCURATE", # High precision mode. / 高精度模式。
    "SIGMA": 0.01,      # Smearing width (eV). / 展宽宽度（eV）。
}

# Default INCAR settings for NVT molecular dynamics.
# NVT 分子动力学的默认 INCAR 设置。
DEFAULT_INCAR_MD: Dict[str, Any] = {
    "IBRION": 0,        # MD mode. / 分子动力学模式。
    "ISMEAR": 0,        # Gaussian smearing. / 高斯展宽。
    "ISIF": 0,          # Compute forces only; no stress tensor. / 仅计算力；不计算应力张量。
    "ISYM": 0,          # Disable symmetry for MD. / 为 MD 禁用对称性。
    "LCHARG": False,    # Do not write CHGCAR every step. / 不在每步输出 CHGCAR。
    "LWAVE": True,      # Write WAVECAR for restart capability. / 输出 WAVECAR 以支持重启。
    "LREAL": True,      # Real-space projection for efficiency. / 实空间投影以提高效率。
    "LSCALU": False,    # Disable LU decomposition scaling. / 禁用 LU 分解缩放。
    "LPLANE": False,    # Disable plane decomposition. / 禁用平面分解。
    "NELMIN": 4,        # Minimum electronic SCF steps. / 最小电子自洽步数。
    "NELM": 500,        # Maximum electronic SCF steps. / 最大电子自洽步数。
    "NBLOCK": 1,        # Write XDATCAR every NBLOCK steps. / 每 NBLOCK 步写入 XDATCAR。
    "KBLOCK": 100,      # Write PCDAT/DOSCAR every KBLOCK*NBLOCK steps. / 每 KBLOCK*NBLOCK 步写入 PCDAT/DOSCAR。
    "NSIM": 4,          # Number of bands optimised simultaneously. / 同时优化的能带数。
    "BMIX": 1,          # Linear mixing parameter. / 线性混合参数。
    "MAXMIX": 20,       # Maximum steps kept in Broyden mixer. / Broyden 混合器保留的最大步数。
    "SMASS": 0,         # NVE-like thermostat (Nosé mass = 0). / 类 NVE 控温（Nosé 质量 = 0）。
    "PREC": "Normal",   # Normal precision mode. / 普通精度模式。
    "ADDGRID": True,    # Add support grid for augmentation charges. / 为增强电荷添加辅助网格。
    "LDAU": False,      # Disable DFT+U. / 禁用 DFT+U。
}

# Default INCAR settings for NPT molecular dynamics.
# NPT 分子动力学的默认 INCAR 设置。
DEFAULT_INCAR_MD_NPT: Dict[str, Any] = {
    "ALGO": "Fast",         # RMM-DIIS algorithm. / RMM-DIIS 算法。
    "IBRION": 0,            # MD mode. / 分子动力学模式。
    "ISIF": 3,              # Relax ions, cell shape, and volume. / 弛豫离子、晶胞形状和体积。
    "ISYM": 0,              # Disable symmetry for MD. / 为 MD 禁用对称性。
    "LCHARG": False,        # Do not write CHGCAR every step. / 不在每步输出 CHGCAR。
    "LWAVE": True,          # Write WAVECAR for restart capability. / 输出 WAVECAR 以支持重启。
    "LREAL": True,          # Real-space projection for efficiency. / 实空间投影以提高效率。
    "LSCALU": False,        # Disable LU decomposition scaling. / 禁用 LU 分解缩放。
    "LPLANE": False,        # Disable plane decomposition. / 禁用平面分解。
    "NELMIN": 4,            # Minimum electronic SCF steps. / 最小电子自洽步数。
    "NELM": 500,            # Maximum electronic SCF steps. / 最大电子自洽步数。
    "NBLOCK": 1,            # Write XDATCAR every NBLOCK steps. / 每 NBLOCK 步写入 XDATCAR。
    "KBLOCK": 100,          # Write PCDAT/DOSCAR every KBLOCK*NBLOCK steps. / 每 KBLOCK*NBLOCK 步写入 PCDAT/DOSCAR。
    "NSIM": 4,              # Number of bands optimised simultaneously. / 同时优化的能带数。
    "BMIX": 1,              # Linear mixing parameter. / 线性混合参数。
    "MAXMIX": 20,           # Maximum steps kept in Broyden mixer. / Broyden 混合器保留的最大步数。
    "SMASS": 0,             # Nosé thermostat mass parameter. / Nosé 控温质量参数。
    "MDALGO": 3,            # Langevin thermostat/barostat algorithm. / Langevin 控温/控压算法。
    "LANGEVIN_GAMMA_L": 1,  # Langevin friction for lattice DOFs (ps⁻¹). / 晶格自由度的 Langevin 摩擦系数（ps⁻¹）。
    "PMASS": 10,            # Fictitious lattice mass for barostat. / 控压器虚拟晶格质量。
    "PSTRESS": 0,           # Target pressure (kBar). / 目标压强（kBar）。
    "PREC": "Low",          # Low precision mode for NPT MD efficiency. / NPT MD 使用低精度模式以提升效率。
    "LDAU": False,          # Disable DFT+U. / 禁用 DFT+U。
}

# ---------------------------------------------------------------------------
# NBO constants
# ---------------------------------------------------------------------------
# NBO 常量
# ---------------------------------------------------------------------------

# Default keyword parameters for the NBO configuration file.
# NBO 配置文件的默认关键字参数。
DEFAULT_NBO_CONFIG_PARAMS: Dict[str, str] = {
    "occ_1c": "1.60",       # Occupancy cutoff for one-center NBOs. / 单中心 NBO 占据数截断。
    "occ_2c": "1.85",       # Occupancy cutoff for two-center NBOs. / 双中心 NBO 占据数截断。
    "print_cube": "F",      # Whether to print .cube visualisation files. / 是否输出 .cube 可视化文件。
    "density": "F",         # Visualise density (T) or wavefunctions (F). / 可视化密度（T）或波函数（F）。
    "vis_start": "0",       # First NBO index for .cube output. / .cube 输出的起始 NBO 索引。
    "vis_end": "-1",        # Last NBO index for .cube output (-1 = all). / .cube 输出的终止 NBO 索引（-1 = 全部）。
    "mesh_x": "0", "mesh_y": "0", "mesh_z": "0",   # Mesh points along each lattice vector. / 沿各晶格矢量的网格点数。
    "box_x": "1", "box_y": "1", "box_z": "1",       # Number of unit cells for .cube file. / .cube 文件使用的晶胞数。
    "origin_fact": "0.00",  # Origin shift factor for .cube file. / .cube 文件的原点偏移因子。
}

# Template string for the NBO configuration file.
# NBO 配置文件的模板字符串。
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

# Absolute path to the directory containing this module.
# 包含本模块的目录的绝对路径。
MODULE_DIR = Path(__file__).resolve().parent

# Path to the ANO-RCC-MB basis set file used by NBO calculations.
# NBO 计算所用 ANO-RCC-MB 基组文件的路径。
NBO_BASIS_PATH = MODULE_DIR / "basis" / "basis-ANO-RCC-MB.txt"
