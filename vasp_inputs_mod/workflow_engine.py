# -*- coding: utf-8 -*-
"""
flow.workflow_engine — VASP input-file dispatch engine
=======================================================

Position in the write pipeline
-------------------------------
::

    FrontendAdapter.from_frontend_dict()    (flow/api.py)
        └─ VaspWorkflowParams.to_workflow_config()
                 └─ WorkflowEngine.run(config)           ← THIS FILE
                          ├─ _get_incar_params()  ← CALC_TYPE_REGISTRY lookup
                          └─ VaspInputMaker.write_*(output_dir, ...)  (flow/maker.py)
                                   └─ InputSet.write_input()  (flow/input_sets.py)
                                            └─ POSCAR / INCAR / KPOINTS / POTCAR (disk)

Responsibilities
----------------
1. ``CalcType`` enum — the single canonical name for each calculation type.
2. ``CALC_TYPE_REGISTRY`` — maps each ``CalcType`` to its INCAR template, WAVECAR
   retention flag, VTST requirement, and script category.  This is the single
   source of truth for per-type defaults.
3. ``WorkflowConfig`` dataclass — all parameters consumed by ``WorkflowEngine``.
4. ``WorkflowEngine.run()`` — selects the correct ``VaspInputMaker.write_*()``
   method via a ``match`` statement and forwards calc-type-specific parameters.

Extension points — where to touch this file when adding a new stage
--------------------------------------------------------------------
1. Add a value to the ``CalcType`` enum.
2. Add a ``CalcTypeConfig(...)`` entry to ``CALC_TYPE_REGISTRY`` referencing the
   correct ``DEFAULT_INCAR_*`` template from ``flow/constants.py``.
3. If the new type needs extra parameters (e.g. ``nbo_config``), add a field to
   ``WorkflowConfig``.
4. Add a ``case CalcType.NEW_TYPE: maker.write_new(...)`` arm to
   ``WorkflowEngine.run()``, passing any new ``WorkflowConfig`` fields.
5. Add a matching ``FrontendAdapter`` extraction in ``flow/api.py``.
"""

import logging
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pymatgen.core import Structure

from .constants import (
    DEFAULT_INCAR_BULK, DEFAULT_INCAR_SLAB,
    DEFAULT_INCAR_NEB, DEFAULT_INCAR_DIMER, DEFAULT_INCAR_LOBSTER,
    DEFAULT_INCAR_NMR_CS, DEFAULT_INCAR_NMR_EFG, DEFAULT_INCAR_MD,
    DEFAULT_INCAR_MD_NPT, DEFAULT_INCAR_FREQ, DEFAULT_INCAR_NBO,
    INCAR_DELTA_STATIC_SP,
    INCAR_DELTA_STATIC_DOS,
    INCAR_DELTA_STATIC_CHG,
    INCAR_DELTA_STATIC_ELF,
    FUNCTIONAL_INCAR_PATCHES,
)
from .calc_type import CalcType
from .maker import VaspInputMaker
from .script import CalcCategory, Script
from .utils import load_structure
from .validator import validate as _validator_validate, ValidationError

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class CalcTypeConfig:
    """每种计算类型的静态配置，合并原先分散的四个查找表。"""
    incar_base: Dict[str, Any]
    incar_delta: Dict[str, Any] = field(default_factory=dict)
    need_wavecharge: bool = False
    need_vtst: bool = False
    script_category: CalcCategory = CalcCategory.RELAX

    def get_merged_incar(self, user_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged = {**self.incar_base, **self.incar_delta}
        if user_overrides:
            merged.update(user_overrides)
        return merged


# CALC_TYPE_REGISTRY — single source of truth for per-type INCAR defaults.
#
# Each entry maps a CalcType to a CalcTypeConfig which bundles:
#   incar_base       – dict merged from DEFAULT_INCAR_* in flow/constants.py
#   incar_delta      – incremental overrides on top of incar_base
#   need_wavecharge  – retain WAVECAR/CHGCAR after the job (True for DOS, Lobster, NBO…)
#   need_vtst        – require VTST-patched VASP binary (NEB, Dimer)
#   beef_compatible  – flag BEEF xc as incompatible (NMR, NBO, Lobster)
#   script_category  – CalcCategory used by flow/script.py PBS template selection
#
# To add a new calc type:
#   1. Add to CalcType enum above.
#   2. Add a CalcTypeConfig entry here with the matching DEFAULT_INCAR_* template.
#   3. Add a case arm in WorkflowEngine.run().
CALC_TYPE_REGISTRY: Dict[CalcType, CalcTypeConfig] = {
    # 结构优化
    CalcType.BULK_RELAX: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_BULK,
        script_category=CalcCategory.RELAX,
    ),
    CalcType.SLAB_RELAX: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_SLAB,
        script_category=CalcCategory.RELAX,
    ),
    # 静态/电子结构 — MPStaticSetEcat 自行处理 DEFAULT_INCAR_STATIC，此处仅存增量
    CalcType.STATIC_SP: CalcTypeConfig(
        incar_base={},
        incar_delta=INCAR_DELTA_STATIC_SP,
        script_category=CalcCategory.STATIC,
    ),
    CalcType.DOS_SP: CalcTypeConfig(
        incar_base={},
        incar_delta=INCAR_DELTA_STATIC_DOS,
        need_wavecharge=True,
        script_category=CalcCategory.STATIC,
    ),
    CalcType.CHG_SP: CalcTypeConfig(
        incar_base={},
        incar_delta=INCAR_DELTA_STATIC_CHG,
        need_wavecharge=True,
        script_category=CalcCategory.STATIC,
    ),
    CalcType.ELF_SP: CalcTypeConfig(
        incar_base={},
        incar_delta=INCAR_DELTA_STATIC_ELF,
        need_wavecharge=True,
        script_category=CalcCategory.STATIC,
    ),
    # 过渡态
    CalcType.NEB: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_NEB,
        need_vtst=True,
        script_category=CalcCategory.NEB,
    ),
    CalcType.DIMER: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_DIMER,
        need_vtst=True,
        script_category=CalcCategory.DIMER,
    ),
    # 频率（IR 增量由 FreqSetEcat 内部 calc_ir 参数处理）
    CalcType.FREQ: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_FREQ,
        script_category=CalcCategory.FREQ,
    ),
    CalcType.FREQ_IR: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_FREQ,
        script_category=CalcCategory.FREQ,
    ),
    # 性质分析
    CalcType.LOBSTER: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_LOBSTER,
        need_wavecharge=True,
        script_category=CalcCategory.LOBSTER,
    ),
    CalcType.NMR_CS: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_NMR_CS,
        script_category=CalcCategory.NMR,
    ),
    CalcType.NMR_EFG: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_NMR_EFG,
        script_category=CalcCategory.NMR,
    ),
    CalcType.NBO: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_NBO,
        need_wavecharge=True,
        script_category=CalcCategory.NBO,
    ),
    # 分子动力学
    CalcType.MD_NVT: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_MD,
        need_wavecharge=True,
        script_category=CalcCategory.MD,
    ),
    CalcType.MD_NPT: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_MD_NPT,
        need_wavecharge=True,
        script_category=CalcCategory.MD,
    ),
}

# 需要 vdw_kernel.bindat 的泛函
_VDW_NEEDED: set[str] = {"BEEF", "BEEFVTST"}

_CALC_TYPE_STR_MAP: Dict[str, CalcType] = {
    "bulk_relax":    CalcType.BULK_RELAX,
    "slab_relax":    CalcType.SLAB_RELAX,
    "static_sp":     CalcType.STATIC_SP,
    "static_dos":    CalcType.DOS_SP,
    "dos":           CalcType.DOS_SP,
    "static_charge": CalcType.CHG_SP,
    "static_elf":    CalcType.ELF_SP,
    "neb":           CalcType.NEB,
    "dimer":         CalcType.DIMER,
    "freq":          CalcType.FREQ,
    "freq_ir":       CalcType.FREQ_IR,
    "lobster":       CalcType.LOBSTER,
    "nmr_cs":        CalcType.NMR_CS,
    "nmr_efg":       CalcType.NMR_EFG,
    "nbo":           CalcType.NBO,
    "md_nvt":        CalcType.MD_NVT,
    "md_npt":        CalcType.MD_NPT,
}

# ── CalcType → 前端字符串映射（用于自动生成输出目录名）──────────────────────
CALC_TYPE_FRONTEND_NAME: Dict[CalcType, str] = {
    v: k for k, v in _CALC_TYPE_STR_MAP.items()
    if k not in ("static_dos", "dos")  # 去重：DOS_SP 优先用 "static_dos"
}
# 手动修正去重后的别名
CALC_TYPE_FRONTEND_NAME[CalcType.DOS_SP] = "static_dos"


@dataclass
class WorkflowConfig:
    """
    工作流配置 - 用户友好接口
    
    相比直接操作 VaspInputMaker，用户只需：
    1. 指定 calc_type：计算类型（必选）
    2. 指定 structure：输入结构（必选）
    3. 指定 prev_dir：前序目录（可选，系统会自动推断）
    
    ================================================================================
    参数分类
    ================================================================================
    
    【前端可配置参数】
      calc_type          - 计算类型（CalcType枚举）
      structure          - 输入结构（文件路径或pymatgen Structure对象）
      functional         - 泛函，默认 PBE
      kpoints_density    - K点密度，默认 50.0
      output_dir         - 输出目录，默认自动生成
      prev_dir           - 前序计算目录（可选）
      
      MD参数: ensemble, start_temp, end_temp, nsteps, time_step
      NEB参数: n_images, use_idpp, start_structure, end_structure
      频率参数: vibrate_indices, calc_ir
      NMR参数: isotopes
      NBO参数: nbo_config
      
    【高级参数】（谨慎使用）
      user_incar_overrides - 直接覆盖INCAR参数
      
    ================================================================================
    """
    # === 核心配置 ===
    calc_type: CalcType                                    # 计算类型（必选）
    structure: Optional[Union[str, Path, Structure]] = None  # 输入结构（neb/dimer 可为 None）
    
    # === 功能参数（前端暴露）===
    functional: str = "PBE"                                # 泛函，默认 PBE
    kpoints_density: float = 50.0                           # K点密度
    output_dir: Optional[Union[str, Path]] = None          # 输出目录
    
    # === 前序依赖（可选，系统可自动推断）===
    prev_dir: Optional[Union[str, Path]] = None
    
    # === MD 专用参数 ===
    ensemble: str = "nvt"
    start_temp: float = 300.0
    end_temp: float = 300.0
    nsteps: int = 1000
    time_step: Optional[float] = None
    
    # === NEB 专用参数 ===
    n_images: int = 6
    use_idpp: bool = True
    start_structure: Optional[Union[str, Path, Structure]] = None
    end_structure: Optional[Union[str, Path, Structure]] = None
    
    # === 频率计算专用 ===
    vibrate_indices: Optional[List[int]] = None
    calc_ir: bool = False
    
    # === NMR 专用 ===
    isotopes: Optional[List[str]] = None
    
    # === NBO 专用 ===
    nbo_config: Optional[Dict[str, Any]] = None

    # === Lobster 专用 ===
    lobster_overwritedict: Optional[Dict[str, Any]] = None
    lobster_custom_lines: Optional[List[str]] = None

    # === 高级覆盖 ===
    user_incar_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # ── calc_type：接受字符串，自动转换为 CalcType 枚举 ──────────────────
        if isinstance(self.calc_type, str):
            key = self.calc_type.lower().strip()
            if key not in _CALC_TYPE_STR_MAP:
                valid = ", ".join(f'"{k}"' for k in sorted(_CALC_TYPE_STR_MAP))
                raise ValueError(
                    f"Unknown calc_type string '{self.calc_type}'. "
                    f"Valid options: {valid}"
                )
            self.calc_type = _CALC_TYPE_STR_MAP[key]

        # ── functional：统一大写 ─────────────────────────────────────────────
        self.functional = self.functional.upper()

        # ── output_dir：统一转为 Path ────────────────────────────────────────
        self.output_dir = Path(self.output_dir) if self.output_dir else None

        # ── user_incar_overrides：确保不为 None ──────────────────────────────
        if self.user_incar_overrides is None:
            self.user_incar_overrides = {}
    
    def auto_detect_prev_dir(self) -> Optional[Path]:
        """
        自动检测前序目录
        
        策略：
        1. 如果已指定 prev_dir，直接返回
        2. 尝试从 output_dir 的兄弟目录推断（假设命名规范）
        3. 尝试从当前工作目录推断
        """
        if self.prev_dir:
            return Path(self.prev_dir).resolve()
        
        # 策略2：从 output_dir 推断
        if self.output_dir:
            parent = self.output_dir.parent
            # 尝试常见的命名模式
            patterns = ["00-relax", "01-relax", "opt", "optimization", "relax"]
            for pattern in patterns:
                candidate = parent / pattern
                if candidate.exists() and (candidate / "CONTCAR").exists():
                    logger.info(f"自动检测到前序目录: {candidate}")
                    return candidate
        
        return None
    
class WorkflowEngine:
    """
    工作流引擎 - 执行标准化工作流
    
    用法示例：
    
    ```python
    # 简单用法
    engine = WorkflowEngine()
    engine.run(
        calc_type=CalcType.STATIC_SP,
        structure="POSCAR",
        functional="PBE",
        output_dir="calc/static"
    )
    
    # 续接前序计算
    engine.run(
        calc_type=CalcType.DOS_SP,
        structure="calc/static",
        prev_dir="calc/static"  # 可省略，会自动检测
    )
    ```
    
    对于脚本生成：
    ```python
    engine.generate_script(
        calc_type=CalcType.NEB,
        ...
    )
    ```
    """
    
    def __init__(
        self,
        maker: Optional[VaspInputMaker] = None,
        script_maker: Optional[Script] = None,
    ):
        self.maker = maker or VaspInputMaker()
        self.script_maker = script_maker or Script()
    
    def _get_incar_params(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Merge INCAR parameters in priority order (lowest → highest):

        1. ``CALC_TYPE_REGISTRY[calc_type].incar_base``  — type-specific defaults
        2. ``CALC_TYPE_REGISTRY[calc_type].incar_delta`` — static increments
        3. ``FUNCTIONAL_INCAR_PATCHES[functional]``      — functional-specific tags
           (e.g. GGA/LUSE_VDW/AGGAC/LASPH for BEEF, METAGGA/ADDGRID for SCAN,
           LHFCALC/AEXX/HFSCREEN for HSE)
        4. ``config.user_incar_overrides``               — user-supplied overrides
           (always wins; user can still override any functional-level default)

        Note: when ``prev_dir`` is set, INCAR/KPOINTS inheritance from the
        previous calculation is handled inside the relevant ``InputSet``
        ``from_prev_calc_ecat()`` class methods (``MPStaticSetEcat``,
        ``FreqSetEcat``, ``LobsterSetEcat``, …) in ``flow/input_sets.py``.
        Those methods merge the inherited INCAR with calc-type-specific deltas
        (e.g. ``DEFAULT_INCAR_STATIC`` on top of the relax INCAR), then apply
        ``user_incar_overrides`` as the final layer.  This function therefore
        does NOT additionally read ``prev_dir/INCAR`` — doing so would put the
        full relax INCAR into ``user_incar_settings``, overriding the static
        defaults (IBRION, NSW, …) that ``from_prev_calc_ecat`` correctly sets.

        Side effect: when MAGMOM is present in the merged result, ISPIN=2 is
        injected automatically unless the user already supplied ISPIN explicitly.

        The result is passed as ``user_incar_settings`` to ``VaspInputMaker``,
        which forwards it to the pymatgen ``InputSet`` as the final INCAR overlay.

        按优先级（从低到高）合并 INCAR 参数：

        1. ``CALC_TYPE_REGISTRY[calc_type].incar_base``  — 计算类型专属默认值
        2. ``CALC_TYPE_REGISTRY[calc_type].incar_delta`` — 静态增量覆盖
        3. ``FUNCTIONAL_INCAR_PATCHES[functional]``      — 泛函专属标记
        4. ``config.user_incar_overrides``               — 用户覆盖（始终最高优先级）

        注意：设置了 ``prev_dir`` 时，INCAR/KPOINTS 继承由各 ``InputSet`` 的
        ``from_prev_calc_ecat()`` 类方法在 ``flow/input_sets.py`` 内部处理，
        本函数不额外读取 ``prev_dir/INCAR``。

        副作用：若合并结果中存在 MAGMOM，且用户未显式设置 ISPIN，则自动注入 ISPIN=2。
        """
        ct_cfg = CALC_TYPE_REGISTRY.get(config.calc_type)
        base = ct_cfg.get_merged_incar({}) if ct_cfg is not None else {}
        func_patch = FUNCTIONAL_INCAR_PATCHES.get(config.functional, {})
        merged: Dict[str, Any] = {**base, **func_patch, **config.user_incar_overrides}
        # Auto-inject ISPIN=2 when MAGMOM is present and the user has not
        # explicitly set ISPIN.  VASP silently ignores MAGMOM when ISPIN=1.
        # 当 MAGMOM 存在且用户未显式设置 ISPIN 时自动注入 ISPIN=2。
        if "MAGMOM" in merged and "ISPIN" not in config.user_incar_overrides:
            merged.setdefault("ISPIN", 2)
        return merged

    def _get_script_context(self, config: WorkflowConfig) -> Dict[str, Any]:
        """生成脚本渲染上下文。"""
        ct_cfg = CALC_TYPE_REGISTRY[config.calc_type]
        need_wavecharge = ct_cfg.need_wavecharge
        cleanup_cmd = (
            ""
            if need_wavecharge
            else "rm REPORT CHG* DOSCAR EIGENVAL IBZKPT PCDAT PROCAR WAVECAR XDATCAR vasprun.xml FORCECAR"
        )
        return {
            "functional": config.functional,
            "cleanup_cmd": cleanup_cmd,
            "need_vdw": any(v in config.functional for v in _VDW_NEEDED),
        }
    
    def _copy_vdw_kernel(self, output_dir: Path) -> None:
        """Copy ``vdw_kernel.bindat`` into *output_dir* for vdW functionals.

        The source path is read from the ``FLOW_VDW_KERNEL`` environment
        variable.  Raises ``FileNotFoundError`` with a clear message when the
        variable is unset or points to a non-existent file — so the user knows
        exactly what to fix before the VASP job is submitted.

        将 ``vdw_kernel.bindat`` 复制到 *output_dir*（用于需要 vdW 核文件的泛函）。

        源路径从环境变量 ``FLOW_VDW_KERNEL`` 读取。若该变量未设置或指向不存在的
        文件，抛出 ``FileNotFoundError`` 并给出明确的错误说明。
        """
        vdw_env = os.environ.get("FLOW_VDW_KERNEL", "").strip()
        if not vdw_env:
            raise FileNotFoundError(
                "BEEF/BEEFVTST functional requires vdw_kernel.bindat in the "
                "output directory, but the FLOW_VDW_KERNEL environment variable "
                "is not set.  Please run:\n"
                "  export FLOW_VDW_KERNEL=/absolute/path/to/vdw_kernel.bindat\n"
                "BEEF/BEEFVTST 泛函需要 vdw_kernel.bindat，但环境变量 "
                "FLOW_VDW_KERNEL 未设置。请执行上述 export 命令后重试。"
            )
        src = Path(vdw_env)
        if not src.is_file():
            raise FileNotFoundError(
                f"vdw_kernel.bindat not found at '{src}' "
                f"(FLOW_VDW_KERNEL={vdw_env}).\n"
                f"vdw_kernel.bindat 在 '{src}' 处不存在，请检查 FLOW_VDW_KERNEL 路径。"
            )
        dest = output_dir / "vdw_kernel.bindat"
        shutil.copy2(src, dest)
        logger.info("Copied vdw_kernel.bindat → %s", dest)

    def _copy_prev_wavecharge(
        self,
        prev_dir: Path,
        output_dir: Path,
        user_incar_overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Copy WAVECAR and/or CHGCAR from *prev_dir* into *output_dir*.

        Files are only copied when they exist and are non-empty.  Missing or
        empty files are silently skipped — the workflow proceeds normally.

        Returns a dict of INCAR tags to add (``ICHARG=1`` when CHGCAR is
        copied, ``ISTART=1`` when only WAVECAR is copied).  Tags already
        present in *user_incar_overrides* are **not** included in the return
        value so the user's explicit settings always take highest priority.

        将 WAVECAR 和/或 CHGCAR 从 *prev_dir* 复制到 *output_dir*。

        仅在文件存在且非空时复制；缺失或空文件静默跳过，工作流继续正常运行。

        返回需追加的 INCAR 标记字典：复制了 CHGCAR 时返回 ``ICHARG=1``，
        仅复制了 WAVECAR 时返回 ``ISTART=1``。若用户在 *user_incar_overrides*
        中已显式设置这些标记，则不覆盖（用户值始终优先）。
        """
        incar_additions: Dict[str, Any] = {}

        chgcar = prev_dir / "CHGCAR"
        wavecar = prev_dir / "WAVECAR"

        chgcar_copied = False
        wavecar_copied = False

        if chgcar.is_file() and chgcar.stat().st_size > 0:
            shutil.copy2(chgcar, output_dir / "CHGCAR")
            chgcar_copied = True
            logger.info("Copied CHGCAR from %s → %s", prev_dir, output_dir)

        if wavecar.is_file() and wavecar.stat().st_size > 0:
            shutil.copy2(wavecar, output_dir / "WAVECAR")
            wavecar_copied = True
            logger.info("Copied WAVECAR from %s → %s", prev_dir, output_dir)

        if chgcar_copied and "ICHARG" not in user_incar_overrides:
            incar_additions["ICHARG"] = 1
        elif wavecar_copied and "ISTART" not in user_incar_overrides:
            incar_additions["ISTART"] = 1

        return incar_additions

    def run(self, config: WorkflowConfig) -> str:
        """Write VASP input files for *config* to ``config.output_dir``.

        Steps
        -----
        1. Auto-detect ``prev_dir`` if not supplied.
        2. Validate ``config`` (raises ``ValueError`` on bad params).
        3. Resolve structure from ``prev_dir`` when the explicit path is absent.
        4. Pre-check ``prev_dir`` for WAVECAR/CHGCAR; append ICHARG/ISTART tags.
        5. Build a fresh ``VaspInputMaker`` with the merged INCAR params.
        6. Dispatch to the correct ``maker.write_*()`` via a ``match`` statement.
        7. Copy WAVECAR/CHGCAR from ``prev_dir`` into the output directory.
        8. Copy ``vdw_kernel.bindat`` if a vdW functional is used.

        To add a new calc type, add a ``case CalcType.NEW_TYPE:`` arm here that
        calls the appropriate ``VaspInputMaker`` method with any required
        ``WorkflowConfig`` fields.

        Returns:
            Absolute path to the output directory as a string.
        """
        # 1. 自动检测 prev_dir
        if config.prev_dir is None:
            config.prev_dir = config.auto_detect_prev_dir()

        prev = Path(config.prev_dir).resolve() if config.prev_dir else None

        # 2. 验证配置
        try:
            _validator_validate(
                calc_type=config.calc_type.value,
                structure=config.structure,
                functional=config.functional,
                kpoints_density=config.kpoints_density,
                output_dir=config.output_dir,
                prev_dir=config.prev_dir,
                incar=config.user_incar_overrides or None,
                # NEB 专用字段通过 **extra 传入
                start_structure=config.start_structure,
                end_structure=config.end_structure,
                neb_images=None,  # WorkflowConfig 不直接暴露 neb_images 列表
            )
        except ValidationError as exc:
            # 将 ValidationError 转为 ValueError 保持 run() 原有异常类型契约
            raise ValueError("配置验证失败:\n" + "\n".join(f"  - {e}" for e in exc.errors)) from exc
        
        # 3. 解析 structure 为具体的文件路径
        #
        # 优先级：
        #   a) structure 是存在的文件            → 直接使用
        #   b) structure 是目录                  → 从目录取 CONTCAR（非空）或 POSCAR
        #   c) structure 文件不存在 / 为 None     → 从 prev_dir 取 CONTCAR（非空）或 POSCAR
        struct = config.structure

        if isinstance(struct, (str, Path)):
            struct_path = Path(struct)

            if struct_path.is_file():
                # 文件存在，直接使用，无需任何处理
                pass

            elif struct_path.is_dir():
                # structure 本身是目录，从中解析 CONTCAR/POSCAR
                contcar = struct_path / "CONTCAR"
                poscar  = struct_path / "POSCAR"
                if contcar.is_file() and contcar.stat().st_size > 0:
                    struct = contcar
                    logger.info(
                        "Structure '%s' is a directory; using CONTCAR: %s",
                        config.structure, contcar,
                    )
                elif poscar.is_file():
                    struct = poscar
                    logger.info(
                        "Structure '%s' is a directory; using POSCAR: %s",
                        config.structure, poscar,
                    )
                else:
                    raise ValueError(
                        f"Structure path '{config.structure}' is a directory but "
                        "contains neither CONTCAR nor POSCAR."
                    )

            else:
                # 路径不存在，尝试从 prev_dir 取
                if prev is not None:
                    contcar = prev / "CONTCAR"
                    poscar_fallback = prev / "POSCAR"
                    if contcar.is_file() and contcar.stat().st_size > 0:
                        struct = contcar
                        logger.info(
                            "Structure '%s' not found; using CONTCAR from prev_dir: %s",
                            config.structure, contcar,
                        )
                    elif poscar_fallback.is_file():
                        struct = poscar_fallback
                        logger.info(
                            "Structure '%s' not found; using POSCAR from prev_dir: %s",
                            config.structure, poscar_fallback,
                        )
                    else:
                        raise ValueError(
                            f"Structure file '{config.structure}' not found and "
                            f"prev_dir '{prev}' contains neither CONTCAR nor POSCAR."
                        )
                # else: structure 不存在且无 prev_dir → validator 已拦截，不会到达此处

        elif struct is None:
            # structure=None，完全依赖 prev_dir
            if prev is not None:
                contcar = prev / "CONTCAR"
                poscar_fallback = prev / "POSCAR"
                if contcar.is_file() and contcar.stat().st_size > 0:
                    struct = contcar
                    logger.info("structure=None; using CONTCAR from prev_dir: %s", contcar)
                elif poscar_fallback.is_file():
                    struct = poscar_fallback
                    logger.info("structure=None; using POSCAR from prev_dir: %s", poscar_fallback)
                else:
                    raise ValueError(
                        f"structure is None and prev_dir '{prev}' contains "
                        "neither CONTCAR nor POSCAR."
                    )

        # 4. 确定输出目录
        output_dir = config.output_dir
        if output_dir is None:
            output_dir = Path.cwd() / f"calc_{config.calc_type.value}"
        output_dir = Path(output_dir).resolve()

        # 5. Pre-check prev_dir for WAVECAR/CHGCAR.
        wavecharge_incar: Dict[str, Any] = {}
        if prev is not None:
            chgcar = prev / "CHGCAR"
            wavecar = prev / "WAVECAR"
            chgcar_avail = chgcar.is_file() and chgcar.stat().st_size > 0
            wavecar_avail = wavecar.is_file() and wavecar.stat().st_size > 0
            if chgcar_avail and "ICHARG" not in config.user_incar_overrides:
                wavecharge_incar["ICHARG"] = 1
            elif wavecar_avail and "ISTART" not in config.user_incar_overrides:
                wavecharge_incar["ISTART"] = 1

        # 6. 生成输入文件
        # 使用 replace() 创建每次调用独立的副本，避免 self.maker 状态污染
        incar_params = {**self._get_incar_params(config), **wavecharge_incar}
        maker = replace(
            self.maker,
            functional=config.functional,
            kpoints_density=config.kpoints_density,
            user_incar_settings=incar_params,
        )

        match config.calc_type:
            case CalcType.BULK_RELAX:
                maker.write_bulk(struct, output_dir)

            case CalcType.SLAB_RELAX:
                maker.write_slab(struct, output_dir)

            case CalcType.STATIC_SP | CalcType.DOS_SP | CalcType.CHG_SP | CalcType.ELF_SP:
                maker.write_noscf(
                    output_dir,
                    structure=struct if prev is None else None,
                    prev_dir=prev,
                )

            case CalcType.NEB:
                maker.write_neb(
                    output_dir,
                    start_structure=config.start_structure,
                    end_structure=config.end_structure,
                    n_images=config.n_images,
                    use_idpp=config.use_idpp,
                )

            case CalcType.DIMER:
                maker.write_dimer(output_dir, neb_dir=prev)

            case CalcType.FREQ:
                maker.write_freq(
                    output_dir,
                    prev_dir=prev,
                    structure=struct,
                    calc_ir=False,
                    vibrate_indices=config.vibrate_indices,
                )

            case CalcType.FREQ_IR:
                maker.write_freq(
                    output_dir,
                    prev_dir=prev,
                    structure=struct,
                    calc_ir=True,
                    vibrate_indices=config.vibrate_indices,
                )

            case CalcType.LOBSTER:
                maker.write_lobster(
                    output_dir,
                    structure=struct if prev is None else None,
                    prev_dir=prev,
                    overwritedict=config.lobster_overwritedict,
                    custom_lobsterin_lines=config.lobster_custom_lines,
                )

            case CalcType.NMR_CS:
                maker.write_nmr(
                    output_dir,
                    mode="cs",
                    isotopes=config.isotopes,
                    prev_dir=prev,
                    structure=struct if prev is None else None,
                )

            case CalcType.NMR_EFG:
                maker.write_nmr(
                    output_dir,
                    mode="efg",
                    isotopes=config.isotopes,
                    prev_dir=prev,
                    structure=struct if prev is None else None,
                )

            case CalcType.NBO:
                _nbo_cfg = dict(config.nbo_config) if config.nbo_config else {}
                _basis = _nbo_cfg.pop("basis_source", None)
                maker.write_nbo(
                    output_dir,
                    basis_source=_basis,
                    prev_dir=prev,
                    structure=struct if prev is None else None,
                    nbo_config=_nbo_cfg or None,
                )

            case CalcType.MD_NVT | CalcType.MD_NPT:
                maker.write_md(
                    output_dir,
                    ensemble="nvt" if config.calc_type == CalcType.MD_NVT else "npt",
                    start_temp=config.start_temp,
                    end_temp=config.end_temp,
                    nsteps=config.nsteps,
                    time_step=config.time_step,
                    prev_dir=prev,
                    structure=struct if prev is None else None,
                )

            case _:
                raise ValueError(f"不支持的计算类型: {config.calc_type}")

        # 7. Copy WAVECAR/CHGCAR from prev_dir (silent no-op if absent/empty).
        # 将 WAVECAR/CHGCAR 从 prev_dir 复制到输出目录（不存在或为空时静默跳过）。
        if prev is not None:
            self._copy_prev_wavecharge(prev, output_dir, config.user_incar_overrides)

        # 8. Copy vdW kernel file for functionals that require it (BEEF, BEEFVTST).
        # 为需要 vdW 核文件的泛函（BEEF、BEEFVTST）复制 vdw_kernel.bindat。
        if any(v in config.functional for v in _VDW_NEEDED):
            self._copy_vdw_kernel(output_dir)

        logger.info(f"工作流完成，输入文件已生成至: {output_dir}")
        return str(output_dir)
    
    def generate_script(
        self,
        config: WorkflowConfig,
        output_dir: Optional[Union[str, Path]] = None,
        calc_category: Optional[CalcCategory] = None,
        cores: Optional[int] = None,
        walltime: Optional[int] = None,
        queue: Optional[str] = None,
        **script_kwargs,
    ) -> List[str]:
        """
        生成 PBS/SLURM 作业脚本。

        Args:
            config:        工作流配置
            output_dir:    脚本生成目录（默认使用 config.output_dir）
            calc_category: 计算类别，None 时由系统从 calc_type 自动推断
            cores:         核数（None 则使用类别默认值）
            walltime:      计算时间，小时（None 则使用类别默认值）
            queue:         队列名（None 则使用集群默认值）
            **script_kwargs: 额外模板变量（最高优先级）
        """
        # 1. 确保输出目录存在
        if output_dir is None:
            output_dir = config.output_dir
        if output_dir is None:
            output_dir = self.run(config)

        # 2. 若调用方未传 calc_category，从注册表直接读取
        if calc_category is None:
            calc_category = self._infer_calc_category_from_config(config)

        # 3. 渲染脚本
        return self.script_maker.render_script(
            folders=[output_dir],
            functional=config.functional,
            calc_category=calc_category,
            cores=cores,
            walltime=walltime,
            queue=queue,
            custom_context=script_kwargs if script_kwargs else None,
        )

    def _infer_calc_category_from_config(self, config: WorkflowConfig) -> CalcCategory:
        """从 CALC_TYPE_REGISTRY 直接读取 script_category，无需本地映射表。"""
        return CALC_TYPE_REGISTRY[config.calc_type].script_category


