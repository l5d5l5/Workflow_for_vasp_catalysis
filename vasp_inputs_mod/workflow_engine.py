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
)
from .maker import VaspInputMaker
from .script import CalcCategory, Script
from .utils import load_structure

logger = logging.getLogger(__name__)


class CalcType(Enum):
    """标准化计算类型枚举 - 用户唯一需要选择的参数"""
    # === 结构优化类 ===
    BULK_RELAX = "bulk_relax"           # 体相结构优化
    SLAB_RELAX = "slab_relax"           # 表面 slab 优化
    
    # === 电子结构计算类 ===
    STATIC_SP = "static_sp"             # 单点能计算（静态）
    DOS_SP = "static_dos"               # 静态 + DOS
    CHG_SP = "static_charge_density"    # 静态 + 电荷密度
    ELF_SP = "static_elf"               # 静态 + 电子局域函数
    
    # === 过渡态搜索类 ===
    NEB = "neb"                         # NEB 过渡态搜索
    DIMER = "dimer"                     # Dimer 方法（需先 NEB）
    
    # === 频率计算类 ===
    FREQ = "frequency"                  # 频率计算（声子）
    FREQ_IR = "frequency_ir"            # 频率 + 红外（介电常数）
    
    # === 性质分析类 ===
    LOBSTER = "lobster"                 # LOBSTER 基态分析
    NMR_CS = "nmr_cs"                   # NMR 化学位移
    NMR_EFG = "nmr_efg"                 # NMR 电场梯度
    NBO = "nbo"                         # NBO 分析
    
    # === 分子动力学类 ===
    MD_NVT = "md_nvt"                   # NVT 分子动力学
    MD_NPT = "md_npt"                   # NPT 分子动力学


@dataclass(frozen=True)
class CalcTypeConfig:
    """每种计算类型的静态配置，合并原先分散的四个查找表。"""
    incar_base: Dict[str, Any]
    incar_delta: Dict[str, Any] = field(default_factory=dict)
    need_wavecharge: bool = False
    need_vtst: bool = False
    beef_compatible: bool = True
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
        beef_compatible=False,
        script_category=CalcCategory.LOBSTER,
    ),
    CalcType.NMR_CS: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_NMR_CS,
        beef_compatible=False,
        script_category=CalcCategory.NMR,
    ),
    CalcType.NMR_EFG: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_NMR_EFG,
        beef_compatible=False,
        script_category=CalcCategory.NMR,
    ),
    CalcType.NBO: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_NBO,
        need_wavecharge=True,
        beef_compatible=False,
        script_category=CalcCategory.NBO,
    ),
    # 分子动力学
    CalcType.MD_NVT: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_MD,
        need_wavecharge=True,
        beef_compatible=False,
        script_category=CalcCategory.MD,
    ),
    CalcType.MD_NPT: CalcTypeConfig(
        incar_base=DEFAULT_INCAR_MD_NPT,
        need_wavecharge=True,
        beef_compatible=False,
        script_category=CalcCategory.MD,
    ),
}

# 需要 vdw_kernel.bindat 的泛函
_VDW_NEEDED: set[str] = {"BEEF", "BEEFVTST"}


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
    structure: Union[str, Path, Structure]                  # 输入结构（必选）
    
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

    # === 高级覆盖（谨慎使用）===
    user_incar_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.functional = self.functional.upper()
        self.output_dir = Path(self.output_dir) if self.output_dir else None
    
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
    
    def validate(self) -> List[str]:
        """
        验证配置完整性，返回错误列表。

        注意：仅验证路径存在性和基本参数有效性，
        不执行实际的结构加载（由 maker.write_*() 统一处理，避免重复 I/O）。
        """
        errors = []

        # 验证结构路径存在（不加载内容，由 maker 层统一加载）
        if isinstance(self.structure, (str, Path)):
            structure_path = Path(self.structure)
            if structure_path.is_dir():
                # 如果是目录，检查 CONTCAR 或 POSCAR
                if not ((structure_path / "CONTCAR").exists() or (structure_path / "POSCAR").exists()):
                    errors.append(f"结构目录 {structure_path} 中没有 CONTCAR 或 POSCAR")
            elif not structure_path.exists():
                errors.append(f"结构文件不存在: {structure_path}")
        elif not isinstance(self.structure, Structure):
            errors.append(f"structure 类型无效: {type(self.structure)}")

        # 验证 NEB 需要 start/end 结构
        if self.calc_type == CalcType.NEB:
            if not self.start_structure or not self.end_structure:
                errors.append(f"计算类型 {self.calc_type.value} 需要提供 start_structure 和 end_structure")

            # 验证 start/end 结构路径
            for name, struct in [("start_structure", self.start_structure),
                                  ("end_structure", self.end_structure)]:
                if struct:
                    path = Path(struct) if isinstance(struct, (str, Path)) else None
                    if path and path.is_dir():
                        if not ((path / "CONTCAR").exists() or (path / "POSCAR").exists()):
                            errors.append(f"{name} 目录 {path} 中没有 CONTCAR 或 POSCAR")
                    elif path and not path.exists():
                        errors.append(f"{name} 文件不存在: {path}")

        # 验证泛函兼容性
        if "BEEF" in self.functional and not CALC_TYPE_REGISTRY[self.calc_type].beef_compatible:
            errors.append(f"泛函 {self.functional} 不适用于计算类型 {self.calc_type.value}")

        # 验证 prev_dir
        detected_prev = self.auto_detect_prev_dir()
        if detected_prev and not (detected_prev / "INCAR").exists():
            errors.append(f"前序目录 {detected_prev} 中没有 INCAR 文件")

        return errors


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
        3. ``config.user_incar_overrides``               — user-supplied overrides

        The result is passed as ``user_incar_settings`` to ``VaspInputMaker``,
        which forwards it to the pymatgen ``InputSet`` as the final INCAR overlay.
        """
        ct_cfg = CALC_TYPE_REGISTRY.get(config.calc_type)
        if ct_cfg is None:
            return dict(config.user_incar_overrides)
        return ct_cfg.get_merged_incar(config.user_incar_overrides)

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
    
    def run(self, config: WorkflowConfig) -> str:
        """Write VASP input files for *config* to ``config.output_dir``.

        Steps
        -----
        1. Auto-detect ``prev_dir`` if not supplied.
        2. Validate ``config`` (raises ``ValueError`` on bad params).
        3. Build a fresh ``VaspInputMaker`` with the merged INCAR params from
           ``CALC_TYPE_REGISTRY`` + ``config.user_incar_overrides``.
        4. Dispatch to the correct ``maker.write_*()`` via a ``match`` statement.
           Each arm passes calc-type-specific parameters (``vibrate_indices``,
           ``nbo_config``, ``lobster_overwritedict``, …).

        To add a new calc type, add a ``case CalcType.NEW_TYPE:`` arm here that
        calls the appropriate ``VaspInputMaker`` method with any required
        ``WorkflowConfig`` fields.

        Returns:
            Absolute path to the output directory as a string.
        """
        # 1. 自动检测 prev_dir
        if config.prev_dir is None:
            config.prev_dir = config.auto_detect_prev_dir()
        
        # 2. 验证配置
        errors = config.validate()
        if errors:
            raise ValueError("配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors))
        
        # 3. 确定输出目录
        output_dir = config.output_dir
        if output_dir is None:
            output_dir = Path.cwd() / f"calc_{config.calc_type.value}"
        output_dir = Path(output_dir).resolve()
        
        # 4. 生成输入文件
        # 使用 replace() 创建每次调用独立的副本，避免 self.maker 状态污染
        maker = replace(
            self.maker,
            functional=config.functional,
            kpoints_density=config.kpoints_density,
            user_incar_settings=self._get_incar_params(config),
        )

        struct = config.structure
        prev = config.prev_dir

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
                maker.write_nbo(
                    output_dir,
                    prev_dir=prev,
                    structure=struct if prev is None else None,
                    nbo_config=config.nbo_config,
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


# === 便捷函数 ===

def quick_relax(structure: Union[str, Path], output_dir: str = "calc_relax", 
                functional: str = "PBE", is_slab: bool = False) -> str:
    """
    快速结构优化工作流
    
    用法:
        quick_relax("POSCAR", "calc/relax")
        quick_relax("POSCAR", "calc/slab", is_slab=True)
    """
    config = WorkflowConfig(
        calc_type=CalcType.SLAB_RELAX if is_slab else CalcType.BULK_RELAX,
        structure=structure,
        functional=functional,
        output_dir=output_dir,
    )
    engine = WorkflowEngine()
    return engine.run(config)


def quick_static(structure: Union[str, Path], output_dir: str = "calc_static",
                 functional: str = "PBE", need_dos: bool = False) -> str:
    """
    快速静态计算工作流
    
    用法:
        quick_static("POSCAR", "calc/static")
        quick_static("calc/relax", "calc/static", need_dos=True)
    """
    calc_type = CalcType.DOS_SP if need_dos else CalcType.STATIC_SP
    config = WorkflowConfig(
        calc_type=calc_type,
        structure=structure,
        functional=functional,
        output_dir=output_dir,
        prev_dir=structure if Path(structure).is_dir() else None,
    )
    engine = WorkflowEngine()
    return engine.run(config)


def quick_neb(start_struct: Union[str, Path], end_struct: Union[str, Path],
              output_dir: str = "calc_neb", functional: str = "PBE", 
              n_images: int = 6) -> str:
    """
    快速 NEB 过渡态搜索工作流
    
    用法:
        quick_neb("start.cif", "end.cif", "calc/neb")
        quick_neb("00-relax", "10-relax", "calc/neb")
    """
    config = WorkflowConfig(
        calc_type=CalcType.NEB,
        structure=start_struct,
        start_structure=start_struct,
        end_structure=end_struct,
        functional=functional,
        output_dir=output_dir,
        n_images=n_images,
    )
    engine = WorkflowEngine()
    return engine.run(config)
