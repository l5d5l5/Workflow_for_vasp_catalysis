# -*- coding: utf-8 -*-
"""
flow.api — Frontend-to-engine adapter layer
============================================

Position in the write pipeline
-------------------------------
::

    Stage.prepare()                     (flow/workflow/stages/*.py)
        └─ BaseStage._write_vasp_inputs()
               └─ FrontendAdapter.from_frontend_dict()   ← THIS FILE
                        └─ VaspWorkflowParams.to_workflow_config()
                                 └─ WorkflowEngine.run()  (flow/workflow_engine.py)
                                          └─ VaspInputMaker.write_*()  (flow/maker.py)
                                                   └─ InputSet.write_input()  (flow/input_sets.py)
                                                            └─ POSCAR / INCAR / KPOINTS / POTCAR  (disk)

Responsibilities
----------------
1. Accept a simple ``frontend_dict`` (calc_type string, xc, kpoints density,
   user_incar_settings, prev_dir, lobsterin, …) and validate/normalise it into
   a typed ``VaspWorkflowParams`` object.
2. Convert ``VaspWorkflowParams`` to the engine's ``WorkflowConfig`` via
   ``to_workflow_config()``, mapping frontend names to ``CalcType`` enum values
   and fanning out sub-module parameters (frequency, lobster, NBO, …).

Extension points — where to touch this file
-------------------------------------------
- **New calc type**: add an entry to ``FRONTEND_CALC_TYPE_MAP`` and a matching
  case in ``VaspWorkflowParams.to_workflow_config()``'s ``calc_type_map``.
- **New frontend param group**: add a ``FrontendXxxParams`` dataclass, extract
  it in ``from_frontend_dict()``, and transfer it in ``to_workflow_config()``.
- **New lobsterin/NBO field**: extend ``FrontendLobsterParams`` /
  ``FrontendNBOParams``, extract from ``data`` in ``from_frontend_dict()``, and
  set the matching ``WorkflowConfig`` field in ``to_workflow_config()``.

本模块是前端数据字典到引擎工作流配置的适配层。

职责
----
1. 接受前端传入的简单字典（包含 calc_type 字符串、交换关联泛函、K 点密度、
   用户 INCAR 设置、前序目录、lobsterin 等），将其验证并规范化为带类型的
   ``VaspWorkflowParams`` 对象。
2. 通过 ``to_workflow_config()`` 将 ``VaspWorkflowParams`` 转换为引擎所需的
   ``WorkflowConfig``，把前端字符串名称映射为 ``CalcType`` 枚举值，并展开各
   子模块参数（频率、Lobster、NBO 等）。

扩展点
------
- **新计算类型**：在 ``FRONTEND_CALC_TYPE_MAP`` 中添加条目，并在
  ``VaspWorkflowParams.to_workflow_config()`` 的 ``calc_type_map`` 中添加对应分支。
- **新前端参数组**：新增 ``FrontendXxxParams`` 数据类，在 ``from_frontend_dict()``
  中提取，并在 ``to_workflow_config()`` 中传递。
- **新 lobsterin/NBO 字段**：扩展 ``FrontendLobsterParams`` /
  ``FrontendNBOParams``，从 ``from_frontend_dict()`` 的 ``data`` 中提取，并在
  ``to_workflow_config()`` 中设置 ``WorkflowConfig`` 的对应字段。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pymatgen.core import Structure

from .calc_type import CalcType
from .workflow_engine import WorkflowConfig, WorkflowEngine
from .script import Script, CalcCategory
from .validator import validate as _validator_validate, ValidationError

logger = logging.getLogger(__name__)


# ============================================================================
# 前端参数定义（FrontendParams）
# ============================================================================

@dataclass
class FrontendStructureInput:
    """Frontend structure input descriptor.

    Supports three source types:
    - ``"file"``: uploaded file content (``content``) or local file path (``id``).
    - ``"library"``: select from a predefined structure library (``id`` is the
      library identifier).
    - ``"task"``: retrieve from a previous task's output directory (``id`` is the
      task directory path).

    Attributes:
        source: Source type — one of ``"file"``, ``"library"``, or ``"task"``.
        id: Identifier — file name, library ID, or task directory path.
        content: Raw file content, used when ``source="file"`` and the file is
            uploaded directly (rather than referenced by path).

    前端结构输入参数。

    支持三种来源：
    - ``"file"``：上传的文件内容（``content``）或本地文件路径（``id``）。
    - ``"library"``：从预定义结构库中选择（``id`` 为库中的结构标识符）。
    - ``"task"``：从之前任务的结果目录中获取（``id`` 为任务目录路径）。

    属性：
        source: 来源类型，取值为 ``"file"``、``"library"`` 或 ``"task"``。
        id: 标识符——文件名、库 ID 或任务路径。
        content: 文件原始内容，当 ``source="file"`` 且直接上传文件（而非路径引用）时使用。
    """
    source: str = "file"
    id: str = ""
    content: str = ""

    def to_path_or_content(self) -> Union[str, Path]:
        """Resolve the structure input to a file path or raw content string.

        Returns:
            - ``source="file"``: returns ``content`` if non-empty, otherwise ``id``.
            - ``source="library"``: returns ``id`` and emits a warning that library
              support must be separately implemented.
            - ``source="task"``: returns ``id`` after verifying the directory exists;
              raises ``FileNotFoundError`` if not found.
            - Unknown source: returns ``id`` with a warning.

        将结构输入解析为文件路径或原始内容字符串。

        返回：
            - ``source="file"``：若 ``content`` 非空则返回之，否则返回 ``id``。
            - ``source="library"``：返回 ``id`` 并发出警告（需要结构库支持）。
            - ``source="task"``：验证目录存在后返回 ``id``；不存在则抛出
              ``FileNotFoundError``。
            - 未知来源：返回 ``id`` 并发出警告。
        """
        if self.source == "file":
            return self.content if self.content else self.id
        elif self.source == "library":
            logger.warning(
                "FrontendStructureInput source='library' 需要结构库支持。"
                "当前返回 id='%s'，请确保该标识符在结构库中有效。", self.id
            )
            return self.id
        elif self.source == "task":
            task_path = Path(self.id)
            if not task_path.exists():
                raise FileNotFoundError(
                    f"FrontendStructureInput source='task'，但任务目录不存在: {self.id}"
                )
            return self.id
        else:
            logger.warning("未知的 FrontendStructureInput source='%s'，回退到使用 id", self.source)
            return self.id


@dataclass
class FrontendPrecisionParams:
    """Precision / convergence parameters exposed to the frontend.

    All fields map directly to VASP INCAR tags.  ``None`` means "use the
    InputSet default" and the field is omitted from ``user_incar_overrides``.

    Attributes:
        encut:  Plane-wave energy cutoff in eV (ENCUT).
        ediff:  Electronic convergence criterion in eV (EDIFF).
        ediffg: Ionic convergence criterion in eV or eV/Å (EDIFFG).
        nedos:  Number of DOS grid points (NEDOS).

    精度/收敛参数——前端暴露。

    所有字段直接映射到 VASP INCAR 标记。``None`` 表示"使用 InputSet 默认值"，
    该字段不会被写入 ``user_incar_overrides``。

    属性：
        encut:  平面波截断能（eV），对应 ENCUT。
        ediff:  电子自洽收敛判据（eV），对应 EDIFF。
        ediffg: 离子弛豫收敛判据（eV 或 eV/Å），对应 EDIFFG。
        nedos:  态密度网格点数，对应 NEDOS。
    """
    encut: Optional[int] = None
    ediff: Optional[float] = None
    ediffg: Optional[float] = None
    nedos: Optional[int] = None


@dataclass
class FrontendKpointParams:
    """K-point sampling parameters exposed to the frontend.

    Attributes:
        density:         Reciprocal-space k-point density (points per Å⁻¹).
        gamma_centered:  If ``True``, use Gamma-centred mesh; otherwise shifted.

    K 点采样参数——前端暴露。

    属性：
        density:         倒空间 K 点密度（每 Å⁻¹ 的点数）。
        gamma_centered:  ``True`` 表示使用 Gamma 中心网格，否则使用偏移网格。
    """
    density: Optional[float] = None
    gamma_centered: bool = True


@dataclass
class FrontendMagmomParams:
    """Magnetic moment parameters exposed to the frontend.

    Supports two input formats:
    - ``per_atom``: ``List[float]`` — site-ordered moments,
      e.g. ``[5.0, 5.0, 3.0, 3.0]``.
    - ``per_element``: ``Dict[str, float]`` — element-keyed moments,
      e.g. ``{"Fe": 5.0, "Co": 3.0}``.

    pymatgen ``user_incar_settings["MAGMOM"]`` expects a per-site
    ``List[float]``.  ``to_pymatgen_format()`` returns the correct type for
    use in ``to_workflow_config()``.

    磁矩参数——前端暴露。

    支持两种格式：
    - ``per_atom``：``List[float]``，按原子顺序排列，如 ``[5.0, 5.0, 3.0, 3.0]``。
    - ``per_element``：``Dict[str, float]``，按元素键值，如 ``{"Fe": 5.0, "Co": 3.0}``。

    pymatgen ``user_incar_settings["MAGMOM"]`` 期望 ``List[float]``（per-site）。
    ``to_pymatgen_format()`` 返回正确类型，供 ``to_workflow_config()`` 使用。
    """
    enabled: bool = False
    per_atom: Optional[List[float]] = None
    per_element: Dict[str, float] = field(default_factory=dict)

    def to_pymatgen_format(self) -> Optional[List[float]]:
        """Return the per-site ``List[float]`` expected by pymatgen.

        - ``per_atom`` takes priority: returned directly.
        - ``per_element``: returns ``None``; the caller should expand the dict
          against the structure's site order after the structure is loaded.

        Returns:
            Per-site moment list, or ``None`` if disabled or only
            ``per_element`` data is available.

        返回 pymatgen 期望的 per-site ``List[float]``。

        - ``per_atom`` 优先：直接返回列表。
        - ``per_element``：返回 ``None``；调用方应在加载结构后按位点顺序展开该字典。

        返回：
            per-site 磁矩列表，若未启用或仅有 ``per_element`` 数据则返回 ``None``。
        """
        if not self.enabled:
            return None
        if self.per_atom:
            return [float(v) for v in self.per_atom]
        return None

    def to_incar_format(self) -> Optional[str]:
        """Return the VASP MAGMOM string for direct INCAR writing.

        Kept for backward compatibility with code paths that write INCAR tags
        directly rather than via pymatgen InputSet.

        Returns:
            Space-separated moment string, or ``None`` if disabled.

        返回 VASP MAGMOM 字符串，用于直接写 INCAR 的兼容场景。

        保留该方法以兼容不经由 pymatgen InputSet 而直接写 INCAR 标记的代码路径。

        返回：
            空格分隔的磁矩字符串，若未启用则返回 ``None``。
        """
        if not self.enabled:
            return None
        if self.per_atom:
            return " ".join(str(v) for v in self.per_atom)
        if self.per_element:
            return " ".join(f"{k} {v}" for k, v in self.per_element.items())
        return None

    @property
    def values(self) -> Dict[str, float]:
        """Compatibility alias for ``per_element``.

        ``per_element`` 的兼容性别名。
        """
        return self.per_element

    @values.setter
    def values(self, val: Dict[str, float]):
        # Redirect legacy attribute writes to the canonical field.
        # 将旧属性写操作重定向到规范字段。
        self.per_element = val


@dataclass
class FrontendDFTPlusUParams:
    """DFT+U (Hubbard U) parameters exposed to the frontend.

    ``values`` format::

        {"Fe": {"LDAUU": 4.0, "LDAUL": 2, "LDAUJ": 0.0},
         "Co": {"LDAUU": 3.0, "LDAUL": 2, "LDAUJ": 0.0}}

    pymatgen ``user_incar_settings`` expects three separate dicts::

        LDAUU = {"Fe": 4.0, "Co": 3.0}   # Dict[str, float]
        LDAUL = {"Fe": 2,   "Co": 2}      # Dict[str, int]
        LDAUJ = {"Fe": 0.0, "Co": 0.0}    # Dict[str, float]

    ``to_pymatgen_format()`` performs this conversion.

    DFT+U（Hubbard U）参数——前端暴露。

    ``values`` 格式::

        {"Fe": {"LDAUU": 4.0, "LDAUL": 2, "LDAUJ": 0.0},
         "Co": {"LDAUU": 3.0, "LDAUL": 2, "LDAUJ": 0.0}}

    pymatgen ``user_incar_settings`` 期望三个独立的字典::

        LDAUU = {"Fe": 4.0, "Co": 3.0}   # Dict[str, float]
        LDAUL = {"Fe": 2,   "Co": 2}      # Dict[str, int]
        LDAUJ = {"Fe": 0.0, "Co": 0.0}    # Dict[str, float]

    ``to_pymatgen_format()`` 执行此转换。
    """
    enabled: bool = False
    values: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_pymatgen_format(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Convert the nested ``values`` dict to three separate pymatgen dicts.

        Returns:
            ``{"LDAUU": {...}, "LDAUL": {...}, "LDAUJ": {...}}``, or ``None``
            when DFT+U is disabled or ``values`` is empty.

        将嵌套的 ``values`` 字典转换为 pymatgen 期望的三个独立字典。

        返回：
            ``{"LDAUU": {...}, "LDAUL": {...}, "LDAUJ": {...}}``，
            若 DFT+U 未启用或 ``values`` 为空则返回 ``None``。
        """
        if not self.enabled or not self.values:
            return None
        ldauu: Dict[str, float] = {}
        ldaul: Dict[str, int]   = {}
        ldauj: Dict[str, float] = {}
        for elem, uda in self.values.items():
            ldauu[elem] = float(uda.get("LDAUU", 0.0))
            ldaul[elem] = int(uda.get("LDAUL", 0))
            ldauj[elem] = float(uda.get("LDAUJ", 0.0))
        return {"LDAUU": ldauu, "LDAUL": ldaul, "LDAUJ": ldauj}


@dataclass
class FrontendVdwParams:
    """van der Waals dispersion correction parameters exposed to the frontend.

    Attributes:
        method: Dispersion correction method name, e.g. ``"None"``, ``"D3"``,
            ``"D3BJ"``.  Resolved from ``FRONTEND_VDW_MAP`` in
            ``FrontendAdapter.from_frontend_dict()``.

    范德华色散校正参数——前端暴露。

    属性：
        method: 色散校正方法名称，如 ``"None"``、``"D3"``、``"D3BJ"``。
            在 ``FrontendAdapter.from_frontend_dict()`` 中通过 ``FRONTEND_VDW_MAP``
            解析。
    """
    method: str = "None"


@dataclass
class FrontendDipoleParams:
    """Dipole correction parameters exposed to the frontend.

    Attributes:
        enabled:   Whether to enable dipole correction (LDIPOL = .TRUE.).
        direction: Cartesian direction along which the correction is applied
            (IDIPOL: 1=x, 2=y, 3=z).

    偶极校正参数——前端暴露。

    属性：
        enabled:   是否启用偶极校正（LDIPOL = .TRUE.）。
        direction: 施加校正的笛卡儿方向（IDIPOL：1=x，2=y，3=z）。
    """
    enabled: bool = True
    direction: int = 3


@dataclass
class FrontendFrequencyParams:
    """Vibrational frequency calculation parameters exposed to the frontend.

    Attributes:
        ibrion:             VASP IBRION tag (5 = finite differences, 7 = DFPT).
        potim:              Finite-difference step size in Å (POTIM).
        nfree:              Number of displacements per atom (NFREE).
        vibrate_mode:       Atom-selection strategy: ``"inherit"`` uses
            ``vibrate_indices``; other values are interpreted by the maker layer.
        adsorbate_formula:  Chemical formula of the adsorbate for automatic
            index selection.
        adsorbate_prefer:   Preferred end of the structure to match when
            resolving the adsorbate (``"tail"`` or ``"head"``).
        vibrate_indices:    Explicit zero-based atom indices to displace.
        calc_ir:            If ``True``, also compute the dielectric tensor
            (sets LEPSILON and IBRION = 7).

    频率计算参数——前端暴露。

    属性：
        ibrion:             VASP IBRION 标记（5 = 有限差分，7 = DFPT）。
        potim:              有限差分步长（Å），对应 POTIM。
        nfree:              每个原子的位移次数，对应 NFREE。
        vibrate_mode:       原子选择策略：``"inherit"`` 使用 ``vibrate_indices``；
            其他值由 maker 层解释。
        adsorbate_formula:  用于自动选取原子索引的吸附质化学式。
        adsorbate_prefer:   解析吸附质时优先匹配结构的哪一端（``"tail"`` 或 ``"head"``）。
        vibrate_indices:    显式指定的零索引原子下标列表（用于位移计算）。
        calc_ir:            若为 ``True``，同时计算介电张量（设置 LEPSILON 和
            IBRION = 7）。
    """
    ibrion: int = 5
    potim: float = 0.015
    nfree: int = 2
    vibrate_mode: str = "inherit"
    adsorbate_formula: Optional[str] = None
    adsorbate_prefer: str = "tail"
    vibrate_indices: Optional[List[int]] = None
    calc_ir: bool = False


@dataclass
class FrontendLobsterParams:
    """LOBSTER chemical-bonding analysis parameters exposed to the frontend.

    Attributes:
        lobsterin_mode:         Template mode: ``"template"`` (auto-generate) or
            ``"custom"`` (use ``custom_lobsterin``).
        custom_lobsterin:       Full lobsterin content string when
            ``lobsterin_mode="custom"``.
        start_energy:           Lower bound of the energy window (eV).
        end_energy:             Upper bound of the energy window (eV).
        cohp_generator:         COHP bond-length range string passed to lobsterin.
        overwritedict:          Key–value pairs that overwrite the generated
            lobsterin dict before writing.
        custom_lobsterin_lines: Verbatim lines appended to the lobsterin file.

    LOBSTER 化学键分析参数——前端暴露。

    属性：
        lobsterin_mode:         模板模式：``"template"``（自动生成）或
            ``"custom"``（使用 ``custom_lobsterin``）。
        custom_lobsterin:       当 ``lobsterin_mode="custom"`` 时使用的完整
            lobsterin 内容字符串。
        start_energy:           能量窗口下界（eV）。
        end_energy:             能量窗口上界（eV）。
        cohp_generator:         传递给 lobsterin 的 COHP 键长范围字符串。
        overwritedict:          写入前覆盖生成的 lobsterin 字典的键值对。
        custom_lobsterin_lines: 逐字追加到 lobsterin 文件末尾的行列表。
    """
    lobsterin_mode: str = "template"
    custom_lobsterin: Optional[str] = None
    start_energy: float = -20.0
    end_energy: float = 20.0
    cohp_generator: str = "from 1.2 to 1.9 orbitalwise"
    overwritedict: Optional[Dict[str, Any]] = None
    custom_lobsterin_lines: Optional[List[str]] = None


@dataclass
class FrontendNBOParams:
    """Natural Bond Orbital (NBO) analysis parameters exposed to the frontend.

    Attributes:
        basis_source:      Basis set identifier (``"ANO-RCC-MB"`` or
            ``"custom"``).
        custom_basis_path: Path to a custom basis set file when
            ``basis_source="custom"``.
        occ_1c:            One-centre occupancy threshold for bond detection.
        occ_2c:            Two-centre occupancy threshold for bond detection.
        print_cube:        Whether to write cube files (``"T"`` / ``"F"``).
        density:           Whether to write density cube (``"T"`` / ``"F"``).
        vis_start:         First orbital index for cube visualisation.
        vis_end:           Last orbital index (``-1`` = last orbital).
        mesh:              Cube-file grid dimensions ``[nx, ny, nz]``.
        box_int:           Integer box extension factors ``[bx, by, bz]``.
        origin_fact:       Fractional origin offset factor.

    自然键轨道（NBO）分析参数——前端暴露。

    属性：
        basis_source:      基组标识符（``"ANO-RCC-MB"`` 或 ``"custom"``）。
        custom_basis_path: 当 ``basis_source="custom"`` 时指向自定义基组文件的路径。
        occ_1c:            单中心占据阈值，用于键的判定。
        occ_2c:            双中心占据阈值，用于键的判定。
        print_cube:        是否输出 cube 文件（``"T"`` / ``"F"``）。
        density:           是否输出密度 cube 文件（``"T"`` / ``"F"``）。
        vis_start:         cube 可视化的起始轨道索引。
        vis_end:           终止轨道索引（``-1`` 表示最后一个轨道）。
        mesh:              cube 文件网格维度 ``[nx, ny, nz]``。
        box_int:           整数盒子扩展因子 ``[bx, by, bz]``。
        origin_fact:       分数原点偏移因子。
    """
    basis_source: str = "ANO-RCC-MB"
    custom_basis_path: Optional[str] = None
    occ_1c: float = 1.60
    occ_2c: float = 1.85
    print_cube: str = "F"
    density: str = "F"
    vis_start: int = 0
    vis_end: int = -1
    mesh: List[int] = field(default_factory=lambda: [0, 0, 0])
    box_int: List[int] = field(default_factory=lambda: [1, 1, 1])
    origin_fact: float = 0.00


@dataclass
class FrontendMDParams:
    """Molecular dynamics parameters exposed to the frontend.

    Attributes:
        ensemble:        Statistical ensemble — ``"nvt"`` (constant N, V, T)
            or ``"npt"`` (constant N, p, T).
        start_temp:      Starting temperature in K (TEBEG).
        end_temp:        Ending temperature in K (TEEND); equal to
            ``start_temp`` for isothermal runs.
        nsteps:          Total number of MD steps (NSW).
        time_step:       Ionic time step in fs (POTIM); ``None`` uses the
            InputSet default.
        langevin_gamma:  Per-element Langevin friction coefficients (ps⁻¹),
            passed as ``{"Fe": 10.0, …}`` when using the Langevin thermostat.

    分子动力学参数——前端暴露。

    属性：
        ensemble:        统计系综——``"nvt"``（恒定 N、V、T）或
            ``"npt"``（恒定 N、p、T）。
        start_temp:      起始温度（K），对应 TEBEG。
        end_temp:        终止温度（K），对应 TEEND；等温模拟时与 ``start_temp`` 相同。
        nsteps:          MD 总步数，对应 NSW。
        time_step:       离子时间步长（fs），对应 POTIM；``None`` 使用 InputSet 默认值。
        langevin_gamma:  各元素的 Langevin 摩擦系数（ps⁻¹），使用 Langevin
            恒温器时以 ``{"Fe": 10.0, …}`` 形式传入。
    """
    ensemble: str = "nvt"
    start_temp: float = 300.0
    end_temp: float = 300.0
    nsteps: int = 1000
    time_step: Optional[float] = None
    langevin_gamma: Optional[Dict[str, float]] = None


@dataclass
class FrontendNEBParams:
    """Nudged Elastic Band (NEB) transition-state search parameters exposed to
    the frontend.

    Attributes:
        n_images:  Number of intermediate images between endpoints.
        use_idpp:  If ``True``, initialise the path with the Image Dependent
            Pair Potential (IDPP) interpolation instead of linear interpolation.

    微动弹性带（NEB）过渡态搜索参数——前端暴露。

    属性：
        n_images:  端点之间的中间像数量。
        use_idpp:  若为 ``True``，使用图像依赖对势（IDPP）内插初始化路径，
            而非线性内插。
    """
    n_images: int = 6
    use_idpp: bool = True


@dataclass
class FrontendResourceParams:
    """Compute resource allocation parameters exposed to the frontend.

    Attributes:
        runtime: Wall-clock time limit in hours.
        cores:   Number of MPI ranks (CPU cores) to request.
        queue:   Scheduler queue / partition name.

    计算资源配置——前端暴露。

    属性：
        runtime: 计算时间上限（小时）。
        cores:   请求的 MPI 进程数（CPU 核心数）。
        queue:   调度器队列/分区名称。
    """
    runtime: int = 72
    cores: int = 72
    queue: str = "low"


# ============================================================================
# 统一API参数类
# ============================================================================

@dataclass
class VaspWorkflowParams:
    """Unified VASP workflow parameters — the complete frontend-supplied configuration.

    This dataclass is the canonical intermediate representation between the raw
    frontend dict and the engine's ``WorkflowConfig``.  All frontend parameter
    groups are collected here as typed sub-objects; ``to_workflow_config()``
    converts them to the engine format.

    Core attributes:
        calc_type:       Frontend calculation type string (resolved to
            ``CalcType`` by ``to_workflow_config()``).
        structure:       Input structure — file path, pymatgen ``Structure``,
            or ``FrontendStructureInput``.

    Optional sub-module attributes:
        precision, kpoints, magmom, dft_u, vdw, dipole, frequency,
        lobster, nbo, md, neb — each holds a typed params dataclass or ``None``.

    统一的 VASP 工作流参数——前端传入的完整配置。

    本数据类是原始前端字典与引擎 ``WorkflowConfig`` 之间的规范中间表示。
    所有前端参数组以类型化子对象形式汇聚于此；``to_workflow_config()`` 负责将其
    转换为引擎格式。

    核心属性：
        calc_type:  前端计算类型字符串（由 ``to_workflow_config()`` 解析为
            ``CalcType``）。
        structure:  输入结构——文件路径、pymatgen ``Structure`` 对象或
            ``FrontendStructureInput``。

    可选子模块属性：
        precision、kpoints、magmom、dft_u、vdw、dipole、frequency、
        lobster、nbo、md、neb——每项持有一个类型化参数数据类或 ``None``。
    """

    # === 核心参数 ===
    calc_type: str
    structure: Union[str, Path, Structure, FrontendStructureInput]

    # === 功能参数 ===
    functional: str = "PBE"
    kpoints_density: float = 50.0

    # === 可选参数 ===
    prev_dir: Optional[Union[str, Path]] = None
    output_dir: Optional[Union[str, Path]] = None

    # === 子模块参数 ===
    precision: Optional[FrontendPrecisionParams] = None
    kpoints: Optional[FrontendKpointParams] = None
    magmom: Optional[FrontendMagmomParams] = None
    dft_u: Optional[FrontendDFTPlusUParams] = None
    vdw: Optional[FrontendVdwParams] = None
    dipole: Optional[FrontendDipoleParams] = None
    frequency: Optional[FrontendFrequencyParams] = None
    lobster: Optional[FrontendLobsterParams] = None
    nbo: Optional[FrontendNBOParams] = None
    nbo_config: Optional[Dict[str, Any]] = None
    md: Optional[FrontendMDParams] = None
    neb: Optional[FrontendNEBParams] = None

    # === 资源配置 ===
    resources: Optional[FrontendResourceParams] = None

    # === 自定义INCAR ===
    custom_incar: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Normalise functional to uppercase and supply default sub-objects.
        # 将泛函规范化为大写，并提供默认子对象。
        self.functional = self.functional.upper()
        if self.precision is None:
            self.precision = FrontendPrecisionParams()
        if self.kpoints is None:
            self.kpoints = FrontendKpointParams()
        if self.resources is None:
            self.resources = FrontendResourceParams()

    def to_workflow_config(self) -> WorkflowConfig:
        """Convert this adapter object to the engine's ``WorkflowConfig``.

        Mapping rules:
        - ``calc_type`` string → ``CalcType`` enum via ``calc_type_map``
        - ``custom_incar`` + precision overrides → ``user_incar_overrides``
        - Sub-module params (frequency, lobster, NBO) set dedicated fields on
          ``WorkflowConfig`` (e.g. ``vibrate_indices``, ``lobster_overwritedict``)

        Extension: when you add a new ``FrontendXxxParams`` group, add a matching
        ``if self.xxx: config.xxx_field = ...`` block at the end of this method,
        and add the corresponding field to ``workflow_engine.WorkflowConfig``.

        将本适配器对象转换为引擎所需的 ``WorkflowConfig``。

        映射规则：
        - ``calc_type`` 字符串通过 ``calc_type_map`` 映射为 ``CalcType`` 枚举值。
        - ``custom_incar`` + 精度覆盖参数合并为 ``user_incar_overrides``。
        - 子模块参数（frequency、lobster、NBO）设置 ``WorkflowConfig`` 的专用字段
          （如 ``vibrate_indices``、``lobster_overwritedict``）。

        扩展：新增 ``FrontendXxxParams`` 参数组时，在本方法末尾添加对应的
        ``if self.xxx: config.xxx_field = ...`` 代码块，并在
        ``workflow_engine.WorkflowConfig`` 中添加相应字段。
        """

        calc_type_map = {
            "bulk_relax":    CalcType.BULK_RELAX,
            "slab_relax":    CalcType.SLAB_RELAX,
            "static_sp":     CalcType.STATIC_SP,
            "static_dos":    CalcType.DOS_SP,
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

        backend_calc_type = calc_type_map.get(self.calc_type)
        if backend_calc_type is None:
            raise ValueError(f"未知的计算类型: {self.calc_type}")

        # ── 构建 INCAR 覆盖参数 ────────────────────────────────────────────
        user_incar_overrides: Dict[str, Any] = {}
        if self.custom_incar:
            user_incar_overrides.update(self.custom_incar)

        if self.precision:
            p = self.precision
            if p.encut:  user_incar_overrides["ENCUT"]  = p.encut
            if p.ediff:  user_incar_overrides["EDIFF"]  = p.ediff
            if p.ediffg: user_incar_overrides["EDIFFG"] = p.ediffg
            if p.nedos:  user_incar_overrides["NEDOS"]  = p.nedos

        # frequency
        if self.frequency and backend_calc_type in (CalcType.FREQ, CalcType.FREQ_IR):
            f = self.frequency
            user_incar_overrides.update({
                "IBRION": f.ibrion,
                "POTIM":  f.potim,
                "NFREE":  f.nfree,
            })
            if f.calc_ir:
                user_incar_overrides["LEPSILON"] = True
                user_incar_overrides["IBRION"]   = 7

        # ── MAGMOM ────────────────────────────────────────────────────────
        if self.magmom and self.magmom.enabled:
            if self.magmom.per_atom:
                # 直接传 per-site 列表，pymatgen 原样写入 INCAR
                # Pass per-site list directly; pymatgen writes it verbatim to INCAR.
                user_incar_overrides["MAGMOM"] = [float(v) for v in self.magmom.per_atom]
            elif self.magmom.per_element:
                # 传 per-element dict，pymatgen 按结构中各元素出现顺序展开
                # Pass per-element dict; pymatgen expands by element order in the structure.
                user_incar_overrides["MAGMOM"] = {
                    k: float(v) for k, v in self.magmom.per_element.items()
                }

        # dipole
        if self.dipole and self.dipole.enabled:
            user_incar_overrides.update({
                "IDIPOL": self.dipole.direction,
                "LDIPOL": True,
            })

        # ── DFT+U ─────────────────────────────────────────────────────────
        if self.dft_u and self.dft_u.enabled:
            dft_u_fmt = self.dft_u.to_pymatgen_format()
            if dft_u_fmt:
                user_incar_overrides["LDAUU"] = dft_u_fmt["LDAUU"]
                user_incar_overrides["LDAUL"] = dft_u_fmt["LDAUL"]
                user_incar_overrides["LDAUJ"] = dft_u_fmt["LDAUJ"]
                user_incar_overrides["LDAU"]  = True

        # MD
        if self.md and backend_calc_type in (CalcType.MD_NVT, CalcType.MD_NPT):
            user_incar_overrides.update({
                "TEBEG": self.md.start_temp,
                "TEEND": self.md.end_temp,
                "NSW":   self.md.nsteps,
            })
            if self.md.time_step:
                user_incar_overrides["POTIM"] = self.md.time_step

        # ── 构建 WorkflowConfig ───────────────────────────────────────────
        config = WorkflowConfig(
            calc_type=backend_calc_type,
            structure=self.structure,
            functional=self.functional,
            kpoints_density=self.kpoints_density,
            output_dir=self.output_dir,
            prev_dir=self.prev_dir,
            user_incar_overrides=user_incar_overrides,
        )

        if self.md:
            config.ensemble   = self.md.ensemble
            config.start_temp = self.md.start_temp
            config.end_temp   = self.md.end_temp
            config.nsteps     = self.md.nsteps
            config.time_step  = self.md.time_step

        if self.neb:
            config.n_images  = self.neb.n_images
            config.use_idpp  = self.neb.use_idpp

        if self.frequency:
            config.vibrate_indices = self.frequency.vibrate_indices
            config.calc_ir         = self.frequency.calc_ir

        if self.lobster:
            config.lobster_overwritedict   = self.lobster.overwritedict
            config.lobster_custom_lines    = self.lobster.custom_lobsterin_lines

        if self.nbo_config is not None:
            config.nbo_config = self.nbo_config

        return config

    def get_script_context(self) -> Dict[str, Any]:
        """Build the template rendering context required by the script generator.

        Returns:
            Dict with keys ``functional``, ``calc_category``, ``cores``,
            ``walltime``, ``queue``, and optionally ``need_vdw``.

        构建脚本生成器所需的模板渲染上下文。

        返回：
            包含 ``functional``、``calc_category``、``cores``、``walltime``、
            ``queue`` 以及可选 ``need_vdw`` 键的字典。
        """
        # Map frontend calc_type strings to CalcCategory for PBS/SLURM template selection.
        # 将前端 calc_type 字符串映射为 CalcCategory，用于 PBS/SLURM 模板选择。
        calc_type_to_category = {
            "bulk_relax":    CalcCategory.RELAX,
            "slab_relax":    CalcCategory.RELAX,
            "static_sp":     CalcCategory.STATIC,
            "static_dos":    CalcCategory.STATIC,
            "static_charge": CalcCategory.STATIC,
            "static_elf":    CalcCategory.STATIC,
            "neb":           CalcCategory.NEB,
            "dimer":         CalcCategory.DIMER,
            "freq":          CalcCategory.FREQ,
            "freq_ir":       CalcCategory.FREQ,
            "lobster":       CalcCategory.LOBSTER,
            "nmr_cs":        CalcCategory.NMR,
            "nmr_efg":       CalcCategory.NMR,
            "nbo":           CalcCategory.NBO,
            "md_nvt":        CalcCategory.MD,
            "md_npt":        CalcCategory.MD,
        }
        calc_category = calc_type_to_category.get(self.calc_type, CalcCategory.STATIC)

        res = self.resources
        context: Dict[str, Any] = {
            "functional":    self.functional,
            "calc_category": calc_category,
            "cores":         res.cores   if res else None,
            "walltime":      res.runtime if res else None,
            "queue":         res.queue   if res else None,
        }

        if self.vdw and self.vdw.method != "None":
            context["need_vdw"] = True
        return context


# ============================================================================
# API 类
# ============================================================================

class VaspAPI:
    """Unified VASP workflow API.

    Wraps ``WorkflowEngine`` and ``Script`` to provide a single entry point
    for executing a complete VASP workflow (input generation + job script
    creation) from a ``VaspWorkflowParams`` object.

    VASP 工作流统一 API。

    封装 ``WorkflowEngine`` 与 ``Script``，提供单一入口点，用于从
    ``VaspWorkflowParams`` 对象执行完整的 VASP 工作流（输入文件生成 + 作业
    脚本创建）。
    """

    def __init__(
        self,
        engine: Optional[WorkflowEngine] = None,
        script_maker: Optional[Script] = None,
    ):
        self.engine = engine or WorkflowEngine()
        self.script_maker = script_maker or Script()

    def run_workflow(
        self,
        params: VaspWorkflowParams,
        generate_script: bool = True,
    ) -> Dict[str, Any]:
        """Execute a full VASP workflow from a ``VaspWorkflowParams`` object.

        Steps:
        1. Convert *params* to ``WorkflowConfig`` and validate.
        2. Call ``WorkflowEngine.run()`` to write VASP input files.
        3. Optionally generate a PBS/SLURM job script.

        Args:
            params:          Typed workflow parameters.
            generate_script: If ``True``, also write a job submission script.

        Returns:
            Dict with keys ``success``, ``output_dir``, ``calc_type``, and
            optionally ``script_paths``.

        Raises:
            ValueError: if ``config.validate()`` returns any errors.

        从 ``VaspWorkflowParams`` 对象执行完整的 VASP 工作流。

        步骤：
        1. 将 *params* 转换为 ``WorkflowConfig`` 并验证。
        2. 调用 ``WorkflowEngine.run()`` 写出 VASP 输入文件。
        3. 可选地生成 PBS/SLURM 作业脚本。

        参数：
            params:          类型化工作流参数。
            generate_script: 若为 ``True``，同时写出作业提交脚本。

        返回：
            包含 ``success``、``output_dir``、``calc_type`` 以及可选
            ``script_paths`` 键的字典。

        抛出：
            ValueError: 若 ``config.validate()`` 返回任何错误。
        """
        logger.info(f"开始执行工作流: calc_type={params.calc_type}")

        config = params.to_workflow_config()
        output_dir = self.engine.run(config)
        result = {
            "success":    True,
            "output_dir": output_dir,
            "calc_type":  params.calc_type,
        }

        if generate_script:
            script_paths = self.engine.generate_script(
                config=config,
                **params.get_script_context()
            )
            result["script_paths"] = script_paths

        logger.info(f"工作流执行完成: {output_dir}")
        return result

    def validate_params(self, params: VaspWorkflowParams) -> List[str]:
        """Validate a ``VaspWorkflowParams`` object without executing the workflow.

        Delegates all checks to ``validator.validate()``.  Returns the error list
        rather than raising, preserving the ``List[str]`` contract for callers.

        Args:
            params: Workflow parameters to validate.

        Returns:
            List of human-readable error strings (empty if valid).

        将所有检查委托给 ``validator.validate()``，以 ``List[str]`` 形式返回错误，
        而非抛出异常，保持调用方兼容性。
        """
        if params is None:
            return ["VaspWorkflowParams object is None"]

        errors: List[str] = []
        try:
            _validator_validate(
                calc_type=params.calc_type,
                structure=params.structure,
                functional=params.functional,
                kpoints_density=params.kpoints_density,
                output_dir=params.output_dir,
                prev_dir=params.prev_dir,
            )
        except ValidationError as exc:
            errors.extend(exc.errors)
        return errors

    @staticmethod
    def from_json(json_str: str) -> VaspWorkflowParams:
        """Deserialise a JSON string to ``VaspWorkflowParams``.

        Delegates to ``from_dict`` after parsing.

        Args:
            json_str: JSON-encoded workflow parameter dict.

        Returns:
            Parsed ``VaspWorkflowParams`` instance.

        将 JSON 字符串反序列化为 ``VaspWorkflowParams``。

        解析后委托给 ``from_dict``。

        参数：
            json_str: JSON 编码的工作流参数字典。

        返回：
            解析后的 ``VaspWorkflowParams`` 实例。
        """
        import json
        data = json.loads(json_str)
        return VaspAPI.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> VaspWorkflowParams:
        """Build a ``VaspWorkflowParams`` from a plain dict.

        If *data* contains ``"type"`` and ``"settings"`` keys it is treated as
        a frontend dict and forwarded to ``FrontendAdapter.from_frontend_dict``.
        Otherwise the direct-mapping path is used.

        Args:
            data: Workflow parameter dict (direct or frontend format).

        Returns:
            Constructed ``VaspWorkflowParams`` instance.

        从普通字典构建 ``VaspWorkflowParams``。

        若 *data* 包含 ``"type"`` 和 ``"settings"`` 键，则视为前端格式，
        转发给 ``FrontendAdapter.from_frontend_dict`` 处理；否则使用直接映射路径。

        参数：
            data: 工作流参数字典（直接格式或前端格式）。

        返回：
            构建的 ``VaspWorkflowParams`` 实例。
        """
        if "type" in data and "settings" in data:
            return FrontendAdapter.from_frontend_dict(data)

        precision = None
        if "precision" in data and data["precision"]:
            p = data["precision"]
            precision = FrontendPrecisionParams(
                encut=p.get("encut"), ediff=p.get("ediff"),
                ediffg=p.get("ediffg"), nedos=p.get("nedos"),
            )

        kpoints = None
        if "kpoints" in data and data["kpoints"]:
            k = data["kpoints"]
            kpoints = FrontendKpointParams(
                density=k.get("density"),
                gamma_centered=k.get("gammaCentered", True),
            )

        resources = None
        if "resource" in data and data["resource"]:
            r = data["resource"]
            # Strip a trailing "h" or "H" from runtime strings like "72h".
            # 从 "72h" 等格式中去除尾部的 "h" 或 "H"。
            runtime_str = str(r.get("runtime", "72"))
            runtime = int(runtime_str.rstrip("hH"))
            resources = FrontendResourceParams(runtime=runtime, cores=r.get("cores", 72))

        structure_input = None
        if "structure" in data and isinstance(data["structure"], dict):
            s = data["structure"]
            structure_input = FrontendStructureInput(
                source=s.get("source", "file"),
                id=s.get("id", ""),
                content=s.get("content", ""),
            )

        md = None
        if "md" in data and data["md"]:
            m = data["md"]
            md = FrontendMDParams(
                ensemble=m.get("ensemble", "nvt"),
                start_temp=m.get("TEBEG", 300),
                end_temp=m.get("TEEND", 300),
                nsteps=m.get("NSW", 1000),
                time_step=m.get("POTIM"),
            )

        frequency = None
        if "frequency" in data and data["frequency"]:
            f = data["frequency"]
            frequency = FrontendFrequencyParams(
                ibrion=f.get("IBRION", 5),
                potim=f.get("POTIM", 0.015),
                nfree=f.get("NFREE", 2),
                vibrate_mode=f.get("vibrate_mode", "inherit"),
                adsorbate_formula=f.get("adsorbate_formula"),
                calc_ir=f.get("calc_ir", False),
            )

        prev_dir = None
        if "prev_dir" in data and data["prev_dir"]:
            prev_dir = data["prev_dir"]
        elif "from_prev_calc" in data and data["from_prev_calc"]:
            prev_dir = data["from_prev_calc"]

        custom_incar = data.get("custom_incar") or None

        return VaspWorkflowParams(
            calc_type=data.get("calc_type", "static_sp"),
            structure=data.get("structure", data.get("structure_id", "")),
            functional=data.get("functional", data.get("xc", "PBE")),
            kpoints_density=data.get("kpoints_density", data.get("kpoints", {}).get("density", 50)),
            prev_dir=prev_dir,
            output_dir=data.get("output_dir"),
            precision=precision,
            kpoints=kpoints,
            resources=resources,
            md=md,
            frequency=frequency,
            custom_incar=custom_incar,
        )


# ============================================================================
# 前端兼容层
# ============================================================================

# Map frontend string names → backend calc_type strings used by WorkflowEngine.
# Add an entry here whenever a new CalcType is registered in workflow_engine.py.
# 将前端字符串名称映射到 WorkflowEngine 使用的后端 calc_type 字符串。
# 每当在 workflow_engine.py 中注册新的 CalcType 时，在此处添加对应条目。
FRONTEND_CALC_TYPE_MAP = {
    "dos":           "static_dos",
    "lobster":       "lobster",
    "freq":          "freq",
    "nbo":           "nbo",
    "bulk_relax":    "bulk_relax",
    "slab_relax":    "slab_relax",
    "relax":         "slab_relax",
    "static_sp":     "static_sp",
    "static_charge": "static_charge",
    "static_elf":    "static_elf",
    "neb":           "neb",
    "dimer":         "dimer",
    "md_nvt":        "md_nvt",
    "md_npt":        "md_npt",
    "nmr_cs":        "nmr_cs",
    "nmr_efg":       "nmr_efg",
}

# Reverse mapping: backend calc_type string → first matching frontend name.
# 反向映射：后端 calc_type 字符串 → 第一个匹配的前端名称。
BACKEND_TO_FRONTEND_CALC_TYPE = {v: k for k, v in FRONTEND_CALC_TYPE_MAP.items()}

# Map frontend XC functional aliases to canonical pymatgen functional names.
# 将前端 XC 泛函别名映射为 pymatgen 规范泛函名称。
FRONTEND_XC_MAP = {
    "PBE": "PBE", "RPBE": "RPBE", "BEEF": "BEEF",
    "SCAN": "SCAN", "HSE": "HSE", "LDA": "LDA", "PBEsol": "PBEsol",
}

# Map frontend vdW method strings to canonical method identifiers.
# 将前端 vdW 方法字符串映射为规范方法标识符。
FRONTEND_VDW_MAP = {
    "None": "None", "D3": "D3", "D3BJ": "D3BJ",
    "DFT-D2": "DFT-D2", "DFT-D3": "D3",
}


class FrontendAdapter:
    """Convert a plain frontend dict into a fully typed ``VaspWorkflowParams``.

    This is the sole public entry point used by ``BaseStage._write_vasp_inputs()``.
    It normalises string calc-type names, resolves XC functional aliases, and
    unpacks every sub-module parameter group (precision, lobster, NBO, frequency,
    MAGMOM, DFT+U, …) from the flat ``settings`` dict.

    To add a new frontend parameter:
      1. Add the key to ``known_keys`` inside ``from_frontend_dict`` so it is
         excluded from ``custom_incar``.
      2. Build the matching ``FrontendXxxParams`` object and attach it to the
         returned ``VaspWorkflowParams``.
      3. Transfer it to ``WorkflowConfig`` in ``to_workflow_config()``.

    将普通前端字典转换为完全类型化的 ``VaspWorkflowParams``。

    这是 ``BaseStage._write_vasp_inputs()`` 唯一使用的公共入口点。
    它规范化 calc_type 字符串名称，解析 XC 泛函别名，并从扁平的 ``settings``
    字典中解包每个子模块参数组（precision、lobster、NBO、frequency、MAGMOM、
    DFT+U 等）。

    新增前端参数的步骤：
      1. 在 ``from_frontend_dict`` 内的 ``known_keys`` 中添加该键，使其不进入
         ``custom_incar``。
      2. 构建对应的 ``FrontendXxxParams`` 对象并附加到返回的
         ``VaspWorkflowParams`` 上。
      3. 在 ``to_workflow_config()`` 中将其传递给 ``WorkflowConfig``。
    """

    @staticmethod
    def from_frontend_dict(data: Dict[str, Any]) -> VaspWorkflowParams:
        """Build a ``VaspWorkflowParams`` from a flat frontend dict.

        Expected keys in *data*:
            calc_type   – string name, resolved via ``FRONTEND_CALC_TYPE_MAP``
            xc / functional – functional alias, resolved via ``FRONTEND_XC_MAP``
            kpoints     – ``{"density": float}``
            settings    – flat INCAR + sub-module params (NEDOS, IBRION, lobsterin_mode, …)
            structure   – ``{"source": "file", "id": "<path>"}``
            prev_dir    – predecessor calculation directory
            lobsterin   – ``Dict[str, Any]`` written to lobsterin (overwritedict)
            lobsterin_custom_lines – ``List[str]`` appended verbatim to lobsterin

        从扁平的前端字典构建 ``VaspWorkflowParams``。

        *data* 中的预期键：
            calc_type              — 字符串名称，通过 ``FRONTEND_CALC_TYPE_MAP`` 解析
            xc / functional        — 泛函别名，通过 ``FRONTEND_XC_MAP`` 解析
            kpoints                — ``{"density": float}``
            settings               — 扁平 INCAR + 子模块参数（NEDOS、IBRION、
                                     lobsterin_mode 等）
            structure              — ``{"source": "file", "id": "<path>"}``
            prev_dir               — 前序计算目录
            lobsterin              — ``Dict[str, Any]``，写入 lobsterin（overwritedict）
            lobsterin_custom_lines — ``List[str]``，逐字追加到 lobsterin
        """
        # 1. calc_type
        # Resolve frontend calc_type alias to the backend string used by WorkflowEngine.
        # 将前端 calc_type 别名解析为 WorkflowEngine 使用的后端字符串。
        frontend_calc_type = data.get("calc_type", "dos")
        backend_calc_type = FRONTEND_CALC_TYPE_MAP.get(frontend_calc_type, frontend_calc_type)

        # 2. 结构
        # Parse the structure descriptor into a FrontendStructureInput.
        # 将结构描述符解析为 FrontendStructureInput。
        struct_data = data.get("structure", {})
        if isinstance(struct_data, dict):
            structure = FrontendStructureInput(
                source=struct_data.get("source", "file"),
                id=struct_data.get("id", ""),
                content=struct_data.get("content", ""),
            )
        else:
            # Non-dict values (e.g. raw path string) are passed through unchanged.
            # 非字典值（如原始路径字符串）直接透传。
            structure = struct_data

        # 3. 泛函
        # Resolve XC alias; fall back to uppercasing the raw string.
        # 解析 XC 别名；若无匹配则将原始字符串转为大写。
        xc = data.get("xc", data.get("functional", "PBE"))
        functional = FRONTEND_XC_MAP.get(xc, xc.upper())

        # 4. settings
        settings = data.get("settings", {})

        # 4.1 精度参数
        precision = FrontendPrecisionParams(
            nedos=_parse_int(settings.get("NEDOS")),
            encut=_parse_int(settings.get("ENCUT")),
            ediff=_parse_float(settings.get("EDIFF")),
            ediffg=_parse_float(settings.get("EDIFFG")),
        )

        # 4.2 自定义 INCAR（非标准字段）
        # Collect all settings keys not in known_keys as raw INCAR overrides.
        # 将所有不在 known_keys 中的 settings 键收集为原始 INCAR 覆盖。
        custom_incar: Dict[str, Any] = {}
        known_keys = {
            "NEDOS", "ENCUT", "EDIFF", "EDIFFG",
            "ISMEAR", "SIGMA", "IBRION", "POTIM", "NFREE",
            "from_prev_calc", "lobsterin_mode", "calc_ir",
            "vibrate_mode", "adsorbate_formula", "adsorbate_formula_prefer",
            "vibrate_indices", "basis_source", "nbo_config",
            # MAGMOM / DFT+U 单独处理，不进 custom_incar
            # MAGMOM / DFT+U are handled separately and excluded from custom_incar.
            "MAGMOM", "LDAUU", "LDAUL", "LDAUJ",
        }
        for key, value in settings.items():
            if key not in known_keys and value not in (None, "", "—"):
                try:
                    # Attempt numeric parsing; fall back to raw string on failure.
                    # 尝试数值解析；失败时保留原始字符串。
                    custom_incar[key] = _parse_number(value)
                except (ValueError, TypeError):
                    custom_incar[key] = value

        # 4.3 ISMEAR / SIGMA
        # Apply smearing parameters only when explicitly provided and non-empty.
        # 仅在显式提供且非空时才应用展宽参数。
        if "ISMEAR" in settings and settings["ISMEAR"] not in (None, "", "—"):
            custom_incar["ISMEAR"] = _parse_int(settings["ISMEAR"])
        if "SIGMA" in settings and settings["SIGMA"] not in (None, "", "—"):
            custom_incar["SIGMA"] = _parse_float(settings["SIGMA"])

        # 5. kpoints
        kpt_data = data.get("kpoints", {})
        if isinstance(kpt_data, dict):
            kpoints = FrontendKpointParams(
                density=_parse_float(kpt_data.get("density")),
                gamma_centered=kpt_data.get("gammaCentered", True),
            )
        else:
            # Scalar value treated as density directly.
            # 标量值直接视为密度。
            kpoints = FrontendKpointParams(density=_parse_float(kpt_data))

        # 6. resource
        # Strip trailing "h"/"H" from runtime strings such as "72h".
        # 从 "72h" 等运行时字符串中去除尾部的 "h"/"H"。
        res_data = data.get("resource", {})
        if isinstance(res_data, dict):
            runtime_str = str(res_data.get("runtime", "72"))
            runtime = int(runtime_str.rstrip("hH"))
            resources = FrontendResourceParams(runtime=runtime, cores=res_data.get("cores", 72))
        else:
            resources = FrontendResourceParams()

        # 7. prev_dir
        # Prefer settings["from_prev_calc"] then top-level prev_dir / prevDir.
        # 优先使用 settings["from_prev_calc"]，其次是顶层的 prev_dir / prevDir。
        prev_dir = (
            settings.get("from_prev_calc")
            or data.get("prev_dir")
            or data.get("prevDir")
        )

        # 8. vdW
        vdw_method = data.get("vdw", "None")
        vdw = FrontendVdwParams(method=FRONTEND_VDW_MAP.get(vdw_method, vdw_method))

        # 9. 偶极校正
        dipole = FrontendDipoleParams(enabled=data.get("dipole", False))

        # 10. 频率参数
        # Frequency parameters are only constructed for frequency calc types.
        # 频率参数仅在计算类型为频率计算时构建。
        frequency = None
        if backend_calc_type in ("freq", "freq_ir"):
            frequency = FrontendFrequencyParams(
                ibrion=_parse_int(settings.get("IBRION", 5)),
                potim=_parse_float(settings.get("POTIM", 0.015)),
                nfree=_parse_int(settings.get("NFREE", 2)),
                vibrate_mode=settings.get("vibrate_mode", "inherit"),
                adsorbate_formula=settings.get("adsorbate_formula"),
                adsorbate_prefer=settings.get("adsorbate_formula_prefer", "tail"),
                calc_ir=settings.get("calc_ir", False),
            )
            if "vibrate_indices" in settings and settings["vibrate_indices"]:
                try:
                    # Parse comma-separated index string to integer list.
                    # 将逗号分隔的索引字符串解析为整数列表。
                    frequency.vibrate_indices = [
                        int(x.strip())
                        for x in str(settings["vibrate_indices"]).split(",")
                    ]
                except ValueError:
                    pass

        # 11. Lobster
        lobster = None
        if backend_calc_type == "lobster":
            lobster = FrontendLobsterParams(
                lobsterin_mode=settings.get("lobsterin_mode", "template"),
                overwritedict=data.get("lobsterin") or None,
                custom_lobsterin_lines=data.get("lobsterin_custom_lines") or None,
            )

        # 12. NBO
        nbo = None
        if backend_calc_type == "nbo":
            nbo_config = settings.get("nbo_config", {})
            nbo = FrontendNBOParams(
                basis_source="ANO-RCC-MB" if settings.get("basis_source") == "default" else "custom",
                custom_basis_path=settings.get("nboBasisPath"),
                occ_1c=_parse_float(nbo_config.get("occ_1c", 1.60)),
                occ_2c=_parse_float(nbo_config.get("occ_2c", 1.85)),
                print_cube=nbo_config.get("print_cube", "F"),
                density=nbo_config.get("density", "F"),
                vis_start=_parse_int(nbo_config.get("vis_start", 0)),
                vis_end=_parse_int(nbo_config.get("vis_end", -1)),
                mesh=nbo_config.get("mesh", [0, 0, 0]) if isinstance(nbo_config.get("mesh"), list) else [0, 0, 0],
                box_int=nbo_config.get("box_int", [1, 1, 1]) if isinstance(nbo_config.get("box_int"), list) else [1, 1, 1],
                origin_fact=_parse_float(nbo_config.get("origin_fact", 0.00)),
            )

        # ── 13. MAGMOM ────────────────────────────────────────────────────
        # 前端可通过两种方式传入：
        #   a) settings["MAGMOM"] = [5.0, 5.0, 3.0]     per-atom 列表
        #   b) settings["MAGMOM"] = "5.0 5.0 3.0"       空格分隔字符串
        #   c) settings["MAGMOM"] = {"Fe": 5.0, "Co": 3.0}  per-element dict
        # The frontend may supply MAGMOM in three forms:
        #   a) List[float]        — per-atom order
        #   b) space-delimited str — per-atom order
        #   c) Dict[str, float]   — per-element mapping
        magmom = None
        raw_magmom = settings.get("MAGMOM")
        if raw_magmom is not None:
            magmom = FrontendMagmomParams(enabled=True)
            if isinstance(raw_magmom, dict):
                # per-element dict
                magmom.per_element = {k: float(v) for k, v in raw_magmom.items()}
            else:
                # per-atom 列表或字符串
                # Per-atom list or space-delimited string.
                magmom.per_atom = _parse_magmom_list(raw_magmom)

        # ── 14. DFT+U ─────────────────────────────────────────────────────
        # 前端传入格式（推荐，按元素顺序）：
        #   settings["LDAUU"] = {"Fe": 4.0, "Co": 3.0}
        #   settings["LDAUL"] = {"Fe": 2,   "Co": 2}
        #   settings["LDAUJ"] = {"Fe": 0.0, "Co": 0.0}
        # 或旧格式（空格字符串，需配合元素列表）：
        #   settings["LDAUU"] = "4.0 3.0"  +  settings["elements"] = ["Fe", "Co"]
        # Recommended frontend format (per-element dicts):
        #   settings["LDAUU"] = {"Fe": 4.0, "Co": 3.0}
        #   settings["LDAUL"] = {"Fe": 2,   "Co": 2}
        #   settings["LDAUJ"] = {"Fe": 0.0, "Co": 0.0}
        # Legacy format (space-separated strings + elements list):
        #   settings["LDAUU"] = "4.0 3.0"  +  settings["elements"] = ["Fe", "Co"]
        dft_u = None
        raw_ldauu = settings.get("LDAUU")
        raw_ldaul = settings.get("LDAUL")
        raw_ldauj = settings.get("LDAUJ")

        if raw_ldauu is not None or raw_ldaul is not None:
            dft_u = FrontendDFTPlusUParams(enabled=True)

            if isinstance(raw_ldauu, dict):
                # ── 推荐格式：直接是 per-element dict ──────────────────────
                # Recommended format: LDAUU is already a per-element dict.
                # 推荐格式：LDAUU 已经是 per-element 字典。
                elements = list(raw_ldauu.keys())
                ldauu_dict = {k: float(v) for k, v in raw_ldauu.items()}
                ldaul_dict = (
                    {k: int(v) for k, v in raw_ldaul.items()}
                    if isinstance(raw_ldaul, dict)
                    else {e: 0 for e in elements}
                )
                ldauj_dict = (
                    {k: float(v) for k, v in raw_ldauj.items()}
                    if isinstance(raw_ldauj, dict)
                    else {e: 0.0 for e in elements}
                )
                dft_u.values = {
                    elem: {
                        "LDAUU": ldauu_dict.get(elem, 0.0),
                        "LDAUL": ldaul_dict.get(elem, 0),
                        "LDAUJ": ldauj_dict.get(elem, 0.0),
                    }
                    for elem in elements
                }

            else:
                # ── 兼容格式：空格字符串 + elements 列表 ──────────────────
                # Legacy format: space-separated string values require an explicit
                # elements list to pair with U values.
                # 兼容格式：空格分隔的字符串值需要显式 elements 列表与 U 值配对。
                elements = settings.get("elements", [])
                ldauu_vals = _parse_number_list(raw_ldauu)
                ldaul_vals = _parse_number_list(raw_ldaul)
                ldauj_vals = _parse_number_list(raw_ldauj)

                if elements and len(elements) == len(ldauu_vals):
                    dft_u.values = {
                        elem: {
                            "LDAUU": float(ldauu_vals[i]) if i < len(ldauu_vals) else 0.0,
                            "LDAUL": int(ldaul_vals[i])   if i < len(ldaul_vals)  else 0,
                            "LDAUJ": float(ldauj_vals[i]) if i < len(ldauj_vals)  else 0.0,
                        }
                        for i, elem in enumerate(elements)
                    }
                else:
                    # 无法匹配元素，禁用 DFT+U 并警告
                    # Cannot pair elements with U values; disable DFT+U and warn.
                    logger.warning(
                        "DFT+U: LDAUU 为字符串格式但未提供 settings['elements'] 列表，"
                        "或元素数量与 U 值数量不匹配，DFT+U 将被忽略。"
                        "推荐使用 dict 格式: settings['LDAUU'] = {'Fe': 4.0, 'Co': 3.0}"
                    )
                    dft_u = None

        return VaspWorkflowParams(
            calc_type=backend_calc_type,
            structure=structure,
            functional=functional,
            kpoints_density=kpoints.density or 50.0,
            prev_dir=prev_dir,
            precision=precision,
            kpoints=kpoints,
            resources=resources,
            vdw=vdw,
            dipole=dipole,
            frequency=frequency,
            lobster=lobster,
            nbo=nbo,
            magmom=magmom,
            dft_u=dft_u,
            custom_incar=custom_incar if custom_incar else None,
        )


# ============================================================================
# 辅助解析函数
# ============================================================================

def _parse_int(value: Any) -> Optional[int]:
    """Safely coerce *value* to ``int``, returning ``None`` on failure.

    Recognises ``None``, empty string, and the em-dash placeholder ``"—"`` as
    absent values.

    安全地将 *value* 转换为 ``int``，失败时返回 ``None``。

    将 ``None``、空字符串以及占位符 ``"—"`` 视为缺失值。
    """
    if value is None or value == "" or value == "—":
        return None
    try:
        return int(str(value))
    except (ValueError, TypeError):
        return None


def _parse_float(value: Any) -> Optional[float]:
    """Safely coerce *value* to ``float``, returning ``None`` on failure.

    Recognises ``None``, empty string, and the em-dash placeholder ``"—"`` as
    absent values.

    安全地将 *value* 转换为 ``float``，失败时返回 ``None``。

    将 ``None``、空字符串以及占位符 ``"—"`` 视为缺失值。
    """
    if value is None or value == "" or value == "—":
        return None
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return None


def _parse_number(value: Any) -> Union[int, float, str]:
    """Safely parse *value* as a number, preferring ``int`` over ``float``.

    Returns the original value unchanged when numeric parsing fails.

    安全地将 *value* 解析为数值，优先返回 ``int``（整型优先）。

    数值解析失败时原样返回原始值。
    """
    if value is None or value == "" or value == "—":
        return value
    try:
        f = float(str(value))
        # Return int when the float is a whole number.
        # 浮点数为整数时返回 int。
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return value


def _parse_number_list(value: Any) -> List[float]:
    """Parse a string or list into a ``List[float]`` for DFT+U legacy format.

    Args:
        value: A ``List``, a space-separated string, or ``None``.

    Returns:
        Parsed list of floats, or an empty list when *value* is ``None``.

    将字符串或列表解析为 ``List[float]``，用于 DFT+U 兼容格式。

    参数：
        value: ``List``、空格分隔的字符串或 ``None``。

    返回：
        解析后的浮点数列表；*value* 为 ``None`` 时返回空列表。
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        return [float(x) for x in value.split()]
    return []


def _parse_magmom_list(value: Any) -> List[float]:
    """Parse a MAGMOM value into a per-site ``List[float]``.

    Handles three input forms:
    - ``List[float]``        — returned directly after float conversion.
    - Space-separated string — split and converted; ``"N*val"`` shorthand is
      expanded (e.g. ``"3*5.0"`` → ``[5.0, 5.0, 5.0]``).
    - Anything else          — returns an empty list.

    将 MAGMOM 值解析为 per-site ``List[float]``。

    支持三种输入形式：
    - ``List[float]``       — float 转换后直接返回。
    - 空格分隔的字符串     — 分割并转换；支持 ``"N*val"`` 简写展开
      （如 ``"3*5.0"`` → ``[5.0, 5.0, 5.0]``）。
    - 其他类型             — 返回空列表。
    """
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        result = []
        for part in value.split():
            if "*" in part:
                # Expand "count*value" shorthand notation.
                # 展开 "count*value" 简写记法。
                count, val = part.split("*")
                result.extend([float(val)] * int(count))
            else:
                result.append(float(part))
        return result
    return []


# ============================================================================
# 主入口 — generate_inputs()
# ============================================================================

def _normalise_dft_u(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalise user-facing DFT+U dict to the internal per-element format.

    Accepts short keys (``"U"``, ``"l"``, ``"J"``) or full VASP tag names
    (``"LDAUU"``, ``"LDAUL"``, ``"LDAUJ"``).  A plain scalar is treated as the
    U value with l=2 (d-shell) and J=0.

    Examples::

        {"Fe": {"U": 4.0, "l": 2, "J": 0.0}}   # short keys
        {"Fe": {"LDAUU": 4.0, "LDAUL": 2}}       # VASP tag names
        {"Fe": 4.0}                               # scalar shorthand
    """
    result: Dict[str, Dict[str, Any]] = {}
    for elem, spec in raw.items():
        if isinstance(spec, (int, float)):
            result[elem] = {"LDAUU": float(spec), "LDAUL": 2, "LDAUJ": 0.0}
        else:
            u = float(spec.get("U", spec.get("LDAUU", 0.0)))
            l = int(spec.get("l", spec.get("L", spec.get("LDAUL", 2))))
            j = float(spec.get("J", spec.get("LDAUJ", 0.0)))
            result[elem] = {"LDAUU": u, "LDAUL": l, "LDAUJ": j}
    return result


def generate_inputs(
    calc_type: str,
    structure: Union[str, Path] = "POSCAR",
    functional: str = "PBE",
    kpoints_density: float = 50.0,
    output_dir: Optional[Union[str, Path]] = None,
    prev_dir: Optional[Union[str, Path]] = None,
    *,
    incar: Optional[Dict[str, Any]] = None,
    magmom: Optional[Union[List[float], Dict[str, float]]] = None,
    dft_u: Optional[Dict[str, Any]] = None,
    cohp_generator: Optional[Union[str, List[str]]] = None,
    lobsterin: Optional[Dict[str, Any]] = None,
    nbo_config: Optional[Dict[str, Any]] = None,
    walltime: Optional[str] = None,
    ncores: Optional[int] = None,
    dry_run: bool = False,
) -> Union[str, Dict[str, Any]]:
    """Generate VASP input files for one calculation — the single user-facing entry point.
    为单次计算生成 VASP 输入文件——唯一面向用户的入口函数。

    All internal complexity (adapter layer, dataclasses, engine dispatch) is
    fully encapsulated.  Callers only need to supply their scientific parameters;
    no knowledge of the internal pipeline is required.
    所有内部复杂性（适配层、数据类、引擎调度）均被完全封装。
    调用方只需提供科学参数，无需了解内部流程。

    Args:
        calc_type (str): Calculation type string.  Controls which VASP input
            template and defaults are used.  Accepted values:
            ``"bulk_relax"`` — bulk structure relaxation (ISIF=3);
            ``"slab_relax"`` — surface slab relaxation (ISIF=2);
            ``"static_sp"``  — single-point energy;
            ``"static_dos"`` — single-point + projected DOS (writes CHGCAR);
            ``"static_charge"`` — single-point + full charge density;
            ``"static_elf"`` — single-point + electron localisation function;
            ``"freq"`` — vibrational frequencies (finite differences, IBRION=5);
            ``"freq_ir"`` — vibrational frequencies + IR intensities (DFPT, IBRION=7);
            ``"lobster"``    — COHP bonding analysis (writes WAVECAR);
            ``"nmr_cs"`` / ``"nmr_efg"`` — NMR chemical shift / EFG;
            ``"nbo"``        — Natural Bond Orbital analysis;
            ``"neb"``        — Nudged Elastic Band transition-state search;
            ``"dimer"``      — Dimer method saddle-point search;
            ``"md_nvt"`` / ``"md_npt"`` — molecular dynamics (NVT / NPT).
        calc_type (str): 计算类型字符串，决定使用哪套 VASP 输入模板和默认参数。
            ``"bulk_relax"`` — 体相结构弛豫（ISIF=3）；
            ``"slab_relax"`` — 表面 slab 弛豫（ISIF=2）；
            ``"static_sp"``  — 单点能；
            ``"static_dos"`` — 单点 + 投影态密度（输出 CHGCAR）；
            ``"static_charge"`` — 单点 + 全电荷密度；
            ``"static_elf"`` — 单点 + 电子局域函数；
            ``"freq"`` — 振动频率（有限差分，IBRION=5）；
            ``"freq_ir"`` — 振动频率 + 红外强度（DFPT，IBRION=7）；
            ``"lobster"``    — COHP 化学键分析（输出 WAVECAR）；
            ``"nmr_cs"`` / ``"nmr_efg"`` — NMR 化学位移 / 电场梯度；
            ``"nbo"``        — 自然键轨道分析；
            ``"neb"``        — NEB 过渡态搜索；
            ``"dimer"``      — Dimer 鞍点搜索；
            ``"md_nvt"`` / ``"md_npt"`` — 分子动力学（NVT / NPT）。

        structure (str | Path): Path to the input structure file (POSCAR, CIF,
            CONTCAR) or to a directory that contains a CONTCAR.
            Default: ``"POSCAR"`` (file in the current working directory).
        structure (str | Path): 输入结构文件（POSCAR、CIF、CONTCAR）的路径，
            或包含 CONTCAR 的目录路径。默认值：``"POSCAR"``（当前工作目录下的文件）。

        functional (str): Exchange-correlation functional label.  The functional
            controls both the pymatgen ``InputSet`` selection and the INCAR patches
            applied on top of the base settings.  Accepted values:
            ``"PBE"`` (default) — standard GGA;
            ``"RPBE"`` — revised PBE (better adsorption energies);
            ``"BEEF"`` — BEEF-vdW non-local van-der-Waals (requires
              vdw_kernel.bindat and a BEEF-patched VASP binary);
            ``"HSE"``  — HSE06 hybrid (accurate band gaps, expensive);
            ``"SCAN"`` — SCAN meta-GGA;
            ``"LDA"``  — local density approximation.
        functional (str): 交换关联泛函标签，同时控制 pymatgen ``InputSet`` 的选择
            和叠加在基础设置之上的 INCAR 补丁。接受的值：
            ``"PBE"``（默认）— 标准 GGA；
            ``"RPBE"`` — 修正 PBE（吸附能更准确）；
            ``"BEEF"`` — BEEF-vdW 非局域范德华（需要 vdw_kernel.bindat 和
              BEEF 补丁版 VASP 二进制）；
            ``"HSE"``  — HSE06 杂化泛函（能隙准确，计算代价高）；
            ``"SCAN"`` — SCAN meta-GGA；
            ``"LDA"``  — 局域密度近似。

        kpoints_density (float): Reciprocal-space k-point sampling density in
            units of Å⁻¹ (points per reciprocal-lattice length).  The actual
            k-mesh is generated automatically by pymatgen using this density.
            Default: ``50.0``.  Typical values: 50 for bulk, 25 for surfaces.
        kpoints_density (float): 倒空间 K 点采样密度（单位 Å⁻¹，即每倒格矢长度
            的网格点数）。pymatgen 据此自动生成 K 网格。默认值：``50.0``。
            典型值：体相 50，表面 25。

        output_dir (str | Path | None): Directory where the generated VASP input
            files (INCAR, KPOINTS, POSCAR, POTCAR, lobsterin, …) are written.
            When ``None`` the engine auto-generates a directory name under the
            current working directory based on the ``calc_type``.
        output_dir (str | Path | None): 生成的 VASP 输入文件（INCAR、KPOINTS、
            POSCAR、POTCAR、lobsterin 等）写入的目标目录。为 ``None`` 时，引擎
            根据 ``calc_type`` 在当前工作目录下自动生成目录名。

        prev_dir (str | Path | None): Directory of a preceding calculation.
            Enables three automatic behaviours:

            1. **Structure extraction** — when no ``structure`` file exists at
               the given path (or the default ``"POSCAR"`` is absent), the
               engine reads ``prev_dir/CONTCAR`` (if non-empty) or falls back
               to ``prev_dir/POSCAR``.

            2. **INCAR inheritance** — settings from ``prev_dir/INCAR`` become
               the base INCAR; workflow type-defaults are **not** applied on
               top so ENCUT, EDIFF, EDIFFG, … carry over automatically.
               The ``functional`` patch and any ``incar={}`` overrides are
               still applied above the inherited values.

            3. **WAVECAR/CHGCAR copy** — if ``prev_dir`` contains a non-empty
               CHGCAR the file is copied into the output directory and
               ``ICHARG=1`` is set automatically; if only a WAVECAR exists,
               it is copied and ``ISTART=1`` is set.  Both tags are
               overridable via ``incar={"ICHARG": X}``.

            Required (or auto-detected) for: ``"static_dos"``,
            ``"static_charge"``, ``"static_elf"``, ``"lobster"``, ``"nbo"``,
            ``"dimer"``.
            Example: ``prev_dir="./01-slab_relax/Fe110"``.
        prev_dir (str | Path | None): 前序计算目录。启用三项自动行为：

            1. **结构提取** — 若指定的 ``structure`` 路径（或默认 ``"POSCAR"``）
               不存在，引擎从 ``prev_dir/CONTCAR``（非空优先）或
               ``prev_dir/POSCAR`` 中读取结构。

            2. **INCAR 继承** — ``prev_dir/INCAR`` 的内容成为基础 INCAR；
               计算类型工作流默认值**不再**叠加覆盖，ENCUT、EDIFF、EDIFFG
               等参数自动延续。泛函补丁和 ``incar={}`` 覆盖项仍应用于继承值之上。

            3. **WAVECAR/CHGCAR 复制** — 若 ``prev_dir`` 包含非空 CHGCAR，
               将其复制到输出目录并自动设置 ``ICHARG=1``；若仅存在 WAVECAR，
               则复制并设置 ``ISTART=1``。两者均可通过 ``incar={"ICHARG": X}``
               覆盖。

            以下计算类型需要或可自动检测：``"static_dos"``、``"static_charge"``、
            ``"static_elf"``、``"lobster"``、``"nbo"``、``"dimer"``。
            示例：``prev_dir="./01-slab_relax/Fe110"``。

        incar (dict | None): **The single channel for all INCAR overrides.**
            Pass any VASP INCAR tag as a plain ``{"TAG": value}`` dict.
            These values are merged on top of *all* other settings
            (calc-type defaults, functional patches, DFT+U, MAGMOM) and
            therefore always win.  Any standard VASP tag is valid — there is
            no whitelist.  Default: ``None`` (use calc-type defaults only).

            Common examples::

                incar={"EDIFFG": -0.01}          # tighter force convergence
                incar={"EDIFF":  1e-7}            # tighter electronic convergence
                incar={"ENCUT":  600}             # raise plane-wave cutoff (eV)
                incar={"LREAL":  False}           # reciprocal-space projectors
                incar={"NPAR": 4, "KPAR": 2}     # parallelisation flags
                incar={"ISMEAR": 0, "SIGMA": 0.05}   # Gaussian smearing
                incar={"NSW": 300, "POTIM": 0.3}     # relax step count / size
                incar={"LORBIT": 11, "NEDOS": 3001}  # projected DOS density

        incar (dict | None): **所有 INCAR 覆盖项的唯一通道。**
            以 ``{"标记": 值}`` 字典的形式传入任意 VASP INCAR 标记。
            这些值将叠加在*所有*其他设置之上（计算类型默认值、泛函补丁、
            DFT+U、MAGMOM），因此具有最高优先级。支持所有标准 VASP 标记，
            无白名单限制。默认值：``None``（仅使用计算类型默认值）。

        magmom (list[float] | dict[str, float] | None): Initial magnetic moments
            for ISPIN=2 calculations.  Two formats are accepted:
            - ``List[float]``: per-site moments in the same order as atoms in
              the structure, e.g. ``[5.0, 5.0, 3.0, 3.0]``.
            - ``Dict[str, float]``: per-element moments; pymatgen expands the
              dict against the structure's site order automatically,
              e.g. ``{"Fe": 5.0, "Co": 3.0, "O": 0.0}``.
            Default: ``None`` (no MAGMOM tag written; pymatgen uses its own
            default if the functional requires spin polarisation).
        magmom (list[float] | dict[str, float] | None): ISPIN=2 计算的初始磁矩。
            支持两种格式：
            - ``List[float]``：按结构中原子顺序排列的 per-site 磁矩列表，
              如 ``[5.0, 5.0, 3.0, 3.0]``。
            - ``Dict[str, float]``：per-element 磁矩字典，pymatgen 会自动按
              结构位点顺序展开，如 ``{"Fe": 5.0, "Co": 3.0, "O": 0.0}``。
            默认值：``None``（不写入 MAGMOM；若泛函要求自旋极化，pymatgen
            使用自身默认值）。

        dft_u (dict | None): DFT+U (Hubbard U) parameters per element.
            ``LDAUTYPE=2`` (Dudarev simplified, U_eff = U − J) is added
            automatically.  Three equivalent input formats are accepted::

                # Recommended — short keys
                {"Fe": {"U": 4.0, "l": 2, "J": 0.0},
                 "Co": {"U": 3.0, "l": 2}}

                # VASP tag names
                {"Fe": {"LDAUU": 4.0, "LDAUL": 2, "LDAUJ": 0.0}}

                # Scalar shorthand — U value only; l=2 (d-orbital), J=0 assumed
                {"Fe": 4.0, "Co": 3.0}

            Key meanings:
            ``"U"`` / ``"LDAUU"`` — Coulomb U in eV;
            ``"l"`` / ``"LDAUL"`` — angular momentum of correlated shell
              (0=s, 1=p, 2=d, 3=f);
            ``"J"`` / ``"LDAUJ"`` — exchange J in eV (usually 0 for Dudarev).
            Default: ``None`` (DFT+U disabled).
        dft_u (dict | None): 各元素的 DFT+U（Hubbard U）参数。
            ``LDAUTYPE=2``（Dudarev 简化，U_eff = U − J）将被自动添加。
            支持三种等价格式（见英文示例）。
            键的含义：
            ``"U"`` / ``"LDAUU"`` — Coulomb U（eV）；
            ``"l"`` / ``"LDAUL"`` — 关联壳层角量子数（0=s,1=p,2=d,3=f）；
            ``"J"`` / ``"LDAUJ"`` — 交换 J（eV），Dudarev 方案通常为 0。
            默认值：``None``（禁用 DFT+U）。

        cohp_generator (str | list[str] | None): COHP bond-length range
            specification(s) passed to the lobsterin file.  Only used when
            ``calc_type="lobster"``.
            - ``str``: a single range entry written to ``cohpGenerator`` in
              the lobsterin overwrite dict,
              e.g. ``"from 1.5 to 1.9 orbitalwise"``.
            - ``List[str]``: multiple entries.  The **first** entry replaces
              the pymatgen-generated ``cohpGenerator`` default; each subsequent
              entry is appended as a raw ``cohpGenerator …`` line at the end
              of the lobsterin file.
            Default: ``None`` (pymatgen generates a default cohpGenerator
            from the structure's shortest bond lengths).
        cohp_generator (str | list[str] | None): 写入 lobsterin 文件的 COHP
            键长范围规格。仅在 ``calc_type="lobster"`` 时生效。
            - ``str``：单条范围，写入 lobsterin 的 ``cohpGenerator`` 覆盖字典，
              如 ``"from 1.5 to 1.9 orbitalwise"``。
            - ``List[str]``：多条范围。**第一条**替换 pymatgen 生成的默认
              ``cohpGenerator``；其余各条作为 ``cohpGenerator …`` 原始行
              追加到 lobsterin 文件末尾。
            默认值：``None``（由 pymatgen 根据结构中最短键长自动生成）。

        lobsterin (dict | None): Additional key-value pairs written directly
            to the lobsterin overwrite dict, complementing ``cohp_generator``.
            Only used when ``calc_type="lobster"``.
            Example: ``{"COHPstartEnergy": -20.0, "COHPendEnergy": 20.0}``.
            Default: ``None``.
        lobsterin (dict | None): 直接写入 lobsterin 覆盖字典的额外键值对，
            与 ``cohp_generator`` 配合使用。仅在 ``calc_type="lobster"`` 时生效。
            示例：``{"COHPstartEnergy": -20.0, "COHPendEnergy": 20.0}``。
            默认值：``None``。

        nbo_config (dict | None): NBO analysis configuration passed directly to
            the NBO input-set builder.  Only used when ``calc_type="nbo"``.
            Supported keys:

            ``"occ_1c"`` (bool) — enable one-centre NBO occupancy analysis;
            ``"occ_2c"`` (bool) — enable two-centre NBO occupancy analysis;
            ``"basis_source"`` (str) — path or identifier for the NBO basis
              set, e.g. ``"ANO-RCC-MB"`` or a path to a custom basis file;
            ``"nbo_keywords"`` (list[str]) — additional raw NBO keyword strings
              appended verbatim to the NBO input file.

            Example::

                nbo_config={
                    "occ_1c":       True,                 # enable 1-centre occupancy / 启用单中心占据
                    "occ_2c":       True,                 # enable 2-centre occupancy / 启用双中心占据
                    "basis_source": "ANO-RCC-MB",         # basis set identifier / 基组标识符
                    "nbo_keywords": ["$NBO BNDIDX $END"], # extra NBO keywords / 额外 NBO 关键字
                }

            Default: ``None`` (NBO input set built with defaults only).
        nbo_config (dict | None): NBO 分析配置，直接传递给 NBO 输入集构造函数。
            仅在 ``calc_type="nbo"`` 时生效。支持的键：

            ``"occ_1c"`` (bool) — 启用单中心 NBO 占据分析；
            ``"occ_2c"`` (bool) — 启用双中心 NBO 占据分析；
            ``"basis_source"`` (str) — NBO 基组路径或标识符，
              如 ``"ANO-RCC-MB"`` 或自定义基组文件路径；
            ``"nbo_keywords"`` (list[str]) — 逐字追加到 NBO 输入文件末尾的
              额外原始 NBO 关键字字符串列表。

            默认值：``None``（使用默认参数构建 NBO 输入集）。

        walltime (str | None): Wall-clock time limit for the PBS submission
            script in ``"HH:MM:SS"`` format, e.g. ``"48:00:00"``.
            ``None`` → automatically chosen per calc_type
            (e.g. ``"124:00:00"`` for relaxations, ``"48:00:00"`` for statics).
        walltime (str | None): PBS 提交脚本的墙钟时间限制，格式为 ``"HH:MM:SS"``，
            如 ``"48:00:00"``。``None`` → 按计算类型自动选择
            （如弛豫默认 ``"124:00:00"``，静态计算默认 ``"48:00:00"``）。

        ncores (int | None): Number of CPU cores for the PBS submission script.
            ``None`` → automatically chosen per calc_type (default: 72).
        ncores (int | None): PBS 提交脚本的 CPU 核数。``None`` → 按计算类型自动
            选择（默认值：72）。

        dry_run (bool): When ``True``, return a configuration preview dict
            **without writing any files**.  Safe to call with a non-existent
            structure path — useful for inspecting or testing parameter
            combinations before a real run.  The returned dict contains:
            ``"incar"`` — merged INCAR key-value pairs (``dict``);
            ``"calc_type"`` — resolved CalcType enum value string;
            ``"functional"`` — uppercased functional string;
            ``"kpoints_density"`` — k-point density (``float``);
            ``"lobsterin"`` — lobsterin overwrite dict (Lobster only, if set);
            ``"lobsterin_custom_lines"`` — extra raw lobsterin lines (if set).
            Default: ``False``.
        dry_run (bool): 为 ``True`` 时，**不写入任何文件**，直接返回配置预览字典。
            允许传入不存在的结构路径——适用于在正式运行前检查或测试参数组合。
            返回字典包含：
            ``"incar"`` — 合并后的 INCAR 键值对（``dict``）；
            ``"calc_type"`` — 解析后的 CalcType 枚举值字符串；
            ``"functional"`` — 大写泛函字符串；
            ``"kpoints_density"`` — K 点密度（``float``）；
            ``"lobsterin"`` — lobsterin 覆盖字典（仅 Lobster，若已设置）；
            ``"lobsterin_custom_lines"`` — 额外的 lobsterin 原始行（若已设置）。
            默认值：``False``。

    Returns:
        str: ``dry_run=False`` (default) — absolute path to the output
        directory containing the generated VASP input files.
        str: ``dry_run=False``（默认）— 包含生成的 VASP 输入文件的输出目录绝对路径。

        dict: ``dry_run=True`` — configuration preview dict (see ``dry_run``
        parameter above for the key listing).
        dict: ``dry_run=True`` — 配置预览字典（键列表见上方 ``dry_run`` 参数说明）。

    Examples::

        from flow.api import generate_inputs

        # 1. Standard PBE bulk relaxation — minimal call
        #    标准 PBE 体相弛豫——最简调用
        out = generate_inputs("bulk_relax", "POSCAR")

        # 2. Custom INCAR settings — any VASP tag via incar=
        #    自定义 INCAR 设置——通过 incar= 传入任意 VASP 标记
        out = generate_inputs(
            "slab_relax", "POSCAR",
            incar={"EDIFFG": -0.01, "ENCUT": 600, "NPAR": 4, "LREAL": False},
        )

        # 3. BEEF functional with DFT+U (Dudarev, short-key format)
        #    BEEF 泛函 + DFT+U（Dudarev，短键格式）
        out = generate_inputs(
            "bulk_relax", "Fe_bulk/POSCAR", functional="BEEF",
            dft_u={"Fe": {"U": 4.0, "l": 2}, "Co": {"U": 3.0, "l": 2}},
            magmom={"Fe": 5.0, "Co": 3.0},
        )

        # 4. Lobster with multiple element-specific COHP ranges
        #    带多条元素专属 COHP 范围的 Lobster 计算
        out = generate_inputs(
            "lobster", "POSCAR", prev_dir="./relax",
            cohp_generator=[
                "from 1.5 to 2.2 type Pt type C orbitalwise",
                "from 1.5 to 2.1 type Pt type O orbitalwise",
            ],
            lobsterin={"COHPstartEnergy": -20.0, "COHPendEnergy": 20.0},
        )

        # 5. Dry run — inspect merged INCAR without writing files
        #    Dry run——在不写文件的情况下检查合并后的 INCAR
        preview = generate_inputs("bulk_relax", "POSCAR", dry_run=True)
        print(preview["incar"]["IBRION"])   # → 2
        print(preview["incar"]["ENCUT"])    # → 520
    """
    # ── 0. Validate all parameters before any transformation or file I/O ───
    _validator_validate(
        calc_type=calc_type,
        structure=structure,
        functional=functional,
        kpoints_density=kpoints_density,
        output_dir=output_dir,
        prev_dir=prev_dir,
        incar=incar,
        magmom=magmom,
        dft_u=dft_u,
        walltime=walltime,
        ncores=ncores,
        dry_run=dry_run,
        # calc_type-specific params forwarded for cross-field warning checks
        lobsterin=lobsterin,
        cohp_generator=cohp_generator,
        nbo_config=nbo_config,
    )

    # ── 1. Build extra INCAR dict; DFT+U adds LDAUTYPE=2 automatically ────
    extra_incar: Dict[str, Any] = dict(incar or {})

    dft_u_params: Optional[FrontendDFTPlusUParams] = None
    if dft_u:
        dft_u_params = FrontendDFTPlusUParams(
            enabled=True,
            values=_normalise_dft_u(dft_u),
        )
        extra_incar.setdefault("LDAUTYPE", 2)

    # ── 2. MAGMOM — accept list (per-site) or dict (per-element) ──────────
    magmom_params: Optional[FrontendMagmomParams] = None
    if magmom is not None:
        if isinstance(magmom, list):
            magmom_params = FrontendMagmomParams(
                enabled=True,
                per_atom=[float(v) for v in magmom],
            )
        else:
            magmom_params = FrontendMagmomParams(
                enabled=True,
                per_element={k: float(v) for k, v in magmom.items()},
            )

    # ── 3. Lobster — build overwritedict and custom lines from cohp_generator
    lobster_params: Optional[FrontendLobsterParams] = None
    if calc_type == "lobster":
        overwrite: Dict[str, Any] = dict(lobsterin or {})
        custom_lines: Optional[List[str]] = None
        if cohp_generator is not None:
            if isinstance(cohp_generator, str):
                overwrite["cohpGenerator"] = cohp_generator
            else:
                gen_list = list(cohp_generator)
                if gen_list:
                    overwrite["cohpGenerator"] = gen_list[0]
                    if len(gen_list) > 1:
                        custom_lines = [f"cohpGenerator {g}" for g in gen_list[1:]]
        lobster_params = FrontendLobsterParams(
            overwritedict=overwrite if overwrite else None,
            custom_lobsterin_lines=custom_lines,
        )

    # ── 3b. NBO config — pass raw dict through to WorkflowConfig.nbo_config ─
    nbo_config_params: Optional[Dict[str, Any]] = dict(nbo_config) if nbo_config else None

    # ── 4. Assemble VaspWorkflowParams ─────────────────────────────────────
    params = VaspWorkflowParams(
        calc_type=calc_type,
        structure=structure,
        functional=functional,
        kpoints_density=kpoints_density,
        output_dir=output_dir,
        prev_dir=prev_dir,
        magmom=magmom_params,
        dft_u=dft_u_params,
        lobster=lobster_params,
        nbo_config=nbo_config_params,
        custom_incar=extra_incar if extra_incar else None,
    )

    # ── 5. Dry run: return preview dict without any file I/O ───────────────
    if dry_run:
        from .script_writer import ScriptWriter as _SW
        config = params.to_workflow_config()
        engine = WorkflowEngine()
        incar_dict = engine._get_incar_params(config)
        preview: Dict[str, Any] = {
            "incar":           incar_dict,
            "calc_type":       config.calc_type.value,
            "functional":      config.functional,
            "kpoints_density": config.kpoints_density,
        }
        if config.lobster_overwritedict:
            preview["lobsterin"] = dict(config.lobster_overwritedict)
        if config.lobster_custom_lines:
            preview["lobsterin_custom_lines"] = list(config.lobster_custom_lines)

        # 1. INCAR preview — printed first
        if incar_dict:
            width = max(len(k) for k in incar_dict) + 1
            print("[dry_run] INCAR preview:")
            for k in sorted(incar_dict):
                print(f"  {k:<{width}}: {incar_dict[k]}")
            print()

        # 2. Script PBS directives — printed last
        _SW().write(
            output_dir=Path(output_dir or f"calc_{calc_type}"),
            calc_type=calc_type,
            functional=functional,
            walltime=walltime,
            ncores=ncores,
            dry_run=True,
        )
        return preview

    # ── 6. Write files ─────────────────────────────────────────────────────
    out_dir_str: str = VaspAPI().run_workflow(params, generate_script=False)["output_dir"]

    # ── 7. Write PBS submission script ────────────────────────────────────
    from .script_writer import ScriptWriter as _SW
    _SW().write(
        output_dir=Path(out_dir_str),
        calc_type=calc_type,
        functional=functional,
        walltime=walltime,
        ncores=ncores,
    )

    return out_dir_str


