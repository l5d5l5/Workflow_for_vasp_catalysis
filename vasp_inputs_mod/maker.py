"""
High-level API: generate different types of VASP input files from configuration objects.

This module exposes VaspInputMaker, a dataclass-based factory that wraps all
supported VASP input-set types (bulk relax, slab, NEB, Lobster, NBO, frequency,
NMR, MD, etc.) behind a unified interface.  Each ``write_*`` method resolves
INCAR / KPOINTS settings by merging global Maker defaults with per-call overrides,
handles MAGMOM format compatibility, ensures the output directory exists, and
delegates to the appropriate ``*SetEcat`` input-set class.

高层 API：通过配置对象生成不同类型的 VASP 输入文件。

本模块暴露 VaspInputMaker，一个基于 dataclass 的工厂类，将所有受支持的
VASP 输入集类型（体相弛豫、slab、NEB、Lobster、NBO、频率、NMR、MD 等）
统一封装在同一接口后。每个 ``write_*`` 方法通过合并全局 Maker 默认值与
单次调用覆盖值来解析 INCAR / KPOINTS 设置，处理 MAGMOM 格式兼容性，
确保输出目录存在，并委托给对应的 ``*SetEcat`` 输入集类。
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
import numpy as np
from typing import Any, Dict, List, Optional, Union

from pymatgen.core import Structure
from pymatgen.io.lobster import Lobsterin

from .input_sets import (
    BulkRelaxSetEcat,
    FreqSetEcat,
    LobsterSetEcat,
    MPStaticSetEcat,
    NEBSetEcat,
    SlabSetEcat,
    DimerSetEcat,
    NBOSetEcat,
    NMRSetEcat,
    MDSetEcat,
)
import logging

from .utils import load_structure, pick_adsorbate_indices_by_formula_strict

# Module-level logger; inherits the application's logging configuration.
# 模块级日志记录器；继承应用程序的日志配置。
logger = logging.getLogger(__name__)


@dataclass
class VaspInputMaker:
    """
    Factory dataclass for generating VASP input files across calculation types.

    All ``write_*`` methods share a common parameter-resolution pipeline:
    global INCAR / KPOINTS settings are merged with per-call overrides, MAGMOM
    values are normalised for pymatgen compatibility, the output directory is
    created on demand, and the appropriate ``*SetEcat`` object writes the files.

    Attributes:
        name: Human-readable label for this maker instance.
        functional: DFT functional string (e.g. ``"PBE"``, ``"BEEF"``).
        kpoints_density: Default k-point density used by automatic KPOINTS generation.
        use_default_incar: Whether to apply the built-in INCAR defaults from the
            input-set class before applying user overrides.
        use_default_kpoints: Whether to use automatic KPOINTS generation.
        user_incar_settings: Global INCAR key-value overrides applied to every
            ``write_*`` call unless superseded by a per-call override.
        user_kpoints_settings: Global KPOINTS object; replaced entirely by a
            per-call value when one is provided.
        user_potcar_functional: POTCAR functional string passed to pymatgen
            (e.g. ``"PBE_54"``).
        extra_kwargs: Additional keyword arguments forwarded to every input-set
            constructor.

    VASP 输入文件生成工厂 dataclass，覆盖所有计算类型。

    所有 ``write_*`` 方法共享同一套参数解析流程：全局 INCAR / KPOINTS 设置
    与单次调用的覆盖值合并，MAGMOM 值被规范化以兼容 pymatgen，输出目录按需
    创建，并由对应的 ``*SetEcat`` 对象写出文件。

    Attributes:
        name: 本 Maker 实例的可读标签。
        functional: DFT 泛函字符串（如 ``"PBE"``、``"BEEF"``）。
        kpoints_density: 自动 KPOINTS 生成时使用的默认 k 点密度。
        use_default_incar: 是否在应用用户覆盖前使用输入集类内置的 INCAR 默认值。
        use_default_kpoints: 是否使用自动 KPOINTS 生成。
        user_incar_settings: 全局 INCAR 键值覆盖，应用于每次 ``write_*`` 调用，
            除非被单次调用的覆盖值取代。
        user_kpoints_settings: 全局 KPOINTS 对象；若单次调用提供了值则完全替换。
        user_potcar_functional: 传递给 pymatgen 的 POTCAR 泛函字符串
            （如 ``"PBE_54"``）。
        extra_kwargs: 转发给每个输入集构造函数的额外关键字参数。
    """

    name: str = "VaspInputMaker"
    functional: str = "PBE"
    kpoints_density: float = 50.0
    use_default_incar: bool = True
    use_default_kpoints: bool = True

    # Global baseline INCAR / KPOINTS settings shared by all write_* calls.
    # 所有 write_* 调用共享的全局基线 INCAR / KPOINTS 设置。
    user_incar_settings: Dict[str, Any] = field(default_factory=dict)
    user_kpoints_settings: Any = None

    user_potcar_functional: str = "PBE_54"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict_ecat(cls, config: Dict[str, Any], **kwargs):
        """
        Safely instantiate a VaspInputMaker from a configuration dictionary.

        Keys that correspond to declared dataclass fields are passed as direct
        constructor arguments.  All unrecognised keys are collected into
        ``extra_kwargs`` so they can be forwarded to input-set constructors
        without raising ``TypeError``.

        Args:
            config: Flat dictionary of configuration parameters.
            **kwargs: Additional key-value pairs that take precedence over
                identically named entries in ``config``.

        Returns:
            A fully initialised ``VaspInputMaker`` instance.

        从配置字典安全地实例化 VaspInputMaker。

        与声明的 dataclass 字段对应的键作为直接构造函数参数传入。所有
        无法识别的键被收集到 ``extra_kwargs`` 中，以便转发给输入集构造函数
        而不引发 ``TypeError``。

        Args:
            config: 扁平配置参数字典。
            **kwargs: 额外键值对，优先级高于 ``config`` 中同名条目。

        Returns:
            完全初始化的 ``VaspInputMaker`` 实例。
        """
        # Merge config and explicit overrides; explicit kwargs win.
        # 合并 config 与显式覆盖；显式 kwargs 优先。
        full_config = {**config, **kwargs}
        class_fields = {f.name for f in fields(cls)}
        main_params = {}
        # Pre-existing extra_kwargs from the config dict are preserved.
        # 保留 config 字典中已有的 extra_kwargs。
        extra_params = full_config.pop("extra_kwargs", {})

        for k, v in full_config.items():
            # Strip accidental whitespace from key names.
            # 去除键名中偶然混入的空白字符。
            key_clean = k.strip()
            if key_clean in class_fields:
                main_params[key_clean] = v
            else:
                # Unknown keys are routed to extra_kwargs for downstream use.
                # 未知键被路由到 extra_kwargs 供下游使用。
                extra_params[key_clean] = v

        return cls(**main_params, extra_kwargs=extra_params)

    def __post_init__(self):
        """
        Normalise mutable fields immediately after dataclass initialisation.

        Ensures ``functional`` is always uppercase, and that ``user_incar_settings``
        and ``extra_kwargs`` are plain dicts even when ``None`` or other falsy
        values were passed at construction time.

        在 dataclass 初始化后立即规范化可变字段。

        确保 ``functional`` 始终大写，并且 ``user_incar_settings`` 和
        ``extra_kwargs`` 即使在构造时传入 ``None`` 或其他假值时也是普通字典。
        """
        self.functional = (self.functional or "PBE").upper()
        self.user_incar_settings = dict(self.user_incar_settings or {})
        self.extra_kwargs = dict(self.extra_kwargs or {})

    def _apply_magmom_compat(
        self,
        structure: Optional[Structure],
        common_kwargs: Dict[str, Any],
    ) -> Optional[Structure]:
        """
        Normalise the MAGMOM INCAR value to the dict format expected by pymatgen.

        Recent pymatgen versions require MAGMOM to be a ``dict`` keyed by
        element symbol rather than a flat list or VASP-style string.  This
        method converts all three input formats (dict, string, list/tuple) to
        the required dict representation.

        When the per-atom list length matches the number of sites, the values
        are also attached to the structure as a ``"magmom"`` site property so
        that ordering information is preserved.  The averaged per-element dict
        is then stored back into ``common_kwargs["user_incar_settings"]``.

        Args:
            structure: The pymatgen ``Structure`` object to annotate.  If
                ``None``, the method returns ``None`` immediately.
            common_kwargs: The merged INCAR / KPOINTS kwargs dict built by
                :meth:`_build_common_kwargs`.  Modified in-place when a
                MAGMOM conversion is required.

        Returns:
            The (possibly modified copy of the) input structure, or ``None``
            if ``structure`` was ``None``.

        Side effects:
            May replace ``common_kwargs["user_incar_settings"]["MAGMOM"]`` with
            a per-element averaged dict.

        将 MAGMOM INCAR 值规范化为 pymatgen 所期望的字典格式。

        较新版本的 pymatgen 要求 MAGMOM 为以元素符号为键的 ``dict``，而非
        扁平列表或 VASP 风格的字符串。本方法将三种输入格式（dict、字符串、
        list/tuple）均转换为所需的字典表示。

        当逐原子列表长度与站点数量匹配时，值也会作为 ``"magmom"`` 站点属性
        附加到结构上，以保留排序信息。随后将按元素平均的字典写回
        ``common_kwargs["user_incar_settings"]``。

        Args:
            structure: 要注释的 pymatgen ``Structure`` 对象。若为 ``None``，
                方法立即返回 ``None``。
            common_kwargs: 由 :meth:`_build_common_kwargs` 构建的合并后
                INCAR / KPOINTS kwargs 字典。需要转换 MAGMOM 时会就地修改。

        Returns:
            （可能已修改副本的）输入结构，若 ``structure`` 为 ``None`` 则返回
            ``None``。

        Side effects:
            可能将 ``common_kwargs["user_incar_settings"]["MAGMOM"]`` 替换为
            按元素平均的字典。
        """
        if structure is None:
            return None

        incar = common_kwargs.get("user_incar_settings")
        if not incar or "MAGMOM" not in incar:
            # No MAGMOM specified; nothing to do.
            # 未指定 MAGMOM；无需处理。
            return structure

        magmom_val = incar.get("MAGMOM")
        mag_list: List[float] = []

        if isinstance(magmom_val, dict):
            # Already in the required format; pass through unchanged.
            # 已经是所需格式；直接传递，不做修改。
            return structure
        elif isinstance(magmom_val, str):
            # Parse VASP-style "n*val" tokens (e.g. "3*0.6 1*5.0").
            # 解析 VASP 风格的 "n*val" 令牌（如 "3*0.6 1*5.0"）。
            for token in magmom_val.split():
                if "*" in token:
                    try:
                        count, val = token.split("*", 1)
                        mag_list.extend([float(val)] * int(count))
                    except ValueError:
                        logger.warning("MAGMOM 字符串解析失败，保持原值: %s", magmom_val)
                        mag_list = []
                        break
                else:
                    mag_list.append(float(token))
        elif isinstance(magmom_val, (list, tuple)):
            # Convert sequence of numeric values directly.
            # 直接转换数值序列。
            mag_list = [float(v) for v in magmom_val]
        else:
            # Unrecognised type; leave the structure unchanged.
            # 无法识别的类型；保持结构不变。
            return structure

        if not mag_list:
            # Parsing produced an empty list; clear MAGMOM to an empty dict.
            # 解析产生空列表；将 MAGMOM 清空为空字典。
            incar["MAGMOM"] = {}
            return structure

        if len(mag_list) == len(structure):
            # Attach per-atom values as a site property when lengths agree.
            # 当长度一致时，将逐原子值作为站点属性附加。
            try:
                structure = structure.copy()
                structure.add_site_property("magmom", mag_list)
            except Exception as exc:
                logger.warning("为结构附加每原子 MAGMOM 失败，将按元素平均: %s", exc)

        # Build per-element averaged dict from the flat mag_list.
        # 从扁平 mag_list 构建按元素平均的字典。
        per_elem: Dict[str, List[float]] = {}
        for idx, site in enumerate(structure):
            # Use the last known value when the list is shorter than the structure.
            # 当列表比结构短时使用最后一个已知值。
            val = mag_list[idx] if idx < len(mag_list) else mag_list[-1]
            per_elem.setdefault(site.species_string, []).append(val)

        incar["MAGMOM"] = {k: sum(v) / len(v) for k, v in per_elem.items()}
        return structure

    def _ensure_dir(self, output_dir: Union[str, Path]) -> Path:
        """
        Resolve ``output_dir`` to an absolute path and create it if absent.

        Args:
            output_dir: Target directory as a string or Path.

        Returns:
            The resolved, existing ``Path`` object.

        将 ``output_dir`` 解析为绝对路径，如不存在则创建。

        Args:
            output_dir: 字符串或 Path 形式的目标目录。

        Returns:
            已解析且存在的 ``Path`` 对象。
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _build_common_kwargs(
        self,
        local_incar: Optional[Dict[str, Any]] = None,
        local_kpoints: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Build the merged keyword-argument dict shared by all input-set constructors.

        INCAR merge strategy: the global ``user_incar_settings`` dict is copied
        first; then any per-call ``local_incar`` entries overwrite it key-by-key.

        KPOINTS merge strategy: ``local_kpoints`` fully replaces the global
        ``user_kpoints_settings`` when it is not ``None``; otherwise the global
        value is used as-is.

        Note:
            ``functional`` is intentionally *not* included here.  Each
            ``write_*`` method passes it explicitly to avoid conflicts when
            inheriting settings from a ``prev_calc`` directory.

        Args:
            local_incar: Per-call INCAR overrides (higher priority than global).
            local_kpoints: Per-call KPOINTS object (replaces global when given).

        Returns:
            A dict containing ``use_default_incar``, ``user_incar_settings``,
            ``user_kpoints_settings``, ``user_potcar_functional``, and any
            entries from ``extra_kwargs``.

        构建所有输入集构造函数共享的合并关键字参数字典。

        INCAR 合并策略：先复制全局 ``user_incar_settings`` 字典，然后将
        单次调用的 ``local_incar`` 条目逐键覆盖。

        KPOINTS 合并策略：当 ``local_kpoints`` 不为 ``None`` 时完全替换全局
        ``user_kpoints_settings``；否则按原样使用全局值。

        Note:
            ``functional`` 有意 *不* 包含在此处。每个 ``write_*`` 方法显式传递
            它，以避免从 ``prev_calc`` 目录继承设置时产生冲突。

        Args:
            local_incar: 单次调用的 INCAR 覆盖（优先级高于全局）。
            local_kpoints: 单次调用的 KPOINTS 对象（提供时替换全局值）。

        Returns:
            包含 ``use_default_incar``、``user_incar_settings``、
            ``user_kpoints_settings``、``user_potcar_functional`` 以及
            ``extra_kwargs`` 中所有条目的字典。
        """
        # INCAR merge: dict update, local overrides global.
        # INCAR 合并策略：字典 Update，局部覆盖全局。
        merged_incar = self.user_incar_settings.copy()
        if local_incar:
            merged_incar.update(local_incar)

        # KPOINTS merge: full replacement, local takes priority when present.
        # KPOINTS 合并策略：直接替换，局部存在则完全无视全局。
        merged_kpoints = local_kpoints if local_kpoints is not None else self.user_kpoints_settings

        common_kwargs = dict(self.extra_kwargs or {})
        common_kwargs.update(
            {
                "use_default_incar": self.use_default_incar,
                "user_incar_settings": merged_incar,
                "user_kpoints_settings": merged_kpoints,
                "user_potcar_functional": self.user_potcar_functional,
            }
        )
        return common_kwargs

    # INCAR settings interface for concrete calculation types.
    # 各具体计算类型的 INCAR 设置接口。
    def write_bulk(
        self,
        structure: Union[str, Structure, Path],
        output_dir: Union[str, Path],
        is_metal: bool = False,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
    ) -> str:
        """
        Write VASP input files for a bulk structure relaxation.

        Args:
            structure: Input structure as a file path, directory path, or
                pymatgen ``Structure`` object.
            output_dir: Directory where the input files will be written.
            is_metal: When ``True``, applies metallic smearing defaults
                (e.g. Methfessel-Paxton) instead of the semiconductor defaults.
            user_incar_settings: Per-call INCAR overrides merged on top of the
                global settings.
            user_kpoints_settings: Per-call KPOINTS object; replaces the global
                setting when provided.

        Returns:
            Absolute path string of the output directory.

        写出体相结构弛豫的 VASP 输入文件。

        Args:
            structure: 输入结构，可为文件路径、目录路径或 pymatgen
                ``Structure`` 对象。
            output_dir: 输入文件写入的目录。
            is_metal: 为 ``True`` 时应用金属展宽默认值（如 Methfessel-Paxton），
                而非半导体默认值。
            user_incar_settings: 单次调用的 INCAR 覆盖，叠加于全局设置之上。
            user_kpoints_settings: 单次调用的 KPOINTS 对象；提供时替换全局设置。

        Returns:
            输出目录的绝对路径字符串。
        """
        output_dir = self._ensure_dir(output_dir)
        struct_obj = load_structure(structure)
        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        struct_obj = self._apply_magmom_compat(struct_obj, common_kwargs) or struct_obj

        input_obj = BulkRelaxSetEcat(
            functional=self.functional,
            structure=struct_obj,
            kpoints_density=self.kpoints_density,
            use_default_kpoints=self.use_default_kpoints,
            is_metal=is_metal,
            **common_kwargs,
        )
        input_obj.write_input(output_dir)
        return str(output_dir)

    def write_slab(
        self,
        structure: Union[str, Structure, Path],
        output_dir: Union[str, Path],
        auto_dipole: bool = True,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
    ) -> str:
        """
        Write VASP input files for a slab (surface) relaxation.

        Args:
            structure: Input slab structure as a file path, directory path,
                or pymatgen ``Structure`` object.
            output_dir: Directory where the input files will be written.
            auto_dipole: When ``True``, automatically sets the dipole-correction
                tags (``LDIPOL``, ``IDIPOL``) based on the slab geometry.
            user_incar_settings: Per-call INCAR overrides.
            user_kpoints_settings: Per-call KPOINTS object.

        Returns:
            Absolute path string of the output directory.

        写出 slab（表面）弛豫的 VASP 输入文件。

        Args:
            structure: 输入 slab 结构，可为文件路径、目录路径或 pymatgen
                ``Structure`` 对象。
            output_dir: 输入文件写入的目录。
            auto_dipole: 为 ``True`` 时根据 slab 几何形状自动设置偶极修正标签
                （``LDIPOL``、``IDIPOL``）。
            user_incar_settings: 单次调用的 INCAR 覆盖。
            user_kpoints_settings: 单次调用的 KPOINTS 对象。

        Returns:
            输出目录的绝对路径字符串。
        """
        output_dir = self._ensure_dir(output_dir)
        struct_obj = load_structure(structure)
        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        struct_obj = self._apply_magmom_compat(struct_obj, common_kwargs) or struct_obj

        input_obj = SlabSetEcat(
            functional=self.functional,
            structure=struct_obj,
            kpoints_density=self.kpoints_density,
            use_default_kpoints=self.use_default_kpoints,
            auto_dipole=auto_dipole,
            **common_kwargs,
        )
        input_obj.write_input(output_dir)
        return str(output_dir)

    def write_noscf(
        self,
        output_dir: Union[str, Path],
        structure: Union[str, Structure, Path, None] = None,
        prev_dir: Optional[Union[str, Path]] = None,
        number_of_docs: Optional[int] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
    ) -> str:
        """
        Write VASP input files for a non-self-consistent (static) calculation.

        When ``prev_dir`` is given, the functional and base INCAR are inherited
        from the previous calculation via ``MPStaticSetEcat.from_prev_calc_ecat``.
        Otherwise, a fresh input set is constructed from ``structure`` using the
        Maker's ``functional``.

        Args:
            output_dir: Directory where the input files will be written.
            structure: Input structure (used when ``prev_dir`` is ``None``).
            prev_dir: Previous calculation directory to inherit settings from.
            number_of_docs: Passed through to the input-set to control band
                documentation output.
            user_incar_settings: Per-call INCAR overrides.
            user_kpoints_settings: Per-call KPOINTS object.

        Returns:
            Absolute path string of the output directory.

        Raises:
            ValueError: If neither ``structure`` nor ``prev_dir`` is provided.

        写出非自洽（静态）计算的 VASP 输入文件。

        当提供 ``prev_dir`` 时，泛函和基础 INCAR 通过
        ``MPStaticSetEcat.from_prev_calc_ecat`` 从前序计算继承。否则使用
        Maker 的 ``functional`` 从 ``structure`` 构建新的输入集。

        Args:
            output_dir: 输入文件写入的目录。
            structure: 输入结构（``prev_dir`` 为 ``None`` 时使用）。
            prev_dir: 用于继承设置的前序计算目录。
            number_of_docs: 传递给输入集以控制能带文档输出。
            user_incar_settings: 单次调用的 INCAR 覆盖。
            user_kpoints_settings: 单次调用的 KPOINTS 对象。

        Returns:
            输出目录的绝对路径字符串。

        Raises:
            ValueError: 若既未提供 ``structure`` 也未提供 ``prev_dir``。
        """
        output_dir = self._ensure_dir(output_dir)

        structure_obj: Optional[Structure] = None
        if prev_dir is not None:
            # Attempt to load the relaxed structure from the previous directory.
            # 尝试从前序目录加载弛豫后的结构。
            try:
                structure_obj = load_structure(prev_dir)
            except Exception:
                structure_obj = None
        elif structure is not None:
            structure_obj = load_structure(structure)

        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        structure_obj = self._apply_magmom_compat(structure_obj, common_kwargs) or structure_obj

        if prev_dir is not None:
            # Functional is determined by the previous INCAR; do not override it here.
            # 泛函由前序 INCAR 决定，此处不覆盖。
            input_set = MPStaticSetEcat.from_prev_calc_ecat(
                prev_dir=Path(prev_dir).resolve(),
                kpoints_density=self.kpoints_density,
                number_of_docs=number_of_docs,
                user_incar_settings=common_kwargs.get("user_incar_settings"),
                user_kpoints_settings=common_kwargs.get("user_kpoints_settings"),
            )
        else:
            if structure_obj is None:
                raise ValueError("Must provide either 'structure' or 'prev_dir' for NoSCF.")
            input_set = MPStaticSetEcat(
                functional=self.functional,
                structure=structure_obj,
                use_default_kpoints=self.use_default_kpoints,
                number_of_docs=number_of_docs,
                **common_kwargs,
            )

        input_set.write_input(output_dir)
        return str(output_dir)

    def write_neb(
        self,
        output_dir: Union[str, Path],
        start_structure: Union[str, Structure, Path],
        end_structure: Union[str, Structure, Path],
        n_images: int = 6,
        use_idpp: bool = True,
        intermediate_structures: Optional[List[Structure]] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
    ) -> str:
        """
        Write VASP input files for a Nudged Elastic Band (NEB) transition-state search.

        When either ``start_structure`` or ``end_structure`` is a directory path,
        the input set is constructed via ``NEBSetEcat.from_prev_calc`` so that
        INCAR / KPOINTS settings are inherited from that prior calculation.
        Otherwise, image structures are generated by linear or IDPP interpolation
        between the two endpoint structures.

        Args:
            output_dir: Root directory for all NEB image sub-directories.
            start_structure: Initial endpoint structure (file/dir path or
                ``Structure``).
            end_structure: Final endpoint structure (file/dir path or
                ``Structure``).
            n_images: Number of intermediate image structures to generate.
            use_idpp: When ``True``, use the Image-Dependent Pair Potential
                (IDPP) interpolation scheme; otherwise use linear interpolation.
            intermediate_structures: Pre-built intermediate structures. When
                provided, interpolation is skipped.
            user_incar_settings: Per-call INCAR overrides.
            user_kpoints_settings: Per-call KPOINTS object.

        Returns:
            Absolute path string of the output directory.

        生成 NEB 过渡态搜索输入文件。

        当 ``start_structure`` 或 ``end_structure`` 为目录路径时，输入集通过
        ``NEBSetEcat.from_prev_calc`` 构建，以便从该前序计算继承 INCAR / KPOINTS
        设置。否则，通过对两个端点结构进行线性或 IDPP 插值来生成像结构。

        Args:
            output_dir: 所有 NEB 像子目录的根目录。
            start_structure: 初始端点结构（文件/目录路径或 ``Structure``）。
            end_structure: 末端端点结构（文件/目录路径或 ``Structure``）。
            n_images: 要生成的中间像结构数量。
            use_idpp: 为 ``True`` 时使用图像相关对势（IDPP）插值方案；
                否则使用线性插值。
            intermediate_structures: 预建中间结构。提供时跳过插值。
            user_incar_settings: 单次调用的 INCAR 覆盖。
            user_kpoints_settings: 单次调用的 KPOINTS 对象。

        Returns:
            输出目录的绝对路径字符串。
        """
        output_dir = self._ensure_dir(output_dir)
        # Try to load a structure for MAGMOM normalisation; fall back gracefully.
        # 尝试加载结构用于 MAGMOM 规范化；优雅降级。
        structure_for_mag: Optional[Structure] = None
        try:
            structure_for_mag = load_structure(start_structure)
        except Exception:
            try:
                structure_for_mag = load_structure(end_structure)
            except Exception:
                structure_for_mag = None

        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        structure_for_mag = self._apply_magmom_compat(structure_for_mag, common_kwargs) or structure_for_mag

        # Detect whether start or end was supplied as an existing directory.
        # 检测 start 或 end 是否作为已有目录传入。
        is_start_dir = isinstance(start_structure, (str, Path)) and Path(start_structure).is_dir()
        is_end_dir = isinstance(end_structure, (str, Path)) and Path(end_structure).is_dir()

        if is_start_dir or is_end_dir:
            # Prefer start_dir as the reference for inheriting prior-calc settings.
            # 优先使用 start_dir 作为继承前序计算设置的参考。
            prev_dir = start_structure if is_start_dir else end_structure
            input_obj = NEBSetEcat.from_prev_calc(
                prev_dir=prev_dir,
                start_structure=start_structure,
                end_structure=end_structure,
                n_images=n_images,
                use_idpp=use_idpp,
                **common_kwargs
            )
        else:
            input_obj = NEBSetEcat(
                start_structure=start_structure,
                end_structure=end_structure,
                n_images=n_images,
                use_idpp=use_idpp,
                intermediate_structures=intermediate_structures,
                **common_kwargs
            )
        input_obj.write_input(output_dir)
        return str(output_dir)

    def write_lobster(
        self,
        output_dir: Union[str, Path],
        structure: Union[str, Structure, Path, None] = None,
        prev_dir: Optional[Union[str, Path]] = None,
        isym: int = 0,
        ismear: int = -5,
        reciprocal_density: Optional[int] = None,
        user_supplied_basis: Optional[dict] = None,
        overwritedict: Optional[Dict[str, Any]] = None,
        custom_lobsterin_lines: Optional[List[str]] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
    ) -> str:
        """
        Write VASP and Lobster input files for a chemical-bonding analysis.

        Generates the VASP INCAR / KPOINTS / POTCAR files tuned for Lobster
        post-processing (dense k-mesh, no symmetry reduction, tetrahedron
        smearing) together with a ``lobsterin`` file.  When ``prev_dir`` is
        provided, settings are inherited via
        ``LobsterSetEcat.from_prev_calc_ecat``; otherwise a fresh input set is
        built from ``structure``.

        Args:
            output_dir: Directory where all input files will be written.
            structure: Input structure (used when ``prev_dir`` is ``None``).
            prev_dir: Previous VASP calculation directory to inherit settings
                from.
            isym: ISYM tag value for symmetry treatment (0 = no symmetry).
            ismear: ISMEAR tag value for smearing method (−5 = tetrahedron).
            reciprocal_density: Explicit reciprocal-space k-point density;
                overrides ``kpoints_density`` when set.
            user_supplied_basis: Mapping of element symbols to basis-set
                strings passed directly to Lobsterin.
            overwritedict: Additional Lobsterin key-value pairs to overwrite
                after the default lobsterin is built.
            custom_lobsterin_lines: Raw lobsterin lines appended verbatim.
            user_incar_settings: Per-call INCAR overrides.
            user_kpoints_settings: Per-call KPOINTS object.

        Returns:
            Absolute path string of the output directory.

        Raises:
            ValueError: If neither ``structure`` nor ``prev_dir`` is provided.

        写出化学键分析（Lobster）的 VASP 和 Lobster 输入文件。

        生成针对 Lobster 后处理调优的 VASP INCAR / KPOINTS / POTCAR 文件
        （密集 k 网格、无对称性约简、四面体展宽）以及 ``lobsterin`` 文件。
        当提供 ``prev_dir`` 时，通过 ``LobsterSetEcat.from_prev_calc_ecat``
        继承设置；否则从 ``structure`` 构建新的输入集。

        Args:
            output_dir: 所有输入文件写入的目录。
            structure: 输入结构（``prev_dir`` 为 ``None`` 时使用）。
            prev_dir: 用于继承设置的前序 VASP 计算目录。
            isym: 对称性处理的 ISYM 标签值（0 = 无对称性）。
            ismear: 展宽方法的 ISMEAR 标签值（−5 = 四面体）。
            reciprocal_density: 显式倒空间 k 点密度；设置时覆盖
                ``kpoints_density``。
            user_supplied_basis: 元素符号到基组字符串的映射，直接传递给
                Lobsterin。
            overwritedict: 构建默认 lobsterin 后要覆盖的额外 Lobsterin 键值对。
            custom_lobsterin_lines: 逐字追加的原始 lobsterin 行。
            user_incar_settings: 单次调用的 INCAR 覆盖。
            user_kpoints_settings: 单次调用的 KPOINTS 对象。

        Returns:
            输出目录的绝对路径字符串。

        Raises:
            ValueError: 若既未提供 ``structure`` 也未提供 ``prev_dir``。
        """
        output_dir = self._ensure_dir(output_dir)

        structure_obj: Optional[Structure] = None
        if prev_dir is not None:
            try:
                structure_obj = load_structure(prev_dir)
            except Exception:
                structure_obj = None
        elif structure is not None:
            structure_obj = load_structure(structure)

        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        structure_obj = self._apply_magmom_compat(structure_obj, common_kwargs) or structure_obj

        if prev_dir is not None:
            input_set = LobsterSetEcat.from_prev_calc_ecat(
                prev_dir=Path(prev_dir).resolve(),
                kpoints_density=self.kpoints_density,
                isym=isym,
                ismear=ismear,
                reciprocal_density=reciprocal_density,
                user_supplied_basis=user_supplied_basis,
                **common_kwargs,
            )
        else:
            if structure_obj is None:
                raise ValueError("Must provide either 'structure' or 'prev_dir' for Lobster.")
            input_set = LobsterSetEcat(
                structure=structure_obj,
                isym=isym,
                ismear=ismear,
                reciprocal_density=reciprocal_density,
                user_supplied_basis=user_supplied_basis,
                **common_kwargs,
            )

        input_set.write_input(
            output_dir,
            overwritedict=overwritedict,
            custom_lobsterin_lines=custom_lobsterin_lines
        )
        return str(output_dir)

    def write_adsorption(
        self,
        output_dir: Union[str, Path],
        structure: Union[str, Structure, Path],
        prev_dir: Union[str, Path],
        auto_dipole: bool = True,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
    ) -> str:
        """
        Write VASP input files for an adsorbate-on-slab relaxation.

        Settings (functional, ENCUT, etc.) are inherited from a prior slab
        calculation via ``SlabSetEcat.ads_from_prev_calc``.  The Maker's own
        ``functional`` field is deliberately excluded from the kwargs forwarded
        to the input set to prevent conflicts with the inherited value.

        Args:
            output_dir: Directory where the input files will be written.
            structure: Adsorbate + slab structure (file path or ``Structure``).
            prev_dir: Previous clean-slab calculation directory whose settings
                are inherited.
            auto_dipole: Automatically apply dipole correction tags.
            user_incar_settings: Per-call INCAR overrides.
            user_kpoints_settings: Per-call KPOINTS object.

        Returns:
            Absolute path string of the output directory.

        写出吸附质-表面弛豫的 VASP 输入文件。

        设置（泛函、ENCUT 等）通过 ``SlabSetEcat.ads_from_prev_calc`` 从前序
        slab 计算继承。Maker 自身的 ``functional`` 字段被有意从转发给输入集的
        kwargs 中排除，以防与继承值冲突。

        Args:
            output_dir: 输入文件写入的目录。
            structure: 吸附质 + slab 结构（文件路径或 ``Structure``）。
            prev_dir: 继承设置的前序干净 slab 计算目录。
            auto_dipole: 自动应用偶极修正标签。
            user_incar_settings: 单次调用的 INCAR 覆盖。
            user_kpoints_settings: 单次调用的 KPOINTS 对象。

        Returns:
            输出目录的绝对路径字符串。
        """
        output_dir = self._ensure_dir(output_dir)
        prev_dir_path = Path(prev_dir).resolve()
        struct_obj = load_structure(structure)
        # Functional for ads-on-slab is inherited from prev_calc, so remove it
        # from common_kwargs to avoid overriding the inherited value.
        # 吸附质-表面弛豫的泛函从 prev_calc 继承，因此从 common_kwargs 中移除
        # 以避免覆盖继承值。
        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        common_kwargs.pop("functional", None)
        struct_obj = self._apply_magmom_compat(struct_obj, common_kwargs) or struct_obj
        input_set = SlabSetEcat.ads_from_prev_calc(
            structure=struct_obj,
            prev_dir=prev_dir_path,
            auto_dipole=auto_dipole,
            **common_kwargs
        )
        input_set.write_input(output_dir)
        return str(output_dir)

    def write_freq(
        self,
        output_dir: Union[str, Path],
        prev_dir: Union[str, Path],
        structure: Union[str, Structure, Path, None] = None,
        mode: str = "inherit",
        vibrate_indices: Optional[List[int]] = None,
        adsorbate_formula: Optional[str] = None,
        adsorbate_formula_prefer: str = "tail",
        calc_ir: bool = False,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
    ) -> str:
        """
        Write VASP input files for a vibrational frequency calculation.

        The set of vibrating atoms (``IBRION = 5`` selective dynamics) is
        determined by the following priority:

        1. Explicitly provided ``vibrate_indices``.
        2. Atoms matching ``adsorbate_formula`` identified by
           :func:`pick_adsorbate_indices_by_formula_strict`.
        3. ``mode="all"`` — all atoms vibrate (selective dynamics tag removed).
        4. ``mode="inherit"`` — selective dynamics from the relaxed structure
           are kept as-is.

        INCAR / KPOINTS settings are inherited from ``prev_dir`` via
        ``FreqSetEcat.from_prev_calc_ecat``.

        Args:
            output_dir: Directory where the input files will be written.
            prev_dir: Previous relaxation directory; used to inherit INCAR
                settings and (optionally) to read the final CONTCAR.
            structure: Override structure.  When ``None``, CONTCAR from
                ``prev_dir`` is used.
            mode: Vibration mode selector.  One of ``"inherit"`` (keep
                existing selective dynamics), ``"all"`` (remove selective
                dynamics so all atoms vibrate).
            vibrate_indices: Explicit zero-based atom indices to set as free
                in the selective dynamics; overrides ``mode`` and
                ``adsorbate_formula``.
            adsorbate_formula: Chemical formula of the adsorbate fragment whose
                atoms should vibrate.  Used when ``vibrate_indices`` is ``None``.
            adsorbate_formula_prefer: Tie-breaking preference when multiple
                adsorbate fragments match the formula (``"tail"`` or ``"head"``).
            calc_ir: When ``True``, request IR-intensity calculation (requires
                LEPSILON or finite-difference dipole derivatives).
            user_incar_settings: Per-call INCAR overrides.
            user_kpoints_settings: Per-call KPOINTS object.

        Returns:
            Absolute path string of the output directory.

        Raises:
            FileNotFoundError: If ``structure`` is ``None`` and no CONTCAR
                exists in ``prev_dir``.
            ValueError: If ``mode`` is not one of the accepted values.

        写出振动频率计算的 VASP 输入文件。

        振动原子集合（``IBRION = 5`` 选择性动力学）由以下优先级确定：

        1. 显式提供的 ``vibrate_indices``。
        2. 由 :func:`pick_adsorbate_indices_by_formula_strict` 识别的匹配
           ``adsorbate_formula`` 的原子。
        3. ``mode="all"`` — 所有原子振动（移除选择性动力学标签）。
        4. ``mode="inherit"`` — 弛豫结构的选择性动力学保持不变。

        INCAR / KPOINTS 设置通过 ``FreqSetEcat.from_prev_calc_ecat`` 从
        ``prev_dir`` 继承。

        Args:
            output_dir: 输入文件写入的目录。
            prev_dir: 前序弛豫目录；用于继承 INCAR 设置以及（可选）读取最终
                CONTCAR。
            structure: 覆盖结构。为 ``None`` 时使用 ``prev_dir`` 中的 CONTCAR。
            mode: 振动模式选择器。可选 ``"inherit"``（保留现有选择性动力学）
                或 ``"all"``（移除选择性动力学使所有原子振动）。
            vibrate_indices: 显式设为自由的零基原子索引；覆盖 ``mode`` 和
                ``adsorbate_formula``。
            adsorbate_formula: 应振动原子的吸附质碎片化学式。在
                ``vibrate_indices`` 为 ``None`` 时使用。
            adsorbate_formula_prefer: 多个吸附质碎片匹配化学式时的平局处理
                偏好（``"tail"`` 或 ``"head"``）。
            calc_ir: 为 ``True`` 时请求 IR 强度计算（需要 LEPSILON 或有限差分
                偶极导数）。
            user_incar_settings: 单次调用的 INCAR 覆盖。
            user_kpoints_settings: 单次调用的 KPOINTS 对象。

        Returns:
            输出目录的绝对路径字符串。

        Raises:
            FileNotFoundError: 若 ``structure`` 为 ``None`` 且 ``prev_dir`` 中
                不存在 CONTCAR。
            ValueError: 若 ``mode`` 不是接受的值之一。
        """
        output_dir = self._ensure_dir(output_dir)
        prev_dir_path = Path(prev_dir).resolve()

        if structure is not None:
            # Use caller-supplied structure directly.
            # 直接使用调用者提供的结构。
            if isinstance(structure, Structure):
                final_structure = structure.copy()
            else:
                final_structure = Structure.from_file(structure)
        else:
            # Fall back to the relaxed CONTCAR from prev_dir.
            # 回退到 prev_dir 中的弛豫 CONTCAR。
            contcar_path = prev_dir_path / "CONTCAR"
            if not contcar_path.exists():
                raise FileNotFoundError(f"CONTCAR not found in {prev_dir_path}")
            final_structure = Structure.from_file(contcar_path)

        final_vibrate_indices: Optional[List[int]] = None

        if vibrate_indices is not None:
            # Explicit index list takes highest priority.
            # 显式索引列表具有最高优先级。
            final_vibrate_indices = vibrate_indices
        elif adsorbate_formula is not None:
            # Identify vibrating atoms by matching the adsorbate chemical formula.
            # 通过匹配吸附质化学式识别振动原子。
            final_vibrate_indices = pick_adsorbate_indices_by_formula_strict(
                final_structure,
                adsorbate_formula=adsorbate_formula,
                prefer=adsorbate_formula_prefer,
            )
            logger.info("Picked vibrate indices by formula %s: %s", adsorbate_formula, final_vibrate_indices)
        elif mode == "all":
            # All atoms vibrate: remove selective_dynamics site property.
            # 所有原子都参与振动：移除 selective_dynamics 站点属性。
            if "selective_dynamics" in final_structure.site_properties:
                final_structure.remove_site_property("selective_dynamics")
        elif mode == "inherit":
            # Keep selective dynamics from the relaxed structure; log a reminder.
            # 保留弛豫结构的选择性动力学；记录提醒。
            logger.warning(
                    "mode='inherit' is set, no fix atoms set"
                    "If this is unintended, ignore "
                )
        else:
            raise ValueError("mode must be one of: 'inherit', 'all', 'adsorbate'.")

        # Functional should be inherited from prev_dir's INCAR, not forced here.
        # 泛函应从 prev_dir 的 INCAR 继承，不在此处强制指定。
        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        final_structure = self._apply_magmom_compat(final_structure, common_kwargs) or final_structure

        input_set = FreqSetEcat.from_prev_calc_ecat(
            prev_dir=prev_dir_path,
            structure=final_structure,
            vibrate_indices=final_vibrate_indices,
            calc_ir=calc_ir,
            **common_kwargs
        )
        input_set.write_input(output_dir)
        return str(output_dir)

    def write_dimer(
        self,
        output_dir: Union[str, Path],
        neb_dir: Optional[Union[str, Path]] = None,
        num_images: Optional[int] = None,
        structure: Union[str, Structure, Path, None] = None,
        modecar: Union[str, Path, np.ndarray, None] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
    ):
        """
        Write VASP input files for a Dimer transition-state search.

        Two construction paths are supported:

        * **From NEB results** (``neb_dir`` provided): the saddle-point
          geometry and MODECAR are extracted from a completed NEB directory via
          ``DimerSetEcat.from_neb_calc``.
        * **From scratch** (``structure`` + ``modecar`` provided): a Dimer
          input set is built directly from a manual structure and MODECAR.

        Args:
            output_dir: Directory where the input files will be written.
            neb_dir: Completed NEB calculation directory.  When provided,
                ``structure`` and ``modecar`` are ignored.
            num_images: Number of NEB images to read when extracting from
                ``neb_dir``; ``None`` means auto-detect.
            structure: Initial structure for the Dimer (used when ``neb_dir``
                is ``None``).
            modecar: Dimer mode vector as a file path or numpy array (used
                when ``neb_dir`` is ``None``).
            user_incar_settings: Per-call INCAR overrides.
            user_kpoints_settings: Per-call KPOINTS object.

        Returns:
            Absolute path string of the output directory.

        Raises:
            ValueError: If ``neb_dir`` is ``None`` and either ``structure`` or
                ``modecar`` is missing.

        写出 Dimer 过渡态搜索的 VASP 输入文件。

        支持两种构建路径：

        * **从 NEB 结果构建**（提供 ``neb_dir``）：通过
          ``DimerSetEcat.from_neb_calc`` 从已完成的 NEB 目录提取鞍点几何形状
          和 MODECAR。
        * **从头构建**（提供 ``structure`` + ``modecar``）：直接从手动结构和
          MODECAR 构建 Dimer 输入集。

        Args:
            output_dir: 输入文件写入的目录。
            neb_dir: 已完成的 NEB 计算目录。提供时忽略 ``structure`` 和
                ``modecar``。
            num_images: 从 ``neb_dir`` 提取时读取的 NEB 像数量；``None``
                表示自动检测。
            structure: Dimer 的初始结构（``neb_dir`` 为 ``None`` 时使用）。
            modecar: Dimer 模式向量，文件路径或 numpy 数组（``neb_dir`` 为
                ``None`` 时使用）。
            user_incar_settings: 单次调用的 INCAR 覆盖。
            user_kpoints_settings: 单次调用的 KPOINTS 对象。

        Returns:
            输出目录的绝对路径字符串。

        Raises:
            ValueError: 若 ``neb_dir`` 为 ``None`` 且缺少 ``structure`` 或
                ``modecar``。
        """
        output_dir = self._ensure_dir(output_dir)

        structure_obj: Optional[Structure] = None
        if neb_dir is not None:
            try:
                structure_obj = load_structure(neb_dir)
            except Exception:
                structure_obj = None
        elif structure is not None:
            structure_obj = load_structure(structure)

        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        structure_obj = self._apply_magmom_compat(structure_obj, common_kwargs) or structure_obj

        if neb_dir is not None:
            logger.info(f"Generating Dimer input from NEB directory: {neb_dir}")
            input_set = DimerSetEcat.from_neb_calc(
                neb_dir=neb_dir,
                num_images=num_images,
                **common_kwargs
            )
        else:
            if structure_obj is None or modecar is None:
                raise ValueError(
                    "Must provide both 'structure' and 'modecar' if 'neb_dir' is not specified."
                )
            logger.info("Generating Dimer input from manually provided structure and MODECAR.")
            input_set = DimerSetEcat(
                structure=structure_obj,
                modecar=modecar,
                **common_kwargs
            )
        input_set.write_input(output_dir)
        return str(output_dir)

    def write_nbo(
        self,
        output_dir: Union[str, Path],
        basis_source: Union[str, Path, Dict[str, str], None] = None,
        prev_dir: Optional[Union[str, Path]] = None,
        structure: Union[str, Structure, Path, None] = None,
        nbo_config: Optional[Dict[str, Any]] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
    ) -> str:
        """
        Write VASP input files for a Natural Bond Orbital (NBO) analysis.

        The final structure is sourced either from the caller-supplied
        ``structure`` argument or from the CONTCAR inside ``prev_dir``.  When
        ``prev_dir`` is provided, INCAR / KPOINTS settings are also inherited
        via ``NBOSetEcat.from_prev_calc``; otherwise a standalone NBO input set
        is constructed.

        Args:
            output_dir: Directory where all input files will be written.
            basis_source: Basis-set specification — a file path, a raw basis
                string, or a ``{element: basis}`` dict.  Passed through to
                ``NBOSetEcat``.
            prev_dir: Previous VASP calculation directory.
            structure: Input structure.  When both ``prev_dir`` and
                ``structure`` are given, ``structure`` is used for the geometry
                while ``prev_dir`` supplies INCAR settings.
            nbo_config: Additional NBO configuration options forwarded to
                ``NBOSetEcat``.
            user_incar_settings: Per-call INCAR overrides.
            user_kpoints_settings: Per-call KPOINTS object.

        Returns:
            Absolute path string of the output directory.

        Raises:
            FileNotFoundError: If CONTCAR is absent in ``prev_dir`` when
                ``structure`` is ``None``.
            ValueError: If neither ``structure`` nor ``prev_dir`` is provided.

        写出自然键轨道（NBO）分析的 VASP 输入文件。

        最终结构来自调用者提供的 ``structure`` 参数，或 ``prev_dir`` 中的
        CONTCAR。当提供 ``prev_dir`` 时，INCAR / KPOINTS 设置也通过
        ``NBOSetEcat.from_prev_calc`` 继承；否则构建独立的 NBO 输入集。

        Args:
            output_dir: 所有输入文件写入的目录。
            basis_source: 基组规格——文件路径、原始基组字符串或
                ``{element: basis}`` 字典。传递给 ``NBOSetEcat``。
            prev_dir: 前序 VASP 计算目录。
            structure: 输入结构。当 ``prev_dir`` 和 ``structure`` 同时提供时，
                ``structure`` 用于几何形状，``prev_dir`` 提供 INCAR 设置。
            nbo_config: 转发给 ``NBOSetEcat`` 的额外 NBO 配置选项。
            user_incar_settings: 单次调用的 INCAR 覆盖。
            user_kpoints_settings: 单次调用的 KPOINTS 对象。

        Returns:
            输出目录的绝对路径字符串。

        Raises:
            FileNotFoundError: 若 ``structure`` 为 ``None`` 且 ``prev_dir``
                中不存在 CONTCAR。
            ValueError: 若既未提供 ``structure`` 也未提供 ``prev_dir``。
        """
        output_dir = self._ensure_dir(output_dir)

        if structure is not None:
            # Caller provided structure takes precedence over prev_dir geometry.
            # 调用者提供的结构优先于 prev_dir 的几何形状。
            if isinstance(structure, Structure):
                final_structure = structure.copy()
            else:
                final_structure = Structure.from_file(structure)
        elif prev_dir is not None:
            prev_dir_path = Path(prev_dir).resolve()
            contcar_path = prev_dir_path / "CONTCAR"
            if not contcar_path.exists():
                raise FileNotFoundError(f"CONTCAR not found in {prev_dir_path}")
            final_structure = Structure.from_file(contcar_path)
        else:
            raise ValueError("Must provide either 'structure' or 'prev_dir' to generate NBO input.")

        logger.info("Generating NBO pure input files...")

        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        final_structure = self._apply_magmom_compat(final_structure, common_kwargs) or final_structure

        if prev_dir is not None:
            # Inherit INCAR / functional from the previous calculation.
            # 从前序计算继承 INCAR / 泛函。
            input_set = NBOSetEcat.from_prev_calc(
                prev_dir=prev_dir,
                basis_source=basis_source,
                nbo_config=nbo_config,
                user_incar_settings=common_kwargs.get("user_incar_settings"),
                user_kpoints_settings=common_kwargs.get("user_kpoints_settings"),
                structure=final_structure
            )
        else:
            input_set = NBOSetEcat(
                structure=final_structure,
                basis_source=basis_source,
                nbo_config=nbo_config,
                **common_kwargs
            )

        input_set.write_input(output_dir)

        return str(output_dir)

    def write_nmr(
        self,
        output_dir: Union[str, Path],
        mode: str = "cs",
        isotopes: Optional[List[str]] = None,
        kpoints_density: int = 100,
        prev_dir: Optional[Union[str, Path]] = None,
        structure: Union[str, Structure, Path, None] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
    ) -> str:
        """
        Write VASP input files for an NMR parameter calculation.

        Supports chemical-shift (CS) and electric-field-gradient (EFG) modes.
        When ``prev_dir`` is given, settings are inherited via
        ``NMRSetEcat.from_prev_calc_ecat``; otherwise a fresh input set is
        built from ``structure`` using the Maker's ``functional``.

        Args:
            output_dir: Output directory.
            mode: NMR calculation mode.
                ``"cs"`` — chemical shift (LCHIMAG enabled);
                ``"efg"`` — electric field gradient (LEFG and QUAD_EFG enabled).
                Defaults to ``"cs"``.
            isotopes: Reference isotopes per element for EFG mode, formatted as
                ``["Li-7", "O-17"]``.  Ignored for CS mode.
            kpoints_density: Default k-point density used when KPOINTS cannot
                be inherited from a previous calculation.  Defaults to 100.
            prev_dir: Previous calculation directory.  When provided, structure,
                functional, and INCAR / KPOINTS settings are inherited from it.
            structure: Input structure.  Must be provided when ``prev_dir`` is
                ``None``.
            user_incar_settings: Per-call INCAR overrides (highest priority).
            user_kpoints_settings: Per-call KPOINTS object.

        Returns:
            Absolute path string of the output directory.

        Raises:
            ValueError: If neither ``structure`` nor ``prev_dir`` is provided.

        写出 NMR 参数计算的 VASP 输入文件。

        支持化学位移（CS）和电场梯度（EFG）两种模式。当提供 ``prev_dir`` 时，
        设置通过 ``NMRSetEcat.from_prev_calc_ecat`` 继承；否则使用 Maker 的
        ``functional`` 从 ``structure`` 构建新的输入集。

        Args:
            output_dir: 输出目录。
            mode: NMR 计算模式。
                ``"cs"`` — 化学位移（启用 LCHIMAG）；
                ``"efg"`` — 电场梯度（启用 LEFG 与 QUAD_EFG）。
                默认 ``"cs"``。
            isotopes: EFG 模式下各元素的参考同位素，格式如
                ``["Li-7", "O-17"]``。对 CS 模式无效。
            kpoints_density: 无法从前序计算继承 KPOINTS 时使用的默认 k 点密度。
                默认 100。
            prev_dir: 前序计算目录。提供时从中继承结构、泛函及
                INCAR / KPOINTS 设置。
            structure: 输入结构。``prev_dir`` 为 ``None`` 时必须提供。
            user_incar_settings: 用户自定义 INCAR 参数（优先级最高）。
            user_kpoints_settings: 用户自定义 KPOINTS 对象。

        Returns:
            输出目录的绝对路径字符串。

        Raises:
            ValueError: 若既未提供 ``structure`` 也未提供 ``prev_dir``。
        """
        output_dir = self._ensure_dir(output_dir)

        structure_obj: Optional[Structure] = None
        if prev_dir is not None:
            try:
                structure_obj = load_structure(prev_dir)
            except Exception:
                structure_obj = None
        elif structure is not None:
            structure_obj = load_structure(structure)

        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        structure_obj = self._apply_magmom_compat(structure_obj, common_kwargs) or structure_obj

        if prev_dir is not None:
            input_set = NMRSetEcat.from_prev_calc_ecat(
                prev_dir=Path(prev_dir).resolve(),
                mode=mode,
                isotopes=isotopes,
                kpoints_density=kpoints_density,
                user_incar_settings=common_kwargs.get("user_incar_settings"),
                user_kpoints_settings=common_kwargs.get("user_kpoints_settings"),
            )
        else:
            if structure_obj is None:
                raise ValueError("Must provide either 'structure' or 'prev_dir' for NMR calculation.")
            input_set = NMRSetEcat(
                structure=structure_obj,
                mode=mode,
                isotopes=isotopes,
                functional=self.functional,
                kpoints_density=kpoints_density,
                **common_kwargs,
            )

        input_set.write_input(output_dir)
        return str(output_dir)

    def write_md(
        self,
        output_dir: Union[str, Path],
        ensemble: str = "nvt",
        start_temp: float = 300.0,
        end_temp: float = 300.0,
        nsteps: int = 1000,
        time_step: Optional[float] = None,
        spin_polarized: bool = False,
        langevin_gamma: Optional[List[float]] = None,
        prev_dir: Optional[Union[str, Path]] = None,
        structure: Union[str, Structure, Path, None] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
    ) -> str:
        """
        Write VASP input files for a molecular-dynamics (MD) simulation.

        Supports NVT (canonical) and NPT (isothermal-isobaric) ensembles.
        NPT uses a Langevin thermostat and barostat (MDALGO=3) and automatically
        sets ENCUT to 1.5 × the maximum default cutoff to eliminate Pulay stress.

        When ``prev_dir`` is provided, the structure and INCAR are inherited
        from it (with geometry-relaxation tags stripped) via
        ``MDSetEcat.from_prev_calc_ecat``; otherwise a fresh MD input set is
        built from ``structure``.

        Args:
            output_dir: Output directory.
            ensemble: MD ensemble type.
                ``"nvt"`` — canonical ensemble (Nosé-Hoover thermostat);
                ``"npt"`` — isothermal-isobaric ensemble (Langevin + barostat).
                Defaults to ``"nvt"``.
            start_temp: Initial temperature in kelvin.  Defaults to 300.0.
            end_temp: Final temperature in kelvin.  Equal to ``start_temp``
                gives constant-temperature MD.  Defaults to 300.0.
            nsteps: Total number of MD steps (NSW).  Defaults to 1000.
            time_step: Time step in femtoseconds.  For NVT, ``None`` triggers
                auto-detection (0.5 fs × 4 × steps when H is present, else
                2.0 fs).  For NPT, defaults to 2.0 fs.
            spin_polarized: Enable spin polarisation (ISPIN=2).  Defaults to
                ``False``.
            langevin_gamma: Langevin damping coefficients per element species
                (LANGEVIN_GAMMA, ps⁻¹) for NPT mode.  Auto-set to
                ``[10.0] × n_elements`` when ``None``.  Ignored for NVT.
            prev_dir: Previous calculation directory.  When provided,
                structure and INCAR settings are inherited (relaxation tags
                removed, then MD defaults merged on top).
            structure: Input structure.  Must be provided when ``prev_dir`` is
                ``None``.
            user_incar_settings: Per-call INCAR overrides (highest priority).
            user_kpoints_settings: Per-call KPOINTS object (defaults to
                Gamma-only 1×1×1 inside the input set).

        Returns:
            Absolute path string of the output directory.

        Raises:
            ValueError: If neither ``structure`` nor ``prev_dir`` is provided.

        写出分子动力学（MD）模拟的 VASP 输入文件。

        支持 NVT（正则系综）和 NPT（等温等压系综）两种模式。NPT 模式使用
        Langevin 热浴与压强控制（MDALGO=3），并自动将 ENCUT 设为
        1.5 × VASP 默认最大截断能以消除 Pulay 应力。

        当提供 ``prev_dir`` 时，结构和 INCAR 通过
        ``MDSetEcat.from_prev_calc_ecat`` 从中继承（剥离几何弛豫标签后合并
        MD 默认值）；否则从 ``structure`` 构建新的 MD 输入集。

        Args:
            output_dir: 输出目录。
            ensemble: MD 系综类型。
                ``"nvt"`` — 正则系综（Nosé-Hoover 控温）；
                ``"npt"`` — 等温等压系综（Langevin 热浴 + 控压）。
                默认 ``"nvt"``。
            start_temp: 起始温度（K）。默认 300.0。
            end_temp: 终止温度（K），与 ``start_temp`` 相同即为恒温 MD。
                默认 300.0。
            nsteps: MD 总步数（NSW）。默认 1000。
            time_step: 时间步长（fs）。NVT 模式下 ``None`` 触发自动检测
                （含 H 原子时为 0.5 fs 且步数 × 4，否则为 2.0 fs）；NPT
                模式默认 2.0 fs。
            spin_polarized: 是否开启自旋极化（ISPIN=2）。默认 ``False``。
            langevin_gamma: NPT 模式下各元素种的 Langevin 阻尼系数
                （LANGEVIN_GAMMA，ps⁻¹）。为 ``None`` 时自动设为
                ``[10.0] × n_elems``。对 NVT 模式无效。
            prev_dir: 前序计算目录。提供时继承结构和 INCAR 设置（移除弛豫
                标签后合并 MD 默认值）。
            structure: 输入结构。``prev_dir`` 为 ``None`` 时必须提供。
            user_incar_settings: 用户自定义 INCAR 参数（优先级最高）。
            user_kpoints_settings: 用户自定义 KPOINTS（默认 Gamma-only
                1×1×1）。

        Returns:
            输出目录的绝对路径字符串。

        Raises:
            ValueError: 若既未提供 ``structure`` 也未提供 ``prev_dir``。
        """
        output_dir = self._ensure_dir(output_dir)

        structure_obj: Optional[Structure] = None
        if prev_dir is not None:
            try:
                structure_obj = load_structure(prev_dir)
            except Exception:
                structure_obj = None
        elif structure is not None:
            structure_obj = load_structure(structure)

        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        structure_obj = self._apply_magmom_compat(structure_obj, common_kwargs) or structure_obj

        # Bundle MD-specific parameters into a single dict for clean forwarding.
        # 将 MD 专属参数打包为单一字典以便整洁地转发。
        md_params = dict(
            ensemble=ensemble,
            start_temp=start_temp,
            end_temp=end_temp,
            nsteps=nsteps,
            time_step=time_step,
            spin_polarized=spin_polarized,
            langevin_gamma=langevin_gamma,
        )

        if prev_dir is not None:
            input_set = MDSetEcat.from_prev_calc_ecat(
                prev_dir=Path(prev_dir).resolve(),
                user_incar_settings=common_kwargs.get("user_incar_settings"),
                user_kpoints_settings=common_kwargs.get("user_kpoints_settings"),
                **md_params,
            )
        else:
            if structure_obj is None:
                raise ValueError("Must provide either 'structure' or 'prev_dir' for MD calculation.")
            input_set = MDSetEcat(
                structure=structure_obj,
                functional=self.functional,
                **md_params,
                **common_kwargs,
            )

        input_set.write_input(output_dir)
        return str(output_dir)
