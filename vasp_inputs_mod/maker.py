"""高层 API：通过配置生成不同类型的 VASP 输入。"""

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
)
import logging

from .utils import load_structure, pick_adsorbate_indices_by_formula_strict

logger = logging.getLogger(__name__)


@dataclass
class VaspInputMaker:
    """VASP 输入文件生成器（封装不同计算类型）。"""

    name: str = "VaspInputMaker"
    functional: str = "PBE"
    kpoints_density: float = 50.0
    use_default_incar: bool = True
    use_default_kpoints: bool = True
    
    # 全局默认设置 (Baseline)
    user_incar_settings: Dict[str, Any] = field(default_factory=dict)
    user_kpoints_settings: Any = None
    
    user_potcar_functional: str = "PBE_54"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict_ecat(cls, config: Dict[str, Any], **kwargs):
        """从字典安全地初始化 Maker，未知的键会被放入 extra_kwargs。"""
        full_config = {**config, **kwargs}
        class_fields = {f.name for f in fields(cls)}
        main_params = {}
        extra_params = full_config.pop("extra_kwargs", {})

        for k, v in full_config.items():
            key_clean = k.strip()
            if key_clean in class_fields:
                main_params[key_clean] = v
            else:
                extra_params[key_clean] = v

        return cls(**main_params, extra_kwargs=extra_params)

    def __post_init__(self):
        self.functional = (self.functional or "PBE").upper()
        self.user_incar_settings = dict(self.user_incar_settings or {})
        self.extra_kwargs = dict(self.extra_kwargs or {})

    def _ensure_dir(self, output_dir: Union[str, Path]) -> Path:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _build_common_kwargs(
        self, 
        local_incar: Optional[Dict[str, Any]] = None, 
        local_kpoints: Optional[Any] = None
    ) -> Dict[str, Any]:
        """提取通用参数，并执行 局部配置 对 全局配置 的覆盖合并。

        注意：functional 由各个 write_* 方法明确传递，避免与 prev_calc 模式之间产生冲突。
        """
        # INCAR 合并策略：字典 Update，局部覆盖全局
        merged_incar = self.user_incar_settings.copy()
        if local_incar:
            merged_incar.update(local_incar)

        # KPOINTS 合并策略：直接替换，局部存在则完全无视全局
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
    
    #INCAR设置接口
    def write_bulk(
        self, 
        structure: Union[str, Structure, Path], 
        output_dir: Union[str, Path], 
        is_metal: bool = False,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
    ) -> str:
        output_dir = self._ensure_dir(output_dir)
        input_obj = BulkRelaxSetEcat(
            functional=self.functional,
            structure=load_structure(structure),
            kpoints_density=self.kpoints_density,
            use_default_kpoints=self.use_default_kpoints,
            is_metal=is_metal,
            **self._build_common_kwargs(user_incar_settings, user_kpoints_settings),
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
        output_dir = self._ensure_dir(output_dir)
        input_obj = SlabSetEcat(
            functional=self.functional,
            structure=load_structure(structure),
            kpoints_density=self.kpoints_density,
            use_default_kpoints=self.use_default_kpoints,
            auto_dipole=auto_dipole,
            **self._build_common_kwargs(user_incar_settings, user_kpoints_settings),
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
        output_dir = self._ensure_dir(output_dir)
        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)

        if prev_dir is not None:
            # 对于 prev_dir 情况，functional 应由 prev INCAR 决定，
            # 且不应将 maker 的 functional/extra_kwargs 作为额外参数传入。
            input_set = MPStaticSetEcat.from_prev_calc_ecat(
                prev_dir=Path(prev_dir).resolve(),
                kpoints_density=self.kpoints_density,
                number_of_docs=number_of_docs,
                user_incar_settings=common_kwargs.get("user_incar_settings"),
                user_kpoints_settings=common_kwargs.get("user_kpoints_settings"),
            )
        else:
            if structure is None:
                raise ValueError("Must provide either 'structure' or 'prev_dir' for NoSCF.")
            input_set = MPStaticSetEcat(
                functional=self.functional,
                structure=load_structure(structure),
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
        output_dir = self._ensure_dir(output_dir)
        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)

        is_start_dir = isinstance(start_structure, (str, Path)) and Path(start_structure).is_dir()
        is_end_dir = isinstance(end_structure, (str, Path)) and Path(end_structure).is_dir()

        if is_start_dir or is_end_dir:
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
        output_dir = self._ensure_dir(output_dir)
        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)

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
            if structure is None:
                raise ValueError("Must provide either 'structure' or 'prev_dir' for Lobster.")
            input_set = LobsterSetEcat(
                structure=load_structure(structure),
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
        output_dir = self._ensure_dir(output_dir)
        prev_dir_path = Path(prev_dir).resolve()
        # Slab adsorbate 从 prev calc 继承 functional，因此需要避免将 Maker 的 functional 传入。
        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        common_kwargs.pop("functional", None)
        input_set = SlabSetEcat.ads_from_prev_calc(
            structure=load_structure(structure),
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
        output_dir = self._ensure_dir(output_dir)
        prev_dir_path = Path(prev_dir).resolve()

        if structure is not None:
            if isinstance(structure, Structure):
                final_structure = structure.copy()
            else:
                final_structure = Structure.from_file(structure)
        else:
            contcar_path = prev_dir_path / "CONTCAR"
            if not contcar_path.exists():
                raise FileNotFoundError(f"CONTCAR not found in {prev_dir_path}")
            final_structure = Structure.from_file(contcar_path)

        final_vibrate_indices: Optional[List[int]] = None

        if vibrate_indices is not None:
            final_vibrate_indices = vibrate_indices
        elif adsorbate_formula is not None:
            final_vibrate_indices = pick_adsorbate_indices_by_formula_strict(
                final_structure,
                adsorbate_formula=adsorbate_formula,
                prefer=adsorbate_formula_prefer,
            )
            logger.info("Picked vibrate indices by formula %s: %s", adsorbate_formula, final_vibrate_indices)
        elif mode == "all":
            #所有原子都参与震动
            if "selective_dynamics" in final_structure.site_properties:
                final_structure.remove_site_property("selective_dynamics")
        elif mode == "inherit":
            #直接继承结构优化后的原子,没有设定固定进行提醒即可
            logger.warning(
                    "mode='inherit' is set, no fix atoms set"
                    "If this is unintended, ignore "
                )
        else:
            raise ValueError("mode must be one of: 'inherit', 'all', 'adsorbate'.")

        # 使用 prev_dir 生成频率计算时，functional 应从 prev INCAR 确定，避免冲突
        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)

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
        output_dir = self._ensure_dir(output_dir)
        common_kwargs = self._build_common_kwargs(user_incar_settings, user_kpoints_settings)
        
        if neb_dir is not None:
            logger.info(f"Generating Dimer input from NEB directory: {neb_dir}")
            input_set = DimerSetEcat.from_neb_calc(
                neb_dir=neb_dir,
                num_images=num_images,
                **common_kwargs
            )
        else:
            if structure is None or modecar is None:
                raise  ValueError(
                     "Must provide both 'structure' and 'modecar' if 'neb_dir' is not specified."
                )
            logger.info("Generating Dimer input from manually provided structure and MODECAR.")
            input_set = DimerSetEcat(
                structure=load_structure(structure),
                modecar=modecar,
                **common_kwargs
            )
        input_set.write_input(output_dir)
        return str(output_dir)

        
