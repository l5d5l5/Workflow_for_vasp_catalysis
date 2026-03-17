"""高层 API：通过配置生成不同类型的 VASP 输入。"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pymatgen.core import Structure

from .input_sets import (
    BulkRelaxSetEcat,
    FreqSetEcat,
    LobsterSetEcat,
    MPStaticSetEcat,
    NEBSetEcat,
    SlabSetEcat,
)
import logging

from .utils import load_structure, pick_adsorbate_indices_by_formula_strict

logger = logging.getLogger(__name__)


@dataclass
class VaspInputMaker:
    """VASP 输入文件生成器（封装不同计算类型）。"""

    name: str = "VaspInputMaker"
    functional: str = "PBE"
    user_incar_settings: Dict[str, Any] = field(default_factory=dict)
    user_kpoints_settings: Any = None
    user_potcar_functional: Any = "PBE_54"
    use_default_incar: bool = True
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    structure: Union[str, Structure, None] = None
    is_metal: bool = False
    use_default_kpoints: bool = True
    auto_dipole: bool = False

    # NEB
    start_structure: Union[str, Structure, None] = None
    end_structure: Union[str, Structure, None] = None
    intermediate_structures: List[Structure] = field(default_factory=list)
    n_images: int = 6

    # NoSCF/Lobster
    number_of_docs: Optional[int] = None
    isym: int = 0
    ismear: int = -5
    kpoints_density: float = 50
    reciprocal_density: Optional[int] = None
    user_supplied_basis: Optional[dict] = None
    overwritedict: Dict[str, Any] = field(default_factory=dict)
    prev_dir: Union[str, Path, None] = None

    @classmethod
    def from_dict_ecat(cls, config: Dict[str, Any], **kwargs):
        full_config = {**config, **kwargs}
        class_fields = {f.name.strip() for f in fields(cls)}
        main_params: Dict[str, Any] = {}
        extra_params: Dict[str, Any] = full_config.pop("extra_kwargs", {})

        for k, v in full_config.items():
            key_clean = k.strip()
            if key_clean in class_fields:
                main_params[key_clean] = v
            else:
                extra_params[key_clean] = v

        return cls(**main_params, extra_kwargs=extra_params)

    def __post_init__(self):
        # Ensure functional normalization and prevent accidental mutable shared state
        self.functional = (self.functional or "PBE").upper()
        self.user_incar_settings = dict(self.user_incar_settings or {})
        self.extra_kwargs = dict(self.extra_kwargs or {})

    def _get_final_structure(self, structure_override: Union[str, Structure, None]) -> Structure:
        struct_source = structure_override if structure_override is not None else self.structure
        if struct_source is None:
            raise ValueError("Structure is required for this calculation.")
        return load_structure(struct_source)

    def _ensure_dir(self, output_dir: Union[str, Path]) -> Path:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _resolve_prev_dir(self, prev_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        if prev_dir is None:
            prev_dir = self.prev_dir
        if prev_dir is None:
            return None
        return Path(prev_dir).resolve()

    def _build_common_kwargs(self, structure: Structure) -> Dict[str, Any]:
        return {
            "structure": structure,
            "functional": self.functional,
            "use_default_incar": self.use_default_incar,
            "user_incar_settings": self.user_incar_settings,
            "user_kpoints_settings": self.user_kpoints_settings,
            "user_potcar_functional": self.user_potcar_functional,
            **self.extra_kwargs,
        }

    def write_bulk(self, output_dir: Union[str, Path], structure: Union[str, Structure, None] = None) -> str:
        final_structure = self._get_final_structure(structure)
        output_dir = self._ensure_dir(output_dir)

        common_kwargs = self._build_common_kwargs(final_structure)
        input_kwargs = {
            "kpoints_density": self.kpoints_density,
            "is_metal": self.is_metal,
            "use_default_kpoints": self.use_default_kpoints,
            **common_kwargs,
        }

        input_obj = BulkRelaxSetEcat(**input_kwargs)
        input_obj.write_input(output_dir)
        return str(output_dir)

    def write_slab(self, output_dir: Union[str, Path], structure: Union[str, Structure, None] = None) -> str:
        final_structure = self._get_final_structure(structure)
        output_dir = self._ensure_dir(output_dir)

        common_kwargs = self._build_common_kwargs(final_structure)
        input_kwargs = {
            "kpoints_density": self.kpoints_density,
            "use_default_kpoints": self.use_default_kpoints,
            "auto_dipole": self.auto_dipole,
            **common_kwargs,
        }

        input_obj = SlabSetEcat(**input_kwargs)
        input_obj.write_input(output_dir)
        return str(output_dir)

    def write_noscf(
        self,
        output_dir: Union[str, Path],
        structure: Union[str, Structure, None] = None,
        prev_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        output_dir = self._ensure_dir(output_dir)
        prev_dir_path = self._resolve_prev_dir(prev_dir)

        if prev_dir_path is not None:
            input_set = MPStaticSetEcat.from_prev_calc_ecat(
                prev_dir=prev_dir_path,
                kpoints_density=self.kpoints_density,
                number_of_docs=self.number_of_docs,
                user_incar_settings=self.user_incar_settings,
                **self.extra_kwargs,
            )
        else:
            final_structure = self._get_final_structure(structure)
            input_set = MPStaticSetEcat(
                structure=final_structure,
                functional=self.functional,
                use_default_incar=self.use_default_incar,
                use_default_kpoints=self.use_default_kpoints,
                number_of_docs=self.number_of_docs,
                user_incar_settings=self.user_incar_settings,
                user_kpoints_settings=self.user_kpoints_settings,
                user_potcar_functional=self.user_potcar_functional,
                **self.extra_kwargs,
            )

        input_set.write_input(output_dir)
        return str(output_dir)

    def write_neb(
        self,
        output_dir: Union[str, Path],
        start_structure: Union[str, Structure, None] = None,
        end_structure: Union[str, Structure, None] = None,
        intermediate_structures: Optional[List[Structure]] = None,
        n_images: Optional[int] = None,
    ) -> str:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        final_start = self._get_final_structure(
            start_structure if start_structure is not None else self.start_structure
        )
        final_end = self._get_final_structure(
            end_structure if end_structure is not None else self.end_structure
        )

        input_obj = NEBSetEcat(
            start_structure=final_start,
            end_structure=final_end,
            n_images=int(n_images) if n_images is not None else int(self.n_images),
            intermediate_structures=intermediate_structures if intermediate_structures is not None else self.intermediate_structures,
            functional=self.functional.upper(),
            use_default_incar=self.use_default_incar,
            user_incar_settings=self.user_incar_settings,
            user_kpoints_settings=self.user_kpoints_settings,
            user_potcar_functional=self.user_potcar_functional,
            **self.extra_kwargs,
        )

        input_obj.write_input(output_dir)
        return str(output_dir)

    def write_lobster(
        self,
        output_dir: Union[str, Path],
        structure: Union[str, Structure, None] = None,
        prev_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        output_dir = self._ensure_dir(output_dir)
        prev_dir_path = self._resolve_prev_dir(prev_dir)

        if prev_dir_path is not None:
            input_set = LobsterSetEcat.from_prev_calc_ecat(
                prev_dir=prev_dir_path,
                kpoints_density=self.kpoints_density,
                user_incar_settings=self.user_incar_settings,
                user_supplied_basis=self.user_supplied_basis,
                **self.extra_kwargs,
            )
        else:
            input_set = LobsterSetEcat(
                structure=self._get_final_structure(structure),
                functional=self.functional,
                isym=self.isym,
                ismear=self.ismear,
                reciprocal_density=self.reciprocal_density,
                user_supplied_basis=self.user_supplied_basis,
                use_default_incar=self.use_default_incar,
                user_incar_settings=self.user_incar_settings,
                user_kpoints_settings=self.user_kpoints_settings,
                user_potcar_functional=self.user_potcar_functional,
                **self.extra_kwargs,
            )

        input_set.write_input(output_dir)

        lb = LobsterSetEcat.standard_calculations_from_vasp_files(
            POSCAR_input=output_dir / "POSCAR",
            INCAR_input=output_dir / "INCAR",
            POTCAR_input=output_dir / "POTCAR",
        )
        lb.write_lobsterin(output_dir / "lobsterin", overwritedict=self.overwritedict)
        return str(output_dir)

    def write_adsorption(
        self,
        output_dir: Union[str, Path],
        structure: Union[str, Structure],
        prev_dir: Union[str, Path],
    ) -> str:
        output_dir = Path(output_dir).resolve()
        if prev_dir is None and self.prev_dir is not None:
            prev_dir = self.prev_dir
        prev_dir_path = Path(prev_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        input_set = SlabSetEcat.ads_from_prev_calc(
            structure=structure,
            prev_dir=prev_dir_path,
            user_incar_settings=self.user_incar_settings,
            user_kpoints_settings=self.user_kpoints_settings,
            auto_dipole=self.auto_dipole,
            user_potcar_functional=self.user_potcar_functional,
            **self.extra_kwargs,
        )
        input_set.write_input(output_dir)
        return str(output_dir)

    def write_freq(
        self,
        output_dir: Union[str, Path],
        prev_dir: Optional[Union[str, Path]] = None,
        structure: Union[str, Structure, None] = None,
        mode: str = "inherit",
        vibrate_indices: Optional[List[int]] = None,
        adsorbate_formula: Optional[str] = None,
        adsorbate_formula_prefer: str = "tail",
    ) -> str:
        output_dir = self._ensure_dir(output_dir)
        prev_dir_path = self._resolve_prev_dir(prev_dir)
        if prev_dir_path is None:
            raise ValueError("Frequency calculation requires 'prev_dir' (needs CONTCAR/INCAR/KPOINTS).")

        if structure is not None:
            final_structure = self._get_final_structure(structure)
        else:
            contcar_path = prev_dir_path / "CONTCAR"
            if not contcar_path.exists():
                raise FileNotFoundError(f"CONTCAR not found in {prev_dir_path}")
            final_structure = Structure.from_file(contcar_path)

        final_vibrate_indices: Optional[List[int]] = None
        inherit_sd = False

        if vibrate_indices is not None:
            final_vibrate_indices = vibrate_indices
            inherit_sd = False
        elif adsorbate_formula is not None:
            final_vibrate_indices = pick_adsorbate_indices_by_formula_strict(
                final_structure,
                adsorbate_formula=adsorbate_formula,
                prefer=adsorbate_formula_prefer,
            )
            inherit_sd = False
            logger.info(
                "Picked vibrate indices by formula %s: %s",
                adsorbate_formula,
                final_vibrate_indices,
            )
        else:
            if mode == "all":
                final_vibrate_indices = list(range(len(final_structure)))
                inherit_sd = False
            elif mode == "inherit":
                sd = final_structure.site_properties.get("selective_dynamics")
                has_sd = isinstance(sd, list) and len(sd) == len(final_structure)
                if not has_sd:
                    raise ValueError(
                        "mode='inherit' requires selective_dynamics in CONTCAR, but none was found. "
                        "Provide vibrate_indices explicitly, or provide adsorbate_formula (non-ambiguous), "
                        "or switch to mode='all'."
                    )
                final_vibrate_indices = None
                inherit_sd = True
            elif mode == "adsorbate":
                raise ValueError(
                    "mode='adsorbate' auto-detect is disabled in strict mode. "
                    "Provide adsorbate_formula (preferred) or vibrate_indices (best)."
                )
            else:
                raise ValueError("mode must be one of: 'inherit', 'all', 'adsorbate'.")

        input_set = FreqSetEcat.from_prev_calc_ecat(
            prev_dir=prev_dir_path,
            structure=final_structure,
            vibrate_indices=final_vibrate_indices,
            inherit_sd=inherit_sd,
            user_incar_settings=getattr(self, "user_incar_settings", None),
            user_kpoints_settings=getattr(self, "user_kpoints_settings", None),
            **getattr(self, "extra_kwargs", {}),
        )
        input_set.write_input(output_dir)
        return str(output_dir)
