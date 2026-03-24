# -*- coding: utf-8 -*-
"""封装各类 VASP 输入集（基于 pymatgen VaspInputSet）。"""

from pathlib import Path
import logging
import subprocess
import shutil
import numpy as np
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Union

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.sets import (
    LobsterSet,
    MPMetalRelaxSet,
    MPStaticSet,
    MVLSlabSet,
    NEBSet,
)

logger = logging.getLogger(__name__)

from .constants import (
    DEFAULT_INCAR_BULK,
    DEFAULT_INCAR_FREQ,
    DEFAULT_INCAR_LOBSTER,
    DEFAULT_INCAR_NEB,
    DEFAULT_INCAR_SLAB,
    DEFAULT_INCAR_STATIC,
    DEFAULT_INCAR_DIMER,
    DEFAULT_INCAR_NBO,
    DEFAULT_NBO_CONFIG_PARAMS,
    NBO_CONFIG_TEMPLATE,
    NBO_BASIS_PATH,
    _BEEF_INCAR,
)
from .kpoints import build_kpoints_by_lengths
from .utils import (
    convert_vasp_format_to_pymatgen_dict,
    infer_functional_from_incar,
    load_structure,
)


class VaspInputSetEcat:
    """Shared helper methods for ECAT-style VASP input sets."""

    @staticmethod
    def _load_structure(struct_source: Union[str, Path, Structure]) -> Structure:
        return load_structure(struct_source)

    def write_input(self, output_dir: Union[str, Path], *args: Any, **kwargs: Any):
        """Wrap write_input to enforce safe LDAU behavior.

        Some upstream VASP input set implementations may default LDAU to True via their
        internal YAML defaults (e.g., MPMetalRelaxSet). In that case, `self.user_incar_settings`
        does not contain the final INCAR values, so we must patch the written file.
        """
        result = super().write_input(output_dir, *args, **kwargs)

        # Post-process the written INCAR to ensure +U is not enabled by accident.
        out_path = Path(output_dir)
        incar_path = out_path / "INCAR"
        if incar_path.exists():
            incar = Incar.from_file(incar_path)
            if self._should_disable_ldau(incar):
                logger.warning(
                    "Detected LDAU=True in written INCAR without any LDAUL/LDAUU/LDAUJ; forcing LDAU=False."
                )
                incar["LDAU"] = False
                incar.write_file(incar_path)

        return result

    @classmethod
    def _build_incar(
        cls, functional: str,
        default_incar: Optional[Dict[str, Any]] = None,
        extra_incar: Optional[Dict[str, Any]] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build INCAR dict, optionally applying defaults and user overrides."""

        incar: Dict[str, Any] = {}
        if default_incar:
            incar.update(default_incar)

        if "BEEF" in functional.upper():
            incar.update(_BEEF_INCAR)

        if extra_incar:
            incar.update(extra_incar)

        if user_incar_settings:
            incar.update(user_incar_settings)

        # 如果用户开启了 LDAU，但未提供 U 值列表，则告警并将 LDAU 设为 False（等同于未开启 +U）。
        if cls._should_disable_ldau(incar):
            logger.warning(
                "LDAU=True but no LDAUL/LDAUU/LDAUJ provided; setting LDAU=False (no +U)."
            )
            incar["LDAU"] = False

        return incar

    @staticmethod
    def _should_disable_ldau(incar: Dict[str, Any]) -> bool:
        """Determine whether we should turn off LDAU when no U values are provided.

        This protects users who explicitly set LDAU=True but forget to set any of the
        U-related parameters (LDAUL/LDAUU/LDAUJ). In that case, VASP will still treat
        it as +U enabled but with undefined U values, which is usually unintended.
        """
        if not incar.get("LDAU"):
            return False

        # 允许多种布尔写法
        val = incar.get("LDAU")
        if isinstance(val, str):
            val = val.strip().upper()
            if val in {"FALSE", ".FALSE.", "0"}:
                return False

        # 如果存在任何有效的 U 相关字段，我们认为是有意使用 +U
        def is_valid_u_list(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, (list, tuple)) and len(value) == 0:
                return False
            if isinstance(value, str) and value.strip() == "":
                return False
            return True

        for k in ("LDAUL", "LDAUU", "LDAUJ"):
            if k in incar and is_valid_u_list(incar[k]):
                return False

        return True

    @classmethod
    def _resolve_kpoints(
        cls,
        structure: Structure,
        use_default_kpoints: bool,
        user_kpoints_settings: Optional[Any],
        default_density: Union[int, float, Sequence[float]],
    ) -> Optional[Kpoints]:
        if user_kpoints_settings is not None:
            return user_kpoints_settings
        if use_default_kpoints:
            return cls._make_kpoints_from_density(structure, default_density, style=2)
        return None

    @classmethod
    def _read_and_convert_incar(
        cls, incar_path: Union[str, Path], structure: Structure
    ) -> tuple[Dict[str, Any], str]:
        incar_path = Path(incar_path)
        incar = {
            **(Incar.from_file(incar_path).as_dict() if incar_path.exists() else {})
        }
        functional = infer_functional_from_incar(incar)

        format_keys = ["MAGMOM", "LDAUU", "LDAUJ", "LDAUL"]
        converted_params: Dict[str, Any] = {}
        for key in format_keys:
            value = incar.get(key)
            if value is None or isinstance(value, dict):
                continue
            conversion_result = convert_vasp_format_to_pymatgen_dict(structure, key, value)
            if conversion_result:
                converted_params.update(conversion_result)
        for key in format_keys:
            incar.pop(key, None)
        incar.update(converted_params)

        return incar, functional

    @classmethod
    def _make_kpoints_from_density(
        cls,
        structure: Structure,
        kpoints_density: Optional[Union[int, float, Sequence[float]]],
        style: int = 2,
    ) -> Optional[Kpoints]:
        if kpoints_density is None:
            return None
        if isinstance(kpoints_density, (int, float)):
            densities = [float(kpoints_density)] * 3
        elif isinstance(kpoints_density, (list, tuple)) and len(kpoints_density) == 3:
            densities = [float(x) for x in kpoints_density]
        else:
            raise ValueError("kpoints_density must be a number or a sequence of 3 numbers.")
        return build_kpoints_by_lengths(structure=structure, length_densities=densities, style=style)


class SlabSetEcat(VaspInputSetEcat, MVLSlabSet):
    """Slab 计算输入集（MVLSlabSet + ECAT 默认值）。"""

    def __init__(
        self,
        structure: Union[str, Structure],
        functional: str = "PBE",
        kpoints_density: int = 25,
        use_default_incar: bool = True,
        use_default_kpoints: bool = True,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        auto_dipole: bool = True,
        **extra_kwargs,
    ):
        loaded_structure = self._load_structure(structure)
        self.functional = functional.upper()

        incar = self._build_incar(
            self.functional,
            DEFAULT_INCAR_SLAB if use_default_incar else None,
            user_incar_settings=user_incar_settings,
        )
        
        kpoints = self._resolve_kpoints(
            loaded_structure, use_default_kpoints, 
            user_kpoints_settings, [kpoints_density, kpoints_density, 1]
        )

        super().__init__(
            structure=loaded_structure,
            auto_dipole=auto_dipole,
            user_incar_settings=incar,
            user_kpoints_settings=kpoints,
            **extra_kwargs,
        )

    @classmethod
    def ads_from_prev_calc(
        cls,
        structure: Union[str, Structure],
        prev_dir: Union[str, Path],
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        auto_dipole: bool = True,
        **extra_kwargs,
    ):
        prev_dir = Path(prev_dir).resolve()
        incar_path = prev_dir / "INCAR"
        if not incar_path.exists():
            raise FileNotFoundError(f"INCAR not found in prev_dir: {prev_dir}")

        loaded_prev_structure = Structure.from_file(
            prev_dir / "CONTCAR" if (prev_dir / "CONTCAR").exists() else prev_dir / "POSCAR"
        )

        base_incar, functional = cls._read_and_convert_incar(incar_path, loaded_prev_structure)

        kpoints = user_kpoints_settings
        if kpoints is None and (prev_dir / "KPOINTS").exists():
            kpoints = Kpoints.from_file(prev_dir / "KPOINTS")

        init_kwargs = extra_kwargs.copy()
        init_kwargs.update(
            {
                "functional": functional,
                "structure": structure,
                "use_default_incar": False,
                "use_default_kpoints": False,
                "auto_dipole": auto_dipole,
                "user_incar_settings": cls._build_incar(functional, base_incar, user_incar_settings=user_incar_settings),
                "user_kpoints_settings": kpoints,
            }
        )

        return cls(**init_kwargs)


class BulkRelaxSetEcat(VaspInputSetEcat, MPMetalRelaxSet):
    """Bulk 结构松弛输入集（MPMetalRelaxSet + ECAT 默认值）。"""

    def __init__(
        self,
        structure: Union[str, Structure],
        functional: str = "PBE",
        is_metal: bool = True,
        kpoints_density: int = 25,
        use_default_incar: bool = True,
        use_default_kpoints: bool = True,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        **extra_kwargs,
    ):
        self.functional = functional.upper()
        loaded_structure = self._load_structure(structure)

        extra_incar = {"ISMEAR": 1, "SIGMA": 0.20} if is_metal else {"ISMEAR": 0, "SIGMA": 0.05}
        incar = self._build_incar(
            self.functional,
            DEFAULT_INCAR_BULK if use_default_incar else None,
            extra_incar=extra_incar,
            user_incar_settings=user_incar_settings
        )

        kpoints = self._resolve_kpoints(
            loaded_structure, use_default_kpoints, user_kpoints_settings, kpoints_density
        )

        super().__init__(
            structure=loaded_structure,
            user_incar_settings=incar,
            user_kpoints_settings=kpoints,
            **extra_kwargs,
        )


class MPStaticSetEcat(VaspInputSetEcat, MPStaticSet):
    """单点计算（MPStaticSet + ECAT 默认值）。"""

    def __init__(
        self,
        structure: Union[str, Structure],
        functional: str = "PBE",
        use_default_incar: bool = True,
        use_default_kpoints: bool = True,
        number_of_docs: Optional[int] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        **extra_kwargs,
    ):
        self.functional = functional.upper()
        loaded_structure = self._load_structure(structure)

        extra_incar = {"NEDOS": number_of_docs} if number_of_docs is not None else None
        incar = self._build_incar(
            self.functional,
            DEFAULT_INCAR_STATIC if use_default_incar else None,
            extra_incar=extra_incar,
            user_incar_settings=user_incar_settings
        )

        kpoints = self._resolve_kpoints(
            loaded_structure, use_default_kpoints, user_kpoints_settings, [40, 40, 40]
        )

        super().__init__(
            structure=loaded_structure,
            user_incar_settings=incar,
            user_kpoints_settings=kpoints,
            **extra_kwargs,
        )

    @classmethod
    def from_prev_calc_ecat(
        cls,
        prev_dir: Union[str, Path],
        kpoints_density: int = 50,
        number_of_docs: Optional[int] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        **extra_kwargs,
    ):
        prev_dir = Path(prev_dir).resolve()
        loaded_structure = cls._load_structure(prev_dir)
        base_incar, functional = cls._read_and_convert_incar(prev_dir / "INCAR", loaded_structure)

        # 从 prev INCAR 继承全部设置，然后用静态计算的默认值覆盖（比如 IBRION/NSW）
        merged_incar = {**base_incar, **DEFAULT_INCAR_STATIC}
        if user_incar_settings:
            merged_incar.update(user_incar_settings)

        if number_of_docs is not None:
            merged_incar["NEDOS"] = int(number_of_docs)

        incar = cls._build_incar(functional, None, user_incar_settings=merged_incar)

        # Prev calc 模式：使用 prev INCAR 中解析到的 functional，避免任何额外 kwargs 干扰。
        init_kwargs = extra_kwargs.copy()
        init_kwargs.update(
            {
                "structure": loaded_structure,
                "functional": functional,
                "use_default_incar": False,
                "use_default_kpoints": False,
                "number_of_docs": number_of_docs,
                "user_incar_settings": incar,
                "user_kpoints_settings": user_kpoints_settings
                or cls._make_kpoints_from_density(loaded_structure, kpoints_density),
            }
        )
        return cls(**init_kwargs)


class LobsterSetEcat(VaspInputSetEcat, LobsterSet):
    def __init__(
        self,
        structure: Union[str, Structure],
        functional: str = "PBE",
        isym: int = 0,
        ismear: int = -5,
        reciprocal_density: Optional[int] = None,
        user_supplied_basis: Optional[dict] = None,
        use_default_incar: bool = True,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        **extra_kwargs,
    ):
        self.functional = functional.upper()
        self.user_supplied_basis = user_supplied_basis 
        loaded_structure = self._load_structure(structure)

        incar = self._build_incar(
            self.functional,
            DEFAULT_INCAR_LOBSTER if use_default_incar else None,
            user_incar_settings=user_incar_settings
        )

        super().__init__(
            structure=loaded_structure,
            isym=isym,
            ismear=ismear,
            reciprocal_density=reciprocal_density,
            user_supplied_basis=user_supplied_basis,
            user_incar_settings=incar,
            user_kpoints_settings=user_kpoints_settings,
            **extra_kwargs,
        )

    def write_input(self, output_dir, overwritedict: Optional[Dict[str, Any]] = None, 
                    custom_lobsterin_lines: Optional[List[str]] = None,
                    *args, **kwargs):
        super().write_input(output_dir, *args, **kwargs)
        output_dir = Path(output_dir).resolve()
        from pymatgen.io.lobster.inputs import Lobsterin
        try:
            lb = Lobsterin.standard_calculations_from_vasp_files(
                POSCAR_input=output_dir / "POSCAR",
                INCAR_input=output_dir / "INCAR",
                POTCAR_input=output_dir / "POTCAR",
                dict_for_basis=self.user_supplied_basis,
            )
            if overwritedict:
                lb.update(overwritedict)
            lobsterin_path = output_dir / "lobsterin"
            lb.write_lobsterin(lobsterin_path)
            if custom_lobsterin_lines:
                with open(lobsterin_path, "a", encoding="utf-8") as f:
                    f.write("\n! --- Custom User Lines ---\n")
                    for line in custom_lobsterin_lines:
                        f.write(f"{line}\n")
        except Exception as e:
            logging.getLogger(__name__).warning(f"生成 lobsterin 文件失败: {e}")

    @classmethod
    def from_prev_calc_ecat(
        cls,
        prev_dir: Union[str, Path],
        kpoints_density: int = 50,
        isym: int = 0,
        ismear: int = -5,
        reciprocal_density: Optional[int] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        user_supplied_basis: Optional[dict] = None,
        **extra_kwargs,
    ):
        prev_dir = Path(prev_dir).resolve()
        loaded_structure = cls._load_structure(prev_dir)
        base_incar, functional = cls._read_and_convert_incar(prev_dir / "INCAR", loaded_structure)

        incar = cls._build_incar(
            functional,
            base_incar,
            extra_incar=DEFAULT_INCAR_LOBSTER,
            user_incar_settings=user_incar_settings,
        )
        if user_kpoints_settings is not None:
            final_kpts = user_kpoints_settings
            final_recip = None
        elif reciprocal_density is not None:
            final_kpts = None
            final_recip = reciprocal_density
        else:
            final_kpts = cls._make_kpoints_from_density(loaded_structure, kpoints_density)
            final_recip = None
        init_kwargs = extra_kwargs.copy()
        init_kwargs.update(
            {
                "structure": loaded_structure,
                "functional": functional,
                "isym": isym,
                "ismear": ismear,
                "reciprocal_density": final_recip,
                "user_supplied_basis": user_supplied_basis,
                "use_default_incar": False,
                "user_incar_settings": incar,
                "user_kpoints_settings": final_kpts,
            }
        )

        return cls(**init_kwargs)

class NEBSetEcat(VaspInputSetEcat, NEBSet):
    def __init__(
        self,
        start_structure: Union[str, Structure],
        end_structure: Union[str, Structure],
        n_images: int = 6,
        intermediate_structures: Optional[List[Structure]] = None,
        functional: str = "PBE",
        use_default_incar: bool = True,
        use_idpp: bool = True,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Dict[str, Any]] = None,
        **extra_kwargs,
    ):
        start_structure = self._load_structure(start_structure)
        end_structure = self._load_structure(end_structure)
        self.functional = functional.upper()

        if intermediate_structures is None:
            if use_idpp:
                try:
                    from pymatgen.analysis.diffusion.neb.pathfinder import IDPPSolver
                    solver = IDPPSolver.from_endpoints(
                        [start_structure, end_structure], n_images=n_images + 2, sort_tol=0.1
                    )
                    intermediate_structures = solver.run(maxiter=2000, tol=1e-5, species=start_structure.species)
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        "IDPPSolver failed (%s). Falling back to linear interpolation.", e
                    )
                    intermediate_structures = start_structure.interpolate(end_structure, n_images + 1)
            else:
                intermediate_structures = start_structure.interpolate(end_structure, n_images + 1)

        incar = self._build_incar(
            self.functional,
            DEFAULT_INCAR_NEB if use_default_incar else None,
            user_incar_settings=user_incar_settings
        )

        super().__init__(
            structures=intermediate_structures,
            user_incar_settings=incar,
            user_kpoints_settings=user_kpoints_settings,
            **extra_kwargs,
        )

    @classmethod
    def from_prev_calc(
        cls,
        prev_dir: Union[str, Path],
        start_structure: Union[str, Structure, Path],
        end_structure: Union[str, Structure, Path],
        n_images: int = 6,
        use_idpp: bool =True,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        **extra_kwargs,
    ):
        prev_dir = Path(prev_dir).resolve()
        
        try:
            base_incar = Incar.from_file(prev_dir / "INCAR")
        except FileNotFoundError:
            logging.getLogger(__name__).warning(f"INCAR not found in {prev_dir}, using default.")
            base_incar = Incar()
        
        try:
            base_kpoints = Kpoints.from_file(prev_dir / "KPOINTS")
        except FileNotFoundError:
            base_kpoints = None
            
        final_incar = dict(base_incar)
        if DEFAULT_INCAR_NEB:
            final_incar.update(DEFAULT_INCAR_NEB)
        if user_incar_settings:
            final_incar.update(user_incar_settings)

        final_kpoints = user_kpoints_settings if user_kpoints_settings is not None else base_kpoints

        init_kwargs = extra_kwargs.copy()
        init_kwargs.update(
            {
                "start_structure": start_structure,  # 直接传原始参数
                "end_structure": end_structure,      # 直接传原始参数
                "n_images": n_images,
                "use_idpp": use_idpp,
                "use_default_incar": False,  
                "user_incar_settings": final_incar,
                "user_kpoints_settings": final_kpoints,
            }
        )
        return cls(**init_kwargs)                  
    

class FreqSetEcat(MPStaticSetEcat):
    @staticmethod
    def _apply_vibrate_indices(structure: Structure, vibrate_indices: List[int]) -> Structure:
        n = len(structure)
        bad = [i for i in vibrate_indices if (not isinstance(i, int)) or i < 0 or i >= n]
        if bad:
            raise IndexError(f"vibrate_indices out of range for {n} sites: {bad}")

        structure = structure.copy()
        sd = [[False, False, False] for _ in range(n)]
        for idx in vibrate_indices:
            sd[idx] = [True, True, True]

        structure.add_site_property("selective_dynamics", sd)
        return structure

    def __init__(
        self,
        structure: Union[str, Structure, Path],
        functional: str = "PBE",
        use_default_incar: bool = True,
        use_default_kpoints: bool = False,
        kpoints_density: Optional[Union[int, float, Sequence[float]]] = None,
        calc_ir: bool = False,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        **extra_kwargs,
    ):
        loaded_structure = self._load_structure(structure)
        functional = (functional or "PBE").upper()
        ir_tags = {"LEPSILON": True, "NWRITE": 3, "IBRION": 7} if calc_ir else {}
        base_freq_incar = {**DEFAULT_INCAR_FREQ, **ir_tags}
        incar = self._build_incar(
            functional,
            base_freq_incar if use_default_incar else None,
            user_incar_settings=user_incar_settings
        )

        kpoints = self._resolve_kpoints(
            loaded_structure, use_default_kpoints, user_kpoints_settings, kpoints_density or 25
        )

        super().__init__(
            structure=loaded_structure,
            functional=functional,
            use_default_incar=False,
            use_default_kpoints=False,
            user_incar_settings=incar,
            user_kpoints_settings=kpoints,
            **extra_kwargs,
        )

    @classmethod
    def from_prev_calc_ecat(
        cls,
        prev_dir: Union[str, Path],
        structure: Optional[Union[str, Structure, Path]] = None,
        vibrate_indices: Optional[List[int]] = None,
        calc_ir: bool = False,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        **extra_kwargs,
    ):
        prev_dir = Path(prev_dir).resolve()
        if structure is None:
            loaded_structure = cls._load_structure(prev_dir)
        else:
            loaded_structure = cls._load_structure(structure) 

        if vibrate_indices is not None:
            loaded_structure = cls._apply_vibrate_indices(loaded_structure, vibrate_indices)

        base_incar, functional = cls._read_and_convert_incar(prev_dir / "INCAR", loaded_structure)
        
        for k in ["IBRION", "NSW", "POTIM", "EDIFF", "EDIFFG", "ISIF", "NPAR", "NCORE"]:
            base_incar.pop(k, None)
        ir_tags = {"LEPSILON": True, "NWRITE": 3, "IBRION": 7} if calc_ir else {}
        extra_incar_combined = {**DEFAULT_INCAR_FREQ, **ir_tags}
        incar = cls._build_incar(
            functional, 
            base_incar, 
            extra_incar=extra_incar_combined, 
            user_incar_settings=user_incar_settings
        )
        

        kpoints = user_kpoints_settings
        if kpoints is None:
            try:
                kpoints = Kpoints.from_file(prev_dir / "KPOINTS")
            except FileNotFoundError:
                logging.getLogger(__name__).warning(f"KPOINTS not found in {prev_dir}, will generate default.")
                kpoints = None

        init_kwargs = extra_kwargs.copy()
        init_kwargs.update(
            {
                "structure": loaded_structure,
                "functional": functional,
                "use_default_incar": False,
                "use_default_kpoints": False,
                "user_incar_settings": incar,
                "user_kpoints_settings": kpoints,
            }
        )

        return cls(**init_kwargs)
    
class DimerSetEcat(MPStaticSetEcat):
    """
    通过调用底层 VTST 脚本生成 Dimer 计算输入文件。
    自动读取 IMAGES、补全端点 OUTCAR、覆盖 POSCAR。
    """

    def __init__(
        self,
        structure: Union[str, Structure, Path],
        modecar: Union[str, Path, np.ndarray],
        **kwargs,
    ):
        #MODECAR以数组的形式存入内存最为合适。
        if modecar is None:
            raise ValueError("CRITICAL ERROR: 'modecar' must be provided! ")
        if isinstance(modecar, (str, Path)):
            modecar_path = Path(modecar).resolve()
            if not modecar_path.exists():
                raise FileNotFoundError(f"Provided MODECAR file does not exist: {modecar_path}")        
            self.modecar_data = np.loadtxt(modecar_path)
        elif isinstance(modecar, np.ndarray):
            self.modecar_data = modecar
        else:
            raise TypeError("'modecar' must be either a numpy array or a valid file path.")
        super().__init__(structure=structure, **kwargs)

    def write_input(self, output_dir: Union[str, Path], **kwargs):
        super().write_input(output_dir, **kwargs)
        
        modecar_path = Path(output_dir) / "MODECAR"
        with open(modecar_path, "w") as f:
            for row in self.modecar_data:
                f.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
        logger.info(f"MODECAR successfully written to {modecar_path}")

    @classmethod
    def from_neb_calc(
        cls,
        neb_dir: Union[str, Path],
        num_images: Optional[int] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        **extra_kwargs,
    ):
        neb_dir = Path(neb_dir).resolve()
        incar_path = neb_dir / "INCAR"
        
        if num_images is not None:
            logger.info(f"User specified num_images={num_images}. Skipping INCAR reading.")
        else:
            logger.info("Auto mode: Reading INCAR to determine IMAGES...")
            if not incar_path.exists():
                raise FileNotFoundError(f"INCAR not found in {neb_dir}")
                
            neb_incar = Incar.from_file(incar_path)
            raw_images = neb_incar.get("IMAGES")
            
            if raw_images is None:
                raise ValueError("IMAGES tag not found in INCAR. Is this a valid NEB directory?")
            
            # 清洗注释
            if isinstance(raw_images, str):
                clean_images = raw_images.split('#')[0].split('!')[0].strip()
            else:
                clean_images = raw_images
                
            try:
                num_images = int(clean_images)
            except ValueError:
                raise ValueError(f"Failed to parse IMAGES value '{raw_images}' as an integer.")
                
        num2 = num_images + 1

        #使用沙盒机制，在临时文件夹中执行VTST脚本
        logger.info("Creating a temporary sandbox to protect original NEB directory...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            if incar_path.exists():
                shutil.copy(incar_path, tmp_path / "INCAR")
            for i in range(num2 + 1):
                src_d = neb_dir / f"{i:02d}"
                dst_d = tmp_path / f"{i:02d}"
                dst_d.mkdir()
                
                outcar_src = src_d / "OUTCAR"
                if not outcar_src.exists():
                    outcar_src = src_d / "OUTCAR.gz"                
                if not outcar_src.exists():
                    if i == 0:
                        outcar_src = neb_dir / "01" / "OUTCAR"
                    elif i == num2:
                        outcar_src = neb_dir / f"{num_images:02d}" / "OUTCAR"
                if outcar_src.exists():
                    shutil.copy(outcar_src, dst_d / "OUTCAR")

                # --- 复制 CONTCAR 并重命名为 POSCAR ---
                contcar_src = src_d / "CONTCAR"
                if not contcar_src.exists():
                    contcar_src = src_d / "CONTCAR.gz"
                    
                if contcar_src.exists() and contcar_src.stat().st_size > 0:
                    shutil.copy(contcar_src, dst_d / "POSCAR")
                else:
                    # 如果 CONTCAR 不存在或为空，退化使用 POSCAR
                    poscar_src = src_d / "POSCAR"
                    if poscar_src.exists():
                        shutil.copy(poscar_src, dst_d / "POSCAR")
            # 在沙盒中调用底层 VTST 脚本
            try:
                logger.info("Running nebresults.pl in sandbox...")
                subprocess.run(["nebresults.pl"], cwd=tmp_path, check=True, capture_output=True)
                
                logger.info("Running neb2dim.pl in sandbox...")
                subprocess.run(["neb2dim.pl"], cwd=tmp_path, check=True, capture_output=True)
            except FileNotFoundError:
                raise RuntimeError("VTST scripts (nebresults.pl, neb2dim.pl) not found in system PATH.")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"VTST script execution failed in sandbox:\n{e.stderr.decode()}")

            # 从沙盒中提取数据到内存
            dim_dir = tmp_path / "dim"
            if not dim_dir.exists():
                raise FileNotFoundError(f"neb2dim.pl failed to create the 'dim' directory in sandbox.")

            # 提取 POSCAR 结构
            saddle_struct = cls._load_structure(dim_dir / "POSCAR")
            
            # 提取 MODECAR 数据为 numpy array
            modecar_file = dim_dir / "MODECAR"
            if not modecar_file.exists():
                raise FileNotFoundError(f"MODECAR not found in sandbox.")
            modecar_data = np.loadtxt(modecar_file)
            logger.info("Sandbox cleaned up successfully. Original NEB directory is untouched.")

        base_incar, functional = cls._read_and_convert_incar(incar_path, saddle_struct)
        
        tags_to_remove = [
            "IMAGES", "SPRING", "LCLIMB", "ICHAIN", 
            "IBRION", "NSW", "POTIM", "EDIFF", "IOPT", "##NEB"
        ]
        for tag in tags_to_remove:
            base_incar.pop(tag, None)
            
        incar = cls._build_incar(
            functional,
            base_incar,
            extra_incar=DEFAULT_INCAR_DIMER,
            user_incar_settings=user_incar_settings
        )
        
        kpoints = user_kpoints_settings
        if kpoints is None:
            try:
                kpoints = Kpoints.from_file(neb_dir / "KPOINTS")
            except FileNotFoundError:
                logger.warning(f"KPOINTS not found in {neb_dir}, will generate default.")
                kpoints = None

        init_kwargs = extra_kwargs.copy()
        init_kwargs.update(
            {
                "structure": saddle_struct,
                "functional": functional,
                "use_default_incar": False,
                "use_default_kpoints": False,
                "user_incar_settings": incar,
                "user_kpoints_settings": kpoints,
                "modecar": modecar_data,
            }
        )
            
        return cls(**init_kwargs)
    
class NBOSetEcat(MPStaticSetEcat):
    """
    用于 VASP-NBO 计算的输入文件生成器。
    支持自动解析全局基组文件，生成严格符合 Fortran 格式的 nbo.config。
    """

    def __init__(
        self,
        structure: Union[str, Structure, Path],
        basis_source: Union[str, Path, Dict[str, str], None] = None,
        nbo_config: Optional[Dict[str, Any]] = None,
        prev_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        # 1. 初始化 NBO 专属参数
        self.nbo_config_params = {**DEFAULT_NBO_CONFIG_PARAMS, **(nbo_config or {})}
        self.prev_dir = Path(prev_dir).resolve() if prev_dir else None

        loaded_structure = self._load_structure(structure)

        # 2. 解析基组
        if basis_source is None:
            basis_source = NBO_BASIS_PATH
            logger.info(f"Using default basis set from: {basis_source}")

        if isinstance(basis_source, dict):
            self.basis_settings = basis_source
        else:
            logger.info("Parsing master basis set file/string...")
            self.basis_settings = self._parse_basis_file(basis_source)

        elements_in_struct = {str(el) for el in loaded_structure.composition.elements}
        missing_elements = elements_in_struct - set(self.basis_settings.keys())
        if missing_elements:
            raise ValueError(
                f"CRITICAL: Missing basis set definitions for elements: {missing_elements}. "
            )

        # 3. 纯粹地调用父类初始化
        super().__init__(structure=loaded_structure, **kwargs)

        ispin = int(self.incar.get("ISPIN", 1))
        if ispin == 2:
            logger.info("ISPIN=2 detected in INCAR. Halving occ_1c (LP) and occ_2c (DP) cutoffs.")
            for key in ["occ_1c", "occ_2c"]:
                if key in self.nbo_config_params:
                    orig_val = float(self.nbo_config_params[key])
                    new_val = orig_val / 2.0
                    formatted_val = f"{new_val:.3f}".rstrip('0').rstrip('.')
                    self.nbo_config_params[key] = formatted_val
                    logger.debug(f"Adjusted {key}: {orig_val} -> {formatted_val}")

    @classmethod
    def from_prev_calc(
        cls,
        prev_dir: Union[str, Path],
        basis_source: Union[str, Path, Dict[str, str], None] = None,
        nbo_config: Optional[Dict[str, Any]] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        **extra_kwargs,
    ):
        """
        从前一步计算（如静态计算或优化）继承参数，生成 NBO 计算任务。
        """
        prev_dir = Path(prev_dir).resolve()
        loaded_structure = cls._load_structure(prev_dir)
        incar_path = prev_dir / "INCAR"
        
        # 1. 继承并修改 INCAR
        if not incar_path.exists():
            raise FileNotFoundError(f"INCAR not found in {prev_dir}")
            
        # 读取旧 INCAR 和泛函
        base_incar, functional = cls._read_and_convert_incar(incar_path, loaded_structure)
        
        incar = cls._build_incar(
            functional,
            base_incar,
            extra_incar=DEFAULT_INCAR_NBO,
            user_incar_settings=user_incar_settings
        )
        
        # 2. 继承 KPOINTS
        kpoints = user_kpoints_settings
        if kpoints is None:
            try:
                kpoints = Kpoints.from_file(prev_dir / "KPOINTS")
                logger.info(f"Inheriting KPOINTS from {prev_dir}")
            except FileNotFoundError:
                logger.warning(f"KPOINTS not found in {prev_dir}, will generate default.")
                kpoints = None

        final_struct = extra_kwargs.pop("structure", loaded_structure) 
        init_kwargs = extra_kwargs.copy()
        init_kwargs.update(
            {
                "structure": final_struct,
                "functional": functional,
                "basis_source": basis_source,
                "nbo_config": nbo_config,
                "prev_dir": prev_dir,
                "use_default_incar": False,    
                "use_default_kpoints": False,
                "user_incar_settings": incar,  
                "user_kpoints_settings": kpoints,
            }
        )
            
        return cls(**init_kwargs)

    def write_input(self, output_dir: Union[str, Path], **kwargs):
        """写入 VASP 输入文件、NBO 专属文件，并拷贝波函数"""
        super().write_input(output_dir, **kwargs)
        output_dir = Path(output_dir).resolve()
        
        self._write_nbo_config(output_dir / "nbo.config")
        self._write_basis_inp(output_dir / "basis.inp")
        
        logger.info(f"NBO specific input files written to {output_dir}")

    @staticmethod
    def _parse_basis_file(source: Union[str, Path]) -> Dict[str, str]:
        basis_dict = {}
        if isinstance(source, Path) or (isinstance(source, str) and Path(source).is_file()):
            with open(source, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            lines = source.splitlines()

        current_element = None
        current_basis = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('!'):
                continue
            if stripped == '****':
                if current_element:
                    basis_dict[current_element] = '\n'.join(current_basis)
                    current_element = None
                    current_basis = []
                continue
            parts = stripped.split()
            if len(parts) == 2 and parts[0].isalpha() and parts[1] == '0' and current_element is None:
                current_element = parts[0].capitalize()
            elif current_element:
                current_basis.append(line.rstrip())

        return basis_dict

    def _write_nbo_config(self, filepath: Path):
        with open(filepath, "w") as f:
            f.write(NBO_CONFIG_TEMPLATE.format(**self.nbo_config_params))

    def _write_basis_inp(self, filepath: Path):
        header = """!----------------------------------------------------------------------
! Basis Set Exchange
! Version 0.12
! https://www.basissetexchange.org
!----------------------------------------------------------------------
!   Basis set: ANO-RCC-MB
! Description: ANO-RCC-MB
!        Role: orbital
!     Version: 1  (Data from OpenMolCAS)
!----------------------------------------------------------------------
"""
        elements_in_struct = [str(el) for el in self.structure.composition.elements]
        
        with open(filepath, "w") as f:
            f.write(header)
            f.write("****\n")
            for el in elements_in_struct:
                f.write(f"{el}     0\n")
                f.write(self.basis_settings[el] + "\n")
                f.write("****\n")
