import os
import sys
import math
import glob
import warnings
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, Set

from pymatgen.core import Structure, Composition
from pymatgen.io.lobster import Lobsterin
from pymatgen.io.vasp.inputs import Kpoints, Incar, Poscar
from pymatgen.io.vasp.sets import MVLSlabSet, VaspInputSet, MPStaticSet, MPMetalRelaxSet, LobsterSet, NEBSet
from pymatgen.analysis.diffusion.neb.pathfinder import IDPPSolver


_BEEF_INCAR = {
                "GGA": "BF",
                "LUSE_VDW": True,
                "AGGAC": 0.0000,
                "LASPH": True,
                "Zab_VDW": -1.8867,
            }

class VaspInputSet_ecat(VaspInputSet):
    # 统一的结构加载函数
    @staticmethod
    def _load_structure(struct_source: Union[str, Path, Structure]) -> Structure:
        if isinstance(struct_source, Structure):
            return struct_source

        p = Path(struct_source).expanduser()

        if p.is_dir():
            # 优先级：CONTCAR > POSCAR > POSCAR.vasp > *.vasp > *.cif
            candidates: List[Path] = []
            for fname in ["CONTCAR", "POSCAR", "POSCAR.vasp"]:
                fp = p / fname
                if fp.exists():
                    candidates.append(fp)

            if not candidates:
                vasp_like = sorted(p.glob("*.vasp"))
                if vasp_like:
                    candidates.extend(vasp_like)

            if not candidates:
                cif_files = sorted(p.glob("*.cif"))
                if len(cif_files) == 1:
                    candidates.append(cif_files[0])
                elif len(cif_files) > 1:
                    raise FileNotFoundError(
                        f"Multiple CIF files found in {p}. Please specify one explicitly: "
                        + ", ".join(str(x.name) for x in cif_files)
                    )

            if not candidates:
                raise FileNotFoundError(f"No structure file found in folder: {p}")

            return Structure.from_file(str(candidates[0]))

        if p.is_file():
            return Structure.from_file(str(p))

        raise ValueError(f"Invalid structure source: {struct_source}")

    # ---------- VASP compressed format parsing ----------
    @staticmethod
    def _parse_vasp_compressed_list(vasp_format: Union[str, Sequence[Any]]) -> Optional[List[float]]:
        """
        解析 VASP 常见压缩写法：
        - "2*1.0 3*0.0 1.5" -> [1.0,1.0,0.0,0.0,0.0,1.5]
        - ["2*1.0", 0, 1] 同样支持
        返回 None 表示解析失败
        """
        def expand_token(tok: Any) -> Optional[List[float]]:
            if tok is None:
                return None
            if isinstance(tok, (int, float)):
                return [float(tok)]
            s = str(tok).strip()
            if not s:
                return None
            if "*" in s:
                parts = s.split("*")
                if len(parts) != 2:
                    return None
                n_str, v_str = parts[0].strip(), parts[1].strip()
                try:
                    n = int(float(n_str))
                    v = float(v_str)
                except ValueError:
                    return None
                if n < 0:
                    return None
                return [v] * n
            else:
                try:
                    return [float(s)]
                except ValueError:
                    return None

        values: List[float] = []
        if isinstance(vasp_format, str):
            tokens = vasp_format.split()
        elif isinstance(vasp_format, (list, tuple)):
            tokens = list(vasp_format)
        else:
            return None

        for t in tokens:
            expanded = expand_token(t)
            if expanded is None:
                return None
            values.extend(expanded)

        return values

    @staticmethod
    def _vasp_species_order(structure: Structure) -> List[str]:
        """
        获取 VASP/POSCAR 的元素顺序（非常关键，不要用 composition 的 dict key 顺序）。
        """
        try:
            return Poscar(structure).site_symbols
        except Exception:
            # 兜底：使用 structure 的出现顺序去重
            seen = set()
            order = []
            for site in structure:
                sp = site.species_string
                if sp not in seen:
                    seen.add(sp)
                    order.append(sp)
            return order

    @classmethod
    def _convert_vasp_format_to_pymatgen_dict(
        cls,
        structure: Structure,
        key: str,
        vasp_format: Union[str, Sequence[Any]]
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        将 VASP 原生格式（支持 N*val 压缩）转换为 pymatgen VaspInputSet 支持的 dict 格式：
        - MAGMOM -> {"MAGMOM": {"Fe": 2.0, "O": 0.6}}
        - LDAUU/LDAUJ/LDAUL 类似
        支持两种长度：
        - 每原子一个值（len == num_sites），要求同一元素值必须一致才能缩并
        - 每元素一个值（len == num_species），按 POSCAR 元素顺序映射
        """
        values = cls._parse_vasp_compressed_list(vasp_format)
        if not values:
            return None

        num_total_atoms = len(structure)
        species_order = cls._vasp_species_order(structure)
        num_unique_species = len(species_order)

        # 1) atom-level
        if len(values) == num_total_atoms:
            # 检查同元素是否一致
            species_map: Dict[str, float] = {}
            for site, v in zip(structure, values):
                sp = site.species_string
                v = float(v)
                if sp not in species_map:
                    species_map[sp] = v
                else:
                    if abs(species_map[sp] - v) > 1e-12:
                        warnings.warn(
                            f"{key}: atom-level values contain different entries for the same species '{sp}'. "
                            f"Cannot safely convert to species dict; leaving as-is."
                        )
                        return None

        # 2) species-level by POSCAR order
        elif len(values) == num_unique_species:
            species_map = dict(zip(species_order, [float(x) for x in values]))

        else:
            warnings.warn(
                f"{key}: value count ({len(values)}) does not match total atoms ({num_total_atoms}) "
                f"or unique species ({num_unique_species}). Cannot convert."
            )
            return None

        if key == "MAGMOM":
            return {"MAGMOM": species_map}
        if key in ["LDAUU", "LDAUJ", "LDAUL"]:
            return {key: species_map}
        return None

class KPOINTS:
    @classmethod
    def automatic_by_lengths(cls, structure: Structure, length_densities: Sequence[float],
                             style: int = 1, shift=(0, 0, 0), comment: Optional[str] = None):
        """
        根据倒格矢长度和密度计算 k-point 网格。
        """
        if len(length_densities) != 3:
            raise ValueError("length_densities must be a sequence of 3 floats")
            
        mode = "Gamma" if style == 1 else "Monkhorst"    
        if comment is None: comment = f"{mode} Kpoint by lengths"    
        abc = structure.lattice.abc
        num_div = tuple(math.ceil(ld / abc[idx]) for idx, ld in enumerate(length_densities))
        
        return Kpoints(comment=comment, style=mode, kpts=[num_div], kpts_shift=shift)

## 调用script，不同functional用不同的脚本
class Script:

    MODE_MAP = ["PBE", "BEEF", "PBE_NEB", "BEEF_NEB", "LOBSTER_PBE", "LOBSTER_BEEF"]
    
    def __init__(self, folders: Union[List[str], str], functional: str = None):
        if isinstance(folders, (str, Path)):
            self.folders = [str(folders)]
        elif isinstance(folders, List):
            self.folders = [str(f) for f in folders]
        else:
            raise TypeError("folders must be a string, Path, or list of them")
        self.functional = functional

    def get_script(self) -> str:
        if self.functional is not None and self.functional not in self.MODE_MAP:
            raise ValueError(f"functional must be one of {self.MODE_MAP}")
        elif self.functional is None:
            self.functional = self.MODE_MAP[0]
        else:
            self.functional = self.functional.upper()

        script_map = {
            "PBE": "/data2/home/luodh/script/bulk/PBE",
            "BEEF": "/data2/home/luodh/script/bulk/BF",
            "BEEF_NEB": "/data2/home/luodh/script/bulk/BF-NEB",
            "PBE_NEB": "/data2/home/luodh/script/bulk/PBE-NEB",
            "LOBSTER_BEEF": "/data2/home/luodh/script/bulk/lobster/BEEF",
            "LOBSTER_PBE": "/data2/home/luodh/script/bulk/lobster/PBE",
        }
        script = script_map[self.functional]
        files = glob.glob(f"{script}/*")
        if not files:
            raise FileNotFoundError(f"No files found in {script}")

        for folder in self.folders:
            os.makedirs(folder, exist_ok=True)
            for f in files:
                subprocess.run(["cp", f, folder], check=True)

class MVLSlabSet_ecat(VaspInputSet_ecat, MVLSlabSet):
    """
    继承版的 MVLSlabSet，添加：
    - 自动 structure 加载（支持目录 / POSCAR / Structure）
    - 自动默认 INCAR（ECAT 模式）
    - 自动默认 KPOINTS（分层 slab）
    - 支持 BF 模式
    - 用户可覆盖所有设置
    """
    def __init__(self,
                structure: Union[str, Structure],
                functional: str = "PBE",
                kpoints_density: int = 25,
                normal_incar_set: bool = True,
                normal_kpoints_set: bool = True,
                user_incar_settings: Optional[Dict[str, Any]] = None,
                user_kpoints_settings: Optional[Any] = None,
                auto_dipole: bool = True,
                **extra_kwargs): # 使用 **kwargs 捕获额外参数
        
        # 加载结构
        loaded_structure = self._load_structure(structure)
        self.functional = functional.upper()
        incar = {}

        ##INCAR设置
        if normal_incar_set:
            incar.update({
                "EDIFFG": -0.02,
                "POTIM": 0.20,
                "EDIFF": 1e-04,
                "IBRION": 2,
                "NSW": 500,
                "LREAL": "Auto",
            })
    
        if self.functional == "BEEF": incar.update(_BEEF_INCAR)   
        if user_incar_settings: incar.update(user_incar_settings)

        ## 准备KPOINTS
        kpoints = user_kpoints_settings
        if not kpoints and normal_kpoints_set:
            kpoints = KPOINTS.automatic_by_lengths(loaded_structure, [kpoints_density, kpoints_density, 1], style=2)

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
        **extra_kwargs):

        prev_dir = Path(prev_dir).resolve()
        
        # 1. 读取原始 INCAR 和 CONTCAR
        incar_path = prev_dir / "INCAR"
        if not incar_path.exists():
             raise FileNotFoundError(f"INCAR not found in prev_dir: {prev_dir}")
             
        base_incar = Incar.from_file(incar_path).as_dict()
        incar = base_incar.copy()
        
        prev_contcar_path = prev_dir / "CONTCAR"
        prev_poscar_path = prev_dir / "POSCAR"
        if prev_contcar_path.exists():
            loaded_prev_structure = Structure.from_file(prev_contcar_path)
        elif prev_poscar_path.exists():
            loaded_prev_structure = Structure.from_file(prev_poscar_path)
        else:
            raise FileNotFoundError(f"Neither CONTCAR nor POSCAR found in prev_dir: {prev_dir}. Cannot proceed.") 
        
        # 2. 推断 Functional (用于初始化)
        is_beef = any(key in base_incar for key in _BEEF_INCAR)
        functional = "BEEF" if is_beef else "PBE"
        
        # 3. 格式敏感参数转换 (使用上一步的 CONTCAR 结构)
        FORMAT_SENSITIVE_KEYS = ["MAGMOM", "LDAUU", "LDAUJ", "LDAUL"]
        converted_params = {}
        for key in FORMAT_SENSITIVE_KEYS:
            value = incar.get(key)
            if value is None or isinstance(value, dict): continue
            conversion_result = cls._convert_vasp_format_to_pymatgen_dict(loaded_prev_structure, key, value)
            if conversion_result: converted_params.update(conversion_result)
            
        for key in FORMAT_SENSITIVE_KEYS: incar.pop(key, None)
        incar.update(converted_params) 
        
        # 4. INCAR 用户覆盖及 Slab 修正
        if user_incar_settings: incar.update(user_incar_settings)

        # 5. KPOINTS 继承 (优先读取文件，其次使用用户设置)
        kpoints_path = prev_dir / "KPOINTS"
        kpoints = None
        
        if user_kpoints_settings:
            kpoints = user_kpoints_settings 
        elif kpoints_path.exists():
            kpoints = Kpoints.from_file(kpoints_path)
             
        return cls(
            functional=functional,
            structure=structure,
            normal_incar_set=False, 
            normal_kpoints_set=False, 
            auto_dipole=auto_dipole,
            user_incar_settings=incar,
            user_kpoints_settings=kpoints,
            **extra_kwargs)  
           
class MPMetalRelaxSet_ecat(VaspInputSet_ecat, MPMetalRelaxSet):
    def __init__(self,
                structure: Union[str, Structure],
                functional: str = "PBE",
                is_metal: bool = True,
                kpoints_density: int = 25,
                normal_incar_set: bool = True,
                normal_kpoints_set: bool = True,
                user_incar_settings: Optional[Dict[str, Any]] = None,
                user_kpoints_settings: Optional[Any] = None,
                **extra_kwargs): # 使用 **kwargs 捕获额外参数
        
        # 加载结构
        self.functional = functional.upper()
        loaded_structure = self._load_structure(structure)

        incar = {}
        if normal_incar_set:
            incar.update({
                "EDIFFG": -0.02,
                "EDIFF": 1E-6,
                "POTIM": 0.20,
                "ENCUT": 520,
                "IBRION": 2,
                "LORBIT": 10,
                "NSW": 500,
                "LREAL": False,
            })
        if self.functional == "BEEF":
            incar.update(_BEEF_INCAR)

        # 用户覆盖 
        if not is_metal:
            incar.update({"ISMEAR": 0, "SIGMA": 0.05})
        else:
            incar.update({"ISMEAR": 1, "SIGMA": 0.20})
        if user_incar_settings:
            incar.update(user_incar_settings)
            
        # 准备KPOINTS
        kpoints = None
        if user_kpoints_settings:
            kpoints = user_kpoints_settings
        elif normal_kpoints_set:
            kpoints = KPOINTS.automatic_by_lengths(
                structure=loaded_structure, length_densities=[kpoints_density] * 3,
                style=2)
            
        super().__init__(
            structure=loaded_structure,
            user_incar_settings=incar,
            user_kpoints_settings=kpoints,
            **extra_kwargs
        )

class MPStaticSet_ecat(VaspInputSet_ecat, MPStaticSet):
    """
    用于单点计算，继承版的 MPStaticSet，添加：
    - 自动 structure 加载（支持目录 / POSCAR / Structure）
    - 自动默认 INCAR，加入计算DOS和计算bader电荷的参数
    - 自动默认 KPOINTS
    - 支持 BF 模式
    - 用户可覆盖所有设置
    """
    
    def __init__(self,
                structure: Union[str, Structure],
                functional: str = "PBE",
                normal_incar_set: bool = True,
                normal_kpoints_set: bool = True,
                number_of_docs: Optional[int] = None,
                user_incar_settings: Optional[Dict[str, Any]] = None,
                user_kpoints_settings: Optional[Any] = None,
                **extra_kwargs):
        # 加载结构
        self.functional = functional.upper()
        loaded_structure = self._load_structure(structure)
        incar = {}
        if normal_incar_set:
            incar.update({
                "EDIFF":1E-6,
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
            })
            
        if self.functional == "BEEF":
            incar.update(_BEEF_INCAR)
        
        if number_of_docs is not None:
            incar["NEDOS"] = number_of_docs

        if user_incar_settings:
            incar.update(user_incar_settings)

        kpoints = None
        if user_kpoints_settings:
            kpoints = user_kpoints_settings
        elif normal_kpoints_set:
            kpoints = KPOINTS.automatic_by_lengths(
                structure=loaded_structure, length_densities=[40, 40, 40], style=2)

        super().__init__(
            structure=loaded_structure,
            user_incar_settings=incar,
            user_kpoints_settings=kpoints,
            **extra_kwargs,
        )

    @classmethod
    def from_prev_calc_ecat(cls,
                            prev_dir: Union[str, Path],
                            kpoints_density: int = 50,
                            number_of_docs: Optional[int] = None,
                            user_incar_settings: Optional[Dict[str, Any]] = None,
                            **extra_kwargs):
        """基于上一次计算的结构设计新的电子结构计算"""
        prev_dir = Path(prev_dir).resolve()
        
        # 加载结构，获取CONTCAR
        structure_path = prev_dir / "CONTCAR"
        if not structure_path.exists():
            raise FileNotFoundError(f"CONTCAR not found in supported {prev_dir}")
        loaded_structure = Structure.from_file(structure_path)
        
        # 读取原始INCAR文件和确定泛函
        incar_path = prev_dir / "INCAR"
        # 使用 base_incar 保存原始 INCAR，并拷贝到 incar 进行修改
        base_incar = Incar.from_file(incar_path).as_dict() if incar_path.exists() else {}
        
        # 从 base_incar 开始，应用默认设置和用户覆盖
        incar = base_incar.copy() 
        
        static_defaults = {
            "EDIFF": 1E-6, "NELM": 200, "IBRION": -1, "LORBIT": 11,
            "NEDOS": 3001, "LCHARG": True, "ISMEAR": -5, 
            "LAECHG": True, "LELF": True, "NSW": 0, "SIGMA": 0.05,
        }
        incar.update(static_defaults)
        
        if number_of_docs is not None:
            incar["NEDOS"] = int(number_of_docs)
            
        if user_incar_settings:
            incar.update(user_incar_settings)
            
        functional = "PBE"
        is_beef = any(key in base_incar for key in _BEEF_INCAR)
        if is_beef:
            functional = "BEEF"
            incar.update(_BEEF_INCAR)

        FORMAT_SENSITIVE_KEYS = ["MAGMOM", "LDAUU", "LDAUJ", "LDAUL"]
        converted_params: Dict[str, Any] = {}

        for key in FORMAT_SENSITIVE_KEYS:
            value = incar.get(key)
            if value is None:
                continue
            if isinstance(value, dict):
                converted_params[key] = value
                continue
            conversion_result = cls._convert_vasp_format_to_pymatgen_dict(loaded_structure, key, value)
            if conversion_result:
                converted_params.update(conversion_result)
            else:
                warnings.warn(f"Failed to convert VASP format for {key}='{value}'. Keeping original value.")

        for key in FORMAT_SENSITIVE_KEYS:
            incar.pop(key, None)
        incar.update(converted_params)

        if isinstance(kpoints_density, (int, float)):
            densities_list = [float(kpoints_density)] * 3
        elif isinstance(kpoints_density, (tuple, list)) and len(kpoints_density) == 3:
            densities_list = [float(x) for x in kpoints_density]
        else:
            raise ValueError("kpoints_density must be a single number or a sequence of 3 numbers.")

        kpoints = KPOINTS.automatic_by_lengths(
            structure=loaded_structure, length_densities=densities_list, style=2)


        return cls(
            structure=loaded_structure,
            functional=functional,
            normal_incar_set=False,
            normal_kpoints_set=False,
            number_of_docs=number_of_docs,
            user_incar_settings=incar,
            user_kpoints_settings=kpoints,
            **extra_kwargs,
        )

class LobsterSet_ecat(VaspInputSet_ecat, LobsterSet):
    """
    用于Lobster计算，继承版的 LobsterSet，添加：
    - 自动 structure 加载（支持目录 / POSCAR / Structure）
    - 自动默认 INCAR,加入lobsterin的相关设置
    - 自动默认 KPOINTS
    - 支持 BF 模式
    - 用户可覆盖所有设置
    """
    
    def __init__(
            self,
            structure: Union[str, Structure],
            functional: str = "PBE",
            isym: Literal[0, -1] = 0,
            ismear: Literal[0, -5] = -5,
            reciprocal_density: int = None,
            user_supplied_basis: dict = None,
            normal_incar_set: bool = True,
            user_incar_settings: dict = None,
            user_kpoints_settings: dict = None,
            **extra_kwargs):
        # 加载结构
        self.functional = functional.upper()
        self.normal_incar_set = normal_incar_set
        loaded_structure = self._load_structure(structure)
        incar = {}
        if self.normal_incar_set:
            incar.update({
                "NELM": 150,
                "NCORE": 6,
                "IBRION": -1,
                "EDIFF": 1E-6,
                "LORBIT": 11,
                "NSW": 0,
            })
            
        if self.functional == "BEEF":
            incar.update(_BEEF_INCAR)
        # 用户覆盖
        if user_incar_settings:
            incar.update(user_incar_settings)
        # 准备KPOINTS
        kpoints = None
        if user_kpoints_settings:
            kpoints = user_kpoints_settings

        # 调用父类初始化（非常关键）
        super().__init__(
            structure=loaded_structure,
            isym=isym,
            ismear=ismear,
            reciprocal_density=reciprocal_density,
            user_supplied_basis=user_supplied_basis,
            user_incar_settings=incar,
            user_kpoints_settings=kpoints,
            **extra_kwargs
        )
    
    @classmethod
    def from_prev_calc_ecat(cls,
                            prev_dir: Union[str, Path],
                            kpoints_density: int = 50,
                            user_incar_settings: Optional[Dict[str, Any]] = None,
                            user_supplied_basis: dict = None,
                            **extra_kwargs):
        """基于上一次VASP计算结果， 设计新的LOBSTER计算任务"""
        prev_dir = Path(prev_dir).resolve()
        structure_path = prev_dir / "CONTCAR"
        if not structure_path.exists():
            raise FileNotFoundError(f"CONTCAR not found in supported {prev_dir}")
        loaded_structure = Structure.from_file(structure_path)
        incar_path = prev_dir / "INCAR"
        base_incar = Incar.from_file(incar_path).as_dict() if incar_path.exists() else {}
        incar = base_incar.copy() 
        lobster_defaults = {
            "EDIFF": 1E-6, "NELM": 150, "IBRION": -1,"NCORE": 6, 
            "LORBIT": 11, "NSW": 0,"ISMEAR": -5,"SIGMA": 0.20,     
        }
        incar.update(lobster_defaults)
        
        if user_incar_settings:
            incar.update(user_incar_settings)
            
        functional = "PBE"
        is_beef = any(key in base_incar for key in _BEEF_INCAR)
        if is_beef:
            functional = "BEEF"
            incar.update(_BEEF_INCAR)

        FORMAT_SENSITIVE_KEYS = ["MAGMOM", "LDAUU", "LDAUJ", "LDAUL"]
        converted_params: Dict[str, Any] = {}
        for key in FORMAT_SENSITIVE_KEYS:
            value = incar.get(key)
            if value is None:
                continue
            if isinstance(value, dict):
                converted_params[key] = value
                continue
            conversion_result = cls._convert_vasp_format_to_pymatgen_dict(loaded_structure, key, value)
            if conversion_result:
                converted_params.update(conversion_result)
            else:
                warnings.warn(f"Failed to convert VASP format for {key}='{value}'. Keeping original value.")

        for key in FORMAT_SENSITIVE_KEYS:
            incar.pop(key, None)
        incar.update(converted_params)

        if isinstance(kpoints_density, (int, float)):
            densities_list = [float(kpoints_density)] * 3
        elif isinstance(kpoints_density, (tuple, list)) and len(kpoints_density) == 3:
            densities_list = [float(x) for x in kpoints_density]
        else:
            raise ValueError("kpoints_density must be a single number or a sequence of 3 numbers.")

        kpoints = KPOINTS.automatic_by_lengths(
            structure=loaded_structure,
            length_densities=densities_list,
            style=2
        )
        final_kwargs = {
            "structure": loaded_structure,
            "functional": functional,
            "isym": 0,
            "ismear": -5,
            "reciprocal_density": None,
            "user_supplied_basis": user_supplied_basis,
            "normal_incar_set": False,
            "user_incar_settings": incar,
            "user_kpoints_settings": kpoints,
            **extra_kwargs
        }
        return cls(**final_kwargs)
         
class NEBSet_ecat(VaspInputSet_ecat, NEBSet):
    """
    用于NEB计算，继承版的 NEBSet，添加：
    - 自动 structure 加载（支持目录 / POSCAR / Structure）
    - 自动默认 INCAR,加入neb相关设置
    - 自动默认 KPOINTS
    - 支持 BF 模式
    - 用户可覆盖所有设置
    """
    def __init__(
            self,
            start_structure: Union[str, Structure],
            end_structure: Union[str, Structure],
            n_images: int = 6,
            intermediate_structures: Optional[List[Structure]] = None,
            functional: str = "PBE",
            normal_incar_set: bool = True,
            use_idpp: bool = True,
            user_incar_settings: Optional[Dict[str, Any]] = None,
            user_kpoints_settings: Optional[Dict[str, Any]] = None,
            **extra_kwargs
    ):
        start_structure = self._load_structure(start_structure)
        end_structure = self._load_structure(end_structure)
        self.functional = functional.upper()
        if intermediate_structures is None:
            if use_idpp:
                try:
                    solver = IDPPSolver.from_endpoints(
                        [start_structure, end_structure], n_images=n_images + 2, sort_tol=0.1
                    )
                    intermediate_structures = solver.run(maxiter=2000, tol=1e-5, species=start_structure.species)
                except Exception as e:
                    intermediate_structures = start_structure.interpolate(end_structure, n_images + 1) # +1 包含端点
            else:
                intermediate_structures = start_structure.interpolate(end_structure, n_images + 1)
       
        incar = {}
        if normal_incar_set:
            incar.update({
                "EDIFF":1E-5,
                "NELM": 150,
                "POTIM": 0.02,
                "ICHAIN": 0,
                "SPRING": -5.0,
                "IBRION": 3,
                "LCLIMB": True,
                "LREAL": "Auto",
            })
        if self.functional == "BEEF":
            incar.update(_BEEF_INCAR)

        if user_incar_settings:
            incar.update(user_incar_settings)

        super().__init__(
            structures=intermediate_structures,
            user_incar_settings=incar,
            user_kpoints_settings=user_kpoints_settings,
            **extra_kwargs
        )

class FreqSet_ecat(MPStaticSet_ecat):
    """
    频率计算输入集（基于 MPStaticSet_ecat）
    - 默认 INCAR：IBRION=5, POTIM=0.015, NFREE=2, NSW=1
    - 结构默认来自 prev_dir/CONTCAR（收敛结构）
    - selective_dynamics：
        * vibrate_indices 给定 -> 覆盖：这些 True，其余 False
        * vibrate_indices None -> 不改结构上已有的 selective_dynamics（若存在则保留）
    """

    @staticmethod
    def _apply_vibrate_indices(structure: Structure, vibrate_indices: List[int]) -> Structure:
        n = len(structure)
        bad = [i for i in vibrate_indices if (not isinstance(i, int)) or i < 0 or i >= n]
        if bad:
            raise IndexError(f"vibrate_indices out of range for {n} sites: {bad}")

        sd = [[False, False, False] for _ in range(n)]
        for idx in vibrate_indices:
            sd[idx] = [True, True, True]

        structure.add_site_property("selective_dynamics", sd)
        return structure


    def __init__(
        self,
        structure: Union[str, Structure],
        functional: str = "PBE",
        normal_incar_set: bool = True,
        normal_kpoints_set: bool = False,
        kpoints_density: Optional[Union[int, float, Sequence[float]]] = None,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        **extra_kwargs,
    ):
        loaded_structure = self._load_structure(structure)
        functional = (functional or "PBE").upper()

        incar: Dict[str, Any] = {}
        if normal_incar_set:
            incar.update({
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
            })

        if functional == "BEEF":
            incar.update(_BEEF_INCAR)

        if user_incar_settings:
            incar.update(user_incar_settings)

        kpoints = user_kpoints_settings
        if (kpoints is None) and normal_kpoints_set:
            if kpoints_density is None:
                densities = [40.0, 40.0, 40.0]
            elif isinstance(kpoints_density, (int, float)):
                densities = [float(kpoints_density)] * 3
            else:
                if len(kpoints_density) != 3:
                    raise ValueError("kpoints_density must be a number or a sequence of 3 numbers.")
                densities = [float(x) for x in kpoints_density]

            kpoints = KPOINTS.automatic_by_lengths(
                structure=loaded_structure, length_densities=densities, style=2
            )

        super().__init__(
            structure=loaded_structure,
            functional=functional,
            normal_incar_set=False,
            normal_kpoints_set=False,
            user_incar_settings=incar,
            user_kpoints_settings=kpoints,
            **extra_kwargs,
        )

    @classmethod
    def from_prev_calc_ecat(
        cls,
        prev_dir: Union[str, Path],
        structure: Optional[Union[str, Structure]] = None,
        vibrate_indices: Optional[List[int]] = None,
        inherit_sd: bool = True,
        user_incar_settings: Optional[Dict[str, Any]] = None,
        user_kpoints_settings: Optional[Any] = None,
        **extra_kwargs,
    ):
        prev_dir = Path(prev_dir).resolve()

        # 1) 结构源头：CONTCAR（除非显式传入 structure）
        if structure is None:
            contcar = prev_dir / "CONTCAR"
            if not contcar.exists():
                raise FileNotFoundError(f"CONTCAR not found in {prev_dir}")
            loaded_structure = Structure.from_file(contcar)
        else:
            loaded_structure = cls._load_structure(structure)

        # 2) selective dynamics：indices 覆盖，否则按 inherit_sd 保留原样
        if vibrate_indices is not None:
            loaded_structure = cls._apply_vibrate_indices(loaded_structure, vibrate_indices)

        # 3) 继承 INCAR
        incar_path = prev_dir / "INCAR"
        base_incar = Incar.from_file(incar_path).as_dict() if incar_path.exists() else {}
        incar = base_incar.copy()

        for k in ["IBRION", "NSW", "POTIM", "EDIFF", "EDIFFG", "ISIF", "NPAR", "NCORE"]:
            incar.pop(k, None)

        incar.update({
            "IBRION": 5,
            "POTIM": 0.015,
            "NFREE": 2,
            "NSW": 1,
            "EDIFF": 1e-7,
            "LCHARG": False,
            "LWAVE": False,
            "LREAL": False,
        })

        if user_incar_settings:
            incar.update(user_incar_settings)

        # 4) functional 推断 + BEEF
        is_beef = any(k in base_incar for k in _BEEF_INCAR)
        functional = "BEEF" if is_beef else "PBE"
        if functional == "BEEF":
            incar.update(_BEEF_INCAR)

        # 5) 格式敏感参数转换
        FORMAT_SENSITIVE_KEYS = ["MAGMOM", "LDAUU", "LDAUJ", "LDAUL"]
        converted_params: Dict[str, Any] = {}
        for key in FORMAT_SENSITIVE_KEYS:
            val = incar.get(key)
            if val is None or isinstance(val, dict):
                continue
            res = cls._convert_vasp_format_to_pymatgen_dict(loaded_structure, key, val)
            if res:
                converted_params.update(res)

        for key in FORMAT_SENSITIVE_KEYS:
            incar.pop(key, None)
        incar.update(converted_params)

        # 6) KPOINTS：用户优先，否则继承
        if user_kpoints_settings is not None:
            kpoints = user_kpoints_settings
        else:
            kp_path = prev_dir / "KPOINTS"
            kpoints = Kpoints.from_file(kp_path) if kp_path.exists() else None

        return cls(
            structure=loaded_structure,
            functional=functional,
            normal_incar_set=False,
            normal_kpoints_set=False,
            user_incar_settings=incar,
            user_kpoints_settings=kpoints,
            **extra_kwargs,
        )

@dataclass
class VaspInputMaker:
    """VASP 输入生成器"""

    name: str = "VaspInputMaker"
    functional: Literal["PBE", "BEEF"] = "PBE"
    user_incar_settings: Dict[str, Any] = field(default_factory=dict)
    user_kpoints_settings: Any = None
    user_potcar_functional: Any = "PBE_54"
    normal_incar_set: bool = True
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    structure: Union[str, Structure, None] = None
    is_metal: bool = False
    normal_kpoints_set: bool = True
    auto_dipole: bool = False

    # NEB
    start_structure: Union[str, Structure, None] = None
    end_structure: Union[str, Structure, None] = None
    intermediate_structures: List[Structure] = field(default_factory=list)
    n_images: int = 6

    # NoSCF/Lobster
    number_of_docs: Optional[int] = None
    isym: Literal[0, -1] = 0
    ismear: Literal[0, -5] = -5
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
        extra_params = full_config.pop("extra_kwargs", {})

        for k, v in full_config.items():
            key_clean = k.strip()
            if key_clean in class_fields:
                main_params[key_clean] = v
            else:
                extra_params[key_clean] = v

        return cls(**main_params, extra_kwargs=extra_params)

    def _get_final_structure(self, structure_override: Union[str, Structure, None]) -> Structure:
        struct_source = structure_override if structure_override is not None else self.structure
        if struct_source is None:
            raise ValueError("Structure is required for this calculation.")
        return VaspInputSet_ecat._load_structure(struct_source)

    def _detect_adsorbate_indices(self, structure: Structure, z_cutoff: float = 2.0) -> List[int]:
        """
        简单的启发式算法，用于识别吸附分子。
        逻辑：
        1. 找出所有原子的 Z 坐标。
        2. 寻找 Z 轴上最大的间隙 (Gap)。
        3. 间隙上方的原子被认为是吸附分子。
        注意：这假设吸附分子在 Slab 上方。
        """
        if len(structure) < 2:
            return [0]
            
        # 获取带索引的 Z 坐标: [(0, z0), (1, z1), ...]
        sites_z = sorted([(i, site.coords[2]) for i, site in enumerate(structure)], key=lambda x: x[1])
        
        max_gap = 0.0
        gap_index = 0
        
        # 寻找最大间隙
        for i in range(len(sites_z) - 1):
            gap = sites_z[i+1][1] - sites_z[i][1]
            if gap > max_gap:
                max_gap = gap
                gap_index = i
        
        adsorbate_indices = [x[0] for x in sites_z[gap_index+1:]]
        
        return adsorbate_indices


    def write_bulk(self, output_dir: Union[str, Path], structure: Union[str, Structure, None] = None):
        final_structure = self._get_final_structure(structure)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        common_kwargs = {
            "structure": final_structure,
            "functional": self.functional.upper(),
            "normal_incar_set": self.normal_incar_set,
            "user_incar_settings": self.user_incar_settings,
            "user_kpoints_settings": self.user_kpoints_settings,
            "user_potcar_functional": self.user_potcar_functional,
            **self.extra_kwargs,
        }

        input_kwargs = {
            "kpoints_density": self.kpoints_density,
            "is_metal": self.is_metal,
            "normal_kpoints_set": self.normal_kpoints_set,
            **common_kwargs,
        }

        input_obj = MPMetalRelaxSet_ecat(**input_kwargs)
        input_obj.write_input(output_dir)
        return str(output_dir)

    def write_slab(self, output_dir: Union[str, Path], structure: Union[str, Structure, None] = None):
        final_structure = self._get_final_structure(structure)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        common_kwargs = {
            "structure": final_structure,
            "functional": self.functional.upper(),
            "normal_incar_set": self.normal_incar_set,
            "user_incar_settings": self.user_incar_settings,
            "user_kpoints_settings": self.user_kpoints_settings,
            "user_potcar_functional": self.user_potcar_functional,
            **self.extra_kwargs,
        }

        input_kwargs = {
            "kpoints_density": self.kpoints_density,
            "normal_kpoints_set": self.normal_kpoints_set,
            "auto_dipole": self.auto_dipole,
            **common_kwargs,
        }

        input_obj = MVLSlabSet_ecat(**input_kwargs)
        input_obj.write_input(output_dir)
        return str(output_dir)

    def write_noscf(
        self,
        output_dir: Union[str, Path],
        structure: Union[str, Structure, None] = None,
        prev_dir: Optional[Union[str, Path]] = None,
    ):
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if prev_dir is None and self.prev_dir is not None:
            prev_dir = self.prev_dir
        if prev_dir is not None:
            prev_dir_path = Path(prev_dir).resolve()
            input_set = MPStaticSet_ecat.from_prev_calc_ecat(
                prev_dir=prev_dir_path,
                kpoints_density=self.kpoints_density,
                number_of_docs=self.number_of_docs,
                user_incar_settings=self.user_incar_settings,
                **self.extra_kwargs,
            )
        else:
            final_structure = self._get_final_structure(structure)
            input_set = MPStaticSet_ecat(
                structure=final_structure,
                functional=self.functional.upper(),
                normal_incar_set=self.normal_incar_set,
                normal_kpoints_set=self.normal_kpoints_set,
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
    ):
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        final_start = self._get_final_structure(start_structure if start_structure is not None else self.start_structure)
        final_end = self._get_final_structure(end_structure if end_structure is not None else self.end_structure)

        input_obj = NEBSet_ecat(
            start_structure=final_start,
            end_structure=final_end,
            n_images=int(n_images) if n_images is not None else int(self.n_images),
            intermediate_structures=intermediate_structures if intermediate_structures is not None else self.intermediate_structures,
            functional=self.functional.upper(),
            normal_incar_set=self.normal_incar_set,
            user_incar_settings=self.user_incar_settings,
            user_kpoints_settings=self.user_kpoints_settings,
            user_potcar_functional=self.user_potcar_functional,
            **self.extra_kwargs,
        )

        input_obj.write_input(output_dir)
        return str(output_dir)

    def write_lobster(self, output_dir: Union[str, Path], structure: Union[str, Structure, None] = None, prev_dir: Optional[Union[str, Path]] = None):
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if prev_dir is None and self.prev_dir is not None:
            prev_dir = self.prev_dir        

        if prev_dir is not None:
            input_set = LobsterSet_ecat.from_prev_calc_ecat(
                prev_dir=Path(prev_dir).resolve(),
                kpoints_density=self.kpoints_density,
                user_incar_settings=self.user_incar_settings,
                user_supplied_basis=self.user_supplied_basis,
                **self.extra_kwargs,
            )
        else:
            input_set = LobsterSet_ecat(
                structure=self._get_final_structure(structure),
                functional=self.functional.upper(),
                isym=self.isym,
                ismear=self.ismear,
                reciprocal_density=self.reciprocal_density,
                user_supplied_basis=self.user_supplied_basis,
                user_incar_settings=self.user_incar_settings,
                user_kpoints_settings=self.user_kpoints_settings,
                user_potcar_functional=self.user_potcar_functional,
                **self.extra_kwargs,
            )

        input_set.write_input(output_dir)

        # 自动生成 lobsterin
        lb = Lobsterin.standard_calculations_from_vasp_files(
            POSCAR_input=output_dir / "POSCAR",
            INCAR_input=output_dir / "INCAR",
            POTCAR_input=output_dir / "POTCAR",
        )
        lb.write_lobsterin(output_dir / "lobsterin", overwritedict=self.overwritedict)
        return str(output_dir)

    def write_adsorption(self, output_dir: Union[str, Path], structure: Union[str, Structure], prev_dir: Union[str, Path]):
        output_dir = Path(output_dir).resolve()
        if prev_dir is None and self.prev_dir is not None:
            prev_dir = self.prev_dir
        prev_dir_path = Path(prev_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        input_set = MVLSlabSet_ecat.ads_from_prev_calc(
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

    @staticmethod
    def _formula_to_counts(formula: str) -> Dict[str, int]:
        comp = Composition(formula)
        counts: Dict[str, int] = {}
        for el, amt in comp.get_el_amt_dict().items():
            if abs(amt - round(amt)) > 1e-8:
                raise ValueError(f"adsorbate_formula must be integer stoichiometry, got: {formula}")
            counts[str(el)] = int(round(amt))
        if not counts:
            raise ValueError(f"adsorbate_formula is empty/invalid: {formula}")
        return counts

    @staticmethod
    def _structure_element_counts(structure: Structure) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for site in structure:
            el = site.species_string
            counts[el] = counts.get(el, 0) + 1
        return counts

    @classmethod
    def _pick_adsorbate_indices_by_formula_strict(
        cls,
        structure: Structure,
        adsorbate_formula: str,
        prefer: str = "tail",  # "tail" or "head"
    ) -> List[int]:
        need = cls._formula_to_counts(adsorbate_formula)
        have = cls._structure_element_counts(structure)

        missing = {el: n for el, n in need.items() if have.get(el, 0) < n}
        if missing:
            raise ValueError(f"Structure does not contain enough atoms for {adsorbate_formula}: {missing}")

        ambiguous = {el: (have.get(el, 0), need[el]) for el in need if have.get(el, 0) > need[el]}
        if ambiguous:
            raise ValueError(
                "adsorbate_formula is ambiguous because these elements also appear elsewhere in the structure: "
                f"{ambiguous}. Please provide vibrate_indices explicitly."
            )

        elem_to_indices: Dict[str, List[int]] = {}
        for i, site in enumerate(structure):
            elem_to_indices.setdefault(site.species_string, []).append(i)

        chosen: List[int] = []
        for el, n in need.items():
            pool = elem_to_indices[el]
            chosen.extend(pool[:n] if prefer == "head" else pool[-n:])
        return sorted(chosen)

    def write_freq(
        self,
        output_dir: Union[str, Path],
        prev_dir: Optional[Union[str, Path]] = None,
        structure: Union[str, Structure, None] = None,
        mode: str = "inherit",  # "inherit" or "all" or "adsorbate"(禁用自动)
        vibrate_indices: Optional[List[int]] = None,
        adsorbate_formula: Optional[str] = None,
        adsorbate_formula_prefer: str = "tail",
    ):
        """
        严格策略：
          1) vibrate_indices 优先
          2) adsorbate_formula 次之（若歧义则报错要求 indices）
          3) mode:
             - inherit: CONTCAR 必须自带 selective_dynamics，否则报错
             - all: 全放开
             - adsorbate: 不提供自动识别（避免误判），要求 formula 或 indices
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if prev_dir is None and getattr(self, "prev_dir", None) is not None:
            prev_dir = self.prev_dir
        if prev_dir is None:
            raise ValueError("Frequency calculation requires 'prev_dir' (needs CONTCAR/INCAR/KPOINTS).")
        prev_dir_path = Path(prev_dir).resolve()

        # 结构源头：CONTCAR（除非用户显式传 structure）
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
            final_vibrate_indices = self._pick_adsorbate_indices_by_formula_strict(
                final_structure,
                adsorbate_formula=adsorbate_formula,
                prefer=adsorbate_formula_prefer,
            )
            inherit_sd = False
            print(f"Picked vibrate indices by formula {adsorbate_formula}: {final_vibrate_indices}")

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

        input_set = FreqSet_ecat.from_prev_calc_ecat(
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