"""
Adsorption site identification and modification for VASP catalysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict, List, Tuple, Any

from pymatgen.core import Structure, Molecule
from pymatgen.core.surface import Slab
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.adsorption import AdsorbateSiteFinder, plot_slab
from pymatgen.io.ase import AseAtomsAdaptor


class AdsorptionModify(AdsorbateSiteFinder):
    """
    增强版吸附位点查找与修饰工具。
    支持传统的单步调用，也支持 Fluent API 链式调用。
    """

    def __init__(
        self,
        slab_source: Union[Slab, Structure, str, Path],
        selective_dynamics: bool = False, 
        height: float = 0.9,
        mi_vec: Optional[np.ndarray] = None,
        save_dir: Optional[Union[str, Path]] = None
    ):
        self.slab, self.source_info = self._resolve_structure_input(slab_source)
        super().__init__(slab=self.slab, selective_dynamics=selective_dynamics, height=height, mi_vec=mi_vec)

        self.save_dir = Path(save_dir) if save_dir else Path.cwd() / f"adsorption_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 内部状态：存储生成的吸附结构及其元数据
        self._generated_structures: List[Dict[str, Any]] = []
        
        print(f"Initialized AdsorptionModify with slab from: {self.source_info}")

    @staticmethod
    def run_from_dict(config: Dict[str, Any]) -> Union[List[Structure], Dict[str, List[np.ndarray]]]:
        """静态方法：通过字典配置执行完整的生成/分析流程。"""
        
        # 1. 基础参数检查
        if "target_slab_source" not in config or not config["target_slab_source"]:
            raise ValueError("Config error: 'target_slab_source' is required.")
            
        mode = config.get("mode", "analyze").lower()
        valid_modes = ["analyze", "relative", "generate"]
        if mode not in valid_modes:
            raise ValueError(f"Config error: Invalid mode '{mode}'. Must be one of {valid_modes}.")

        gen_params = config.get("generate_params", {})
        
        init_kwargs = {
            "slab_source": config["target_slab_source"],
            "save_dir": config.get("save_dir"),
            "selective_dynamics": gen_params.get("selective_dynamics", False)
        }
        
        modifier = AdsorptionModify(**init_kwargs)
        print(f"Running in mode: {mode}")

        # --- 1. 分析模式 ---
        if mode == "analyze":
            return modifier.analyze(
                plot=config.get("plot", True),
                plot_params=config.get("plot_params", {})
            )
            
        # --- 2. 相对放置模式 (增加严格的参数校验) ---
        elif mode == "relative":
            rel_params = config.get("relative_params")
            if not rel_params:
                raise ValueError(" Config error: 'relative_params' dictionary is missing for 'relative' mode.")
                
            # 检查 relative 模式必须的三个参数
            required_rel_keys = ["reference_slab_source", "adsorbate_indices", "adsorbate_anchor_indices"]
            for key in required_rel_keys:
                if key not in rel_params or rel_params[key] is None:
                    raise ValueError(f" Config error: '{key}' is required in 'relative_params'.")

            modifier.place_relative(
                reference_slab_source=rel_params["reference_slab_source"],
                adsorbate_indices=rel_params["adsorbate_indices"],
                adsorbate_anchor_indices=rel_params["adsorbate_anchor_indices"],
                find_args=rel_params.get("find_args", {}), 
                movable_adsorbate_indices=rel_params.get("movable_adsorbate_indices")
            )
            
        # --- 3. 批量生成模式 (增加严格的参数校验) ---
        elif mode == "generate":
            if "molecule_formula" not in gen_params or not gen_params["molecule_formula"]:
                raise ValueError("Config error: 'molecule_formula' is required in 'generate_params' for 'generate' mode.")
                
            modifier.generate(
                molecule_formula=gen_params["molecule_formula"],
                find_args=gen_params.get("find_args", {}),
                reorient=gen_params.get("reorient", True),
                plot=config.get("plot", True),
                plot_params=config.get("plot_params", {})
            )

        # 统一保存逻辑
        save_opts = config.get("save_options", {})
        if save_opts.get("save", True) and modifier._generated_structures:
            modifier.save_all(
                filename_prefix=save_opts.get("filename", "POSCAR"),
                fmt=save_opts.get("fmt", "poscar"),
                as_subdirs=save_opts.get("as_subdirs", True)
            )
            
        return modifier.get_structures()

    def analyze(self, plot: bool = True, plot_params: dict = None) -> Dict[str, List[np.ndarray]]:
        """查找并直接返回吸附位点坐标字典 (剔除 'all' 键)。"""
        sites = self.find_adsorption_sites()
        
        # 剔除冗余的 'all' 键
        if "all" in sites:
            del sites["all"]
            
        print(f"Found sites: { {k: len(v) for k, v in sites.items()} }")
        
        if plot:
            self._plot_and_show(plot_params or {}, sites)
            
        return sites

    def generate(self, molecule_formula: str, find_args: dict = None, reorient: bool = True, plot: bool = False, plot_params: dict = None) -> 'AdsorptionModify':
        """在各个位点上生成吸附结构 (支持链式调用)。"""
        if not molecule_formula: 
            raise ValueError("molecule_formula is required for generation.")
            
        try:
            # 1. 优先尝试从 ASE 内置数据库获取 (如 "CO", "OH")
            molecule = self.ase2pmg(molecule_formula)
        except Exception as e_ase:
            file_path = Path(molecule_formula)
            if file_path.exists():
                try:
                    # 2. 尝试直接作为分子读取 (适用于 .xyz, .mol 等格式)
                    molecule = Molecule.from_file(file_path)
                except Exception:
                    # 3. 如果失败 (说明可能是 POSCAR/CONTCAR)，则作为周期性结构读取，并强制转换为分子
                    try:
                        temp_struct = Structure.from_file(file_path)
                        # 剥离晶格，只保留原子种类和笛卡尔坐标，转换为 Molecule
                        molecule = Molecule(temp_struct.species, temp_struct.cart_coords)
                        print(f"Loaded molecule from periodic structure file: {file_path.name}")
                    except Exception as e_struct:
                        raise ValueError(f"Failed to parse '{file_path}' as either Molecule or Structure. Error: {e_struct}")
            else:
                raise ValueError(f"Failed to load molecule '{molecule_formula}': Not in ASE database and file not found.")

        find_args = (find_args or {}).copy()
        requested_positions = find_args.pop("positions", []) 
        
        all_sites = self.find_adsorption_sites(**find_args)
        
        if "all" in all_sites:
            del all_sites["all"]
            
        self._generated_structures = []
        
        molecule_name = Path(molecule_formula).stem if Path(molecule_formula).exists() else molecule_formula
        print(f"Start generating structures for {molecule_name}...")

        for site_type, site_coords_list in all_sites.items():
            if requested_positions and site_type not in requested_positions:
                continue
            
            for i, site_coords in enumerate(site_coords_list):
                adsorbed_structure = self.add_adsorbate(molecule, site_coords, reorient=reorient)
                
                self._generated_structures.append({
                    "structure": adsorbed_structure,
                    "site_type": site_type,
                    "index": i,
                    "molecule": molecule_name # 使用文件名作为前缀
                })

        print(f"Successfully generated {len(self._generated_structures)} structures.")
        
        if plot:
            self._plot_and_show(plot_params or {}, all_sites)

        return self

    def place_relative(self, reference_slab_source: Union[Structure, str, Path], adsorbate_indices: List[int], adsorbate_anchor_indices: List[int], find_args: dict = None, movable_adsorbate_indices: List[int] = None) -> 'AdsorptionModify':
        """基于参考结构进行相对位置的吸附放置，自动匹配目标载体的吸附位点。"""
        if not reference_slab_source:
            raise ValueError("reference_slab_source is required.")
            
        ref_struct, _ = self._resolve_structure_input(reference_slab_source)
        
        # 1. 自动寻找目标载体上的吸附位点
        find_args = (find_args or {}).copy()
        requested_positions = find_args.pop("positions", []) 
        
        if "distance" in find_args:
            print("提示: 在 relative 模式下，'distance' 参数已被忽略。分子高度将 100% 继承自参考结构。")
        find_args["distance"] = 0.0 
        
        all_sites = self.find_adsorption_sites(**find_args)
        if "all" in all_sites:
            del all_sites["all"]
            
        self._generated_structures = []
        print(f"Start placing relative adsorbate...")

        # 2. 遍历所有找到的位点，进行批量相对放置
        for site_type, site_coords_list in all_sites.items():
            if requested_positions and site_type not in requested_positions:
                continue
            
            for i, site_coords in enumerate(site_coords_list):
                final_struct, _ = self._place_relative_adsorbate_logic(
                    relative_slab_structure=ref_struct,
                    target_slab_structure=self.slab,
                    adsorbate_indices=adsorbate_indices,
                    adsorbate_anchor_indices=adsorbate_anchor_indices,
                    target_coords=site_coords,  # 此时的 site_coords 是没有被垫高的真实坐标
                    movable_adsorbate_indices=movable_adsorbate_indices
                )
                
                self._generated_structures.append({
                    "structure": final_struct,
                    "site_type": site_type,
                    "index": i,
                    "molecule": "relative"
                })
        
        print(f"Successfully placed relative adsorbate on {len(self._generated_structures)} sites.")
        return self

    def get_structures(self) -> List[Structure]:
        """获取所有生成的结构列表。"""
        return [item["structure"].copy() for item in self._generated_structures]

    def save_all(self, filename_prefix: str = "POSCAR", fmt: str = "poscar", as_subdirs: bool = True):
        """保存所有生成的结构。"""
        if not self._generated_structures:
            print("Warning: No structures to save. Call generate() or place_relative() first.")
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        for item in self._generated_structures:
            struct = item["structure"]
            # 极简命名：位点类型_序号 (例如: ontop_0)
            identifier = f"{item['site_type']}_{item['index']}"
            
            if as_subdirs:
                site_dir = self.save_dir / f"{item['molecule']}_{identifier}"
                site_dir.mkdir(parents=True, exist_ok=True)
                out_path = site_dir / filename_prefix
            else:
                out_path = self.save_dir / f"{filename_prefix}_{identifier}"
            
            if fmt.lower() == "poscar":
                # 显式指定 encoding="utf-8" 解决 PEP 686 警告
                with open(out_path, mode="wt", encoding="utf-8") as f:
                    f.write(Poscar(struct).get_string())
            else:
                struct.to(filename=str(out_path), fmt=fmt)
                
        print(f"Saved {len(self._generated_structures)} structures to {self.save_dir}")

    # ==========================================
    # 内部辅助方法
    # ==========================================

    def _resolve_structure_input(self, source: Union[Structure, Slab, str, Path]) -> Tuple[Structure, str]:
        if isinstance(source, (Structure, Slab)): 
            return source, "In-memory Object"
        
        path = Path(source)
        if not path.exists(): 
            raise FileNotFoundError(f"Source not found: {path}")
        if path.is_file(): 
            return Structure.from_file(path), str(path)

        if path.is_dir():
            search_patterns = ["CONTCAR", "POSCAR", "*CONTCAR*", "*POSCAR*", "*.vasp", "*.cif"]
            for pattern in search_patterns:
                matches = sorted(list(path.glob(pattern)))
                for fpath in matches:
                    if fpath.is_file() and fpath.stat().st_size > 0:
                        try:
                            return Structure.from_file(fpath), str(fpath)
                        except Exception:
                            continue
            raise ValueError(f"No valid structure files found in directory: {path}.")

        raise ValueError(f"Invalid structure source: {source}")

    def _plot_and_show(self, plot_params: dict, sites: dict):
        params = plot_params.copy()
        figsize = params.pop("figsize", (6, 6))
        fig, ax = plt.subplots(figsize=figsize)
        self.plot_slab_with_labels(self.slab, ax=ax, ads_sites=sites, **params)
        out_p = self.save_dir / "adsorption_sites.png"
        fig.savefig(out_p, dpi=300, bbox_inches='tight')
        plt.show()

    @classmethod
    def ase2pmg(cls, adsorption_molecule: str) -> Molecule:
        from ase.build import molecule
        ase_molecule = molecule(adsorption_molecule)
        return AseAtomsAdaptor.get_molecule(ase_molecule)

    @staticmethod
    def _get_anchor_coords(structure, indices):
        coords = [structure[i].coords for i in indices]
        return coords[0] if len(coords) == 1 else np.mean(coords, axis=0)

    def _place_relative_adsorbate_logic(self, relative_slab_structure, target_slab_structure, adsorbate_indices, adsorbate_anchor_indices, target_coords, movable_adsorbate_indices=None):
        ads_anchor_coords = self._get_anchor_coords(relative_slab_structure, adsorbate_anchor_indices)
        relative_positions = [relative_slab_structure[i].coords - ads_anchor_coords for i in adsorbate_indices]
        adsorbate_species = [relative_slab_structure[i].specie for i in adsorbate_indices]

        combined = target_slab_structure.copy()
        adsorbate_indices_in_combined = []
        for i, rel_pos in enumerate(relative_positions):
            ads_pos = target_coords + rel_pos # 直接使用传入的坐标
            combined.append(adsorbate_species[i], ads_pos, coords_are_cartesian=True)
            adsorbate_indices_in_combined.append(len(combined) - 1)
        
        if target_slab_structure.site_properties.get("selective_dynamics"):
            sd = [list(x) for x in target_slab_structure.site_properties["selective_dynamics"]]
        else:
            sd = [[False, False, False] for _ in range(len(target_slab_structure))]

        n_ads = len(adsorbate_species)
        movable_set = set(movable_adsorbate_indices) if movable_adsorbate_indices is not None else set(range(n_ads))
        for i in range(n_ads):
            sd.append([True, True, True] if i in movable_set else [False, False, False])
                
        combined.add_site_property("selective_dynamics", sd)
        return combined, adsorbate_indices_in_combined
    

    @staticmethod
    def _rotation_matrix_from_vectors(vec1, vec2):
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if np.allclose(v, 0) and c > 0.999999: return np.eye(3)
        if np.allclose(v, 0) and c < -0.999999:
            orth = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9: orth = np.array([0.0, 1.0, 0.0])
            v = np.cross(a, orth)
            v = v / np.linalg.norm(v)
            return -np.eye(3) + 2 * np.outer(v, v)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

    @classmethod
    def _get_rot_matrix_for_slab(cls, slab: Slab) -> np.ndarray:
        if hasattr(slab, "normal"):
            normal = np.array(slab.normal)
            return cls._rotation_matrix_from_vectors(normal, np.array([0.0, 0.0, 1.0]))
        return np.eye(3)

    @classmethod
    def plot_slab_with_labels(
        cls,
        slab: Slab,
        ax: Optional[plt.Axes] = None,
        scale: float = 0.8,
        repeat: Tuple[int, int, int] = (1, 1, 1),
        window: float = 1.5,
        decay: float = 0.2,
        adsorption_sites: bool = True,
        inverse: bool = False,
        label_offset: Tuple[float, float] = (0.0, 0.0),
        ads_sites: Optional[dict] = None,
    ):
        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)

        working = slab.copy()
        rx, ry, rz = 1, 1, 1
        if repeat is not None:
            rx, ry, rz = repeat
            working.make_supercell([rx, ry, rz])

        try:
            plot_slab(
                working, ax=ax, scale=scale, repeat=1, window=window,
                draw_unit_cell=True, decay=decay, adsorption_sites=False, inverse=inverse
            )
        except TypeError:
            plot_slab(working, ax=ax, scale=scale, adsorption_sites=False, inverse=inverse)

        if not adsorption_sites:
            ax.set_aspect("equal")
            return ax

        if ads_sites is None:
            asf = AdsorbateSiteFinder(slab)
            found = asf.find_adsorption_sites()
            ads_sites = found["ontop"] + found["bridge"] + found["hollow"]
        elif isinstance(ads_sites, dict):
            # 这里同样剔除 'all' 键，避免绘图时重复绘制
            ads_sites = ads_sites.get("ontop", []) + ads_sites.get("bridge", []) + ads_sites.get("hollow", [])

        if not ads_sites: return ax

        R = cls._get_rot_matrix_for_slab(working)
        original_lattice = slab.lattice.matrix
        vec_a = original_lattice[0]
        vec_b = original_lattice[1]

        ads_2d_list = []
        labels_list = []

        for idx, site in enumerate(ads_sites):
            site_coords = np.array(site)
            for i in range(rx):
                for j in range(ry):
                    shifted_coords = site_coords + i * vec_a + j * vec_b
                    rotated = np.dot(R, shifted_coords)
                    ads_2d_list.append(rotated[:2])
                    labels_list.append(str(idx))

        if ads_2d_list:
            xs = [p[0] for p in ads_2d_list]
            ys = [p[1] for p in ads_2d_list]
            ax.plot(xs, ys, linestyle="", marker="x", markersize=6, mew=1.5, color="red", zorder=500)
            for (x, y), label in zip(ads_2d_list, labels_list):
                ax.text(x + label_offset[0], y + label_offset[1], label, 
                        fontsize=9, color="blue", fontweight='bold', zorder=501,
                        ha='center', va='bottom')

        ax.set_aspect("equal")
        return ax