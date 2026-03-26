"""
Nanoparticle Structure Generator for VASP catalysis simulations.
Supports ASE-based geometric shapes and Wulff construction via pymatgen.
"""
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict, List, Tuple, Any

import numpy as np
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.wulff import WulffShape

# ASE imports (optional, for geometric particle generation)
try:
    from ase.cluster import FaceCenteredCubic, BodyCenteredCubic, SimpleCubic, HexagonalClosedPacked
    from ase.cluster.icosahedron import Icosahedron
    from ase.cluster.decahedron import Decahedron
    from ase.cluster.octahedron import Octahedron
    from ase.io import write as ase_write
    from pymatgen.io.ase import AseAtomsAdaptor
    _ASE_AVAILABLE = True
except ImportError:
    _ASE_AVAILABLE = False
    warnings.warn("ASE not found. Geometric particle generation (sphere, octahedron, etc.) will be unavailable.")


# ==========================================
# 内部辅助：ASE -> pymatgen 转换
# ==========================================

def _ase_to_pmg(ase_atoms, vacuum: float = 15.0) -> Structure:
    """
    将 ASE Atoms 对象转换为带真空层的 pymatgen Structure。

    Parameters
    ----------
    ase_atoms : ASE Atoms 对象
    vacuum    : 每侧真空层厚度 (Å)，总真空 = vacuum * 2
    """
    if not _ASE_AVAILABLE:
        raise ImportError("ASE is required for this operation.")

    symbols   = ase_atoms.get_chemical_symbols()
    positions = ase_atoms.get_positions()

    center    = positions.mean(axis=0)
    positions = positions - center

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    box_size = (bbox_max - bbox_min) + vacuum * 2

    lattice     = Lattice.orthorhombic(*box_size)
    cart_in_box = positions + box_size / 2
    frac_coords = cart_in_box / box_size

    return Structure(lattice, symbols, frac_coords, coords_are_cartesian=False)


# ==========================================
# 核心类
# ==========================================

class ParticleGenerator:
    """
    纳米粒子结构生成器。
    """

    # 扩充的内置金属数据库：包含催化常用过渡金属的 晶格常数 a (Å) 和 晶格类型
    _ELEMENT_DATABASE: Dict[str, Dict[str, Union[float, str]]] = {
        # === FCC ===
        "Al": {"a": 4.050, "type": "fcc"},
        "Ni": {"a": 3.524, "type": "fcc"},
        "Cu": {"a": 3.615, "type": "fcc"},
        "Rh": {"a": 3.803, "type": "fcc"},
        "Pd": {"a": 3.891, "type": "fcc"},
        "Ag": {"a": 4.085, "type": "fcc"},
        "Ir": {"a": 3.839, "type": "fcc"},
        "Pt": {"a": 3.924, "type": "fcc"},
        "Au": {"a": 4.078, "type": "fcc"},
        "Pb": {"a": 4.950, "type": "fcc"},
        # === BCC ===
        "V":  {"a": 3.030, "type": "bcc"},
        "Cr": {"a": 2.880, "type": "bcc"},
        "Fe": {"a": 2.867, "type": "bcc"},
        "Nb": {"a": 3.300, "type": "bcc"},
        "Mo": {"a": 3.147, "type": "bcc"},
        "Ta": {"a": 3.300, "type": "bcc"},
        "W":  {"a": 3.165, "type": "bcc"},
        # === HCP === (注: c 值在内部由 a * sqrt(8/3) 自动推导)
        "Ti": {"a": 2.951, "type": "hcp"},
        "Co": {"a": 2.507, "type": "hcp"},
        "Zn": {"a": 2.664, "type": "hcp"},
        "Y":  {"a": 3.647, "type": "hcp"},
        "Zr": {"a": 3.232, "type": "hcp"},
        "Ru": {"a": 2.706, "type": "hcp"},
        "Cd": {"a": 2.979, "type": "hcp"},
        "Hf": {"a": 3.195, "type": "hcp"},
        "Re": {"a": 2.761, "type": "hcp"},
        "Os": {"a": 2.735, "type": "hcp"},
    }

    def __init__(
        self,
        element: str,
        lattice_constant: Optional[float] = None,
        lattice_type: Optional[str] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Parameters
        ----------
        element         : 元素符号，如 "Au", "Ru"
        lattice_constant: 晶格常数 (Å)。若为 None，且元素在内置库中，则自动获取；否则报错。
        lattice_type    : "fcc" | "bcc" | "sc" | "hcp"。若为 None，则从内置库推导。
        save_dir        : 默认输出目录
        """
        self.element = element
        db_info = self._ELEMENT_DATABASE.get(element, {})

        # 1. 确定晶格类型 (用户指定 > 数据库推导 > 默认 fcc)
        if lattice_type is not None:
            self.lattice_type = lattice_type.lower()
        elif "type" in db_info:
            self.lattice_type = db_info["type"]
        else:
            self.lattice_type = "fcc"
            warnings.warn(f"Unknown element '{element}', defaulting lattice_type to 'fcc'.")

        # 2. 确定晶格常数 (用户指定 > 数据库推导 > 报错)
        if lattice_constant is not None:
            self.lattice_constant = lattice_constant
        elif "a" in db_info:
            self.lattice_constant = db_info["a"]
        else:
            raise ValueError(
                f"Element '{element}' is not in the built-in database. "
                f"You MUST explicitly provide 'lattice_constant' (e.g., lattice_constant=3.5)."
            )

        self.save_dir = (
            Path(save_dir) if save_dir
            else Path.cwd() / f"particle_{element}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self._structures: List[Dict[str, Any]] = []

        print(f"Initialized ParticleGenerator: {element}, a={self.lattice_constant:.3f} Å, type={self.lattice_type}")

    # ==========================================
    # run_from_dict：配置驱动入口
    # ==========================================

    @staticmethod
    def run_from_dict(config: Dict[str, Any]) -> List[Structure]:
        element = config.get("element")
        if not element:
            raise ValueError("Config error: 'element' is required.")

        mode = config.get("mode", "wulff").lower()
        valid_modes = ["wulff", "sphere", "octahedron", "decahedron", "icosahedron", "fcc_cube", "rod"]
        if mode not in valid_modes:
            raise ValueError(f"Config error: Invalid mode '{mode}'. Must be one of {valid_modes}.")

        # 注意：这里使用 get，如果字典里没有，会传入 None，从而触发 __init__ 中的智能推导
        gen = ParticleGenerator(
            element=element,
            lattice_constant=config.get("lattice_constant"),
            lattice_type=config.get("lattice_type"),
            save_dir=config.get("save_dir"),
        )

        if mode == "wulff":
            params = config.get("wulff_params", {})
            if not params.get("surface_energies"):
                raise ValueError("Config error: 'surface_energies' is required in 'wulff_params'.")
            gen.wulff(
                surface_energies=params["surface_energies"],
                size=params.get("size", 30.0),
                vacuum=params.get("vacuum", 15.0),
            )

        elif mode == "sphere":
            params = config.get("sphere_params", {})
            if "radius" not in params:
                raise ValueError("Config error: 'radius' is required in 'sphere_params'.")
            gen.sphere(**params)

        elif mode == "octahedron":
            params = config.get("octahedron_params", {})
            if "layers" not in params:
                raise ValueError("Config error: 'layers' is required in 'octahedron_params'.")
            gen.octahedron(**params)

        elif mode == "decahedron":
            params = config.get("decahedron_params", {})
            for k in ["p", "q", "r"]:
                if k not in params:
                    raise ValueError(f"Config error: '{k}' is required in 'decahedron_params'.")
            gen.decahedron(**params)

        elif mode == "icosahedron":
            params = config.get("icosahedron_params", {})
            if "n_shells" not in params:
                raise ValueError("Config error: 'n_shells' is required in 'icosahedron_params'.")
            gen.icosahedron(**params)

        elif mode == "fcc_cube":
            params = config.get("fcc_cube_params", {})
            if "surfaces" not in params or "layers" not in params:
                raise ValueError("Config error: 'surfaces' and 'layers' are required in 'fcc_cube_params'.")
            gen.fcc_cube(**params)

        elif mode == "rod":
            params = config.get("rod_params", {})
            for k in ["radius", "length"]:
                if k not in params:
                    raise ValueError(f"Config error: '{k}' is required in 'rod_params'.")
            gen.rod(**params)

        save_opts = config.get("save_options", {})
        if save_opts.get("save", True) and gen._structures:
            gen.save_all(
                filename=save_opts.get("filename", "POSCAR"),
                fmt=save_opts.get("fmt", "poscar"),
            )

        return gen.get_structures()
    
    # ==========================================
    # Wulff 构型（核心功能）
    # ==========================================

    def wulff(
        self,
        surface_energies: Dict[Union[str, Tuple], float],
        size: float = 30.0,
        vacuum: float = 15.0,
    ) -> "ParticleGenerator":
        """
        基于 Wulff 构型生成平衡形状纳米粒子。

        Parameters
        ----------
        surface_energies : 晶面 -> 表面能 (J/m²) 的映射。
                           支持格式："111", "1,1,1", (1,1,1), [1,1,1]
        size             : 粒子近似半径 (Å)，控制粒子大小
        vacuum           : 每侧真空层厚度 (Å)，默认 15.0 Å
        """
        miller_list, energy_list = self._parse_surface_energies(surface_energies)

        pmg_lattice = self._build_pmg_lattice()
        wulff_shape = WulffShape(pmg_lattice, miller_list, energy_list)

        print(f"Wulff shape built. Anisotropy: {wulff_shape.anisotropy:.4f}, "
              f"Weighted surface energy: {wulff_shape.weighted_surface_energy:.4f} J/m²")
        print(f"Area fractions: { {str(k): round(v, 3) for k, v in wulff_shape.area_fraction_dict.items()} }")

        all_points  = self._generate_lattice_points(size)
        inside_mask = self._filter_by_wulff(all_points, wulff_shape, size)
        positions   = all_points[inside_mask]

        if len(positions) == 0:
            raise ValueError("No atoms inside Wulff shape. Try increasing 'size'.")

        structure = self._build_structure(positions, vacuum=vacuum)
        label     = f"wulff_n{len(positions)}"

        self._structures.append({"structure": structure, "label": label})
        print(f"Generated Wulff particle: {len(positions)} atoms, "
              f"vacuum={vacuum} Å/side, label='{label}'")
        return self

    # ==========================================
    # ASE 几何形状方法
    # ==========================================

    def sphere(self, radius: float, vacuum: float = 15.0) -> "ParticleGenerator":
        """
        生成球形纳米粒子。

        Parameters
        ----------
        radius : 球半径 (Å)
        vacuum : 每侧真空层厚度 (Å)，默认 15.0
        """
        all_points = self._generate_lattice_points(radius)
        distances  = np.linalg.norm(all_points, axis=1)
        positions  = all_points[distances <= radius]

        structure = self._build_structure(positions, vacuum=vacuum)
        self._structures.append({"structure": structure, "label": f"sphere_r{radius:.1f}"})
        print(f"Generated sphere: {len(positions)} atoms, r={radius} Å, vacuum={vacuum} Å/side")
        return self

    def octahedron(
        self,
        layers: List[int],
        latticeconstant: Optional[float] = None,
        vacuum: float = 15.0,
    ) -> "ParticleGenerator":
        """
        生成八面体纳米粒子（ASE）。

        Parameters
        ----------
        layers          : [length, cutoff]，cutoff 默认 0（完整八面体）
        latticeconstant : 晶格常数，None 时使用初始化值
        vacuum          : 每侧真空层厚度 (Å)，默认 15.0
        """
        if not _ASE_AVAILABLE:
            raise ImportError("ASE is required for octahedron generation.")

        a      = latticeconstant or self.lattice_constant
        length = layers[0]
        cutoff = layers[1] if len(layers) > 1 else 0

        ase_atoms = Octahedron(self.element, length=length, cutoff=cutoff, latticeconstant=a)
        structure = _ase_to_pmg(ase_atoms, vacuum=vacuum)

        label = f"octahedron_l{length}_c{cutoff}"
        self._structures.append({"structure": structure, "label": label})
        print(f"Generated octahedron: {len(ase_atoms)} atoms, "
              f"length={length}, cutoff={cutoff}, vacuum={vacuum} Å/side")
        return self

    def decahedron(
        self,
        p: int,
        q: int,
        r: int,
        latticeconstant: Optional[float] = None,
        vacuum: float = 15.0,
    ) -> "ParticleGenerator":
        """
        生成十面体纳米粒子（ASE Ino decahedron）。

        Parameters
        ----------
        p               : (100) 面的层数
        q               : (111) 面的层数
        r               : 腰部重构层数（0 = 无重构）
        latticeconstant : 晶格常数
        vacuum          : 每侧真空层厚度 (Å)，默认 15.0
        """
        if not _ASE_AVAILABLE:
            raise ImportError("ASE is required for decahedron generation.")

        a         = latticeconstant or self.lattice_constant
        ase_atoms = Decahedron(self.element, p=p, q=q, r=r, latticeconstant=a)
        structure = _ase_to_pmg(ase_atoms, vacuum=vacuum)

        label = f"decahedron_p{p}_q{q}_r{r}"
        self._structures.append({"structure": structure, "label": label})
        print(f"Generated decahedron: {len(ase_atoms)} atoms, "
              f"p={p}, q={q}, r={r}, vacuum={vacuum} Å/side")
        return self

    def icosahedron(
        self,
        n_shells: int,
        latticeconstant: Optional[float] = None,
        vacuum: float = 15.0,
    ) -> "ParticleGenerator":
        """
        生成二十面体纳米粒子（ASE Mackay icosahedron）。

        Parameters
        ----------
        n_shells        : 壳层数（1=13原子, 2=55原子, 3=147原子）
        latticeconstant : 最近邻距离，None 时从晶格常数推导（FCC: a/√2）
        vacuum          : 每侧真空层厚度 (Å)，默认 15.0
        """
        if not _ASE_AVAILABLE:
            raise ImportError("ASE is required for icosahedron generation.")

        a         = latticeconstant or (self.lattice_constant / np.sqrt(2))
        ase_atoms = Icosahedron(self.element, noshells=n_shells, latticeconstant=a)
        structure = _ase_to_pmg(ase_atoms, vacuum=vacuum)

        label = f"icosahedron_n{n_shells}"
        self._structures.append({"structure": structure, "label": label})
        print(f"Generated icosahedron: {len(ase_atoms)} atoms, "
              f"shells={n_shells}, vacuum={vacuum} Å/side")
        return self

    def fcc_cube(
        self,
        surfaces: List[List[int]],
        layers: List[int],
        latticeconstant: Optional[float] = None,
        vacuum: float = 15.0,
    ) -> "ParticleGenerator":
        """
        生成 FCC 立方体纳米粒子（ASE FaceCenteredCubic）。

        Parameters
        ----------
        surfaces        : 晶面列表，如 [[1,0,0], [1,1,0], [1,1,1]]
        layers          : 各晶面对应的层数，如 [4, 3, 2]
        latticeconstant : 晶格常数
        vacuum          : 每侧真空层厚度 (Å)，默认 15.0
        """
        if not _ASE_AVAILABLE:
            raise ImportError("ASE is required for fcc_cube generation.")
        if self.lattice_type != "fcc":
            warnings.warn(f"fcc_cube is designed for FCC lattice, but lattice_type='{self.lattice_type}'.")

        a         = latticeconstant or self.lattice_constant
        ase_atoms = FaceCenteredCubic(self.element, surfaces=surfaces, layers=layers, latticeconstant=a)
        structure = _ase_to_pmg(ase_atoms, vacuum=vacuum)

        label = f"fcc_cube_{'x'.join(map(str, layers))}"
        self._structures.append({"structure": structure, "label": label})
        print(f"Generated FCC cube: {len(ase_atoms)} atoms, vacuum={vacuum} Å/side")
        return self

    def rod(
        self,
        radius: float,
        length: float,
        vacuum: float = 15.0,
    ) -> "ParticleGenerator":
        """
        生成纳米棒（基于晶格点筛选，沿 z 轴）。

        Parameters
        ----------
        radius : 截面半径 (Å)
        length : 棒长 (Å)
        vacuum : 每侧真空层厚度 (Å)，默认 15.0
        """
        max_dim    = max(radius, length / 2) * 1.2
        all_points = self._generate_lattice_points(max_dim)

        xy_dist   = np.linalg.norm(all_points[:, :2], axis=1)
        z_dist    = np.abs(all_points[:, 2])
        positions = all_points[(xy_dist <= radius) & (z_dist <= length / 2)]

        structure = self._build_structure(positions, vacuum=vacuum)
        label     = f"rod_r{radius:.1f}_l{length:.1f}"
        self._structures.append({"structure": structure, "label": label})
        print(f"Generated rod: {len(positions)} atoms, "
              f"r={radius} Å, l={length} Å, vacuum={vacuum} Å/side")
        return self

    # ==========================================
    # 状态获取
    # ==========================================

    def get_structures(self) -> List[Structure]:
        """获取所有已生成结构的列表。"""
        return [item["structure"].copy() for item in self._structures]

    def get_structure(self, index: int = -1) -> Structure:
        """获取指定索引的结构（默认最后一个）。"""
        if not self._structures:
            raise ValueError("No structures generated yet.")
        return self._structures[index]["structure"].copy()

    # ==========================================
    # 保存
    # ==========================================

    def save_all(
        self,
        filename: Union[str, Path] = "POSCAR",
        fmt: str = "poscar",
        as_subdirs: bool = True,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> "ParticleGenerator":
        """
        保存所有已生成的结构。

        路径解析优先级（从高到低）：
        1. filename 为绝对路径或含父目录的路径 → 直接使用（忽略 save_dir）
        2. save_dir 参数显式指定 → 保存到 save_dir / filename
        3. 初始化时的 self.save_dir → 保存到 self.save_dir / filename

        Parameters
        ----------
        filename  : 文件名或完整路径
        fmt       : 输出格式，"poscar" 或 pymatgen 支持的其他格式
        as_subdirs: 多结构时是否按 label 建子目录（单结构时自动忽略）
        save_dir  : 临时覆盖默认输出目录，不修改 self.save_dir
        """
        if not self._structures:
            print("Warning: No structures to save.")
            return self

        filename             = Path(filename)
        effective_save_dir   = Path(save_dir) if save_dir else self.save_dir
        filename_has_dir     = filename.is_absolute() or (filename.parent != Path("."))

        for item in self._structures:
            struct = item["structure"]
            label  = item["label"]
            n      = len(self._structures)

            if filename_has_dir:
                out_path = filename if n == 1 else filename.parent / f"{self.element}_{label}" / filename.name
            else:
                if n == 1 or not as_subdirs:
                    out_path = effective_save_dir / filename
                else:
                    out_path = effective_save_dir / f"{self.element}_{label}" / filename

            out_path.parent.mkdir(parents=True, exist_ok=True)

            if fmt.lower() == "poscar":
                with open(out_path, mode="wt", encoding="utf-8") as f:
                    f.write(Poscar(struct).get_string())
            else:
                struct.to(filename=str(out_path), fmt=fmt)

            print(f"  Saved: {out_path}")

        print(f"Saved {len(self._structures)} structure(s).")
        return self

    def to_xyz(self, index: int = -1, filename: Optional[Union[str, Path]] = None) -> str:
        """
        将指定结构导出为 XYZ 格式字符串（可选写入文件）。

        Parameters
        ----------
        index    : 结构索引，默认最后一个
        filename : 输出文件路径，None 时仅返回字符串；父目录不存在时自动创建
        """
        struct = self.get_structure(index)
        lines  = [str(len(struct)), self._structures[index]["label"]]
        for site in struct:
            x, y, z = site.coords
            lines.append(f"{site.specie.symbol:2s} {x:15.8f} {y:15.8f} {z:15.8f}")
        content = "\n".join(lines)
        if filename:
            filename = Path(filename)
            filename.parent.mkdir(parents=True, exist_ok=True)  # ← 修复点
            filename.write_text(content, encoding="utf-8")
        return content

    # ==========================================
    # 内部辅助方法
    # ==========================================

    def _build_pmg_lattice(self) -> Lattice:
        """根据晶格类型构建 pymatgen Lattice 对象。"""
        a = self.lattice_constant
        if self.lattice_type in ("fcc", "bcc", "sc"):
            return Lattice.cubic(a)
        elif self.lattice_type == "hcp":
            c = a * np.sqrt(8 / 3)
            return Lattice.hexagonal(a, c)
        else:
            warnings.warn(f"Unknown lattice_type '{self.lattice_type}', falling back to cubic.")
            return Lattice.cubic(a)

    def _generate_lattice_points(self, size: float) -> np.ndarray:
        """生成覆盖指定尺寸范围的晶格点（笛卡尔坐标）。"""
        a = self.lattice_constant
        n = int(np.ceil(size / a)) + 2

        basis_map = {
            "fcc": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
                             [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]),
            "bcc": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            "sc":  np.array([[0.0, 0.0, 0.0]]),
            "hcp": np.array([[0.0, 0.0, 0.0], [1/3, 2/3, 0.5]]),
        }
        basis = basis_map.get(self.lattice_type, basis_map["fcc"])

        if self.lattice_type == "hcp":
            c       = a * np.sqrt(8 / 3)
            lat_mat = np.array([[a, 0, 0], [a * 0.5, a * np.sqrt(3) / 2, 0], [0, 0, c]])
        else:
            lat_mat = np.eye(3) * a

        points = []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                for k in range(-n, n + 1):
                    for b in basis:
                        cart = lat_mat.T @ (np.array([i, j, k]) + b)
                        points.append(cart)

        return np.array(points)

    def _filter_by_wulff(
        self,
        points: np.ndarray,
        wulff_shape: "WulffShape",
        size: float,
    ) -> np.ndarray:
        """筛选位于 Wulff 形状内部的晶格点。"""
        normals   = []
        distances = []

        for facet in wulff_shape.facets:
            n    = np.array(facet.normal)
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            normals.append(n / norm)
            distances.append(facet.e_surf)

        if not normals:
            return np.linalg.norm(points, axis=1) <= size

        normals          = np.array(normals)
        distances        = np.array(distances)
        scale            = size / distances.min()
        scaled_distances = distances * scale

        projections = points @ normals.T
        return np.all(projections <= scaled_distances[np.newaxis, :], axis=1)

    @staticmethod
    def _parse_surface_energies(
        surface_energies: Dict[Union[str, Tuple], float]
    ) -> Tuple[List[Tuple], List[float]]:
        """解析表面能字典，返回 (miller_list, energy_list)。"""
        miller_list = []
        energy_list = []

        for key, energy in surface_energies.items():
            if isinstance(key, str):
                key_clean = key.replace(",", " ").replace("(", "").replace(")", "")
                parts     = key_clean.split()
                if len(parts) == 1 and len(parts[0]) == 3:
                    hkl = tuple(int(c) for c in parts[0])
                elif len(parts) == 3:
                    hkl = tuple(int(p) for p in parts)
                else:
                    raise ValueError(f"Cannot parse miller index from string: '{key}'")
            elif isinstance(key, (tuple, list)):
                hkl = tuple(int(x) for x in key)
            else:
                raise ValueError(f"Unsupported miller index type: {type(key)}")

            if len(hkl) != 3:
                raise ValueError(f"Miller index must have 3 components, got: {hkl}")

            miller_list.append(hkl)
            energy_list.append(float(energy))

        return miller_list, energy_list

    def _build_structure(self, positions: np.ndarray, vacuum: float = 15.0) -> Structure:
        """
        将笛卡尔坐标点集构建为 pymatgen Structure（非周期性大盒子）。

        Parameters
        ----------
        positions : 原子笛卡尔坐标 (N, 3)
        vacuum    : 每侧真空层厚度 (Å)，盒子尺寸 = 粒子范围 + vacuum * 2
        """
        if len(positions) == 0:
            raise ValueError("Cannot build structure from empty positions.")

        center    = positions.mean(axis=0)
        positions = positions - center

        bbox_min    = positions.min(axis=0)
        bbox_max    = positions.max(axis=0)
        box_size    = (bbox_max - bbox_min) + vacuum * 2

        lattice     = Lattice.orthorhombic(*box_size)
        cart_in_box = positions + box_size / 2
        frac_coords = cart_in_box / box_size

        species = [self.element] * len(positions)
        return Structure(lattice, species, frac_coords, coords_are_cartesian=False)