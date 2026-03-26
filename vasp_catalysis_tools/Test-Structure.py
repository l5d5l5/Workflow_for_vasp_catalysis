import pytest
import warnings
from pathlib import Path
from typing import List

from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import Slab
from pymatgen.io.vasp import Poscar

# ⚠️ 注意：这里请确保导入路径与你 Jupyter 当前工作目录下的实际路径一致！
# 如果你的 BulkToSlabGenerator 和 load_structure 就在当前 Notebook 里定义了，
# 你可能需要把它们也复制到这个测试文件里，或者确保它们在 src/ 目录下。
from vasp_catalysis_tools.utils.structure_utils import load_structure
from vasp_catalysis_tools.core.bulk_to_slab import BulkToSlabGenerator

@pytest.fixture
def dummy_bulk() -> Structure:
    lattice = Lattice.cubic(3.61)
    coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    return Structure(lattice, ["Cu"] * 4, coords)

@pytest.fixture
def dummy_poscar_file(tmp_path: Path, dummy_bulk: Structure) -> Path:
    file_path = tmp_path / "POSCAR"
    Poscar(dummy_bulk).write_file(file_path)
    return file_path

class TestLoadStructure:
    def test_load_from_structure_object(self, dummy_bulk):
        loaded = load_structure(dummy_bulk)
        assert isinstance(loaded, Structure)
        assert loaded is not dummy_bulk
        assert len(loaded) == len(dummy_bulk)

    def test_load_from_file(self, dummy_poscar_file):
        loaded = load_structure(dummy_poscar_file)
        assert isinstance(loaded, Structure)
        assert len(loaded) == 4

    def test_load_from_directory_priority(self, tmp_path, dummy_bulk):
        Poscar(dummy_bulk).write_file(tmp_path / "POSCAR")
        contcar_struct = dummy_bulk.copy()
        contcar_struct.make_supercell([2, 1, 1])
        Poscar(contcar_struct).write_file(tmp_path / "CONTCAR")
        loaded = load_structure(tmp_path)
        assert len(loaded) == 8

    def test_load_empty_file_handling(self, tmp_path, dummy_bulk):
        (tmp_path / "CONTCAR").touch()
        Poscar(dummy_bulk).write_file(tmp_path / "POSCAR")
        loaded = load_structure(tmp_path)
        assert len(loaded) == 4

    def test_load_multiple_cif_error(self, tmp_path):
        (tmp_path / "1.cif").touch()
        (tmp_path / "2.cif").touch()
        with pytest.raises(FileNotFoundError, match="Multiple CIF files found"):
            load_structure(tmp_path)

class TestBulkToSlabGenerator:
    def test_initialization(self, dummy_bulk):
        gen = BulkToSlabGenerator(dummy_bulk)
        assert isinstance(gen.bulk_structure, Structure)
        gen_no_std = BulkToSlabGenerator(dummy_bulk, standardize=False)
        assert gen_no_std.bulk_structure == dummy_bulk

    def test_normalize_miller_indices(self, dummy_bulk):
        gen = BulkToSlabGenerator(dummy_bulk)
        assert gen._normalize_miller_indices(111) == (1, 1, 1)
        assert gen._normalize_miller_indices("1 -1 0") == (1, -1, 0)
        with pytest.raises(ValueError):
            gen._normalize_miller_indices("invalid")

    def test_generate_and_fluent_api(self, dummy_bulk):
        gen = BulkToSlabGenerator(dummy_bulk)
        result = (
            gen.generate(miller_indices="111", target_layers=3, vacuum_thickness=10.0)
               .make_supercell("2x2x1")
               .set_fixation(fix_bottom_layers=1)
        )
        assert result is gen
        slabs = gen.get_slabs()
        assert len(slabs) > 0
        slab = gen.get_slab(0)
        assert "selective_dynamics" in slab.site_properties
        sd = slab.site_properties["selective_dynamics"]
        assert [False, False, False] in sd
        assert [True, True, True] in sd

    def test_trim_too_thin_error(self, dummy_bulk):
        """测试如果要求的层数大于实际层数，_trim_to_target_layers 是否会抛出异常"""
        gen = BulkToSlabGenerator(dummy_bulk)
        
        # 1. 先正常生成一个只有 3 层的 slab
        gen.generate(miller_indices="111", target_layers=3)
        small_slab = gen.get_slab(0)
        
        # 2. 故意尝试将这个 3 层的 slab 裁剪到 10 层，这必然会触发 ValueError
        with pytest.raises(ValueError, match="Slab too thin"):
            gen._trim_to_target_layers(small_slab, target_layers=10)

    def test_select_termination(self, dummy_bulk):
        gen = BulkToSlabGenerator(dummy_bulk).generate("111", target_layers=3)
        if len(gen.get_slabs()) > 1:
            gen.select_termination(0)
            assert len(gen.get_slabs()) == 1

    def test_save_slab(self, tmp_path, dummy_bulk):
        gen = BulkToSlabGenerator(dummy_bulk, save_dir=tmp_path)
        gen.generate("100", target_layers=2)
        gen.save_slab(gen.get_slab(0), "POSCAR_test")
        assert (tmp_path / "POSCAR_test").exists()

    def test_run_from_dict(self, tmp_path, dummy_poscar_file):
        config = {
            "structure_source": str(dummy_poscar_file),
            "save_dir": str(tmp_path),
            "standardize_bulk": True,
            "generate_params": {
                "miller_indices": "111",
                "target_layers": 3,
                "vacuum_thickness": 12.0,
                "supercell_matrix": "2x2",
                "fix_bottom_layers": 1
            },
            "save_options": {"save": True, "filename_prefix": "TEST_SLAB"}
        }
        slabs = BulkToSlabGenerator.run_from_dict(config)
        assert len(slabs) > 0
        assert len(list(tmp_path.glob("TEST_SLAB_111_3L*.vasp"))) == len(slabs)