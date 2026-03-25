import sys
from pymatgen.core import Structure
from pathlib import Path
cwd = Path.cwd()
if cwd.name == "vasp_catalysis_tools":
    sys.path.insert(0, str(cwd.parent))
elif (cwd / "vasp_catalysis_tools").exists():
    sys.path.insert(0, str(cwd))
from vasp_catalysis_tools.core.structure_modify import StructureModify


output_path = Path("/data2/home/luodh/Test/VaspStructure-Test/").resolve()
sture = Structure.from_file("/data2/home/luodh/Test/VaspStructure-Test/POSCAR")
modifier = StructureModify(sture)
##单使用-supercell
sture2 = modifier.make_supercell("2x2x2").get_structure()
sture2.to(output_path / "supercell-POSCAR", fmt="poscar")
##单使用-insert
strure3 = modifier.reset_to_initial().insert_atom("Pd", [0,0,0]).get_structure()
strure3.to(output_path / "insert-POSCAR", fmt="POSCAR")
##单使用-修改元素
strure4 = modifier.reset_to_initial().replace_species_all({"Al":"Cu"}).get_structure()
strure4.to(output_path / "replace-POSCAR", fmt="POSCAR")
##单使用-精准原子操作
strure5 = modifier.reset_to_initial().modify_atom_element(index=0, new_element="Cu").get_structure()
strure5.to(output_path / "replace-single-POSCAR", fmt="POSCAR")
##原子位置移动使用
sture6 = modifier.reset_to_initial().modify_atom_coords(index=0, coords=[0, 0, 0.1], frac_coords=True).get_structure()
sture6.to(output_path / "move-POSCAR", fmt="POSCAR")
##多原子位置移动使用
sture7 = modifier.reset_to_initial().batch_modify_coords(indices=[1, 2], coords_list=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.6]]).get_structure()
sture7.to(output_path / "batch-move-POSCAR", fmt="POSCAR")
##晶格矩阵的转变
sture8 = modifier.reset_to_initial().modify_lattice(a=4.0, c=15.0).get_structure()
sture8.to(output_path / "lattice-POSCAR", fmt="POSCAR")
##施加变换矩阵
strure9 = modifier.reset_to_initial().transform_structure([[1, 0, 0], 
                              [0, 2, 0], 
                              [0, 0, 1]]).get_structure()
strure9.to(output_path / "transform-POSCAR", fmt="POSCAR")