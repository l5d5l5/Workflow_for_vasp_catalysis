#!/usr/bin/env python
"""
基本用法示例 - VASP催化工具包

本示例展示了vasp_catalysis_tools包的三个主要组件的基本用法：
1. BulkToSlabGenerator - 从体相结构生成表面
2. AdsorptionModify - 吸附位点分析与修饰
3. StructureModify - 结构修改（掺杂、空位等）
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Element
from pymatgen.io.vasp import Poscar

# 导入我们的工具包
from vasp_catalysis_tools import BulkToSlabGenerator, AdsorptionModify, StructureModify


def bulk_to_slab_example(bulk_file='POSCAR', output_dir='output_slabs'):
    """展示如何从体相结构生成表面"""
    print("\n=== 从体相结构生成表面 ===")
    
    # 方法1：使用类实例直接调用方法
    gen = BulkToSlabGenerator(
        structure_source=bulk_file, 
        save_dir=output_dir,
        standardize=True
    )
    
    slabs = gen.generate(
        miller_indices="111", 
        target_layers=4,
        vacuum_thickness=15.0, 
        supercell_matrix="2x2",
        fix_bottom_layers=2
    )
    
    print(f"生成了 {len(slabs)} 个表面结构")
    
    # 方法2：使用配置字典（适合批量处理）
    config = {
        "structure_source": bulk_file,
        "save_dir": output_dir,
        "standardize_bulk": True,
        "generate_params": {
            "miller_indices": "100",
            "target_layers": 4,
            "vacuum_thickness": 15.0,
            "supercell_matrix": "2x2",
            "fix_bottom_layers": 2
        },
        "save_options": {
            "save": True,
            "filename_prefix": "POSCAR_slab"
        }
    }
    
    slabs2 = BulkToSlabGenerator.run_from_dict(config)
    print(f"通过配置字典生成了 {len(slabs2)} 个表面结构")
    
    return slabs[0] if slabs else None


def adsorption_example(slab_file='output_slabs/POSCAR_111_4L.vasp'):
    """展示如何分析吸附位点与添加吸附物"""
    print("\n=== 吸附位点分析与修饰 ===")
    
    # 初始化吸附分析工具
    ads_tool = AdsorptionModify(
        slab_source=slab_file,
        save_dir="adsorption_results"
    )
    
    # 方法1：查找吸附位点
    sites = ads_tool.find_adsorption_sites()
    print(f"找到的吸附位点数量: ontop={len(sites['ontop'])}, "
          f"bridge={len(sites['bridge'])}, hollow={len(sites['hollow'])}")
    
    # 分析特定吸附位点
    if sites['ontop']:
        site_info = ads_tool.describe_adsorption_site(ads_tool.slab, sites['ontop'][0])
        print(f"Top位点配位环境: {site_info['species_count']}")
    
    # 方法2：通过配置字典（适合批量处理）
    config = {
        "target_slab_source": slab_file,
        "save_dir": "adsorption_results",
        "mode": "generate",
        "generate_params": {
            "molecule_formula": "CO",  # 使用ase生成CO分子
            "reorient": True
        },
        "plot": True,
        "plot_params": {
            "figsize": (8, 8)
        }
    }
    
    try:
        result = AdsorptionModify.run_from_dict(config)
        print(f"生成了 {result.get('generated_count', 0)} 个吸附结构")
    except Exception as e:
        print(f"生成吸附结构时出错: {e}")


def structure_modify_example(structure_file='POSCAR'):
    """展示如何修改结构（掺杂、空位等）"""
    print("\n=== 结构修改 ===")
    
    # 加载结构
    structure = Structure.from_file(structure_file)
    modifier = StructureModify(structure)
    
    # 获取原子层信息
    layers = modifier.get_layers()
    print(f"结构中共有 {len(layers)} 个原子层")
    
    # 随机生成掺杂结构
    metal_element = next((el for el in structure.composition.elements 
                         if el.is_metal), Element("Pt"))
    
    print(f"生成随机掺杂结构，替换 {metal_element} 为 Au")
    doped_structures = StructureModify.generate(
        structure=structure,
        substitute_element=metal_element,
        dopant="Au",
        dopant_num=1,
        num_structs=3
    )
    
    # 生成空位结构
    print(f"生成随机空位结构，移除 {metal_element}")
    vacancy_structures = StructureModify.generate(
        structure=structure,
        substitute_element=metal_element,
        dopant=None,  # None表示生成空位
        dopant_num=1,
        num_structs=3
    )
    
    # 保存一个示例结构
    if doped_structures:
        output_dir = Path("modified_structures")
        output_dir.mkdir(parents=True, exist_ok=True)
        Poscar(doped_structures[0]).write_file(output_dir / "doped_structure.vasp")
        print(f"掺杂结构已保存到 {output_dir / 'doped_structure.vasp'}")
    
    return doped_structures


if __name__ == "__main__":
    # 查找默认的POSCAR文件
    poscar_file = "POSCAR"
    for file in ["POSCAR", "CONTCAR", "../POSCAR", "examples/POSCAR"]:
        if os.path.exists(file):
            poscar_file = file
            break
    
    print(f"使用结构文件: {poscar_file}")
    
    try:
        # 运行体相到表面的示例
        slab = bulk_to_slab_example(poscar_file)
        
        # 如果成功生成了表面，使用它来运行吸附示例
        if slab is not None:
            first_slab_file = Path("output_slabs").glob("*_111_*L*.vasp")
            first_slab_file = next(first_slab_file, None)
            if first_slab_file:
                adsorption_example(first_slab_file)
        
        # 运行结构修改示例
        structure_modify_example(poscar_file)
        
        print("\n所有示例运行完成!")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()