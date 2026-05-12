import os
import sys
import importlib
from pathlib import Path

# Ensure the package root is on sys.path
cwd = Path.cwd()
if cwd.name == "vasp_inputs_mod":
    sys.path.insert(0, str(cwd.parent))
elif (cwd / "vasp_inputs_mod").exists():
    sys.path.insert(0, str(cwd))
else:
    print(f"[Warning] Unexpected cwd: {cwd}")

# 导入模块
import kpoints
import vasp_inputs_mod.api as api_module
importlib.reload(api_module)                    # 强制重载最新代码
from vasp_inputs_mod.api import generate_inputs
from vasp_inputs_mod.workflow_engine import WorkflowEngine, WorkflowConfig

structure_path = "/data2/home/luodh/Test/VaspInput-Test/Slab-calc/slab-optim"
def test_md_nvt():
    """测试 NVT 系综 MD 输入生成"""
    print("=" * 50)
    print("Testing md_nvt...")

    out = generate_inputs(
        "md_nvt",
        structure=structure_path,
        functional="PBE",
        kpoints_density=1.0,
        incar={
            "TEBEG":  300,
            "TEEND":  1000,
            "NSW":    10000,
            "POTIM":  2.0,
            "MDALGO": 2,
            "SMASS":  -3,
            "NPAR":   4,
        },
    )

    assert out is not None, "md_nvt: generate_inputs 返回 None"
    print("[PASS] md_nvt generate_inputs succeeded.")
    print(f"  Output: {out}")
    return out


def test_md_npt():
    """测试 NPT 系综 MD 输入生成"""
    print("=" * 50)
    print("Testing md_npt...")

    out = generate_inputs(
        "md_npt",
        structure=structure_path,
        prev_dir=structure_path,
        functional="beef",
        kpoints_density=1.0,
        incar={
            "TEBEG":   300,
            "TEEND":   300,
            "NSW":     5000,
            "POTIM":   2.0,
            "PSTRESS": 0.0,
            "NPAR":    4,
        },
    )

    assert out is not None, "md_npt: generate_inputs 返回 None"
    print("[PASS] md_npt generate_inputs succeeded.")
    print(f"  Output: {out}")
    return out

def test_freq_ir():
    out = generate_inputs(
    "freq_ir",
    structure=structure_path,
    functional="beef",
    kpoints_density=50.0,
    incar={
        "POTIM": 0.015,
        "NFREE": 2,
        "NPAR":  4,
    },
)

    assert out is not None, "freq_ir generate failed, 返回None"
    print("[PASS] freq_ir generate_inputs succeeded.")
    print(f"  Output: {out}")
    return out

def test_nbo():
    out = generate_inputs(
    "nbo",
    structure=structure_path,   # 显式结构；提供 prev_dir 时可省略
    prev_dir=structure_path, #传入prev_dir，复制基础的INCAR设置进入此。
    functional="BEEF",
    kpoints_density=50.0,
    incar={
        "ENCUT":  520,
        "EDIFF":  1e-6,
        "LWAVE":  True,           # 写出 WAVECAR 供 NBO 后处理使用
        "NPAR":   4,
    },
    nbo_config={
        "occ_1c": 1.60,           # 单中心占据阈值
        "occ_2c": 1.85,           # 双中心占据阈值
    },
)

    assert out is not None, "neb generate failed, 返回None"
    print("[PASS] nbo generate_inputs succeeded.")
    print(f"  Output: {out}")
    return out

def test_lobster():
    out = generate_inputs(
    "lobster",
    structure=structure_path,
    prev_dir=structure_path,
    functional="BEEF",
    kpoints_density=50.0,
    incar={
        "ENCUT":  520,
        "NPAR":   4,
    },
    cohp_generator=[
        "from 1.8 to 2.3 type Fe type O orbitalwise",
        "from 1.1 to 1.5 type C  type O orbitalwise",
    ],
    lobsterin={
        "COHPstartEnergy": -20.0,
        "COHPendEnergy":    20.0,
    },
)
    assert out is not None, "lobster generate failed, 返回None"
    print("[PASS] lobster generate_inputs succeeded.")
    print(f"  Output: {out}")
    return out

def test_neb():
    engine = WorkflowEngine()
    out = engine.run(WorkflowConfig(
        calc_type="neb",
        start_structure=structure_path,   # 初态结构
        end_structure=structure_path,     # 末态结构
        n_images=6,                           # 中间像数量
        use_idpp=False,                    
        functional="BEEF",
        kpoints_density=25.0,
        user_incar_overrides={
            "SPRING": -5,    # NEB 弹簧常数（eV/Å²）；负值 = 切向弹簧
            "NPAR":    4,
            "MAGMOM":   {"Ti": 3.0, "O": 0.4},
            "ISPIN":    2,
            "LDAU":     True,
            "LDAUTYPE": 2,
            "LDAUU":    {"Ti": 3.0, "O": 0.0},
            "LDAUL":    {"Ti": 2,   "O": -1},
            "LDAUJ":    {"Ti": 0.0, "O": 0.0},
        },
        output_dir="neb_run/",
    ))
    assert out is not None, "neb generate failed, 返回None"
    print("[PASS] neb generate_inputs succeeded.")
    print(f"  Output: {out}")
    return out

def test_slab_relax():
    out = generate_inputs(
        "slab_relax",
        structure=structure_path,
        functional="SCAN",
        kpoints_density=50.0,
        prev_dir=structure_path,
        magmom = {"Ti":3.0, "O":0.4},
        dft_u = {"Ti":3.0, "O":0.2},
        incar={
            "NSW":55,
            "EDIFF":1E5,
            "ENCUT":520,
        },
    )
    assert out is not None, "slab_relax generate failed."
    print("[PASS] slab_relax generate_inputs succeeded.")
    print(f" Ouput: {out}")
    return out

def test_bulk_relax():
    out = generate_inputs(
        "bulk_relax",
        structure=structure_path,
        functional="BEEF",
        kpoints_density=60.0,
        prev_dir=structure_path,
        magmom = {"Ti":3.0, "O":0.4},
        dft_u = {"Ti":3.0, "O":0.2},
        incar={
            "NSW":55,
            "EDIFF":1E-5,
            "ENCUT":520,
        },
    )
    assert out is not None, "bulk_relax generate failed."
    print("[PASS] bulk_relax generate_inputs succeeded.")
    print(f" Ouput: {out}")
    return out

def test_static_dos():
    out = generate_inputs(
        "static_dos",
        structure=structure_path,
        functional="BEEF",
        kpoints_density=60.0,
        prev_dir=structure_path,
        magmom = {"Ti":3.0, "O":0.4},
        dft_u = {"Ti":3.0, "O":0.2},
        incar={
            "NSW":55,
            "EDIFF":1E-5,
            "ENCUT":520,
        },
    )
    assert out is not None, "static_dos generate failed."
    print("[PASS] static_dos generate_inputs succeeded.")
    print(f" Ouput: {out}")
    return out

def test_static_charge():
    out = generate_inputs(
        "static_charge",
        structure=structure_path,
        functional="BEEF",
        kpoints_density=60.0,
        prev_dir=structure_path,
        magmom = {"Ti":3.0, "O":0.4},
        dft_u = {"Ti":3.0, "O":0.2},
        incar={
            "NSW":55,
            "EDIFF":1E-5,
            "ENCUT":520,
        },
    )
    assert out is not None, "static_charge generate failed."
    print("[PASS] static_charge generate_inputs succeeded.")
    print(f" Ouput: {out}")
    return out

def test_static_elf():
    out = generate_inputs(
        "static_elf",
        structure=structure_path,
        functional="BEEF",
        kpoints_density=60.0,
        prev_dir=structure_path,
        magmom = {"Ti":3.0, "O":0.4},
        dft_u = {"Ti":3.0, "O":0.2},
        incar={
            "NSW":55,
            "EDIFF":1E-5,
            "ENCUT":520,
        },
    )
    assert out is not None, "static_elf generate failed."
    print("[PASS] static_elf generate_inputs succeeded.")
    print(f" Ouput: {out}")
    return out

def test_freq_ir_workflow():
    engine = WorkflowEngine()
    out = engine.run(WorkflowConfig(
        calc_type="freq_ir",     # IBRION=7 / LEPSILON=True / NWRITE=3 自动注入
        prev_dir=structure_path,
        functional="BEEF",
        kpoints_density=25.0,
        vibrate_indices=[12, 13, 14],
        user_incar_overrides={"POTIM": 0.015, "NFREE": 2, "NPAR": 4},
        output_dir="run_freq_ir",
    ))
    assert out is not None, "freq_ir_run generate failed."
    print("[PASS] freq_ir_run generate_inputs succeeded.")
    print(f" Ouput: {out}")
    return out

def run_all_tests():
    results = {}
    tests = {
        "md_nvt": test_md_nvt,
        "md_npt": test_md_npt,
        "freq_ir": test_freq_ir,
        "nbo": test_nbo,
        "lobster": test_lobster,
        "neb": test_neb,
        "slab_relax": test_slab_relax,
        "bulk_relax":test_bulk_relax,
        "static_dos": test_static_dos,
        "static_charge": test_static_charge,
        "static_elf": test_static_elf,
        "freq_ir_run": test_freq_ir_workflow,

    }

    for name, func in tests.items():
        try:
            func()
            results[name] = "PASS"
        except AssertionError as e:
            print(f"[FAIL] {name}: AssertionError - {e}")
            results[name] = "FAIL"
        except Exception as e:
            print(f"[ERROR] {name}: {type(e).__name__} - {e}")
            results[name] = "ERROR"

    # 汇总结果
    print("\n" + "=" * 50)
    print("Test Summary:")
    for name, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}: {status}")


if __name__ == "__main__":
    run_all_tests()