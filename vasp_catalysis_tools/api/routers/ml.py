"""
/ml 路由 — ML 结构预测
使用 ASE EMT（轻量经验势，支持常见金属：Ag/Al/Au/Cu/Fe/Ni/Pd/Pt）
输出：能量（eV）、每原子能量（eV/atom）、最大原子力（eV/Å）
前端展示在 quickOptOut 字段
"""
import os
import logging
import tempfile
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("catalyst_workbench.ml")
router = APIRouter()

# EMT 支持的元素集合
EMT_SUPPORTED = {
    "H","Al","Si","Cu","Ag","Au","Ni","Pd","Pt",
    "C","N","O","Fe","Co","Zn","Ga","Ge","In","Sn","Sb"
}


class MLRequest(BaseModel):
    structure_str: str  = Field(..., description="结构字符串（POSCAR/XYZ/CIF）")
    fmt:           str  = Field("poscar", description="输入格式")
    relax:         bool = Field(False, description="是否做简单弛豫（BFGS，最多50步）")


class MLResult(BaseModel):
    energy:           float          # 总能量 eV
    energy_per_atom:  float          # 每原子能量 eV/atom
    fmax:             float          # 最大原子力 eV/Å
    converged:        Optional[bool] # 弛豫是否收敛（relax=True 时有效）
    unsupported_elements: list       # 不支持的元素列表（EMT 限制）
    display:          str            # 前端 quickOptOut 展示字符串


def _load_ase_atoms(structure_str: str, fmt: str):
    """结构字符串 → ASE Atoms 对象"""
    import ase.io
    suffix_map = {
        "poscar": ".vasp", "vasp": ".vasp",
        "xyz": ".xyz", "cif": ".cif"
    }
    suffix = suffix_map.get(fmt.lower(), ".vasp")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(structure_str)
        tmp_path = tmp.name
    try:
        atoms = ase.io.read(tmp_path)
    except Exception as e:
        raise HTTPException(422, f"ASE 结构解析失败: {e}")
    finally:
        os.unlink(tmp_path)
    return atoms


@router.post("/predict", summary="EMT 单点能量与力预测", response_model=MLResult)
def predict(req: MLRequest) -> MLResult:
    """
    对当前结构做 EMT 单点计算，返回：
    - 总能量 / 每原子能量 / 最大原子力
    - 前端展示字符串（显示在 quickOptOut）
    
    注意：EMT 仅支持部分金属元素，不支持的元素会在响应中列出。
    """
    try:
        from ase.calculators.emt import EMT
    except ImportError:
        raise HTTPException(503, "ASE 未安装，请运行: pip install ase")

    atoms = _load_ase_atoms(req.structure_str, req.fmt)

    # 检查不支持的元素
    symbols = set(atoms.get_chemical_symbols())
    unsupported = sorted(symbols - EMT_SUPPORTED)

    if unsupported:
        # 有不支持的元素时返回警告，不报错，让前端展示提示
        return MLResult(
            energy=0.0,
            energy_per_atom=0.0,
            fmax=0.0,
            converged=None,
            unsupported_elements=unsupported,
            display=f"⚠️ EMT 不支持元素: {', '.join(unsupported)}",
        )

    try:
        atoms.calc = EMT()

        if req.relax:
            # 简单 BFGS 弛豫
            from ase.optimize import BFGS
            import io as _io
            log_buf = _io.StringIO()
            opt = BFGS(atoms, logfile=log_buf)
            converged = opt.run(fmax=0.05, steps=50)
        else:
            converged = None

        energy          = float(atoms.get_potential_energy())
        forces          = atoms.get_forces()
        fmax            = float((forces**2).sum(axis=1).max()**0.5)
        n               = len(atoms)
        energy_per_atom = energy / n

        if req.relax:
            display = (
                f"E = {energy:.3f} eV  |  "
                f"E/atom = {energy_per_atom:.3f} eV  |  "
                f"F_max = {fmax:.3f} eV/Å  |  "
                f"{'收敛 ✓' if converged else '未收敛（50步）'}"
            )
        else:
            display = (
                f"E = {energy:.3f} eV  |  "
                f"E/atom = {energy_per_atom:.3f} eV  |  "
                f"F_max = {fmax:.3f} eV/Å"
            )

        return MLResult(
            energy=round(energy, 4),
            energy_per_atom=round(energy_per_atom, 4),
            fmax=round(fmax, 4),
            converged=converged,
            unsupported_elements=[],
            display=display,
        )

    except Exception as e:
        logger.error(f"EMT 计算失败: {e}", exc_info=True)
        raise HTTPException(500, f"EMT 计算失败: {e}")


@router.get("/supported_elements", summary="返回 EMT 支持的元素列表")
def supported_elements():
    """前端可用此接口提示用户哪些元素支持 ML 预测"""
    return {"supported": sorted(EMT_SUPPORTED)}