"""
/bulk 路由 — 晶体结构修改操作
所有接口：接收结构字符串 → 修改 → 返回新结构字符串 + 结构信息
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Union

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from ...core.structure_modify import StructureModify
from ...core.utils.structure_utils import load_structure

router = APIRouter()

# ── 公共辅助 ──────────────────────────────────────────

def _modifier_from_str(structure_str: str, fmt: str) -> StructureModify:
    """从结构字符串构造 StructureModify 对象"""
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
        structure = load_structure(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"结构解析失败: {e}")
    finally:
        os.unlink(tmp_path)
    return StructureModify(structure)

def _ok(modifier: StructureModify, output_fmt: str) -> dict:
    """统一成功响应格式"""
    return {
        "info": modifier.get_structure_info(),
        "structure_str": modifier.to_string(output_fmt),
        "fmt": output_fmt,
    }

# ── 请求体模型 ────────────────────────────────────────

class StructBase(BaseModel):
    structure_str: str  = Field(..., description="结构文件字符串")
    fmt:           str  = Field("poscar", description="输入格式: poscar|xyz|cif")
    output_fmt:    str  = Field("poscar", description="输出格式: poscar|xyz|cif")

class SupercellReq(StructBase):
    nx: int = Field(2, ge=1)
    ny: int = Field(2, ge=1)
    nz: int = Field(2, ge=1)

class TransformReq(StructBase):
    matrix: List[List[float]] = Field(..., description="3×3 变换矩阵")

class ModifyElementReq(StructBase):
    index:       int = Field(..., ge=0)
    new_element: str

class ModifyCoordReq(StructBase):
    index:       int          = Field(..., ge=0)
    coords:      List[float]  = Field(..., min_items=3, max_items=3)
    frac_coords: bool         = Field(True)

class RemoveAtomReq(StructBase):
    index: int = Field(..., ge=0)

class ModifyLatticeReq(StructBase):
    a:     Optional[float] = None
    b:     Optional[float] = None
    c:     Optional[float] = None
    alpha: Optional[float] = None
    beta:  Optional[float] = None
    gamma: Optional[float] = None

class InsertAtomReq(StructBase):
    element:    str
    frac_coord: List[float] = Field(..., min_items=3, max_items=3)

class ReplaceAllReq(StructBase):
    species_map: dict = Field(..., description="如 {'Pt':'Pd'}")

class DefectBatchReq(StructBase):
    substitute_element: str
    dopant:             Optional[str]        = None
    dopant_num:         Union[int, float]    = 1
    num_structs:        int                  = Field(3, ge=1, le=50)
    random_seed:        Optional[int]        = None

class DefectStepReq(StructBase):
    substitute_element: str
    dopant:             Optional[str]  = None
    max_steps:          int            = Field(3, ge=1, le=10)
    max_structures_num: int            = Field(15, ge=1, le=50)
    return_all_steps:   bool           = False
    random_seed:        Optional[int]  = None

# ── 路由 ─────────────────────────────────────────────

@router.post("/upload", summary="上传结构文件")
async def upload(file: UploadFile = File(...)):
    """上传 xyz/cif/poscar 文件，返回结构信息 + POSCAR 字符串"""
    suffix = Path(file.filename).suffix.lower()
    fmt_map = {".xyz":"xyz", ".cif":"cif", ".poscar":"poscar",
               ".vasp":"poscar", "":"poscar"}
    fmt = fmt_map.get(suffix)
    if fmt is None:
        raise HTTPException(400, f"不支持的文件格式: {suffix}")
    content = (await file.read()).decode("utf-8")
    modifier = _modifier_from_str(content, fmt)
    return _ok(modifier, "poscar")


@router.post("/info", summary="获取结构信息")
def info(req: StructBase):
    return _modifier_from_str(req.structure_str, req.fmt).get_structure_info()


@router.post("/download", summary="导出为指定格式")
def download(req: StructBase):
    modifier = _modifier_from_str(req.structure_str, req.fmt)
    try:
        content = modifier.to_string(req.output_fmt)
    except ValueError as e:
        raise HTTPException(400, str(e))
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=content, media_type="text/plain")


@router.post("/supercell", summary="生成超胞")
def supercell(req: SupercellReq):
    modifier = _modifier_from_str(req.structure_str, req.fmt)
    try:
        modifier.make_supercell(f"{req.nx}x{req.ny}x{req.nz}")
    except Exception as e:
        raise HTTPException(400, str(e))
    return _ok(modifier, req.output_fmt)


@router.post("/transform", summary="应用 3×3 变换矩阵")
def transform(req: TransformReq):
    modifier = _modifier_from_str(req.structure_str, req.fmt)
    try:
        modifier.transform_structure(req.matrix)
    except Exception as e:
        raise HTTPException(400, str(e))
    return _ok(modifier, req.output_fmt)


@router.post("/modify/element", summary="修改单个原子元素")
def modify_element(req: ModifyElementReq):
    modifier = _modifier_from_str(req.structure_str, req.fmt)
    try:
        modifier.modify_atom_element(req.index, req.new_element)
    except (IndexError, ValueError) as e:
        raise HTTPException(400, str(e))
    return _ok(modifier, req.output_fmt)


@router.post("/modify/coords", summary="修改单个原子坐标")
def modify_coords(req: ModifyCoordReq):
    modifier = _modifier_from_str(req.structure_str, req.fmt)
    try:
        modifier.modify_atom_coords(req.index, req.coords, req.frac_coords)
    except (IndexError, ValueError) as e:
        raise HTTPException(400, str(e))
    return _ok(modifier, req.output_fmt)


@router.post("/modify/lattice", summary="修改晶格参数")
def modify_lattice(req: ModifyLatticeReq):
    modifier = _modifier_from_str(req.structure_str, req.fmt)
    try:
        modifier.modify_lattice(req.a, req.b, req.c, req.alpha, req.beta, req.gamma)
    except Exception as e:
        raise HTTPException(400, str(e))
    return _ok(modifier, req.output_fmt)


@router.post("/modify/replace_all", summary="批量替换所有指定元素")
def replace_all(req: ReplaceAllReq):
    modifier = _modifier_from_str(req.structure_str, req.fmt)
    try:
        modifier.replace_species_all(req.species_map)
    except Exception as e:
        raise HTTPException(400, str(e))
    return _ok(modifier, req.output_fmt)


@router.post("/modify/insert", summary="插入原子")
def insert_atom(req: InsertAtomReq):
    modifier = _modifier_from_str(req.structure_str, req.fmt)
    try:
        modifier.insert_atom(req.element, req.frac_coord)
    except Exception as e:
        raise HTTPException(400, str(e))
    return _ok(modifier, req.output_fmt)


@router.post("/modify/remove", summary="删除原子")
def remove_atom(req: RemoveAtomReq):
    modifier = _modifier_from_str(req.structure_str, req.fmt)
    try:
        modifier.remove_atom(req.index)
    except IndexError as e:
        raise HTTPException(400, str(e))
    return _ok(modifier, req.output_fmt)


@router.post("/defects/batch", summary="批量随机生成掺杂/空位结构")
def defects_batch(req: DefectBatchReq):
    modifier = _modifier_from_str(req.structure_str, req.fmt)
    try:
        structures = modifier.generate_defects_batch(
            substitute_element=req.substitute_element,
            dopant=req.dopant,
            dopant_num=req.dopant_num,
            num_structs=req.num_structs,
            random_seed=req.random_seed,
        )
    except Exception as e:
        raise HTTPException(400, str(e))
    results = []
    for i, s in enumerate(structures):
        m = StructureModify(s)
        results.append({
            "index": i,
            "info": m.get_structure_info(),
            "structure_str": m.to_string(req.output_fmt),
            "fmt": req.output_fmt,
        })
    return {"count": len(results), "structures": results}


@router.post("/defects/step", summary="逐步层级生成掺杂结构")
def defects_step(req: DefectStepReq):
    modifier = _modifier_from_str(req.structure_str, req.fmt)
    try:
        output = modifier.generate_defects_step_by_step(
            substitute_element=req.substitute_element,
            dopant=req.dopant,
            max_steps=req.max_steps,
            max_structures_num=req.max_structures_num,
            return_all_steps=req.return_all_steps,
            random_seed=req.random_seed,
        )
    except Exception as e:
        raise HTTPException(400, str(e))

    def _serialize(s):
        m = StructureModify(s)
        return {
            "info": m.get_structure_info(),
            "structure_str": m.to_string(req.output_fmt),
            "fmt": req.output_fmt,
        }

    if req.return_all_steps:
        return {
            "steps": [
                {"step": i+1, "structures": [_serialize(s) for s in level]}
                for i, level in enumerate(output)
            ]
        }
    return {"count": len(output), "structures": [_serialize(s) for s in output]}