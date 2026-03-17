#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import yaml
from jinja2 import Environment, StrictUndefined


# =============================================================================
# Utilities
# =============================================================================

def die(msg: str, code: int = 2) -> None:
    print(f"[hook] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate()
    return p.returncode, (out or "").strip(), (err or "").strip()


def render_jinja(template_file: Path, ctx: Dict[str, Any]) -> str:
    text = template_file.read_text(encoding="utf-8")
    env = Environment(undefined=StrictUndefined)
    tpl = env.from_string(text)
    return tpl.render(**ctx)


# =============================================================================
# Lock
# =============================================================================

class DirLock:
    """Atomic lock using mkdir (good enough on shared FS)."""

    def __init__(self, lock_dir: Path):
        self.lock_dir = lock_dir
        self.acquired = False

    def acquire(self) -> bool:
        try:
            self.lock_dir.mkdir(parents=False, exist_ok=False)
            self.acquired = True
            (self.lock_dir / "meta.json").write_text(
                json.dumps({"pid": os.getpid(), "time": now_ts()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except FileExistsError:
            return False

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            for x in self.lock_dir.glob("*"):
                try:
                    x.unlink()
                except Exception:
                    pass
            self.lock_dir.rmdir()
        except Exception:
            pass
        self.acquired = False


# =============================================================================
# Markers
# =============================================================================

def done_marker(workdir: Path) -> Path:
    return workdir / "done.ok"


def submitted_marker(workdir: Path) -> Path:
    return workdir / "submitted.json"


def is_done(workdir: Path) -> bool:
    return done_marker(workdir).exists()


def is_submitted(workdir: Path) -> bool:
    return submitted_marker(workdir).exists()


def write_done(workdir: Path, meta: Dict[str, Any]) -> None:
    done_marker(workdir).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# OUTCAR success check
# =============================================================================

def outcar_shows_normal_termination(outcar: Path) -> bool:
    if not outcar.exists():
        return False
    try:
        tail = outcar.read_text(errors="ignore")[-20000:]
    except Exception:
        return False

    patterns = [
        r"total cpu time used",
        r"voluntary context switches",
        r"General timing and accounting informations",
    ]
    return any(re.search(p, tail, flags=re.IGNORECASE) for p in patterns)

def lobster_success(workdir: Path, params: Dict[str, Any]) -> bool:
    """
    LOBSTER stage here = VASP single-point + LOBSTER
    Success criteria:
      - OUTCAR shows normal termination (SP done)
      - lobsterout exists and non-empty
      - success_files exist (default: ICOHPLIST.lobster)
    """
    if not outcar_shows_normal_termination(workdir / "OUTCAR"):
        return False

    lobsterout = workdir / "lobsterout"
    if (not lobsterout.exists()) or lobsterout.stat().st_size <= 0:
        return False

    cfg = (params.get("lobster", {}) or {})
    success_files = cfg.get("success_files", ["ICOHPLIST.lobster"])
    if isinstance(success_files, list):
        for fn in success_files:
            p = workdir / str(fn)
            if (not p.exists()) or p.stat().st_size <= 0:
                return False
    return True


def detect_stage_from_workdir(workdir: Path) -> str:
    sj = submitted_marker(workdir)
    if sj.exists():
        try:
            meta = load_json(sj)
            st = str(meta.get("stage", "")).strip()
            if st:
                return st
        except Exception:
            pass

    for s in STAGE_ORDER:
        if s in workdir.parts:
            return s
    return ""

# =============================================================================
# Params / paths
# =============================================================================

STAGE_ORDER = [
    "bulk_relax", "bulk_dos",
    "bulk_lobster", "slab_relax",
    "slab_dos", "slab_lobster",
    "adsorption", "adsorption_freq", "adsorption_lobster",
]

def enabled_stages(params: Dict[str, Any]) -> List[str]:
    st = (params.get("workflow", {}) or {}).get("stages", {}) or {}
    out: List[str] = []
    for s in STAGE_ORDER:
        if bool(st.get(s, False)):
            out.append(s)
    return out


def run_root(params: Dict[str, Any]) -> Path:
    rr = (params.get("project", {}) or {}).get("run_root")
    if not rr:
        die("params.yaml 缺少 project.run_root")
    return Path(rr).expanduser().resolve()


def project_root(params: Dict[str, Any]) -> Path:
    pr = (params.get("project", {}) or {}).get("project_root")
    if not pr:
        die("params.yaml 缺少 project.project_root")
    return Path(pr).expanduser().resolve()


def get_stage_vasp_config(params: Dict[str, Any], stage: str) -> Dict[str, Any]:
    if stage == "bulk_relax":
        return ((params.get("bulk", {}) or {}).get("vasp", {}) or {})
    if stage == "bulk_dos":
        return ((params.get("bulk_dos", {}) or {}).get("vasp", {}) or {})
    if stage == "slab_relax":
        return ((params.get("slab", {}) or {}).get("vasp", {}) or {})
    if stage == "slab_dos":
        return ((params.get("slab_dos", {}) or {}).get("vasp", {}) or {})
    if stage == "adsorption":
        return ((params.get("adsorption", {}) or {}).get("vasp", {}) or {})
    if stage == "adsorption_freq":
        return ((params.get("freq", {}) or {}).get("vasp", {}) or {})    
    if stage in ("bulk_lobster", "slab_lobster", "adsorption_lobster"):
        return ((params.get("lobster", {}) or {}).get("vasp_singlepoint", {}) or {})
    die(f"未知 stage: {stage}")
    return {}


# =============================================================================
# Structure scanning (user provides ONE directory)
# =============================================================================

def _sanitize_id(s: str) -> str:
    s = re.sub(r"\s+", "_", (s or "").strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("_")
    return s or "bulk"


def _extract_bulk_id_from_filename(name: str) -> str:
    """
    POSCAR_PtSnCu -> PtSnCu
    CONTCAR_PtSnCu -> PtSnCu
    fallback -> stem
    """
    for prefix in ("POSCAR_", "CONTCAR_"):
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):]
    for prefix in ("POSCAR.", "CONTCAR."):
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):]
    return Path(name).stem


def get_bulk_sources(params: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    structure 必须是一个路径字符串：
      - 若为目录：扫描 POSCAR_* / CONTCAR_* / POSCAR.* / CONTCAR.*
      - 若为文件：当作单个 bulk

    返回：
      [{id: "PtSnCu", path: "/dir/POSCAR_PtSnCu"}, ...]
    """
    struct = params.get("structure", None)
    if not struct or not isinstance(struct, str):
        die(
            "params.yaml 顶层必须提供 structure（一个路径字符串）。例如：\n"
            "  structure: \"/path/to/dir_contains_POSCAR_*\"\n"
        )

    p = Path(struct).expanduser().resolve()
    if not p.exists():
        die(f"structure 路径不存在: {p}")

    files: List[Path] = []
    if p.is_file():
        files = [p]
    else:
        cand: List[Path] = []
        cand += sorted(p.glob("POSCAR_*"))
        cand += sorted(p.glob("CONTCAR_*"))
        cand += sorted(p.glob("POSCAR.*"))
        cand += sorted(p.glob("CONTCAR.*"))
        files = [x for x in cand if x.is_file() and x.stat().st_size > 0]
        if not files:
            die(
                f"structure 目录下未找到结构文件。目录: {p}\n"
                "要求文件名类似：POSCAR_PtSnCu / POSCAR_PtSnFe / POSCAR_PtSnAu\n"
                "（也支持 CONTCAR_* 或 POSCAR.xxx / CONTCAR.xxx）"
            )

    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for f in files:
        bid = _sanitize_id(_extract_bulk_id_from_filename(f.name))

        # 去重
        if bid in seen:
            k = 2
            bid2 = f"{bid}_{k}"
            while bid2 in seen:
                k += 1
                bid2 = f"{bid}_{k}"
            bid = bid2

        seen.add(bid)
        out.append({"id": bid, "path": str(f)})

    return out


# =============================================================================
# Import user modules
# =============================================================================

def import_input_maker(pr: Path):
    sys.path.insert(0, str(pr))
    mod_name = os.environ.get("HOOK_INPUT_MODULE", "vasp_inputs")
    try:
        mod = __import__(mod_name, fromlist=["VaspInputMaker"])
    except Exception as e:
        die(
            f"无法导入输入生成模块 '{mod_name}'。\n"
            f"请确认文件存在：{pr}/{mod_name}.py\n"
            f"或设置环境变量 HOOK_INPUT_MODULE。\n"
            f"原始错误: {e}"
        )
    if not hasattr(mod, "VaspInputMaker"):
        die(f"模块 {mod_name} 内找不到 VaspInputMaker。")
    return mod


def import_structure_tools(pr: Path):
    sys.path.insert(0, str(pr))
    mod_name = os.environ.get("HOOK_STRUCT_MODULE", "structure_modify")
    try:
        mod = __import__(mod_name, fromlist=["BulkToSlabGenerator", "AdsorptionModify"])
    except Exception as e:
        die(
            f"无法导入结构修改模块 '{mod_name}'。\n"
            f"请确认文件存在：{pr}/{mod_name}.py\n"
            f"或设置环境变量 HOOK_STRUCT_MODULE。\n"
            f"原始错误: {e}"
        )
    if not hasattr(mod, "BulkToSlabGenerator"):
        die(f"模块 {mod_name} 内找不到 BulkToSlabGenerator。")
    if not hasattr(mod, "AdsorptionModify"):
        die(f"模块 {mod_name} 内找不到 AdsorptionModify。")
    return mod


# =============================================================================
# PBS
# =============================================================================

def build_pbs_context(params: Dict[str, Any], stage: str, workdir: Path, task: Dict[str, Any]) -> Dict[str, Any]:
    pbs = params.get("pbs", {}) or {}
    project = params.get("project", {}) or {}
    vasp_runtime = params.get("vasp_runtime", {}) or {}
    python_runtime = params.get("python_runtime", {}) or {}

    bulk_id = (task.get("meta", {}) or {}).get("bulk_id", "")
    job_name = f"{pbs.get('job_name_prefix', 'job')}_{stage}"
    if bulk_id:
        job_name = (job_name + "_" + str(bulk_id))[:15]

    return {
        "project_root": str(project.get("project_root", "")),
        "run_root": str(project.get("run_root", "")),
        "stage": stage,
        "workdir": str(workdir),
        "task_id": str(task.get("id", "")),
        "bulk_id": str(bulk_id),
        "params_file": str(params.get("_params_file", "")),
        "queue": pbs.get("queue", ""),
        "nodes": int(pbs.get("nodes", 1)),
        "ppn": int(pbs.get("ppn", 1)),
        "walltime": str(pbs.get("walltime", "24:00:00")),
        "job_name": job_name,
        "VER": vasp_runtime.get("VER", ""),
        "TYPE1": vasp_runtime.get("TYPE1", ""),
        "TYPE2": vasp_runtime.get("TYPE2", ""),
        "OPT": vasp_runtime.get("OPT", ""),
        "COMPILER": vasp_runtime.get("COMPILER", ""),
        "conda_sh": python_runtime.get("conda_sh", ""),
        "conda_env": python_runtime.get("conda_env", ""),
        "python_bin": python_runtime.get("python_bin", ""),
        "params": params,
    }


def write_pbs_script(params: Dict[str, Any], stage: str, workdir: Path, task: Dict[str, Any]) -> Path:
    pbs = params.get("pbs", {}) or {}
    template_file = Path(pbs.get("template_file", "")).expanduser()
    if not template_file.exists():
        die(f"PBS 模板文件不存在: {template_file}")

    ctx = build_pbs_context(params, stage, workdir, task)
    script_text = render_jinja(template_file, ctx)
    script_path = workdir / "job.pbs"
    script_path.write_text(script_text, encoding="utf-8")
    return script_path


def qsub(script_path: Path, cwd: Path) -> str:
    rc, out, err = run_cmd(["qsub", str(script_path)], cwd=cwd)
    if rc != 0:
        die(f"qsub 失败 (rc={rc}). stdout='{out}' stderr='{err}'")
    return out.strip()


# =============================================================================
# Manifest + task deps
# =============================================================================

def manifest_path(params: Dict[str, Any]) -> Path:
    return run_root(params) / "manifest.json"


def ensure_manifest(params: Dict[str, Any]) -> Dict[str, Any]:
    mp = manifest_path(params)
    m = load_json(mp) if mp.exists() else {}
    if not m:
        m = {
            "schema_version": 1,
            "created_at": now_ts(),
            "params_file": str(params.get("_params_file", "")),
            "tasks": {}
        }
    if "tasks" not in m:
        m["tasks"] = {}
    return m


def save_manifest(params: Dict[str, Any], m: Dict[str, Any]) -> None:
    dump_json(manifest_path(params), m)


def task_deps_satisfied(m: Dict[str, Any], task: Dict[str, Any]) -> bool:
    for dep_id in task.get("deps", []):
        dep = m["tasks"].get(dep_id)
        if not dep:
            return False
        if not is_done(Path(dep["workdir"])):
            return False
    return True


# ---- task ids + workdirs ----

def hkl_to_str(hkl: List[int]) -> str:
    return f"hkl_{hkl[0]}{hkl[1]}{hkl[2]}"


def bulk_relax_tid(bulk_id: str) -> str:
    return f"bulk_relax:{bulk_id}"


def bulk_dos_tid(bulk_id: str) -> str:
    return f"bulk_dos:{bulk_id}"

def bulk_lobster_tid(bulk_id: str) -> str:
    return f"bulk_lobster:{bulk_id}"

def slab_relax_tid(bulk_id: str, hkl: List[int], layers: int, term: int) -> str:
    return f"slab_relax:{bulk_id}:{hkl_to_str(hkl)}:{layers}L:term{term}"

def slab_dos_tid(bulk_id: str, hkl: List[int], layers: int, term: int) -> str:
    return f"slab_dos:{bulk_id}:{hkl_to_str(hkl)}:{layers}L:term{term}"


def slab_lobster_tid(bulk_id: str, hkl: List[int], layers: int, term: int) -> str:
    return f"slab_lobster:{bulk_id}:{hkl_to_str(hkl)}:{layers}L:term{term}"


def ads_tid(bulk_id: str, hkl: List[int], layers: int, term: int, site_type: str, site_index: int) -> str:
    return f"adsorption:{bulk_id}:{hkl_to_str(hkl)}:{layers}L:term{term}:{site_type}:{site_index:03d}"

def ads_freq_tid(bulk_id: str, hkl: List[int], layers: int, term: int, site_type: str, site_index: int) -> str:
    return f"adsorption_freq:{bulk_id}:{hkl_to_str(hkl)}:{layers}L:term{term}:{site_type}:{site_index:03d}"

def ads_lobster_tid(bulk_id: str, hkl: List[int], layers: int, term: int, site_type: str, site_index: int) -> str:
    return f"adsorption_lobster:{bulk_id}:{hkl_to_str(hkl)}:{layers}L:term{term}:{site_type}:{site_index:03d}"

def bulk_workdir(params: Dict[str, Any], stage: str, bulk_id: str) -> Path:
    return run_root(params) / stage / bulk_id


def slab_workdir(params: Dict[str, Any], stage: str, bulk_id: str, hkl: List[int], layers: int, term: int) -> Path:
    return run_root(params) / stage / bulk_id / hkl_to_str(hkl) / f"{layers}L" / f"term{term}"


def ads_workdir(params: Dict[str, Any], stage: str, bulk_id: str, hkl: List[int], layers: int, term: int, site_type: str, site_index: int) -> Path:
    return run_root(params) / stage / bulk_id / hkl_to_str(hkl) / f"{layers}L" / f"term{term}" / site_type / f"{site_index:03d}"

def get_best_structure_path_from_dir(d: Path) -> Optional[Path]:
    for name in ["CONTCAR", "POSCAR"]:
        p = d / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def resolve_bulk_source_for_slab(params: Dict[str, Any], bulk_id: str, fallback_structure_file: str) -> str:
    """
    slab 的 bulk 来源：优先使用该 bulk_id 的 bulk_relax 结果（CONTCAR>POSCAR）
    """
    br = bulk_workdir(params, "bulk_relax", bulk_id)
    if is_done(br):
        p = get_best_structure_path_from_dir(br)
        if p:
            return str(p)
        die(f"{br} 已 done，但未找到 CONTCAR/POSCAR")
    return fallback_structure_file


# =============================================================================
# Expand manifest (the only place we "create" tasks)
# =============================================================================

def expand_manifest(params: Dict[str, Any]) -> Dict[str, Any]:
    ensure_dir(run_root(params))

    pr = project_root(params)
    struct_mod = import_structure_tools(pr)

    stages_on = set(enabled_stages(params))
    if not stages_on:
        die("workflow.stages 未启用任何 stage。")

    bulks = get_bulk_sources(params)

    m = ensure_manifest(params)
    tasks = m["tasks"]

    # --- 1) bulk_relax & bulk_dos (per bulk) ---
    for b in bulks:
        bid = b["id"]
        bfile = b["path"]

        if "bulk_relax" in stages_on:
            tid = bulk_relax_tid(bid)
            if tid not in tasks:
                w = bulk_workdir(params, "bulk_relax", bid)
                tasks[tid] = {
                    "id": tid,
                    "stage": "bulk_relax",
                    "workdir": str(w),
                    "deps": [],
                    "meta": {"bulk_id": bid, "structure": bfile},
                }

        if "bulk_dos" in stages_on:
            tid = bulk_dos_tid(bid)
            if tid not in tasks:
                w = bulk_workdir(params, "bulk_dos", bid)
                tasks[tid] = {
                    "id": tid,
                    "stage": "bulk_dos",
                    "workdir": str(w),
                    "deps": [bulk_relax_tid(bid)],
                    "meta": {"bulk_id": bid},
                }
        
        if "bulk_lobster" in stages_on:
            tid = bulk_lobster_tid(bid)
            if tid not in tasks:
                w = bulk_workdir(params, "bulk_lobster", bid)
                tasks[tid] = {
                    "id": tid,
                    "stage": "bulk_lobster",
                    "workdir": str(w),
                    "deps": [bulk_relax_tid(bid)],
                    "meta": {"bulk_id": bid, "prev": str(bulk_workdir(params, "bulk_relax", bid))},
                }

    # --- 2) slab_relax fan-out gate: only after bulk_relax(bid) DONE ---
    if "slab_relax" in stages_on:
        slab_cfg = params.get("slab", {}) or {}
        miller_list = slab_cfg.get("miller_list")
        slabgen = slab_cfg.get("slabgen", {}) or {}

        if not miller_list or not isinstance(miller_list, list):
            die("slab.miller_list 缺失或不是 list，例如：slab: { miller_list: [[1,1,0]] }")
        if "target_layers" not in slabgen:
            die("slab.slabgen.target_layers 缺失")

        layers = int(slabgen["target_layers"])

        from pymatgen.io.vasp import Poscar

        for b in bulks:
            bid = b["id"]
            br = tasks.get(bulk_relax_tid(bid))
            if not br:
                continue
            if not is_done(Path(br["workdir"])):
                continue  # gate

            bulk_source = resolve_bulk_source_for_slab(params, bid, b["path"])

            for miller in miller_list:
                hkl = [int(x) for x in miller]
                
                prefix = f"slab_relax:{bid}:{hkl_to_str(hkl)}:{layers}L:"
                if any(tid.startswith(prefix) for tid in tasks.keys()):
                    continue
                gen_params: Dict[str, Any] = {
                    "miller_indices": hkl,
                    "target_layers": layers,
                    "vacuum_thickness": float(slabgen.get("vacuum_thickness", 15.0)),
                    "supercell_matrix": slabgen.get("supercell_matrix", None),
                    "fix_bottom_layers": int(slabgen.get("fix_bottom_layers", 0)),
                    "fix_top_layers": int(slabgen.get("fix_top_layers", 0)),
                    "all_fix": bool(slabgen.get("all_fix", False)),
                    "symmetric": bool(slabgen.get("symmetric", False)),
                    "center": bool(slabgen.get("center", True)),
                    "primitive": bool(slabgen.get("primitive", True)),
                    "lll_reduce": bool(slabgen.get("lll_reduce", True)),
                    "hcluster_cutoff": float(slabgen.get("hcluster_cutoff", 0.25)),
                }
                save_dir = run_root(params) / "_generated_slabs" / bid / hkl_to_str(hkl) / f"{layers}L"
                ensure_dir(save_dir)

                gen_lock = DirLock(save_dir / ".slabgen.lock")
                if not gen_lock.acquire():
                    continue

                try:
                    # 双重保险：拿到锁后再检查一次 prefix（避免等锁期间别人已经生成并写入 tasks）
                    prefix2 = f"slab_relax:{bid}:{hkl_to_str(hkl)}:{layers}L:"
                    if any(tid.startswith(prefix2) for tid in tasks.keys()):
                        continue

                    cfg = {
                        "structure_source": bulk_source,
                        "save_dir": str(run_root(params) / "_generated_slabs" / bid / hkl_to_str(hkl) / f"{layers}L"),
                        "standardize_bulk": bool(slabgen.get("standardize_bulk", True)),
                        "log_to_file": True,
                        "generate_params": gen_params,
                        "save_options": {"save": True, "filename_prefix": "POSCAR"},
                    }

                    slabs = struct_mod.BulkToSlabGenerator.run_from_dict(cfg)
                    if not slabs:
                        continue

                    for term, slab in enumerate(slabs):
                        tid = slab_relax_tid(bid, hkl, layers, term)
                        if tid in tasks:
                            continue

                        w = slab_workdir(params, "slab_relax", bid, hkl, layers, term)
                        ensure_dir(w)
                        Poscar(slab).write_file(w / "POSCAR")

                        tasks[tid] = {
                            "id": tid,
                            "stage": "slab_relax",
                            "workdir": str(w),
                            "deps": [bulk_relax_tid(bid)],
                            "meta": {"bulk_id": bid, "hkl": hkl, "layers": layers, "term": term},
                        }
                finally:
                    gen_lock.release()

    # --- 3) slab_dos + adsorption fan-out gate: only after slab_relax(term) DONE ---
    for tid, t in list(tasks.items()):
        if t.get("stage") != "slab_relax":
            continue
        slab_dir = Path(t["workdir"])
        if not is_done(slab_dir):
            continue

        bid = t["meta"]["bulk_id"]
        hkl = t["meta"]["hkl"]
        layers = int(t["meta"]["layers"])
        term = int(t["meta"]["term"])

        if "slab_dos" in stages_on:
            did = slab_dos_tid(bid, hkl, layers, term)
            if did not in tasks:
                w = slab_workdir(params, "slab_dos", bid, hkl, layers, term)
                ensure_dir(w)
                tasks[did] = {
                    "id": did,
                    "stage": "slab_dos",
                    "workdir": str(w),
                    "deps": [tid],
                    "meta": {"bulk_id": bid, "prev": str(slab_dir), "hkl": hkl, "layers": layers, "term": term},
                }
        
        if "slab_lobster" in stages_on:
            lid = slab_lobster_tid(bid, hkl, layers, term)
            if lid not in tasks:
                w = slab_workdir(params, "slab_lobster", bid, hkl, layers, term)
                ensure_dir(w)
                tasks[lid] = {
                    "id": lid,
                    "stage": "slab_lobster",
                    "workdir": str(w),
                    "deps": [tid],
                    "meta": {"bulk_id": bid, "prev": str(slab_dir), "hkl": hkl, "layers": layers, "term": term},
                }

        if "adsorption" in stages_on:
            ads_cfg = params.get("adsorption", {}) or {}
            build = ads_cfg.get("build")
            if not isinstance(build, dict):
                die("adsorption.build 缺失或不是 dict")

            mode = str(build.get("mode", "site")).lower()
            if mode != "site":
                continue

            mol = build.get("molecule_formula")
            if not mol:
                die("adsorption.build.molecule_formula 缺失")

            enum = ads_cfg.get("enumerate", {}) or {}
            site_types = enum.get("site_types", [str(build.get("site_type", "ontop")).lower()])
            max_per_type = int(enum.get("max_per_type", 10))
            start_index = int(enum.get("start_index", 0))

            height = float(build.get("height", 0.9))
            ads_prefix = f"adsorption:{bid}:{hkl_to_str(hkl)}:{layers}L:term{term}:"
            if any(x.startswith(ads_prefix) for x in tasks.keys()):
                continue

            ads_save_dir = run_root(params) / "_generated_ads" / bid / hkl_to_str(hkl) / f"{layers}L" / f"term{term}"
            ensure_dir(ads_save_dir)

            ads_lock = DirLock(ads_save_dir / ".adsgen.lock")
            if not ads_lock.acquire():
                # 另一个进程正在生成该 slab(term) 的吸附构型，本轮跳过即可
                continue
            try:
                ads_prefix2 = f"adsorption:{bid}:{hkl_to_str(hkl)}:{layers}L:term{term}:"
                if any(x.startswith(ads_prefix2) for x in tasks.keys()):
                    continue
                modifier = struct_mod.AdsorptionModify(
                    slab_source=str(slab_dir),
                    selective_dynamics=bool(build.get("selective_dynamics", False)),
                    height=height,
                    save_dir=str(run_root(params) / "_generated_ads" / bid / hkl_to_str(hkl) / f"{layers}L" / f"term{term}"),
                    log_to_file=True,
                )
            

                find_args = build.get("find_args", {}) or {}
                sites = modifier.find_adsorption_sites(**find_args)

                from pymatgen.core import Molecule
                try:
                    molecule = modifier.ase2pmg(str(mol))
                except Exception:
                    mp = Path(str(mol))
                    if mp.exists():
                        molecule = Molecule.from_file(str(mp))
                    else:
                        die(f"无法解析 molecule_formula='{mol}'（既不是 ASE 名称也不是分子文件路径）")

                from pymatgen.io.vasp import Poscar

                for stype in site_types:
                    stype = str(stype).lower()
                    coords_list = sites.get(stype, [])
                    if not coords_list:
                        continue

                    end = min(len(coords_list), start_index + max_per_type)
                    for idx in range(start_index, end):
                        aid = ads_tid(bid, hkl, layers, term, stype, idx)
                        if aid in tasks:
                            continue

                        w = ads_workdir(params, "adsorption", bid, hkl, layers, term, stype, idx)
                        ensure_dir(w)

                        coords = coords_list[idx]
                        reorient = bool(build.get("reorient", True))
                        final_struct = modifier.add_adsorbate(molecule, coords, reorient=reorient)
                        Poscar(final_struct).write_file(w / "POSCAR")

                        tasks[aid] = {
                            "id": aid,
                            "stage": "adsorption",
                            "workdir": str(w),
                            "deps": [tid],
                            "meta": {
                                "bulk_id": bid,
                                "prev": str(slab_dir),
                                "hkl": hkl, "layers": layers, "term": term,
                                "site_type": stype, "site_index": idx,
                            },
                        }
            finally:
                ads_lock.release()
                
    if "adsorption_freq" in stages_on:
        for tid, t in list(tasks.items()):
            if t.get("stage") != "adsorption":
                continue
            ads_dir = Path(t["workdir"])
            if not is_done(ads_dir):
                continue  # gate

            bid = t["meta"]["bulk_id"]
            hkl = t["meta"]["hkl"]
            layers = int(t["meta"]["layers"])
            term = int(t["meta"]["term"])
            stype = t["meta"]["site_type"]
            sidx = int(t["meta"]["site_index"])

            fid = ads_freq_tid(bid, hkl, layers, term, stype, sidx)
            if fid in tasks:
                continue

            w = ads_workdir(params, "adsorption_freq", bid, hkl, layers, term, stype, sidx)
            ensure_dir(w)
            ads_build = (params.get("adsorption", {}) or {}).get("build", {}) or {}
            mol = ads_build.get("molecule_formula", None)
            
            tasks[fid] = {
                "id": fid,
                "stage": "adsorption_freq",
                "workdir": str(w),
                "deps": [t["id"]],  # after adsorption
                "meta": {
                    "bulk_id": bid,
                    "prev": str(ads_dir),  # use adsorption directory (CONTCAR preferred)
                    "hkl": hkl, "layers": layers, "term": term,
                    "site_type": stype, "site_index": sidx,
                    "adsorbate_formula": mol,
                },
            }
    if "adsorption_lobster" in stages_on:
        for tid, t in list(tasks.items()):
            if t.get("stage") != "adsorption":
                continue
            ads_dir = Path(t["workdir"])
            if not is_done(ads_dir):
                continue  # gate

            bid = t["meta"]["bulk_id"]
            hkl = t["meta"]["hkl"]
            layers = int(t["meta"]["layers"])
            term = int(t["meta"]["term"])
            stype = t["meta"]["site_type"]
            sidx = int(t["meta"]["site_index"])

            lid = ads_lobster_tid(bid, hkl, layers, term, stype, sidx)
            if lid in tasks:
                continue

            w = ads_workdir(params, "adsorption_lobster", bid, hkl, layers, term, stype, sidx)
            ensure_dir(w)

            tasks[lid] = {
                "id": lid,
                "stage": "adsorption_lobster",
                "workdir": str(w),
                "deps": [t["id"]],  # <-- SERIAL after adsorption
                "meta": {
                    "bulk_id": bid,
                    "prev": str(ads_dir),  # <-- use adsorption directory (CONTCAR preferred)
                    "hkl": hkl, "layers": layers, "term": term,
                    "site_type": stype, "site_index": sidx,
                },
            }
            
    save_manifest(params, m)
    return m


# =============================================================================
# Input generation per task
# =============================================================================

def generate_inputs_for_task(input_mod, params: Dict[str, Any], task: Dict[str, Any]) -> None:
    stage = task["stage"]
    workdir = Path(task["workdir"])
    ensure_dir(workdir)

    if stage in ("bulk_lobster", "slab_lobster", "adsorption_lobster"):
        vasp_cfg = get_stage_vasp_config(params, stage)  # lobster.vasp_singlepoint
        maker = input_mod.VaspInputMaker.from_dict_ecat(vasp_cfg)

        prev_dir = Path((task.get("meta", {}) or {}).get("prev", "")).expanduser()
        if not str(prev_dir):
            die(f"{task['id']} 缺少 meta.prev")
        if not is_done(prev_dir):
            die(f"{task['id']} 需要 prev 已完成(done.ok)，但未满足: {prev_dir}")

        pos_src = get_best_structure_path_from_dir(prev_dir)  # CONTCAR > POSCAR
        if not pos_src:
            die(f"{task['id']} prev_dir 未找到 CONTCAR/POSCAR: {prev_dir}")

        if not hasattr(maker, "write_lobster"):
            die(
                "VaspInputMaker 缺少 write_lobster(output_dir, structure, prev_dir)。\n"
            )

        maker.write_lobster(output_dir=workdir, structure=str(pos_src), prev_dir=prev_dir)
        return
    
    if stage == "adsorption_freq":
        vasp_cfg = get_stage_vasp_config(params, stage)  # freq.vasp
        maker = input_mod.VaspInputMaker.from_dict_ecat(vasp_cfg)

        prev_dir = Path((task.get("meta", {}) or {}).get("prev", "")).expanduser()
        if not str(prev_dir):
            die(f"{task['id']} 缺少 meta.prev")
        if not is_done(prev_dir):
            die(f"{task['id']} 需要 adsorption 已完成(done.ok)，但未满足: {prev_dir}")

        fcfg = (params.get("freq", {}) or {}).get("settings", {}) or {}

        if not hasattr(maker, "write_freq"):
            die("VaspInputMaker 缺少 write_freq(output_dir, prev_dir, ...)。\n")
        mode = str(fcfg.get("mode", "inherit"))
        vibrate_indices = fcfg.get("vibrate_indices", None)
        ads_formula = fcfg.get("adsorbate_formula", None)
        prefer = str(fcfg.get("adsorbate_formula_prefer", "tail"))

        if ads_formula is None:
            ads_formula = (task.get("meta", {}) or {}).get("adsorbate_formula", None)
        if ads_formula is None:
            ads_build = (params.get("adsorption", {}) or {}).get("build", {}) or {}
            ads_formula = ads_build.get("molecule_formula", None)
        if vibrate_indices is None and ads_formula is None and mode == "inherit":
            die(
                "adsorption_freq: mode='inherit' 且未提供 adsorbate_formula/vibrate_indices。\n"
                "在 slab 全 fix 的 CONTCAR 下将导致所有原子都不振动。\n"
                "请在 freq.settings 提供 adsorbate_formula 或 vibrate_indices，"
                "或确保 adsorption 的 CONTCAR 自带正确的 selective_dynamics。"
            )
        maker.write_freq(
            output_dir=workdir,
            prev_dir=prev_dir,
            mode=mode,
            vibrate_indices=vibrate_indices,
            adsorbate_formula=ads_formula,
            adsorbate_formula_prefer=prefer,
        )
        return
    # vasp stages
    vasp_cfg = get_stage_vasp_config(params, stage)
    maker = input_mod.VaspInputMaker.from_dict_ecat(vasp_cfg)

    if stage == "bulk_relax":
        structure = (task.get("meta", {}) or {}).get("structure")
        if not structure:
            die(f"{task['id']} 缺少 meta.structure")
        maker.write_bulk(output_dir=workdir, structure=str(structure))
        return

    if stage == "bulk_dos":
        bid = (task.get("meta", {}) or {}).get("bulk_id")
        if not bid:
            die(f"{task['id']} 缺少 meta.bulk_id")
        prev = bulk_workdir(params, "bulk_relax", str(bid))
        if not is_done(prev):
            die(f"{task['id']} 需要对应 bulk_relax 已完成(done.ok)，但未满足: {prev}")
        maker.write_noscf(output_dir=workdir, prev_dir=prev)
        return

    if stage == "slab_relax":
        poscar = workdir / "POSCAR"
        if not poscar.exists():
            die(f"{task['id']} 缺少 POSCAR（应由 expand_manifest 写入）: {poscar}")
        maker.write_slab(output_dir=workdir, structure=str(poscar))
        return

    if stage == "slab_dos":
        prev = Path((task.get("meta", {}) or {}).get("prev", ""))
        if not str(prev):
            die(f"{task['id']} 缺少 meta.prev")
        if not is_done(prev):
            die(f"{task['id']} 需要对应 slab_relax 已完成(done.ok)，但未满足: {prev}")
        maker.write_noscf(output_dir=workdir, prev_dir=prev)
        return

    if stage == "adsorption":
        prev = Path((task.get("meta", {}) or {}).get("prev", ""))
        if not str(prev):
            die(f"{task['id']} 缺少 meta.prev")
        if not is_done(prev):
            die(f"{task['id']} 需要对应 slab_relax 已完成(done.ok)，但未满足: {prev}")
        poscar = workdir / "POSCAR"
        if not poscar.exists():
            die(f"{task['id']} 缺少 POSCAR（应由 expand_manifest 写入）: {poscar}")
        maker.write_adsorption(output_dir=workdir, structure=str(poscar), prev_dir=prev)
        return

    die(f"未实现的 stage: {stage}")


# =============================================================================
# Submit
# =============================================================================

def submit_task(params: Dict[str, Any], m: Dict[str, Any], task: Dict[str, Any], force: bool = False) -> bool:
    """
    返回 True 表示本次实际提交了 qsub。
    返回 False 表示跳过（done/submitted/locked/deps不满足等）。
    """
    pr = project_root(params)
    input_mod = import_input_maker(pr)

    stage = task["stage"]
    workdir = Path(task["workdir"])
    ensure_dir(workdir)

    lock = DirLock(workdir / ".lock")
    if not lock.acquire():
        print(f"[hook] task locked, skip: id={task['id']} workdir={workdir}")
        return False

    try:
        if is_done(workdir) and not force:
            print(f"[hook] task done, skip: id={task['id']}")
            return False
        if is_submitted(workdir) and not force:
            print(f"[hook] task submitted, skip: id={task['id']}")
            return False

        if not force and not task_deps_satisfied(m, task):
            # 不再 die：批量提交时依赖不满足很正常，跳过即可
            print(f"[hook] deps not satisfied, skip: id={task['id']}")
            return False

        generate_inputs_for_task(input_mod, params, task)
        pbs_script = write_pbs_script(params, stage, workdir, task)
        job_id = qsub(pbs_script, cwd=workdir)

        meta = {"task_id": task["id"], "stage": stage, "workdir": str(workdir), "job_id": job_id, "time": now_ts()}
        dump_json(submitted_marker(workdir), meta)

        print(f"[hook] submitted: task_id={task['id']} stage={stage} job_id={job_id} workdir={workdir}")
        return True

    finally:
        lock.release()


def auto_submit_workflow(params: Dict[str, Any], force: bool = False, stage_filter: str = "") -> None:
    """
    兼容原行为：只提交第一个 eligible task。
    """
    m = expand_manifest(params)
    tasks = m["tasks"]

    pri = {s: i for i, s in enumerate(STAGE_ORDER)}

    def key(t: Dict[str, Any]) -> Tuple[int, str]:
        return (pri.get(t["stage"], 999), t["id"])

    for t in sorted(tasks.values(), key=key):
        if stage_filter and t["stage"] != stage_filter:
            continue

        w = Path(t["workdir"])
        if is_done(w):
            continue
        if is_submitted(w) and not force:
            continue
        if not force and not task_deps_satisfied(m, t):
            continue

        submit_task(params, m, t, force=force)
        return

    print("[hook] 无可提交任务：都 done/submitted 或依赖未满足。")


def submit_all_ready(params: Dict[str, Any], force: bool = False, stage_filter: str = "", limit: int = 0) -> int:
    """
    你要的“并发提交”入口：提交所有当前 eligible 的任务。
    - limit=0 表示不限制提交数量
    """
    m = expand_manifest(params)
    tasks = m["tasks"]

    pri = {s: i for i, s in enumerate(STAGE_ORDER)}

    def key(t: Dict[str, Any]) -> Tuple[int, str]:
        return (pri.get(t["stage"], 999), t["id"])

    submitted = 0
    skipped = 0

    for t in sorted(tasks.values(), key=key):
        if stage_filter and t["stage"] != stage_filter:
            continue

        # 尽量在这里先筛一遍，减少锁竞争
        w = Path(t["workdir"])
        if is_done(w):
            skipped += 1
            continue
        if is_submitted(w) and not force:
            skipped += 1
            continue
        if not force and not task_deps_satisfied(m, t):
            skipped += 1
            continue

        ok = submit_task(params, m, t, force=force)
        if ok:
            submitted += 1
            if limit and submitted >= limit:
                break
        else:
            skipped += 1

    print(f"[hook] submit-all finished: submitted={submitted}, skipped={skipped}, limit={limit}")
    return submitted


# =============================================================================
# Mark done
# =============================================================================

def mark_done_if_success_by_workdir(workdir: Path, params: Dict[str, Any]) -> None:
    stage = detect_stage_from_workdir(workdir)
    if not stage:
        die(f"无法从 workdir 推断 stage: {workdir}")

    if stage in ("bulk_lobster", "slab_lobster", "adsorption_lobster"):
        if lobster_success(workdir, params):
            write_done(workdir, {"workdir": str(workdir), "time": now_ts(), "success_check": "OUTCAR+LOBSTER", "stage": stage})
            print(f"[hook] marked done: {workdir} ({stage})")
            return
        die(f"LOBSTER stage 未通过成功检查，拒绝写 done.ok: {workdir}")

    outcar = workdir / "OUTCAR"
    if outcar_shows_normal_termination(outcar):
        write_done(workdir, {"workdir": str(workdir), "time": now_ts(), "success_check": "OUTCAR", "stage": stage})
        print(f"[hook] marked done: {workdir}")
    else:
        die(f"OUTCAR 未显示正常结束，拒绝写 done.ok: {outcar}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="HT workflow hook (manifest deps + PBS submission + done markers)")
    ap.add_argument("--params", type=str, required=True, help="Path to params.yaml")

    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("expand", help="Only expand/write manifest.json")

    s_auto = sub.add_parser("auto", help="Expand manifest and submit the first eligible task")
    s_auto.add_argument("--force", action="store_true")
    s_auto.add_argument("--stage", type=str, default="", help="Optional stage filter")

    s_all = sub.add_parser("submit-all", help="Expand manifest and submit ALL eligible tasks")
    s_all.add_argument("--force", action="store_true")
    s_all.add_argument("--stage", type=str, default="", help="Optional stage filter")
    s_all.add_argument("--limit", type=int, default=0)

    s_mark = sub.add_parser("mark-done", help="Check OUTCAR/LOBSTER and write done.ok for a workdir")
    s_mark.add_argument("--workdir", type=str, required=True)

    args = ap.parse_args()

    params_path = Path(args.params).expanduser().resolve()
    if not params_path.exists():
        die(f"params.yaml 不存在: {params_path}")

    params = load_yaml(params_path)
    params["_params_file"] = str(params_path)

    if args.cmd == "expand":
        expand_manifest(params)
        print(f"[hook] manifest updated: {manifest_path(params)}")
        return

    if args.cmd == "auto":
        if args.stage and args.stage not in STAGE_ORDER:
            die(f"--stage 必须是 {STAGE_ORDER} 之一（或留空）")
        auto_submit_workflow(params, force=bool(args.force), stage_filter=str(args.stage))
        return

    if args.cmd == "submit-all":
        if args.stage and args.stage not in STAGE_ORDER:
            die(f"--stage 必须是 {STAGE_ORDER} 之一（或留空）")
        submit_all_ready(params, force=bool(args.force), stage_filter=str(args.stage), limit=int(args.limit))
        return

    if args.cmd == "mark-done":
        wd = Path(args.workdir).expanduser().resolve()
        if not wd.exists():
            die(f"workdir 不存在: {wd}")
        mark_done_if_success_by_workdir(wd, params=params)
        return

    die(f"未知命令: {args.cmd}")

if __name__ == "__main__":
    main()
