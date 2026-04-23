# -*- coding: utf-8 -*-
"""
flow.validator — Stateless parameter validation for generate_inputs()
======================================================================

Three-layer validation architecture:
  Layer 1: field-level  — each parameter independently
  Layer 2: cross-field  — constraints involving two or more parameters
  Layer 3: business     — per calc_type rules

All errors are collected before raising; the caller receives a single
ValidationError listing every problem at once.

本模块为 generate_inputs() 提供无状态的参数校验，分三层：
  第 1 层：字段级——独立验证每个参数
  第 2 层：跨字段——涉及多个参数的约束
  第 3 层：业务逻辑——按计算类型分类的规则

所有错误均先汇总再一次性抛出，调用方通过单个 ValidationError 获取所有问题。
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# CalcType import is safe: workflow_engine.py does not import validator.py,
# so there is no circular dependency.
from .workflow_engine import CalcType
from .constants import SUPPORTED_FUNCTIONALS

logger = logging.getLogger(__name__)


# ============================================================================
# Custom exception
# ============================================================================

class ValidationError(Exception):
    """Raised when one or more validation checks fail.

    Attributes:
        errors: List of individual error strings.
    """

    def __init__(self, errors: List[str]) -> None:
        self.errors = list(errors)
        count = len(errors)
        detail = "\n".join(f"  {e}" for e in errors)
        super().__init__(f"{count} validation error(s) found:\n{detail}")


# ============================================================================
# Supported value sets
# ============================================================================

# Derived from CalcType enum — single source of truth.
# Three frontend aliases differ from their CalcType.value counterpart; they are
# included here until the public generate_inputs() API aligns with internal names:
#   "freq"          ↔  CalcType.FREQ.value      == "frequency"
#   "freq_ir"       ↔  CalcType.FREQ_IR.value   == "frequency_ir"
#   "static_charge" ↔  CalcType.CHG_SP.value    == "static_charge_density"
_VALID_CALC_TYPES = frozenset(ct.value for ct in CalcType) | frozenset({
    "freq", "freq_ir", "static_charge",
})

_VALID_DFT_U_KEYS = frozenset({"LDAUU", "LDAUL", "LDAUJ", "LDAUTYPE"})

# calc_type values that do NOT use the standard 'structure' field at all.
# - neb:   uses start_structure + end_structure (separate endpoint files)
# - dimer: uses prev_dir pointing to a completed NEB calculation output
# Add new entries here only when a calc_type has its own dedicated input model.
NO_STRUCTURE_CALC_TYPES: frozenset = frozenset({"neb", "dimer"})

# calc_types where prev_dir is mandatory — its absence is a hard error.
# dimer always continues from a prior NEB run; without neb_dir it cannot build inputs.
# Note: "neb" was removed here — NEB needs start_structure + end_structure, not prev_dir.
_REQUIRES_PREV_DIR = frozenset({"dimer"})

# Maps calc_type-specific parameters to the calc_type(s) they are valid for.
# Used to warn users when a parameter is set but will have no effect for the
# chosen calc_type (the parameter is silently ignored and no file is generated).
CALC_TYPE_SPECIFIC_PARAMS: Dict[str, frozenset] = {
    "lobsterin":       frozenset({"lobster"}),
    "cohp_generator":  frozenset({"lobster"}),
    "nbo_config":      frozenset({"nbo"}),
}


# ============================================================================
# Layer 1 — field-level validators
# ============================================================================

def _check_calc_type(calc_type: str, errors: List[str]) -> None:
    """[field] calc_type must be one of the known strings."""
    if calc_type not in _VALID_CALC_TYPES:
        supported = ", ".join(sorted(_VALID_CALC_TYPES))
        errors.append(
            f'[field]    calc_type: "{calc_type}" is not recognised. '
            f"Supported: {supported}"
        )


def _check_functional(functional: str, errors: List[str]) -> None:
    """[field] functional must be one of the supported values (warning, not error)."""
    if functional.upper() not in SUPPORTED_FUNCTIONALS:
        warnings.warn(
            f'functional: "{functional}" is not in the known set '
            f"({', '.join(sorted(SUPPORTED_FUNCTIONALS))}). "
            "It will be passed through as-is.",
            stacklevel=6,
        )


def _check_kpoints_density(kpoints_density: float, errors: List[str]) -> None:
    """[field] kpoints_density must be a positive float."""
    try:
        val = float(kpoints_density)
    except (TypeError, ValueError):
        errors.append(
            f"[field]    kpoints_density: expected a positive float, "
            f"got {kpoints_density!r}"
        )
        return
    if val <= 0:
        errors.append(
            f"[field]    kpoints_density: must be positive, got {val}"
        )


def _check_magmom(
    magmom: Optional[Any],
    errors: List[str],
) -> None:
    """[field] magmom must be dict[str, float] or list[float] when provided."""
    if magmom is None:
        return
    if isinstance(magmom, dict):
        for k, v in magmom.items():
            if not isinstance(k, str):
                errors.append(
                    f"[field]    magmom: dict keys must be strings (element symbols), "
                    f"got key {k!r}"
                )
            try:
                float(v)
            except (TypeError, ValueError):
                errors.append(
                    f"[field]    magmom: dict values must be numeric, "
                    f"got {v!r} for key {k!r}"
                )
    elif isinstance(magmom, list):
        for i, v in enumerate(magmom):
            try:
                float(v)
            except (TypeError, ValueError):
                errors.append(
                    f"[field]    magmom: list entries must be numeric, "
                    f"got {v!r} at index {i}"
                )
    else:
        errors.append(
            f"[field]    magmom: must be dict[str, float] or list[float], "
            f"got {type(magmom).__name__}"
        )


def _check_dft_u(dft_u: Optional[Any], errors: List[str]) -> None:
    """[field] dft_u must be dict; inner dicts may only use recognised keys."""
    if dft_u is None:
        return
    if not isinstance(dft_u, dict):
        errors.append(
            f"[field]    dft_u: must be a dict, got {type(dft_u).__name__}"
        )
        return
    short_keys = frozenset({"U", "l", "L", "J"})
    for elem, spec in dft_u.items():
        if isinstance(spec, (int, float)):
            continue
        if not isinstance(spec, dict):
            errors.append(
                f"[field]    dft_u[{elem!r}]: must be a dict or scalar, "
                f"got {type(spec).__name__}"
            )
            continue
        unknown = set(spec.keys()) - _VALID_DFT_U_KEYS - short_keys
        if unknown:
            errors.append(
                f"[field]    dft_u[{elem!r}]: unrecognised keys {sorted(unknown)}. "
                f"Allowed: LDAUU, LDAUL, LDAUJ, LDAUTYPE (or short forms U, l, J)"
            )


def _check_prev_dir(
    prev_dir: Optional[Any], errors: List[str], dry_run: bool = False
) -> None:
    """[field] prev_dir must exist and be a directory when provided."""
    if prev_dir is None:
        return
    if dry_run:
        return
    p = Path(prev_dir)
    if not p.exists():
        errors.append(f"[field]    prev_dir: path does not exist: {prev_dir}")
    elif not p.is_dir():
        errors.append(f"[field]    prev_dir: path is not a directory: {prev_dir}")


def _check_structure(
    structure: Any,
    prev_dir: Optional[Any],
    errors: List[str],
    dry_run: bool = False,
) -> None:
    """[field] structure path must exist and be readable when provided (and prev_dir absent)."""
    if structure is None:
        return
    if not isinstance(structure, (str, Path)):
        return
    # In dry_run mode the structure file may not exist yet — skip existence check.
    if dry_run:
        return
    p = Path(structure)
    if p.is_dir():
        if not ((p / "CONTCAR").exists() or (p / "POSCAR").exists()):
            errors.append(
                f"[field]    structure: directory {p} contains neither CONTCAR nor POSCAR"
            )
    elif not p.exists():
        if not prev_dir:
            errors.append(f"[field]    structure: file does not exist: {p}")


def _check_incar(incar: Optional[Any], errors: List[str]) -> None:
    """[field] incar must be a flat dict[str, Any] when provided."""
    if incar is None:
        return
    if not isinstance(incar, dict):
        errors.append(
            f"[field]    incar: must be a flat dict, got {type(incar).__name__}"
        )
        return
    for k, v in incar.items():
        if not isinstance(k, str):
            errors.append(f"[field]    incar: keys must be strings, got {k!r}")
        if isinstance(v, dict):
            errors.append(
                f"[field]    incar: values must be scalars (not dicts); "
                f'key "{k}" has a nested dict'
            )


def _check_output_dir(output_dir: Optional[Any], errors: List[str]) -> None:
    """[field] output_dir parent must be writable when the path is given."""
    if output_dir is None:
        return
    p = Path(output_dir)
    parent = p.parent
    if parent.exists() and not parent.is_dir():
        errors.append(
            f"[field]    output_dir: parent path is not a directory: {parent}"
        )


import re as _re
_WALLTIME_RE = _re.compile(r"^\d+:\d{2}:\d{2}(:\d{2})?$")


def _check_walltime(walltime: Optional[Any], errors: List[str]) -> None:
    """[field] walltime must match HH:MM:SS or DD:HH:MM:SS format when provided."""
    if walltime is None:
        return
    if not isinstance(walltime, str) or not _WALLTIME_RE.match(walltime):
        errors.append(
            f"[field]    walltime: {walltime!r} is not a valid format.\n"
            "            Expected HH:MM:SS or DD:HH:MM:SS "
            "(e.g., \"48:00:00\" or \"2:00:00:00\")"
        )


def _check_ncores(ncores: Optional[Any], errors: List[str]) -> None:
    """[field] ncores must be a positive integer when provided."""
    if ncores is None:
        return
    try:
        val = int(ncores)
    except (TypeError, ValueError):
        errors.append(
            f"[field]    ncores: must be a positive integer, got {ncores!r}"
        )
        return
    if val <= 0:
        errors.append(
            f"[field]    ncores: must be a positive integer, got {val}"
        )


# ============================================================================
# Layer 2 — cross-field validators
# ============================================================================

def _cross_prev_dir_structure(
    calc_type: str,
    structure: Any,
    prev_dir: Optional[Any],
    errors: List[str],
    dry_run: bool = False,
) -> None:
    """[combo] If prev_dir given and structure absent → prev_dir must contain POSCAR/CONTCAR."""
    if prev_dir is None or dry_run:
        return
    has_structure = (
        structure is not None
        and isinstance(structure, (str, Path))
        and Path(structure).exists()
    )
    if has_structure:
        return
    p = Path(prev_dir)
    if p.is_dir():
        if not ((p / "CONTCAR").exists() or (p / "POSCAR").exists()):
            errors.append(
                f"[combo]    prev_dir provided but structure is absent: "
                f"{p} contains neither CONTCAR nor POSCAR"
            )


def _cross_magmom_site_count(
    magmom: Optional[Any],
    structure: Any,
    prev_dir: Optional[Any],
    errors: List[str],
) -> None:
    """[combo] If magmom is a per-site list its length must match the structure site count."""
    if not isinstance(magmom, list):
        return
    # Only check when we can resolve the structure cheaply (no I/O for prev_dir).
    if structure is None or not isinstance(structure, (str, Path)):
        return
    p = Path(structure)
    if not p.is_file():
        return
    try:
        from pymatgen.core import Structure as _S
        s = _S.from_file(str(p))
        if len(magmom) != len(s):
            errors.append(
                f"[combo]    magmom list length ({len(magmom)}) does not match "
                f"structure site count ({len(s)})"
            )
    except Exception:
        pass


def _cross_neb_dimer_requires_prev_dir(
    calc_type: str,
    prev_dir: Optional[Any],
    errors: List[str],
) -> None:
    """[combo] calc_type 'neb' or 'dimer' requires prev_dir."""
    if calc_type in _REQUIRES_PREV_DIR and not prev_dir:
        errors.append(
            f'[combo]    calc_type "{calc_type}" requires prev_dir to be provided'
        )

def _cross_lobster_lwave(
    calc_type: str,
    incar: Optional[Any],
    errors: List[str],
) -> None:
    """[combo] lobster: incar must not explicitly set LWAVE=False."""
    if calc_type != "lobster":
        return
    if not isinstance(incar, dict):
        return
    lwave = incar.get("LWAVE")
    if lwave is False or str(lwave).upper() in ("FALSE", ".FALSE.", "F"):
        errors.append(
            "[combo]    calc_type \"lobster\": incar must not set LWAVE=False "
            "(Lobster requires WAVECAR)"
        )


def _cross_dft_u_not_with_hse(
    functional: str,
    dft_u: Optional[Any],
    errors: List[str],
) -> None:
    """[combo] DFT+U is not compatible with HSE."""
    if dft_u is not None and functional.upper() == "HSE":
        errors.append(
            "[combo]    dft_u is not compatible with functional=\"HSE\""
        )


def _cross_hse_kpoints_warning(
    functional: str,
    kpoints_density: float,
) -> None:
    """[combo] HSE + high kpoints_density: emit a warning (not an error)."""
    try:
        val = float(kpoints_density)
    except (TypeError, ValueError):
        return
    if functional.upper() == "HSE" and val > 20:
        warnings.warn(
            f"functional=\"HSE\" with kpoints_density={val} is very expensive. "
            "Consider kpoints_density ≤ 30 for HSE calculations.",
            stacklevel=6,
        )


def _cross_structure_required(
    calc_type: str,
    structure: Any,
    prev_dir: Optional[Any],
    errors: List[str],
    dry_run: bool = False,
) -> None:
    """[combo] Non-neb/dimer calc_types need either structure or prev_dir."""
    if calc_type in NO_STRUCTURE_CALC_TYPES or dry_run:
        return
    if structure is None and not prev_dir:
        errors.append(
            f'[combo]    calc_type "{calc_type}" requires either \'structure\' '
            "or 'prev_dir' to be provided"
        )


def _cross_neb_requires_start_end_structures(
    calc_type: str,
    extra: Dict[str, Any],
    errors: List[str],
    dry_run: bool = False,
) -> None:
    """[combo] neb requires both start_structure and end_structure (presence + path existence).

    Path existence check mirrors the removed WorkflowConfig.validate() ③.
    Skipped in dry_run mode — files may not exist yet.
    """
    if calc_type != "neb":
        return
    start = extra.get("start_structure")
    end   = extra.get("end_structure")
    if not start or not end:
        errors.append(
            '[combo]    calc_type "neb" requires both \'start_structure\' '
            "and 'end_structure'"
        )
        return  # no point checking paths if values are missing
    if dry_run:
        return
    for field_name, val in (("start_structure", start), ("end_structure", end)):
        if not isinstance(val, (str, Path)):
            continue
        p = Path(val)
        if p.is_dir():
            if not ((p / "CONTCAR").exists() or (p / "POSCAR").exists()):
                errors.append(
                    f"[combo]    neb {field_name}: directory {p} contains "
                    "neither CONTCAR nor POSCAR"
                )
        elif not p.exists():
            errors.append(
                f"[combo]    neb {field_name}: path does not exist: {p}"
            )


def _cross_no_structure_calc_type_warns_if_structure_set(
    calc_type: str,
    structure: Any,
) -> None:
    """[combo] structure is ignored for neb/dimer — emit a warning if supplied."""
    if calc_type in NO_STRUCTURE_CALC_TYPES and structure is not None:
        warnings.warn(
            f"[combo] 'structure' is ignored for calc_type \"{calc_type}\" — "
            f"see documentation for the correct input fields "
            f"({'start_structure + end_structure' if calc_type == 'neb' else 'prev_dir'}).",
            stacklevel=6,
        )


def _cross_calc_type_specific_params_warning(
    calc_type: str,
    extra: Dict[str, Any],
) -> None:
    """[combo] Warn when a calc_type-specific parameter is set for a non-matching calc_type."""
    for param_name, valid_calc_types in CALC_TYPE_SPECIFIC_PARAMS.items():
        val = extra.get(param_name)
        if val is None:
            continue
        # Empty dicts/lists count as "not set" — no warning needed.
        if isinstance(val, (dict, list)) and len(val) == 0:
            continue
        if calc_type not in valid_calc_types:
            warnings.warn(
                f"[combo] Parameter '{param_name}' is set but has no effect for "
                f'calc_type "{calc_type}". '
                f"This parameter is only used for: {sorted(valid_calc_types)}. "
                "No corresponding file will be generated. The calculation will proceed normally.",
                stacklevel=6,
            )


# ============================================================================
# Layer 3 — per-calc-type business logic
# ============================================================================

# Registry: calc_type string → validator(params_dict, errors) -> None
_BUSINESS_REGISTRY: Dict[str, Callable[[Dict[str, Any], List[str]], None]] = {}


def _register(*calc_types: str):
    def decorator(fn):
        for ct in calc_types:
            _BUSINESS_REGISTRY[ct] = fn
        return fn
    return decorator


@_register("neb")
def _biz_neb(p: Dict[str, Any], errors: List[str]) -> None:
    """[business] neb: either neb_images (≥3) or start_structure+end_structure required."""
    images = p.get("neb_images")
    start = p.get("start_structure")
    end = p.get("end_structure")
    if images is not None:
        if not isinstance(images, (list, tuple)) or len(images) < 3:
            errors.append(
                f"[business] calc_type \"neb\": at least 3 image structures required, "
                f"got {len(images) if isinstance(images, (list, tuple)) else 0}"
            )
    elif not (start and end):
        errors.append(
            "[business] calc_type \"neb\": image structures must be provided"
        )


@_register("freq", "freq_ir")
def _biz_freq(p: Dict[str, Any], errors: List[str]) -> None:
    """[business] freq/freq_ir without prev_dir: selective dynamics must be present."""
    if p.get("prev_dir"):
        return
    structure = p.get("structure")
    if structure is None or not isinstance(structure, (str, Path)):
        return
    path = Path(structure)
    if not path.is_file():
        return
    try:
        content = path.read_text(errors="replace")
        lines = content.splitlines()
        # POSCAR line 8 (index 7) is "Selective dynamics" if present.
        if len(lines) >= 8 and lines[7].strip().lower().startswith("s"):
            return  # selective dynamics block found
        errors.append(
            "[business] calc_type \"freq\"/\"freq_ir\" without prev_dir: "
            "selective dynamics must be set in the structure file so the "
            "engine knows which atoms to vibrate"
        )
    except Exception:
        pass


@_register("md_nvt", "md_npt")
def _biz_md(p: Dict[str, Any], errors: List[str]) -> None:
    """[business] MD: temperature and nsteps must be positive integers."""
    temp = p.get("temperature")
    nsteps = p.get("nsteps")
    if temp is not None:
        try:
            if int(temp) <= 0:
                errors.append(
                    f"[business] calc_type MD: temperature must be a positive integer, got {temp}"
                )
        except (TypeError, ValueError):
            errors.append(
                f"[business] calc_type MD: temperature must be a positive integer, got {temp!r}"
            )
    if nsteps is not None:
        try:
            if int(nsteps) <= 0:
                errors.append(
                    f"[business] calc_type MD: nsteps must be a positive integer, got {nsteps}"
                )
        except (TypeError, ValueError):
            errors.append(
                f"[business] calc_type MD: nsteps must be a positive integer, got {nsteps!r}"
            )


@_register("lobster")
def _biz_lobster(p: Dict[str, Any], errors: List[str]) -> None:
    """[business] lobster: warn if NBANDS is not explicitly set in incar."""
    incar = p.get("incar")
    if not isinstance(incar, dict) or "NBANDS" not in incar:
        warnings.warn(
            "calc_type=\"lobster\": NBANDS is not set in incar. "
            "Lobster typically requires an increased NBANDS for accurate COHP. "
            "Consider adding incar={\"NBANDS\": <N>}.",
            stacklevel=6,
        )


def _biz_vdw_kernel(p: Dict[str, Any], errors: List[str]) -> None:
    """[business] If functional requires vdW kernel, check it exists in SCRIPT_DIR."""
    functional = p.get("functional", "")
    if functional.upper() not in {"BEEF", "BEEFVTST"}:
        return
    try:
        from .script_writer import SCRIPT_DIR
        kernel = SCRIPT_DIR / "vdw_kernel.bindat"
        if not kernel.is_file():
            errors.append(
                f"[business] functional \"{functional}\" requires vdw_kernel.bindat but "
                f"it was not found at {kernel}. "
                "Place the file there before running."
            )
    except ImportError:
        pass


def _run_business_layer(params_dict: Dict[str, Any], errors: List[str]) -> None:
    _biz_vdw_kernel(params_dict, errors)
    calc_type = params_dict.get("calc_type", "")
    validator = _BUSINESS_REGISTRY.get(calc_type)
    if validator is None and calc_type in _VALID_CALC_TYPES:
        pass  # no registered validator → no warning needed, calc_type is still valid
    elif validator is None:
        pass  # unknown calc_type already caught by layer 1
    else:
        validator(params_dict, errors)


# ============================================================================
# Public entry point
# ============================================================================

def validate(
    calc_type: str,
    structure: Any = None,
    functional: str = "PBE",
    kpoints_density: float = 50.0,
    output_dir: Any = None,
    prev_dir: Any = None,
    incar: Optional[Dict[str, Any]] = None,
    magmom: Any = None,
    dft_u: Any = None,
    walltime: Optional[str] = None,
    ncores: Optional[int] = None,
    dry_run: bool = False,
    **extra: Any,
) -> None:
    """Run all three validation layers and raise ValidationError if any fail.

    Must be called as the very first action in generate_inputs(), before any
    parameter defaulting, transformation, or file I/O.

    Args:
        calc_type: Calculation type string.
        structure:  Path to structure file or directory.
        functional: Exchange-correlation functional label.
        kpoints_density: K-point sampling density (must be positive).
        output_dir: Output directory path (parent must be writable).
        prev_dir:   Previous calculation directory.
        incar:      Flat INCAR override dict.
        magmom:     Per-site list or per-element dict of magnetic moments.
        dft_u:      DFT+U specification dict.
        walltime:   Wall-clock time in ``"HH:MM:SS"`` format, e.g. ``"24:00:00"``.
        ncores:     Number of CPU cores (positive integer).
        **extra:    Additional calc-type-specific parameters forwarded to
                    the business-logic layer (e.g. neb_images, temperature,
                    nsteps).

    Raises:
        ValidationError: If any validation check fails; all errors are bundled.

    本函数在 generate_inputs() 中必须作为第一步调用，在任何参数默认化、
    转换或文件 I/O 之前执行。无错误时静默返回；有错误时抛出 ValidationError。
    """
    # Normalize CalcType.value strings to public API aliases used throughout this module.
    # WorkflowEngine passes CalcType.value (e.g. "frequency"); the registry and rules
    # expect the frontend alias (e.g. "freq").
    # 将 CalcType.value 字符串规范化为本模块使用的公开 API 别名。
    _ALIAS_MAP = {
        "frequency":              "freq",
        "frequency_ir":           "freq_ir",
        "static_charge_density":  "static_charge",
    }
    calc_type = _ALIAS_MAP.get(calc_type, calc_type)

    errors: List[str] = []

    # ── Layer 1: field-level ─────────────────────────────────────────────────
    _check_calc_type(calc_type, errors)
    _check_functional(functional, errors)  # warning only, never adds to errors
    _check_kpoints_density(kpoints_density, errors)
    _check_magmom(magmom, errors)
    _check_dft_u(dft_u, errors)
    _check_prev_dir(prev_dir, errors, dry_run=dry_run)
    if calc_type not in NO_STRUCTURE_CALC_TYPES:
        _check_structure(structure, prev_dir, errors, dry_run=dry_run)
    _check_incar(incar, errors)
    _check_output_dir(output_dir, errors)
    _check_walltime(walltime, errors)
    _check_ncores(ncores, errors)

    # ── Layer 2: cross-field ─────────────────────────────────────────────────
    _cross_prev_dir_structure(calc_type, structure, prev_dir, errors, dry_run=dry_run)
    _cross_structure_required(calc_type, structure, prev_dir, errors, dry_run=dry_run)
    _cross_magmom_site_count(magmom, structure, prev_dir, errors)
    _cross_neb_dimer_requires_prev_dir(calc_type, prev_dir, errors)
    _cross_neb_requires_start_end_structures(calc_type, extra, errors, dry_run=dry_run) 
    _cross_no_structure_calc_type_warns_if_structure_set(calc_type, structure)  # warning only
    _cross_lobster_lwave(calc_type, incar, errors)
    _cross_dft_u_not_with_hse(functional, dft_u, errors)
    _cross_hse_kpoints_warning(functional, kpoints_density)  # warning only
    _cross_calc_type_specific_params_warning(calc_type, {  # warning only
        "lobsterin": extra.get("lobsterin"),
        "cohp_generator": extra.get("cohp_generator"),
        "nbo_config": extra.get("nbo_config"),
    })

    # ── Layer 3: business logic ──────────────────────────────────────────────
    params_dict: Dict[str, Any] = {
        "calc_type":       calc_type,
        "structure":       structure,
        "functional":      functional,
        "kpoints_density": kpoints_density,
        "prev_dir":        prev_dir,
        "incar":           incar,
        "magmom":          magmom,
        "dft_u":           dft_u,
        **extra,
    }
    _run_business_layer(params_dict, errors)

    if errors:
        raise ValidationError(errors)
