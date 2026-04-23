# -*- coding: utf-8 -*-
"""
flow.script_writer — PBS submission script writer for VASP workflows
=====================================================================

Responsibilities:
  - Select the correct template from SCRIPT_DIR based on calc_type (table-driven).
  - Substitute all placeholders and write ``submit.sh`` to the output directory.
  - Copy the vdW kernel file (``vdw_kernel.bindat``) when the functional requires it.

ScriptWriter has zero knowledge of VASP input generation; it only handles
templates and file copying.

本模块仅负责 PBS 脚本模板渲染和 vdW 核文件复制，与 VASP 输入生成完全解耦。
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from .script import (
    CalcCategory,
    Script,
    CALC_TYPE_TO_CATEGORY,
    _CATEGORY_CONFIG,
    _FUNCTIONAL_TYPE_MAP,
)

logger = logging.getLogger(__name__)

# ── Single configurable base path ────────────────────────────────────────────
SCRIPT_DIR: Path = Path(__file__).resolve().parent / "script"

# ── Table-driven template selection ──────────────────────────────────────────
# All calc_types currently share one template.  Add entries here when a
# dedicated per-calc_type template is added to SCRIPT_DIR.
_CALC_TYPE_TEMPLATE_MAP: Dict[str, str] = {
    "bulk_relax":     "script.txt",
    "slab_relax":     "script.txt",
    "static_sp":      "script.txt",
    "static_dos":     "script.txt",
    "static_charge":  "script.txt",
    "static_elf":     "script.txt",
    "neb":            "script.txt",
    "dimer":          "script.txt",
    "freq":           "script.txt",
    "freq_ir":        "script.txt",
    "lobster":        "script.txt",
    "nmr_cs":         "script.txt",
    "nmr_efg":        "script.txt",
    "nbo":            "script.txt",
    "md_nvt":         "script.txt",
    "md_npt":         "script.txt",
}

# Functionals that require vdw_kernel.bindat to be present alongside the inputs.
_VDW_FUNCTIONALS: frozenset = frozenset({"BEEF", "BEEFVTST"})


def _hours_to_hms(hours: int) -> str:
    """Convert integer hours to HH:MM:SS string (minutes/seconds always 00)."""
    return f"{hours:02d}:00:00"


class ScriptWriter:
    """Write PBS submission scripts from templates; copy vdW kernel when needed.

    Zero knowledge of VASP input generation — only templates and file I/O.
    """

    # ------------------------------------------------------------------
    def write(
        self,
        output_dir: Path,
        calc_type: str,
        functional: str = "PBE",
        walltime: Optional[str] = None,
        ncores: Optional[int] = None,
        dry_run: bool = False,
    ) -> Optional[str]:
        """Render and write ``submit.sh`` into *output_dir*.

        Args:
            output_dir: Directory where the script is written.
            calc_type:  Workflow calc_type string (e.g. ``"bulk_relax"``).
            functional: Exchange-correlation functional (e.g. ``"PBE"``).
            walltime:   Wall-clock time in ``"HH:MM:SS"`` format.
                        ``None`` → use the calc_type default from
                        ``_CATEGORY_CONFIG``.
            ncores:     Number of CPU cores.  ``None`` → use the calc_type
                        default from ``_CATEGORY_CONFIG``.
            dry_run:    When ``True`` print the rendered script content to
                        stdout without writing any files.

        Returns:
            Absolute path to the written script, or ``None`` when
            ``dry_run=True``.
        """
        template_path = self._resolve_template(calc_type)
        template_content = template_path.read_text(encoding="utf-8")

        context = self._build_context(
            calc_type=calc_type,
            functional=functional,
            walltime=walltime,
            ncores=ncores,
            output_dir=output_dir,
        )

        rendered = self._render(template_content, context)

        if dry_run:
            job_name = context.get("JOB_NAME", Path(output_dir).name)
            directives = [
                line.rstrip()
                for line in rendered.splitlines()
                if line.lstrip().startswith(("#PBS", "#SBATCH"))
            ]
            print(f"[dry_run] Submission script — PBS directives for {job_name}:")
            for line in directives:
                print(f"  {line.lstrip()}")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        script_path = output_dir / "submit.sh"
        script_path.write_text(rendered, encoding="utf-8")
        try:
            os.chmod(script_path, 0o755)
        except OSError:
            pass

        logger.info("Wrote submission script: %s", script_path)
        self._copy_vdw_kernel(functional, output_dir)
        return str(script_path)

    # ------------------------------------------------------------------
    def _resolve_template(self, calc_type: str) -> Path:
        filename = _CALC_TYPE_TEMPLATE_MAP.get(calc_type, "script.txt")
        return SCRIPT_DIR / filename

    # ------------------------------------------------------------------
    def _build_context(
        self,
        calc_type: str,
        functional: str,
        walltime: Optional[str],
        ncores: Optional[int],
        output_dir: Path,
    ) -> Dict[str, Any]:
        # Derive calc_category → category config → auto defaults.
        calc_category = CALC_TYPE_TO_CATEGORY.get(calc_type)
        cat_cfg = (
            _CATEGORY_CONFIG.get(calc_category, _CATEGORY_CONFIG[CalcCategory.RELAX])
            if calc_category
            else _CATEGORY_CONFIG[CalcCategory.RELAX]
        )

        # Helper: map functional to TYPE1.
        functional_upper = functional.upper()
        type1 = "org"
        for key, val in _FUNCTIONAL_TYPE_MAP.items():
            if key in functional_upper:
                type1 = val
                break

        # Cleanup and EXTRA_CMD come from the existing Script helper.
        _script = Script(template_path=self._resolve_template(calc_type))
        resolved_cores = ncores if ncores is not None else cat_cfg["typical_cores"]
        extra_cmd = _script._build_extra_cmd(
            calc_category or CalcCategory.RELAX,
            cores_for_cmd=resolved_cores,
        )
        cleanup_cmd = (
            "rm REPORT CHG* DOSCAR EIGENVAL IBZKPT PCDAT PROCAR "
            "WAVECAR XDATCAR vasprun.xml FORCECAR"
            if cat_cfg["cleanup"]
            else ""
        )

        # Walltime: user-supplied HH:MM:SS takes priority; else convert hours.
        resolved_walltime = (
            walltime
            if walltime is not None
            else _hours_to_hms(cat_cfg["typical_walltime"])
        )

        return {
            "JOB_NAME":   Path(output_dir).name,
            "CORES":      resolved_cores,
            "WALLTIME":   resolved_walltime,
            "QUEUE":      "low",
            "TYPE1":      type1,
            "COMPILER":   cat_cfg["compiler"],
            "CLEANUP_CMD": cleanup_cmd,
            "EXTRA_CMD":  extra_cmd,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _render(template: str, context: Dict[str, Any]) -> str:
        result = template
        for key, value in context.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    # ------------------------------------------------------------------
    # Low-level filesystem helpers — all file I/O for the script package
    # lives here so that script.py contains no direct filesystem operations.
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        """Create *path* and all parents; no-op if it already exists."""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_script_file(path: Path, content: str, make_executable: bool = True) -> None:
        """Write *content* to *path* and optionally set executable bits."""
        path.write_text(content, encoding="utf-8")
        if make_executable:
            try:
                os.chmod(path, 0o755)
            except OSError:
                pass

    @staticmethod
    def _copy_file(src: Path, dst: Path) -> None:
        """Copy a single file from *src* to *dst*."""
        shutil.copy2(src, dst)

    # ------------------------------------------------------------------
    def _copy_vdw_kernel(self, functional: str, output_dir: Path) -> None:
        if functional.upper() not in _VDW_FUNCTIONALS:
            return
        src = SCRIPT_DIR / "vdw_kernel.bindat"
        if not src.is_file():
            raise FileNotFoundError(
                f"vdW kernel file required for functional '{functional}' not found.\n"
                f"Expected location: {SCRIPT_DIR / 'vdw_kernel.bindat'}\n"
                "Please place the file at the expected location and retry."
            )
        dst = output_dir / "vdw_kernel.bindat"
        if not dst.exists():
            shutil.copy2(src, dst)
            logger.info("Copied vdw_kernel.bindat → %s", dst)

