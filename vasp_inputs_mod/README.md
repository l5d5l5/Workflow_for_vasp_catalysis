# VASP Flow Module — Code Logic Reference

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Architecture and Data Flow](#2-architecture-and-data-flow)
3. [Module Details](#3-module-details)
   - [api.py — Frontend Adapter](#31-apipy--frontend-adapter)
   - [workflow_engine.py — Workflow Engine](#32-workflow_enginepy--workflow-engine)
   - [maker.py — Input File Factory](#33-makerpy--input-file-factory)
   - [input_sets.py — Input Set Wrappers](#34-input_setspy--input-set-wrappers)
   - [constants.py — Constants and Defaults](#35-constantspy--constants-and-defaults)
   - [kpoints.py — K-point Generator](#36-kpointspy--k-point-generator)
   - [utils.py — Utility Functions](#37-utilspy--utility-functions)
   - [script.py — Job Script Generator](#38-scriptpy--job-script-generator)
4. [Supported Calculation Types](#4-supported-calculation-types)
5. [Parameter Priority and Merge Rules](#5-parameter-priority-and-merge-rules)
6. [Extension Guide](#6-extension-guide)
7. [Usage Examples](#7-usage-examples)

---

## 1. Module Overview

| File | Responsibility | Key classes / functions |
|------|---------------|------------------------|
| `api.py` | Frontend dict → typed parameter objects | `FrontendAdapter`, `VaspWorkflowParams` |
| `workflow_engine.py` | Calc-type registry + dispatch engine | `CalcType`, `CALC_TYPE_REGISTRY`, `WorkflowEngine` |
| `maker.py` | Per-type input file factory | `VaspInputMaker` |
| `input_sets.py` | pymatgen InputSet subclasses | `BulkRelaxSetEcat`, `SlabSetEcat`, … |
| `constants.py` | INCAR defaults, functional patches | `DEFAULT_INCAR_*`, `FUNCTIONAL_INCAR_PATCHES` |
| `kpoints.py` | K-point mesh generation | `build_kpoints_by_lengths` |
| `utils.py` | Structure loading, format conversion | `load_structure`, `infer_functional_from_incar` |
| `script.py` | PBS/SLURM script rendering | `Script`, `CalcCategory` |

---

## 2. Architecture and Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  External caller / workflow stage                               │
│  frontend_dict = { "calc_type": "bulk_relax", … }              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  api.py — FrontendAdapter.from_frontend_dict()                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  1. Parse structure input  (FrontendStructureInput)     │    │
│  │  2. Extract sub-param groups:                           │    │
│  │     MAGMOM, DFT+U, vdW, dipole, lobster, NBO,          │    │
│  │     frequency, NMR, MD, NEB                             │    │
│  │  3. Normalise → VaspWorkflowParams                      │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ .to_workflow_config()
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  workflow_engine.py — WorkflowConfig dataclass                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  CalcType enum value + all engine params                │    │
│  │  (structure, dirs, INCAR overrides, …)                  │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ WorkflowEngine().run(config)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  workflow_engine.py — WorkflowEngine.run()                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  1. Look up CALC_TYPE_REGISTRY for INCAR template       │    │
│  │  2. Merge registry defaults + user_incar_overrides      │    │
│  │  3. match CalcType → call maker.write_*()               │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  maker.py — VaspInputMaker.write_*()                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  1. Merge global INCAR + per-call overrides             │    │
│  │  2. Normalise MAGMOM (list/str → per-element dict)      │    │
│  │  3. Ensure output directory exists                      │    │
│  │  4. Instantiate the matching *SetEcat object            │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ .write_input(output_dir)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  input_sets.py — *SetEcat.write_input()                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  1. _build_incar(): defaults → functional patch →       │    │
│  │     user overrides                                      │    │
│  │  2. Call pymatgen parent to write files                 │    │
│  │  3. Strip @CLASS/@MODULE metadata comments              │    │
│  │  4. Enforce LDAU=False if no U values provided          │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
            POSCAR / INCAR / KPOINTS / POTCAR / lobsterin (disk)
```

---

## 3. Module Details

### 3.1 `api.py` — Frontend Adapter

**Role:** Convert an unstructured frontend dict into strongly typed parameter objects, decoupling the frontend from the engine internals.

#### Frontend parameter dataclasses

```
FrontendStructureInput       Structure source (file / library / task)
FrontendPrecisionParams      ENCUT, EDIFF, EDIFFG, NEDOS
FrontendKpointParams         K-point density, gamma-centred flag
FrontendMagmomParams         MAGMOM — per_atom list OR per_element dict
FrontendDFTPlusUParams       DFT+U — nested {elem: {LDAUU, LDAUL, LDAUJ}}
FrontendVdwParams            vdW correction method (None / D3 / D3BJ / …)
FrontendDipoleParams         LDIPOL / IDIPOL settings
FrontendFrequencyParams      IBRION, POTIM, NFREE, vibrate_indices, calc_ir
FrontendLobsterParams        lobsterin_mode, overwritedict, custom_lobsterin_lines
FrontendNBOParams            NBO program config (basis, occ thresholds, …)
FrontendMDParams             MD ensemble, temperatures, steps, time_step
FrontendNEBParams            n_images, use_idpp, start/end structures
FrontendResourceParams       cores, walltime, queue
VaspWorkflowParams           Top-level container holding all of the above
```

#### Structure input resolution (`FrontendStructureInput.to_path_or_content`)

```
source="file"    → returns content (if non-empty) or id (file path)
source="library" → returns id (requires external library implementation)
source="task"    → verifies directory exists, then returns id
```

#### Frontend string → CalcType mapping (`FRONTEND_CALC_TYPE_MAP`)

Frontend strings (e.g. `"bulk_relax"`) are mapped to `CalcType` enum values via
this dict, decoupling user-visible names from internal enum naming.

#### Main entry point: `FrontendAdapter.from_frontend_dict(data)`

```
Input: dict (raw frontend data)
  ├── extract calc_type → look up FRONTEND_CALC_TYPE_MAP
  ├── extract xc / kpoints_density / custom_incar
  ├── extract structure → FrontendStructureInput
  ├── extract MAGMOM  → FrontendMagmomParams (per_atom or per_element)
  ├── extract LDAUU / LDAUL / LDAUJ → FrontendDFTPlusUParams
  ├── extract lobster params → FrontendLobsterParams
  ├── extract nbo_params → FrontendNBOParams
  ├── extract frequency_params → FrontendFrequencyParams
  ├── extract nmr_params → FrontendNMRParams
  └── extract md_params  → FrontendMDParams
Output: VaspWorkflowParams
```

#### MAGMOM handling in `to_workflow_config()`

| Input format | Resulting `user_incar_overrides["MAGMOM"]` |
|---|---|
| `per_atom` list | `List[float]` — per-site, passed directly |
| `per_element` dict | `Dict[str, float]` — pymatgen expands by element order |

`VaspInputMaker._apply_magmom_compat()` further normalises list/string formats
to a per-element dict before the value reaches the pymatgen InputSet.

#### DFT+U handling in `to_workflow_config()`

`FrontendDFTPlusUParams.to_pymatgen_format()` converts the nested per-element
dict to three separate dicts and injects them into `user_incar_overrides`:

```python
user_incar_overrides["LDAUU"] = {"Fe": 4.0, "Co": 3.0}   # Dict[str, float]
user_incar_overrides["LDAUL"] = {"Fe": 2,   "Co": 2}      # Dict[str, int]
user_incar_overrides["LDAUJ"] = {"Fe": 0.0, "Co": 0.0}    # Dict[str, float]
user_incar_overrides["LDAU"]  = True
```

pymatgen expands these per-element dicts to per-site arrays in the INCAR.

---

### 3.2 `workflow_engine.py` — Workflow Engine

#### `CalcType` enum

The single canonical name for every supported calculation type:

| Enum value | Frontend string | Description |
|---|---|---|
| `BULK_RELAX` | `bulk_relax` | Bulk ionic + cell relaxation |
| `SLAB_RELAX` | `slab_relax` | Surface slab relaxation |
| `STATIC_SP`  | `static_sp` | Single-point (no charge/wave output) |
| `DOS_SP`     | `static_dos` | Static + DOS (retains CHGCAR) |
| `CHG_SP`     | `static_charge` | Static + charge density |
| `ELF_SP`     | `static_elf` | Static + ELF |
| `NEB`        | `neb` | Minimum energy path (VTST) |
| `DIMER`      | `dimer` | Dimer transition-state (VTST) |
| `FREQ`       | `freq` | Vibrational frequency |
| `FREQ_IR`    | `freq_ir` | Frequency + IR (DFPT) |
| `LOBSTER`    | `lobster` | COHP chemical-bonding analysis |
| `NMR_CS`     | `nmr_cs` | NMR chemical shift |
| `NMR_EFG`    | `nmr_efg` | NMR electric field gradient |
| `NBO`        | `nbo` | Natural Bond Orbital analysis |
| `MD_NVT`     | `md_nvt` | NVT molecular dynamics |
| `MD_NPT`     | `md_npt` | NPT molecular dynamics |

#### `CALC_TYPE_REGISTRY`

Maps each `CalcType` to a `CalcTypeConfig`:

```python
@dataclass(frozen=True)
class CalcTypeConfig:
    incar_base:      Dict[str, Any]   # DEFAULT_INCAR_* from constants.py
    incar_delta:     Dict[str, Any]   # incremental overrides on top of base
    need_wavecharge: bool             # retain WAVECAR/CHGCAR after job
    need_vtst:       bool             # require VTST-patched VASP binary
    beef_compatible: bool             # False for NMR, NBO, LOBSTER
    script_category: CalcCategory     # PBS template category
```

`get_merged_incar(user_overrides)` merges `incar_base + incar_delta + user_overrides`
and is called by `WorkflowEngine._get_incar_params()`.

#### `WorkflowConfig` dataclass

Complete parameter set consumed by `WorkflowEngine.run()`:

```
Core:            calc_type, structure, functional, kpoints_density,
                 output_dir, prev_dir
MD:              ensemble, start_temp, end_temp, nsteps, time_step
NEB:             n_images, use_idpp, start_structure, end_structure
Frequency:       vibrate_indices, calc_ir
NMR:             isotopes
NBO:             nbo_config
Lobster:         lobster_overwritedict, lobster_custom_lines
Advanced:        user_incar_overrides
```

#### `WorkflowEngine.run(config)` dispatch

```python
engine = WorkflowEngine()
engine.run(config)
```

Dispatch sequence:
```
1. Auto-detect prev_dir if not supplied
2. Validate config (raises ValueError on bad params)
3. Build maker via replace() with merged INCAR from _get_incar_params()
4. match config.calc_type:
     BULK_RELAX              → maker.write_bulk(struct, output_dir)
     SLAB_RELAX              → maker.write_slab(struct, output_dir)
     STATIC_SP / DOS_SP /
       CHG_SP / ELF_SP       → maker.write_noscf(output_dir, ...)
     NEB                     → maker.write_neb(output_dir, ...)
     DIMER                   → maker.write_dimer(output_dir, neb_dir=prev)
     FREQ                    → maker.write_freq(output_dir, ..., calc_ir=False)
     FREQ_IR                 → maker.write_freq(output_dir, ..., calc_ir=True)
     LOBSTER                 → maker.write_lobster(output_dir, ...)
     NMR_CS / NMR_EFG        → maker.write_nmr(output_dir, ...)
     NBO                     → maker.write_nbo(output_dir, ...)
     MD_NVT / MD_NPT         → maker.write_md(output_dir, ...)
```

---

### 3.3 `maker.py` — Input File Factory

#### `VaspInputMaker` dataclass attributes

| Attribute | Default | Description |
|---|---|---|
| `functional` | `"PBE"` | DFT functional |
| `kpoints_density` | `50.0` | K-point density (points per Å⁻¹) |
| `use_default_incar` | `True` | Apply built-in INCAR defaults |
| `use_default_kpoints` | `True` | Auto-generate KPOINTS |
| `user_incar_settings` | `{}` | Global INCAR overrides |
| `user_potcar_functional` | `"PBE_54"` | POTCAR functional tag |

#### Common flow in every `write_*` method

```
1. _build_common_kwargs(): merge global user_incar_settings + per-call overrides
2. _apply_magmom_compat(): normalise MAGMOM list/string → per-element dict
3. _ensure_dir(): mkdir -p output_dir
4. Instantiate matching *SetEcat object
5. Call .write_input(output_dir)
```

#### `write_*` methods

| Method | Calc types served |
|---|---|
| `write_bulk(structure, output_dir)` | `BULK_RELAX` |
| `write_slab(structure, output_dir)` | `SLAB_RELAX` |
| `write_noscf(output_dir, structure, prev_dir)` | `STATIC_SP`, `DOS_SP`, `CHG_SP`, `ELF_SP` |
| `write_neb(output_dir, start, end, n_images, use_idpp)` | `NEB` |
| `write_dimer(output_dir, neb_dir)` | `DIMER` |
| `write_lobster(output_dir, structure, prev_dir, overwritedict, custom_lobsterin_lines)` | `LOBSTER` |
| `write_freq(output_dir, prev_dir, structure, calc_ir, vibrate_indices)` | `FREQ`, `FREQ_IR` |
| `write_nbo(output_dir, structure, prev_dir, nbo_config)` | `NBO` |
| `write_nmr(output_dir, structure, isotopes)` | `NMR_CS`, `NMR_EFG` |
| `write_md(output_dir, structure, ensemble, …)` | `MD_NVT`, `MD_NPT` |
| `write_adsorption(output_dir, structure, prev_dir)` | adsorption relaxation |

---

### 3.4 `input_sets.py` — Input Set Wrappers

#### Base class `VaspInputSetEcat`

Shared base for all `*SetEcat` classes, providing two key methods:

**`_build_incar(functional, default_incar, extra_incar, user_incar_settings)`**

```
INCAR build priority (lowest → highest):
┌──────────────────────┐
│  default_incar       │  DEFAULT_INCAR_* from constants.py
├──────────────────────┤
│  FUNCTIONAL_PATCH    │  Per-functional patch (BEEF / SCAN / …)
├──────────────────────┤
│  extra_incar         │  Caller-supplied extras
├──────────────────────┤
│  user_incar_settings │  User final overrides (highest priority)
└──────────────────────┘
```

**`write_input(output_dir)` post-processing:**
1. Call pymatgen parent to write files
2. Strip `@CLASS` / `@MODULE` metadata comment lines
3. If `LDAU=True` but no U values, force `LDAU=False`

#### SetEcat class hierarchy

| Class | pymatgen parent | Calculation |
|---|---|---|
| `BulkRelaxSetEcat` | `MPMetalRelaxSet` | Bulk relaxation |
| `SlabSetEcat` | `MVLSlabSet` | Surface slab |
| `MPStaticSetEcat` | `MPStaticSet` | Static single-point |
| `LobsterSetEcat` | `LobsterSet` | Lobster COHP |
| `NEBSetEcat` | `NEBSet` | NEB |
| `FreqSetEcat` | `MPStaticSetEcat` | Frequency / vibrational |
| `DimerSetEcat` | `MPStaticSetEcat` | Dimer transition-state |
| `NBOSetEcat` | `MPStaticSetEcat` | NBO analysis |
| `NMRSetEcat` | `MPStaticSetEcat` | NMR (CS or EFG) |
| `MDSetEcat` | `MPStaticSetEcat` | Molecular dynamics |

> Note: `FreqSetEcat`, `DimerSetEcat`, `NBOSetEcat`, `NMRSetEcat`, and `MDSetEcat`
> all inherit from the project's own `MPStaticSetEcat` (not directly from pymatgen's
> `MPStaticSet`), so they pick up the `_build_incar` post-processing logic.

---

### 3.5 `constants.py` — Constants and Defaults

#### `FUNCTIONAL_INCAR_PATCHES`

Per-functional INCAR patches applied automatically in `_build_incar`:

| Functional | Key settings |
|---|---|
| `BEEF` | `GGA=BF`, `LUSE_VDW=True`, `AGGAC=0.0` |
| `BEEFVTST` | Same as BEEF (with VTST transition-state support) |
| `SCAN` | `METAGGA=SCAN`, `LASPH=True`, `ADDGRID=True` |

#### `DEFAULT_INCAR_*` templates

One dict per calculation type, imported by `workflow_engine.py`:

```
DEFAULT_INCAR_BULK      Bulk relaxation
DEFAULT_INCAR_SLAB      Surface slab
DEFAULT_INCAR_STATIC    Static calculation (base; *SetEcat applies this internally)
DEFAULT_INCAR_NEB       NEB
DEFAULT_INCAR_DIMER     Dimer
DEFAULT_INCAR_FREQ      Frequency / vibrational
DEFAULT_INCAR_LOBSTER   Lobster single-point
DEFAULT_INCAR_NBO       NBO analysis
DEFAULT_INCAR_NMR_CS    NMR chemical shift
DEFAULT_INCAR_NMR_EFG   NMR electric field gradient
DEFAULT_INCAR_MD        MD (NVT)
DEFAULT_INCAR_MD_NPT    MD (NPT)
```

`INCAR_DELTA_STATIC_SP / _DOS / _CHG / _ELF` — incremental overrides applied on
top of the static base for the four static sub-types.

#### Other constants

| Name | Purpose |
|---|---|
| `DEFAULT_NBO_CONFIG_PARAMS` | Default NBO program config parameters |
| `NBO_CONFIG_TEMPLATE` | NBO config file template string |
| `NBO_BASIS_PATH` | Default NBO basis set file path |

---

### 3.6 `kpoints.py` — K-point Generator

#### `build_kpoints_by_lengths(structure, density)`

Auto-derives a Monkhorst-Pack mesh from lattice vector lengths and a target density:

```
For each lattice direction i:
    n_i = max(1, round(density / |a_i|))

Returns: Kpoints object (Gamma-centred or Monkhorst-Pack)
```

Callers only need to supply a density value — no manual grid specification required.

---

### 3.7 `utils.py` — Utility Functions

#### `load_structure(struct_source)`

Smart structure loading supporting multiple input forms:

```
Structure object  → returned directly
File path         → parsed by pymatgen (POSCAR / CIF / CONTCAR / …)
Directory path    → searched in priority order:
                    CONTCAR > POSCAR > POSCAR.vasp > *.vasp > *.cif
String content    → parsed as POSCAR format
```

#### Other utility functions

| Function | Description |
|---|---|
| `convert_vasp_format_to_pymatgen_dict` | VASP format string → Python dict |
| `infer_functional_from_incar` | Infer functional from INCAR file |
| `pick_adsorbate_indices_by_formula_strict` | Select adsorbate atom indices by formula |
| `get_best_structure_path` | Return CONTCAR if it exists, else POSCAR |

---

### 3.8 `script.py` — Job Script Generator

#### `CalcCategory` enum

Calculation category used for PBS/SLURM template selection:

```
RELAX     → relaxation jobs
STATIC    → static single-point
NEB       → NEB / transition-state
DIMER     → Dimer transition-state
LOBSTER   → Lobster post-processing
NBO       → NBO post-processing
FREQ      → frequency calculations
NMR       → NMR calculations
MD        → molecular dynamics
```

#### `Script` class

Renders a PBS/SLURM script from a Jinja2 template.

**Parameter priority (lowest → highest):**
```
Cluster-wide defaults (cluster_defaults)
    ↓
CalcCategory auto-derived values (walltime / cores / compiler)
    ↓
Explicit user values (cores / walltime / queue)
    ↓
custom_context (full override)
```

---

## 4. Supported Calculation Types

| Calc type | Frontend string | CalcType enum | Description |
|---|---|---|---|
| Bulk relaxation | `bulk_relax` | `BULK_RELAX` | Crystal structure optimisation |
| Slab relaxation | `slab_relax` | `SLAB_RELAX` | Surface model optimisation |
| Static SP | `static_sp` | `STATIC_SP` | Single-point energy |
| Static DOS | `static_dos` | `DOS_SP` | Static + DOS (non-SCF) |
| Static charge | `static_charge` | `CHG_SP` | Static + charge density |
| Static ELF | `static_elf` | `ELF_SP` | Static + electron localisation |
| NEB | `neb` | `NEB` | Minimum energy path |
| Dimer | `dimer` | `DIMER` | Transition-state search |
| Frequency | `freq` | `FREQ` | Vibrational frequency / ZPE |
| Frequency IR | `freq_ir` | `FREQ_IR` | Frequency + IR (DFPT) |
| Lobster | `lobster` | `LOBSTER` | COHP chemical-bonding analysis |
| NMR CS | `nmr_cs` | `NMR_CS` | NMR chemical shift |
| NMR EFG | `nmr_efg` | `NMR_EFG` | NMR electric field gradient |
| NBO | `nbo` | `NBO` | Natural Bond Orbital analysis |
| MD NVT | `md_nvt` | `MD_NVT` | NVT molecular dynamics |
| MD NPT | `md_npt` | `MD_NPT` | NPT molecular dynamics |

---

## 5. Parameter Priority and Merge Rules

### INCAR merge (lowest → highest priority)

```
constants.py DEFAULT_INCAR_*            ← calc-type base template
     ↓ overridden by
constants.py INCAR_DELTA_STATIC_*       ← static sub-type increments
     ↓ overridden by
FUNCTIONAL_INCAR_PATCHES                ← functional-specific patch (BEEF/SCAN/…)
     ↓ overridden by
WorkflowConfig.user_incar_overrides     ← merged in WorkflowEngine._get_incar_params()
     ↓ forwarded as
VaspInputMaker.user_incar_settings      ← passed to *SetEcat constructor
     ↓ overridden by
write_*() per-call local_incar          ← highest priority, per-call override
```

### LDAU safety check

`VaspInputSetEcat.write_input()` enforces:
> If `LDAU=True` is in the INCAR but none of `LDAUL`/`LDAUU`/`LDAUJ` are
> provided, `LDAU` is silently forced to `False` and a warning is logged.

This prevents VASP from reading an incomplete DFT+U specification.

---

## 6. Extension Guide

### Adding a new calculation type

**Step 1** — `workflow_engine.py`: add a `CalcType` enum value
```python
class CalcType(Enum):
    NEW_TYPE = "new_type"
```

**Step 2** — `workflow_engine.py`: register in `CALC_TYPE_REGISTRY`
```python
CalcType.NEW_TYPE: CalcTypeConfig(
    incar_base=DEFAULT_INCAR_NEW,
    script_category=CalcCategory.STATIC,
),
```

**Step 3** — `constants.py`: add `DEFAULT_INCAR_NEW` dict

**Step 4** — `input_sets.py`: create `NewTypeSetEcat` (inherit from the most
appropriate `*SetEcat` parent)

**Step 5** — `maker.py`: add `write_new_type()` method following the shared
`_build_common_kwargs` + `_apply_magmom_compat` + `_ensure_dir` pattern

**Step 6** — `workflow_engine.py`: add a `case` arm in `WorkflowEngine.run()`
```python
case CalcType.NEW_TYPE:
    maker.write_new_type(output_dir, ...)
```

**Step 7** — `api.py`: add entry to `FRONTEND_CALC_TYPE_MAP` and handle any
new sub-params in `from_frontend_dict()` / `to_workflow_config()`

### Adding a new functional

Add an entry in `constants.py` → `FUNCTIONAL_INCAR_PATCHES`.  `_build_incar`
applies the matching patch automatically — no other changes required.

---

## 7. Usage Examples

### Example 1: Bulk relaxation (minimal call)

```python
from flow.api import FrontendAdapter
from flow.workflow_engine import WorkflowEngine

frontend_dict = {
    "calc_type": "bulk_relax",
    "xc": "PBE",
    "kpoints": {"density": 50.0},
    "structure": {
        "source": "file",
        "id": "/path/to/POSCAR",
    },
    "settings": {
        "ENCUT": 520,
        "EDIFF": 1e-5,
    },
}

params = FrontendAdapter.from_frontend_dict(frontend_dict)
params.output_dir = "/path/to/output"
WorkflowEngine().run(params.to_workflow_config())
# Writes: POSCAR  INCAR  KPOINTS  POTCAR
```

### Example 2: DFT+U bulk relaxation

```python
frontend_dict = {
    "calc_type": "bulk_relax",
    "xc": "PBE",
    "kpoints": {"density": 50.0},
    "structure": {"source": "file", "id": "/path/to/Fe3O4_POSCAR"},
    "settings": {
        "ISPIN":   2,
        "LMAXMIX": 4,
        "LDAU":    True,
        "LDAUTYPE": 2,
        "LDAUU": {"Fe": 4.0, "O": 0.0},
        "LDAUL": {"Fe": 2,   "O": -1},
        "LDAUJ": {"Fe": 0.0, "O": 0.0},
        "MAGMOM": {"Fe": 5.0, "O": 0.6},
    },
}

params = FrontendAdapter.from_frontend_dict(frontend_dict)
params.output_dir = "/path/to/output"
WorkflowEngine().run(params.to_workflow_config())
```

### Example 3: Lobster with multiple cohpGenerator entries

```python
frontend_dict = {
    "calc_type": "lobster",
    "xc": "PBE",
    "structure": {"source": "task", "id": "/prev/task/dir"},
    "prev_dir": "/prev/task/dir",
    "lobsterin": {
        "COHPstartEnergy": -20.0,
        "COHPendEnergy": 5.0,
        "cohpGenerator": "from 1.5 to 1.9 type Pt type C orbitalwise",
    },
    "lobsterin_custom_lines": [
        "cohpGenerator from 1.5 to 2.1 type Pt type O orbitalwise",
    ],
}

params = FrontendAdapter.from_frontend_dict(frontend_dict)
params.output_dir = "/path/to/lobster_output"
WorkflowEngine().run(params.to_workflow_config())
```

### Example 4: Using VaspInputMaker directly (low-level)

```python
from flow.maker import VaspInputMaker
from pymatgen.core import Structure

maker = VaspInputMaker(
    functional="SCAN",
    kpoints_density=60.0,
    user_incar_settings={"ENCUT": 600, "LASPH": True},
    user_potcar_functional="PBE_54",
)

structure = Structure.from_file("POSCAR")
maker.write_bulk(structure=structure, output_dir="/output/scan_relax")
```

### Example 5: Job script generation

```python
from flow.script import Script, CalcCategory

script = Script(
    calc_category=CalcCategory.RELAX,
    functional="BEEF",
    cores=32,
    walltime=24,
    queue="low",   # supported: low / debug / high / batch / ultrahigh
)
script.render_script(
    output_path="/path/to/submit.pbs",
    job_name="bulk_relax_Fe3O4",
    workdir="/path/to/workdir",
    vasp_cmd="mpirun vasp_std",
    extra_cmd="",
)
```
