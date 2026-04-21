# -*- coding: utf-8 -*-
"""
flow — VASP input-generation and workflow-orchestration package.
flow — VASP 输入文件生成与工作流编排顶层包。

Package layout / 包结构:
  flow/                     Core VASP I/O layer  /  核心 VASP 输入输出层
    api.py                  FrontendAdapter: dict → VaspWorkflowParams
    workflow_engine.py      WorkflowEngine: CalcType dispatch + INCAR merge
    maker.py                VaspInputMaker: write_bulk, write_slab, …
    input_sets.py           pymatgen InputSet subclasses
    constants.py            INCAR templates and shared constants
    kpoints.py              KPOINTS generation helpers
    script.py               PBS/SLURM script rendering
    utils.py                Structure loading utilities

  flow/workflow/            Orchestration layer  /  编排层
    config.py               params.yaml → WorkflowConfig dataclasses
    hook.py                 Manifest expansion + PBS job submission
    stages/                 Per-stage prepare() / check_success() classes
    structure/              pymatgen structure generation helpers
    extract.py              Standalone result-extraction CLI
    markers.py              done.ok / submitted.json helpers
    pbs.py                  DirLock, render_template, submit_job

Import note / 导入说明:
  flow.api and flow.workflow_engine are not re-exported here; every stage
  class imports them directly via lazy imports inside _write_vasp_inputs().
  flow.api 和 flow.workflow_engine 不在此处重导出；
  每个 stage 类通过 _write_vasp_inputs() 内部的延迟导入直接引用它们。
"""
