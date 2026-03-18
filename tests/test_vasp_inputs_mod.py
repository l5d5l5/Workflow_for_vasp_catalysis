import os
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from vasp_inputs_mod import (
    build_kpoints_by_lengths,
    convert_vasp_format_to_pymatgen_dict,
    load_structure,
    ScriptRenderer,
)


def test_parse_vasp_compressed_list_and_convert():
    s = Lattice.cubic(3.0)
    struct = Structure(s, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # atom-level string should convert to species dict
    magmom = "2*3.0"
    out = convert_vasp_format_to_pymatgen_dict(struct, "MAGMOM", magmom)
    assert out == {"MAGMOM": {"Fe": 3.0, "O": 3.0}}

    # list input should also work
    out2 = convert_vasp_format_to_pymatgen_dict(struct, "MAGMOM", [2, 2])
    assert out2 == {"MAGMOM": {"Fe": 2.0, "O": 2.0}}


def test_load_structure_prefers_contcar(tmp_path: Path):
    lattice = Lattice.cubic(2.5)
    struct = Structure(lattice, ["Fe"], [[0, 0, 0]])

    # create both POSCAR and CONTCAR
    (tmp_path / "POSCAR").write_text(struct.to(fmt="POSCAR"))
    (tmp_path / "CONTCAR").write_text(struct.to(fmt="POSCAR"))

    loaded = load_structure(tmp_path)
    assert isinstance(loaded, Structure)
    assert len(loaded) == 1


def test_script_renderer_renders_and_copies_vdw(tmp_path: Path):
    # template with placeholder
    template = "#!/bin/bash\n# JOB: {{JOB_NAME}}\n# TYPE: {{TYPE1}}\n"
    tpl_file = tmp_path / "script.template"
    tpl_file.write_text(template, encoding="utf-8")

    # create dummy vdw_kernel
    vdw = tmp_path / "vdw_kernel.bindat"
    vdw.write_text("vdw", encoding="utf-8")

    out_dir = tmp_path / "run"
    renderer = ScriptRenderer(template_path=tpl_file, vdw_path=vdw)
    rendered_paths = renderer.render_script(
        folders=out_dir,
        functional="beef",
        output_filename="run_script.sh",
    )

    assert len(rendered_paths) == 1
    out_file = Path(rendered_paths[0])
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "# JOB: run" in content
    assert "# TYPE: beef" in content

    # should copy vdw_kernel.bindat
    assert (out_dir / "vdw_kernel.bindat").exists()


def test_build_kpoints_by_lengths_basic():
    lattice = Lattice.cubic(3.0)
    struct = Structure(lattice, ["Fe"], [[0, 0, 0]])
    kpoints = build_kpoints_by_lengths(struct, [10, 10, 10], style=1)
    # style is an Enum type in pymatgen
    assert str(kpoints.style).endswith("Gamma")


if __name__ == "__main__":
    pytest.main([__file__])
