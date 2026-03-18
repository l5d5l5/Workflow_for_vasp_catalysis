"""KPOINTS生成，基于xyz的大小"""

import math
from typing import Optional, Sequence, Union

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Kpoints


def build_kpoints_by_lengths(
    structure: Structure,
    length_densities: Union[float, Sequence[float]],
    gamma_centered: bool = True,
    style: Optional[int] = None,
    shift: tuple = (0, 0, 0),
    comment: Optional[str] = None,
) -> Kpoints:

    """
    根据晶格长度与目标密度自动生成 KPOINTS。
    
    :param structure: Pymatgen Structure 对象
    :param length_densities: 目标长度密度 (K点数 * 晶格长度)。可传入单个浮点数(各向同性)或包含3个浮点数的序列(各向异性)。
    :param gamma_centered: 是否使用 Gamma 中心网格 (默认 True)。若为 False 则使用 Monkhorst-Pack。
    :param shift: K点网格的平移，默认 (0, 0, 0)
    :param comment: KPOINTS 文件的首行注释
    :return: Pymatgen Kpoints 对象
    """
    if isinstance(length_densities, (int, float)):
        length_densities = (float(length_densities),) * 3
    elif len(length_densities) != 3:
        raise ValueError("length_densities 必须是单个浮点数或包含 3 个浮点数的序列")

    if style is not None:
        # backward compatibility: 1->Gamma, 2->Monkhorst
        gamma_centered = (style == 1)
    mode = "Gamma" if gamma_centered else "Monkhorst"
    if comment is None:
        comment = f"{mode} Kpoint by lengths"

    abc = structure.lattice.abc
    num_div = tuple(max(1, math.ceil(ld / abc[idx])) for idx, ld in enumerate(length_densities))

    return Kpoints(comment=comment, style=mode, kpts=[num_div], kpts_shift=shift)
