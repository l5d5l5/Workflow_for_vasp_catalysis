"""
KPOINTS generation utilities based on lattice vector lengths.
Provides a helper that derives the k-point mesh from target length densities
rather than requiring the caller to specify grid divisions explicitly.

基于晶格矢量长度的 KPOINTS 生成工具。
提供一个辅助函数，根据目标长度密度推导 k 点网格，
调用方无需显式指定网格分割数。
"""

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
    """Build a VASP KPOINTS object by dividing each lattice axis by a target length density.

    The number of k-points along each axis is computed as:
        n_i = max(1, ceil(length_density_i / a_i))
    where a_i is the length of the i-th lattice vector.

    根据目标长度密度为每个晶格轴生成 VASP KPOINTS 对象。

    每个轴的 k 点数按以下公式计算：
        n_i = max(1, ceil(length_density_i / a_i))
    其中 a_i 为第 i 个晶格矢量的长度。

    Args:
        structure: Pymatgen Structure object used to read lattice parameters.
                   用于读取晶格参数的 pymatgen Structure 对象。
        length_densities: Target product of (k-point count * lattice length).
                          Pass a single float for isotropic grids, or a
                          sequence of three floats for anisotropic grids.
                          目标（k 点数 * 晶格长度）乘积。
                          传入单个浮点数表示各向同性网格，传入含 3 个浮点数的序列
                          表示各向异性网格。
        gamma_centered: Use a Gamma-centered grid when True (default); use
                        Monkhorst-Pack when False.
                        为 True 时使用 Gamma 中心网格（默认）；
                        为 False 时使用 Monkhorst-Pack 网格。
        style: Legacy integer style selector (1 = Gamma, 2 = Monkhorst-Pack).
               When provided, overrides gamma_centered.
               旧式整数风格选择器（1 = Gamma，2 = Monkhorst-Pack）。
               若提供则覆盖 gamma_centered。
        shift: K-point grid shift vector, default (0, 0, 0).
               k 点网格平移矢量，默认 (0, 0, 0)。
        comment: Comment line written at the top of the KPOINTS file.
                 Defaults to "<mode> Kpoint by lengths".
                 写入 KPOINTS 文件首行的注释。
                 默认为 "<mode> Kpoint by lengths"。

    Returns:
        A pymatgen Kpoints object representing the generated mesh.
        代表所生成网格的 pymatgen Kpoints 对象。

    Raises:
        ValueError: If length_densities is a sequence whose length is not 3.
                    若 length_densities 为长度不等于 3 的序列时抛出。
    """
    # Normalise scalar input to a uniform 3-tuple.
    # 将标量输入规范化为统一的 3 元组。
    if isinstance(length_densities, (int, float)):
        length_densities = (float(length_densities),) * 3
    elif len(length_densities) != 3:
        raise ValueError("length_densities 必须是单个浮点数或包含 3 个浮点数的序列")

    # Handle legacy integer style parameter for backward compatibility.
    # 处理旧式整数 style 参数以保持向后兼容性。
    if style is not None:
        # backward compatibility: 1->Gamma, 2->Monkhorst
        # 向后兼容：1->Gamma，2->Monkhorst
        gamma_centered = (style == 1)

    # Determine mesh style label for comment construction.
    # 确定用于构建注释的网格风格标签。
    mode = "Gamma" if gamma_centered else "Monkhorst"
    if comment is None:
        comment = f"{mode} Kpoint by lengths"

    # Retrieve the three lattice vector lengths (a, b, c).
    # 获取三个晶格矢量长度（a, b, c）。
    abc = structure.lattice.abc

    # Compute grid divisions: ceil(density / length), minimum 1.
    # 计算网格分割数：ceil(密度 / 长度)，最小值为 1。
    num_div = tuple(max(1, math.ceil(ld / abc[idx])) for idx, ld in enumerate(length_densities))

    return Kpoints(comment=comment, style=mode, kpts=[num_div], kpts_shift=shift)
