from __future__ import annotations

from pathlib import Path
from typing import Union

from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS
from pymatgen.core import Structure


# ─────────────────────────────────────────────────────────────
#  辐射源解析（支持字符串名称 / 数值波长 / 模糊匹配）
# ─────────────────────────────────────────────────────────────

def resolve_radiation(radiation: str | float) -> tuple[str, float]:
    """
    将用户输入解析为 (label, wavelength_Å) 二元组。

    支持三种输入形式：
      1. 精确名称   "CuKa"   → 直接查表
      2. 模糊名称   "cuka"   → 大小写不敏感匹配
      3. 数值字符串 "1.54"   → 解析为波长（Å），label 显示为输入值
      4. float/int           → 直接作为波长（Å）

    Raises:
        ValueError: 无法识别的输入，并在错误信息中列出所有可用名称。
    """
    # ── 数值输入（float / int）────────────────────────────────
    if isinstance(radiation, (float, int)):
        wl = float(radiation)
        if wl <= 0:
            raise ValueError(f"波长必须为正数，收到: {wl}")
        return str(radiation), wl

    # ── 字符串输入 ────────────────────────────────────────────
    s = str(radiation).strip()

    # 尝试解析为数值（如 "1.54184"、"0.71"）
    try:
        wl = float(s)
        if wl <= 0:
            raise ValueError(f"波长必须为正数，收到: {wl}")
        return s, wl
    except ValueError:
        pass

    # 精确匹配
    if s in WAVELENGTHS:
        return s, WAVELENGTHS[s]

    # 大小写不敏感匹配
    s_lower = s.lower()
    for key in WAVELENGTHS:
        if key.lower() == s_lower:
            return key, WAVELENGTHS[key]

    # 前缀模糊匹配（如 "Cu" → 优先返回 "CuKa"）
    candidates = [key for key in WAVELENGTHS if key.lower().startswith(s_lower)]
    if len(candidates) == 1:
        return candidates[0], WAVELENGTHS[candidates[0]]
    if len(candidates) > 1:
        # 多个候选时返回最短的（最通用的，如 CuKa 优先于 CuKa1/CuKa2）
        best = min(candidates, key=len)
        return best, WAVELENGTHS[best]

    # 完全无法识别
    available = ", ".join(WAVELENGTHS.keys())
    raise ValueError(
        f"无法识别的辐射源: '{s}'。\n"
        f"可直接输入波长数值（单位 Å），或使用以下名称:\n{available}"
    )


# ─────────────────────────────────────────────────────────────
#  路径解析
# ─────────────────────────────────────────────────────────────

def resolve_structure_path(path: Union[str, Path]) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        for name in ("CONTCAR", "POSCAR"):
            candidate = p / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"目录 '{p}' 下未找到 POSCAR 或 CONTCAR，请确认 VASP 计算已完成。"
        )
    raise FileNotFoundError(f"路径不存在: '{p}'")


# ─────────────────────────────────────────────────────────────
#  核心解析函数
# ─────────────────────────────────────────────────────────────

def get_xrd_data(
    path: Union[str, Path],
    radiation: str | float = "CuKa",
    two_theta_min: float = 10.0,
    two_theta_max: float = 90.0,
    symprec: float = 0.1,
) -> dict:
    """
    XRD 图谱计算入口，供路由层调用。

    Args:
        path:          结构文件路径或包含 POSCAR/CONTCAR 的目录
        radiation:     辐射源名称（如 "CuKa"）或波长数值（如 1.54184）
        two_theta_min: 衍射角范围下限（度）
        two_theta_max: 衍射角范围上限（度）
        symprec:       对称性精度，0 表示不精化
    """
    # 解析辐射源
    radiation_label, wavelength = resolve_radiation(radiation)

    # 角度范围校验
    if two_theta_min >= two_theta_max:
        raise ValueError(
            f"角度范围无效: min={two_theta_min} >= max={two_theta_max}"
        )

    # 读取结构文件
    structure_path = resolve_structure_path(path)
    try:
        structure = Structure.from_file(str(structure_path))
    except Exception as e:
        raise ValueError(f"结构文件解析失败: {e}")

    # 调用 pymatgen XRD 计算器
    calculator = XRDCalculator(wavelength=wavelength, symprec=symprec)
    pattern = calculator.get_pattern(
        structure,
        scaled=True,
        two_theta_range=(two_theta_min, two_theta_max),
    )

    # 整理峰数据
    peaks = []
    for two_theta, intensity, hkl_list, d_hkl in zip(
        pattern.x, pattern.y, pattern.hkls, pattern.d_hkls
    ):
        primary_hkl = hkl_list[0]["hkl"]
        hkl_str     = " ".join(str(h) for h in primary_hkl)
        peaks.append({
            "theta":     round(float(two_theta), 3),
            "intensity": round(float(intensity), 3),
            "hkl":       hkl_str,
            "d_hkl":     round(float(d_hkl), 4),
            "hkl_families": [
                {
                    "hkl":          " ".join(str(h) for h in fam["hkl"]),
                    "multiplicity": fam["multiplicity"],
                }
                for fam in hkl_list
            ],
        })

    sorted_peaks = sorted(peaks, key=lambda p: p["intensity"], reverse=True)
    major_peaks  = [p for p in peaks if p["intensity"] > 5.0]
    strongest    = sorted_peaks[0] if sorted_peaks else None

    return {
        "radiation":       radiation_label,
        "wavelength":      round(wavelength, 6),
        "formula":         structure.composition.reduced_formula,
        "spacegroup":      structure.get_space_group_info()[0],
        "two_theta_range": [two_theta_min, two_theta_max],
        "total_peaks":     len(peaks),
        "major_peaks":     len(major_peaks),
        "max_peak": {
            "theta":     strongest["theta"]     if strongest else None,
            "hkl":       strongest["hkl"]       if strongest else None,
            "intensity": strongest["intensity"] if strongest else None,
        },
        "peaks":  peaks,
        "source": str(structure_path),
    }
