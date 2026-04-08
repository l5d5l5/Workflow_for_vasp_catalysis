from __future__ import annotations

import re
import numpy as np
from pathlib import Path
from typing import Union

from monty.io import reverse_readfile
from monty.re import regrep


# ─────────────────────────────────────────────────────────────
#  路径解析
# ─────────────────────────────────────────────────────────────

def resolve_outcar_path(path: Union[str, Path]) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        outcar = p / "OUTCAR"
        if outcar.is_file():
            return outcar
        raise FileNotFoundError(
            f"目录 '{p}' 下未找到 OUTCAR，请确认 VASP 计算已完成。"
        )
    raise FileNotFoundError(f"路径不存在: '{p}'")


# ─────────────────────────────────────────────────────────────
#  核心解析类
# ─────────────────────────────────────────────────────────────

class VaspNMRParser:
    """
    VASP NMR 高性能解析器。

    扫描策略：
      - NIONS / 计算类型检测  → regrep 正向扫描，terminate_on_match 精确控制
      - 化学屏蔽 (CS)         → reverse_readfile 反向定位最后一个完整块
      - 电场梯度 (EFG)        → reverse_readfile 反向定位最后一个完整块
      - 芯电子贡献 / G=0      → reverse_readfile 反向定位（避免误触发）
    """

    def __init__(self, path: Union[str, Path]):
        self.outcar_path = resolve_outcar_path(path)
        self.filename    = str(self.outcar_path)

        self.n_ions:  int  = 0
        self.has_cs:  bool = False
        self.has_efg: bool = False

        # CS 结果
        self.cs_valence_and_core: list[dict]       = []
        self.cs_valence_only:     list[dict]       = []
        self.cs_tensors:          list[np.ndarray] = []
        self.cs_core_contrib:     dict[str, float] = {}
        self.cs_g0_contrib:       list[list[float]] = []

        # EFG 结果
        self.efg_params:  list[dict]       = []
        self.efg_tensors: list[np.ndarray] = []

    # ─────────────────────────────────────────────────────────
    #  公开接口
    # ─────────────────────────────────────────────────────────

    def parse(self) -> dict:
        self._detect_flags()
        if not self.has_cs and not self.has_efg:
            raise ValueError(
                "OUTCAR 中未检测到 NMR 数据。\n"
                "化学屏蔽需设置 LCHIMAG = .TRUE.\n"
                "电场梯度需含四极核元素。"
            )
        if self.has_cs:
            self._parse_cs_table()
            self._parse_cs_tensors()
            self._parse_cs_core_contrib()
            self._parse_cs_g0_contrib()
        if self.has_efg:
            self._parse_efg_table()
            self._parse_efg_tensors()
        return self._build_result()

    # ─────────────────────────────────────────────────────────
    #  Step 0: 检测标志位 + NIONS
    #  修复 P5：NIONS 用 terminate_on_match，EFG 用数据区专属关键词
    # ─────────────────────────────────────────────────────────

    def _detect_flags(self):
        # ── NIONS：头部即停 ───────────────────────────────────
        nions_match = regrep(
            filename=self.filename,
            patterns={"nions": r"NIONS\s*=\s*(\d+)"},
            terminate_on_match=True,
            postprocess=int,
        )
        hits = nions_match.get("nions", [])
        if not hits:
            raise ValueError("OUTCAR 中未找到 NIONS，文件可能不完整。")
        self.n_ions = hits[0][0][0]

        # ── LCHIMAG：头部参数区扫描 ───────────────────────────
        cs_match = regrep(
            filename=self.filename,
            patterns={"nmr_cs": r"LCHIMAG\s*=\s*T"},
            terminate_on_match=True,
            postprocess=str,
        )
        self.has_cs = bool(cs_match.get("nmr_cs"))

        # ── EFG：用数据区专属表头行，避免参数说明区误触发 ────
        # "ion   Cq(MHz)   eta    Q (mb)" 只出现在数据输出区
        efg_match = regrep(
            filename=self.filename,
            patterns={"nmr_efg": r"ion\s+Cq\(MHz\)\s+eta\s+Q\s+\(mb\)"},
            terminate_on_match=True,
            postprocess=str,
        )
        self.has_efg = bool(efg_match.get("nmr_efg"))

    # ─────────────────────────────────────────────────────────
    #  Step 1-A: 化学屏蔽表格（Maryland 符号，含/不含 G=0）
    # ─────────────────────────────────────────────────────────

    def _parse_cs_table(self):
        """
        反向扫描，定位最后一个 CSA tensor 表格块。

        OUTCAR 中该块结构：
            CSA tensor (J. Mason ...)
            --------------------------------------------------
            EXCLUDING G=0 CONTRIBUTION    INCLUDING G=0 CONTRIBUTION
            ...
            ATOM  ISO_SHIFT  SPAN  SKEW   ISO_SHIFT  SPAN  SKEW
            --------------------------------------------------
            (absolute, valence only)
              1    σ_iso   Ω    κ    σ_iso   Ω    κ
              ...
            --------------------------------------------------
            (absolute, valence and core)
              1    σ_iso   Ω    κ    σ_iso   Ω    κ
              ...
            --------------------------------------------------

        策略：反向收集直到 "CSA tensor" 为止，反转后正向解析。
        """
        block: list[str] = []
        collecting = False

        for line in reverse_readfile(self.filename):
            stripped = line.strip()
            # 遇到块头标记 → 停止（已拿到完整块）
            if "CSA tensor" in stripped:
                break
            # 开始收集（从文件尾部第一个分割线开始）
            if not collecting and re.match(r"-{20,}", stripped):
                collecting = True
                continue
            if collecting:
                block.append(stripped)

        if not block:
            raise ValueError("未找到 CSA tensor 表格，请确认 LCHIMAG = .TRUE.")

        block.reverse()

        # 每行格式：行号 + 6 个浮点（valence_only 3列 + valence_and_core 3列）
        row_pat = re.compile(
            r"^\s*\d+\s+"
            r"([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+"
            r"([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)"
        )

        valence_only:     list[dict] = []
        valence_and_core: list[dict] = []
        in_vc_table = False

        for line in block:
            stripped = line.strip()
            if "valence and core" in stripped:
                in_vc_table = True
                continue
            m = row_pat.match(stripped)
            if m and in_vc_table:
                valence_only.append({
                    "iso_shift": round(float(m.group(1)), 4),
                    "span":      round(float(m.group(2)), 4),
                    "skew":      round(float(m.group(3)), 4),
                })
                valence_and_core.append({
                    "iso_shift": round(float(m.group(4)), 4),
                    "span":      round(float(m.group(5)), 4),
                    "skew":      round(float(m.group(6)), 4),
                })

        self.cs_valence_and_core = valence_and_core
        self.cs_valence_only     = valence_only

    # ─────────────────────────────────────────────────────────
    #  Step 1-B: 化学屏蔽原始非对称张量
    #  修复 P4：先追加再清空的顺序问题
    # ─────────────────────────────────────────────────────────

    def _parse_cs_tensors(self):
        """
        反向扫描，定位最后一个 UNSYMMETRIZED TENSORS 块。

        块结构：
            UNSYMMETRIZED TENSORS
            ion  1
              σ11  σ12  σ13
              σ21  σ22  σ23
              σ31  σ32  σ33
            ion  2
              ...
            SYMMETRIZED TENSORS    ← 块结束标记
        """
        block: list[str] = []
        collecting = False

        for line in reverse_readfile(self.filename):
            stripped = line.strip()
            if "UNSYMMETRIZED TENSORS" in stripped:
                collecting = True
                break
            if not collecting and "SYMMETRIZED TENSORS" in stripped:
                collecting = True
                continue
            if collecting:
                block.append(stripped)

        if not block:
            raise ValueError("未找到 UNSYMMETRIZED TENSORS 块。")

        block.reverse()

        row_pat = re.compile(
            r"^\s*([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s*$"
        )
        ion_pat = re.compile(r"^ion\s+\d+")

        tensors: list[np.ndarray] = []
        current_rows: list[list[float]] = []

        for line in block:
            stripped = line.strip()
            if ion_pat.match(stripped):
                # 修复 P4：遇到新 ion 时先保存上一个（若有），再重置
                if current_rows:
                    tensors.append(np.array(current_rows))
                current_rows = []
                continue
            m = row_pat.match(stripped)
            if m:
                current_rows.append([
                    float(m.group(1)),
                    float(m.group(2)),
                    float(m.group(3))
                ])

        # 收尾最后一个 ion
        if current_rows:
            tensors.append(np.array(current_rows))

        self.cs_tensors = tensors

    # ─────────────────────────────────────────────────────────
    #  Step 1-C: 芯电子贡献
    #  修复 P1：补上 self.cs_core_contrib 赋值
    # ─────────────────────────────────────────────────────────

    def _parse_cs_core_contrib(self):
        """
        反向扫描，定位 Core NMR properties 块。

        块结构：
            Core NMR properties
            typ  El  Core shift (ppm)
            --------------------
            1    C   -200.50
            2    O   -271.10
            --------------------
        """
        core_contrib: dict[str, float] = {}
        collecting = False

        for line in reverse_readfile(self.filename):
            stripped = line.strip()
            # 遇到块头 → 停止
            if "Core NMR properties" in stripped:
                break
            # 遇到数据行（格式：数字 元素 浮点数）
            m = re.match(r"^\s*\d+\s+([A-Z][a-z]?)\s+([-]?\d+\.\d+)", stripped)
            if m:
                collecting = True
                element = m.group(1)
                shift   = float(m.group(2))
                core_contrib[element] = round(shift, 4)

        # 修复 P1：确保赋值到实例属性
        self.cs_core_contrib = core_contrib

    # ─────────────────────────────────────────────────────────
    #  Step 1-D: G=0 贡献矩阵
    #  修复 P2：用精确标题行定位，不依赖通用分割线
    # ─────────────────────────────────────────────────────────

    def _parse_cs_g0_contrib(self):
        """
        反向扫描，定位最后一个 G=0 CONTRIBUTION 块。

        块结构：
            G=0 CONTRIBUTION TO CHEMICAL SHIFT (field along BDIR)
            --------------------------------------------------
            BDIR    X       Y       Z
            --------------------------------------------------
            1    x1    y1    z1
            2    x2    y2    z2
            3    x3    y3    z3
            --------------------------------------------------

        策略：反向收集数据行，遇到块头标题行停止。
        """
        block: list[str] = []
        collecting = False

        for line in reverse_readfile(self.filename):
            stripped = line.strip()
            # 遇到块头标题 → 停止
            if "G=0 CONTRIBUTION TO CHEMICAL SHIFT" in stripped:
                break
            # 跳过分割线和列标题行
            if re.match(r"-{20,}", stripped) or stripped.startswith("BDIR"):
                if collecting:
                    continue
                else:
                    collecting = True
                    continue
            if collecting:
                block.append(stripped)

        block.reverse()

        row_pat = re.compile(
            r"^\s*\d+\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)"
        )
        g0_matrix: list[list[float]] = []
        for line in block:
            m = row_pat.match(line.strip())
            if m:
                g0_matrix.append([
                    float(m.group(1)),
                    float(m.group(2)),
                    float(m.group(3))
                ])

        self.cs_g0_contrib = g0_matrix

    # ─────────────────────────────────────────────────────────
    #  Step 2-A: EFG 参数表（Cq, η, Q）
    #  修复 P3：用精确表头行定位，不依赖 dash_count
    # ─────────────────────────────────────────────────────────

    def _parse_efg_table(self):
        """
        反向扫描，定位最后一个 NMR quadrupolar parameters 数据块。

        块结构：
            --------------------------------------------------
            ion   Cq(MHz)   eta    Q (mb)
            --------------------------------------------------
              1    0.100    0.200   0.300
              ...
            --------------------------------------------------

        策略：反向收集数据行，遇到 "ion   Cq(MHz)" 表头行停止。
        """
        block: list[str] = []
        collecting = False

        for line in reverse_readfile(self.filename):
            stripped = line.strip()
            # 遇到表头行 → 停止（已收集完整数据区）
            if re.match(r"ion\s+Cq\(MHz\)\s+eta\s+Q\s+\(mb\)", stripped):
                break
            # 跳过分割线
            if re.match(r"-{20,}", stripped):
                collecting = True
                continue
            if collecting:
                block.append(stripped)

        if not block:
            raise ValueError("未找到 NMR quadrupolar parameters 数据块。")

        block.reverse()

        row_pat = re.compile(
            r"^\s*\d+\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)"
        )
        efg_params: list[dict] = []
        for line in block:
            m = row_pat.match(line.strip())
            if m:
                efg_params.append({
                    "cq_mhz":                    round(float(m.group(1)), 4),
                    "eta":                       round(float(m.group(2)), 4),
                    "nuclear_quadrupole_moment": round(float(m.group(3)), 4),
                })

        self.efg_params = efg_params

    # ─────────────────────────────────────────────────────────
    #  Step 2-B: EFG 原始张量
    #  修复 P3：用精确表头行定位
    # ─────────────────────────────────────────────────────────

    def _parse_efg_tensors(self):
        """
        反向扫描，定位最后一个 Electric field gradients 数据块。

        块结构：
            Electric field gradients (V/A^2)
            --------------------------------------------------
            ion   V_xx   V_yy   V_zz   V_xy   V_xz   V_yz
            --------------------------------------------------
              1   vxx    vyy    vzz    vxy    vxz    vyz
              ...
            --------------------------------------------------

        上三角 → 对称 3×3 张量。
        """
        block: list[str] = []
        collecting = False

        for line in reverse_readfile(self.filename):
            stripped = line.strip()
            # 遇到列标题行 → 停止
            if re.match(r"ion\s+V_xx\s+V_yy\s+V_zz", stripped):
                break
            # 跳过分割线
            if re.match(r"-{20,}", stripped):
                collecting = True
                continue
            if collecting:
                block.append(stripped)

        if not block:
            raise ValueError("未找到 Electric field gradients 数据块。")

        block.reverse()

        row_pat = re.compile(
            r"^\s*\d+\s+"
            r"([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+"
            r"([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)"
        )
        tensors: list[np.ndarray] = []
        for line in block:
            m = row_pat.match(line.strip())
            if m:
                vxx = float(m.group(1)); vyy = float(m.group(2)); vzz = float(m.group(3))
                vxy = float(m.group(4)); vxz = float(m.group(5)); vyz = float(m.group(6))
                tensors.append(np.array([
                    [vxx, vxy, vxz],
                    [vxy, vyy, vyz],
                    [vxz, vyz, vzz],
                ]))

        self.efg_tensors = tensors

    # ─────────────────────────────────────────────────────────
    #  张量分析（纯 numpy）
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _analyze_cs_tensor(tensor: np.ndarray) -> dict:
        sym     = (tensor + tensor.T) / 2
        eigvals = np.sort(np.linalg.eigvalsh(sym))
        sigma_iso             = float(eigvals.sum() / 3.0)
        sigma_11, sigma_22, sigma_33 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])

        sorted_by_dev = sorted(eigvals, key=lambda x: abs(x - sigma_iso))
        s_yy, s_xx, s_zz = float(sorted_by_dev[0]), float(sorted_by_dev[1]), float(sorted_by_dev[2])
        zeta  = s_zz - sigma_iso
        eta_h = (s_yy - s_xx) / zeta if abs(zeta) > 1e-10 else 0.0
        omega = sigma_33 - sigma_11
        kappa = (3.0 * (sigma_22 - sigma_iso) / omega) if abs(omega) > 1e-10 else 0.0

        return {
            "haeberlen": {
                "sigma_iso":       round(sigma_iso,                        4),
                "delta_sigma_iso": round(s_zz - 0.5 * (s_xx + s_yy),      4),
                "zeta":            round(zeta,                             4),
                "eta":             round(float(eta_h),                     4),
            },
            "mehring": {
                "sigma_iso": round(sigma_iso,  4),
                "sigma_11":  round(sigma_11,   4),
                "sigma_22":  round(sigma_22,   4),
                "sigma_33":  round(sigma_33,   4),
            },
            "maryland": {
                "sigma_iso": round(sigma_iso,       4),
                "omega":     round(float(omega),    4),
                "kappa":     round(float(kappa),    4),
            },
        }

    @staticmethod
    def _analyze_efg_tensor(tensor: np.ndarray) -> dict:
        eigvals     = np.linalg.eigvalsh(tensor)
        sorted_abs  = sorted(eigvals, key=lambda x: abs(x))
        v_xx, v_yy, v_zz = float(sorted_abs[0]), float(sorted_abs[1]), float(sorted_abs[2])
        eta = abs(v_yy - v_xx) / abs(v_zz) if abs(v_zz) > 1e-10 else 0.0
        return {
            "V_xx":      round(v_xx,        6),
            "V_yy":      round(v_yy,        6),
            "V_zz":      round(v_zz,        6),
            "asymmetry": round(float(eta),  6),
        }

    # ─────────────────────────────────────────────────────────
    #  构建最终结果
    #  修复 P6/P7：张量缺失时提供安全默认值，避免 Pydantic 422
    # ─────────────────────────────────────────────────────────

    _DEFAULT_HAEBERLEN = {"sigma_iso": 0.0, "delta_sigma_iso": 0.0, "zeta": 0.0, "eta": 0.0}
    _DEFAULT_MEHRING   = {"sigma_iso": 0.0, "sigma_11": 0.0, "sigma_22": 0.0, "sigma_33": 0.0}
    _DEFAULT_MARYLAND  = {"sigma_iso": 0.0, "omega": 0.0, "kappa": 0.0}
    _DEFAULT_PAS       = {"V_xx": 0.0, "V_yy": 0.0, "V_zz": 0.0, "asymmetry": 0.0}

    def _build_result(self) -> dict:
        result: dict = {
            "source":             str(self.outcar_path),
            "n_ions":             self.n_ions,
            "has_cs":             self.has_cs,
            "has_efg":            self.has_efg,
            "chemical_shielding": None,
            "efg":                None,
        }

        # ── 化学屏蔽 ──────────────────────────────────────────
        if self.has_cs:
            cs_ions = []
            for idx in range(len(self.cs_valence_and_core)):
                if idx < len(self.cs_tensors):
                    tensor       = self.cs_tensors[idx]
                    tensor_entry = self._analyze_cs_tensor(tensor)
                    raw_tensor   = [[round(float(v), 6) for v in row] for row in tensor.tolist()]
                else:
                    # 修复 P6：张量缺失时填充安全默认值
                    tensor_entry = {
                        "haeberlen": self._DEFAULT_HAEBERLEN.copy(),
                        "mehring":   self._DEFAULT_MEHRING.copy(),
                        "maryland":  self._DEFAULT_MARYLAND.copy(),
                    }
                    raw_tensor = [[0.0, 0.0, 0.0]] * 3

                cs_ions.append({
                    "ion":              idx + 1,
                    "valence_and_core": self.cs_valence_and_core[idx],
                    "valence_only":     self.cs_valence_only[idx] if idx < len(self.cs_valence_only) else
                                        {"iso_shift": 0.0, "span": 0.0, "skew": 0.0},
                    "haeberlen":        tensor_entry["haeberlen"],
                    "mehring":          tensor_entry["mehring"],
                    "maryland":         tensor_entry["maryland"],
                    "raw_tensor":       raw_tensor,
                })

            result["chemical_shielding"] = {
                "n_ions":            len(cs_ions),
                "core_contribution": self.cs_core_contrib,
                "g0_contribution":   self.cs_g0_contrib if self.cs_g0_contrib else [],
                "ions":              cs_ions,
            }

        # ── 电场梯度 ──────────────────────────────────────────
        if self.has_efg:
            efg_ions = []
            for idx, params in enumerate(self.efg_params):
                if idx < len(self.efg_tensors):
                    tensor     = self.efg_tensors[idx]
                    pas        = self._analyze_efg_tensor(tensor)
                    raw_tensor = [[round(float(v), 6) for v in row] for row in tensor.tolist()]
                else:
                    # 修复 P7：张量缺失时填充安全默认值
                    pas        = self._DEFAULT_PAS.copy()
                    raw_tensor = [[0.0, 0.0, 0.0]] * 3

                efg_ions.append({
                    "ion":                       idx + 1,
                    "cq_mhz":                    params["cq_mhz"],
                    "eta":                       params["eta"],
                    "nuclear_quadrupole_moment": params["nuclear_quadrupole_moment"],
                    "principal_axis":            pas,
                    "raw_tensor":                raw_tensor,
                })

            result["efg"] = {
                "n_ions": len(efg_ions),
                "ions":   efg_ions,
            }

        return result


# ─────────────────────────────────────────────────────────────
#  对外服务函数
# ─────────────────────────────────────────────────────────────

def get_nmr_data(path: Union[str, Path]) -> dict:
    """NMR 解析入口，供路由层或脚本直接调用。"""
    parser = VaspNMRParser(path)
    return parser.parse()
