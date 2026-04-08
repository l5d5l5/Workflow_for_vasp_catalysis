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

class VaspIRParser:
    """
    VASP IR 红外光谱高性能解析器。

    全部使用 reverse_readfile 反向扫描策略：
      - NIONS        → regrep 正向扫描，头部找到即停
      - Born Charges → reverse_readfile 反向定位最后一个完整块
      - 特征向量     → reverse_readfile 反向定位 SQRT(mass) 后正向解析
    """

    def __init__(self, path: Union[str, Path]):
        self.outcar_path = resolve_outcar_path(path)
        self.filename    = str(self.outcar_path)
        self.n_ions:       int        = 0
        self.born_charges: np.ndarray = None
        self.frequencies:  np.ndarray = None
        self.eigenvectors: np.ndarray = None
        self.intensities:  np.ndarray = None

    def parse(self) -> list[dict]:
        self._extract_nions()
        self._extract_born_charges()
        self._extract_eigenvectors()
        self._calculate_intensities()
        return self.get_results()

    # ── Step 1: NIONS ─────────────────────────────────────────
    def _extract_nions(self):
        """regrep 正向扫描，头部找到即停，几乎零开销。"""
        matches = regrep(
            filename=self.filename,
            patterns={"nions": r"NIONS\s*=\s*(\d+)"},
            terminate_on_match=True,
            postprocess=int,
        )
        hits = matches.get("nions", [])
        if not hits:
            raise ValueError("OUTCAR 中未找到 NIONS，文件可能不完整。")
        self.n_ions = hits[0][0][0]

    # ── Step 2: Born Charges —— reverse_readfile 反向扫描 ─────
    def _extract_born_charges(self):
        """
        从文件尾部反向扫描，收集最后一个完整的 Born Charges 块。

        OUTCAR 中 Born Charges 块结构：
            BORN EFFECTIVE CHARGES (...)
            -------------------------
            ion   1
              1   Zxx  Zxy  Zxz
              2   Zyx  Zyy  Zyz
              3   Zzx  Zzy  Zzz
            ion   2
              ...
            -------------------------   ← 遇到此分割线表示块结束

        反向扫描遇到块尾分割线开始收集，遇到 "BORN EFFECTIVE CHARGES" 停止。
        收集完成后反转即为正向顺序，再逐行解析。
        """
        born_block  = []
        collecting  = False

        for line in reverse_readfile(self.filename):
            stripped = line.strip()

            # 遇到块头标记 → 停止收集（已拿到完整的最后一个块）
            if "BORN EFFECTIVE CHARGES" in stripped:
                if collecting:
                    break

            # 遇到块尾分割线 → 开始收集
            if not collecting and re.match(r"^-{20,}$", stripped):
                collecting = True
                continue

            if collecting:
                born_block.append(stripped)

        if not born_block:
            raise ValueError(
                "未找到 BORN EFFECTIVE CHARGES。\n"
                "请确认 INCAR 中设置了 LEPSILON=.TRUE. 或 LCALCEPS=.TRUE."
            )

        # 反转为正向顺序后解析
        born_block.reverse()
        self.born_charges = self._parse_born_block(born_block)

    def _parse_born_block(self, lines: list[str]) -> np.ndarray:
        """将 Born Charges 文本块解析为 (N_ions, 3, 3) 数组。"""
        born      = []
        current   = None
        row_pat   = re.compile(r"^\s*([1-3])\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)")
        ion_pat   = re.compile(r"^ion\s+(\d+)")

        for line in lines:
            if ion_pat.match(line):
                current = np.zeros((3, 3))
                born.append(current)
                continue
            if current is not None:
                m = row_pat.match(line)
                if m:
                    row_idx = int(m.group(1)) - 1
                    current[row_idx, :] = [
                        float(m.group(2)),
                        float(m.group(3)),
                        float(m.group(4))
                    ]

        if not born:
            raise ValueError("Born Charges 块解析失败，未找到有效 ion 数据。")

        return np.array(born)   # (N_ions, 3, 3)

    # ── Step 3: 特征向量 —— reverse_readfile 反向定位 ─────────
    def _extract_eigenvectors(self):
        """
        从文件尾部反向扫描，遇到 SQRT(mass) 即停，
        收集的尾部内容反转后正向解析所有振动模式。
        """
        tail_lines = []
        for line in reverse_readfile(self.filename):
            tail_lines.append(line)
            if "SQRT(mass)" in line:
                break
        else:
            raise ValueError(
                "未找到 'SQRT(mass)'，请确认 INCAR 中设置了 NWRITE=3。"
            )

        tail_lines.reverse()
        start_idx = next(
            i + 1 for i, l in enumerate(tail_lines) if "SQRT(mass)" in l
        )

        frequencies, eigenvectors = [], []
        i = start_idx

        while i < len(tail_lines):
            line = tail_lines[i].strip()
            if "cm-1" in line:
                is_imaginary = "f/i" in line
                freq_match   = re.search(r"([\d.]+)\s*cm-1", line)
                if freq_match:
                    freq = float(freq_match.group(1))
                    if is_imaginary:
                        freq = -freq
                    frequencies.append(freq)

                    mode_vec = np.zeros((self.n_ions, 3))
                    for ion in range(self.n_ions):
                        data_idx = i + 2 + ion
                        if data_idx < len(tail_lines):
                            parts = tail_lines[data_idx].split()
                            if len(parts) >= 6:
                                mode_vec[ion, :] = [
                                    float(parts[3]),
                                    float(parts[4]),
                                    float(parts[5])
                                ]
                    eigenvectors.append(mode_vec)
                    i += 1 + self.n_ions
            i += 1

        if not eigenvectors:
            raise ValueError("未能提取振动模式数据，文件可能被截断。")

        self.frequencies  = np.array(frequencies)
        self.eigenvectors = np.array(eigenvectors)

    # ── Step 4: 计算红外强度 ──────────────────────────────────
    def _calculate_intensities(self):
        dipole_deriv    = np.einsum('sab,vsb->va', self.born_charges, self.eigenvectors)
        raw_intensities = np.sum(dipole_deriv ** 2, axis=1)
        max_int         = np.max(raw_intensities)
        self.intensities = raw_intensities / max_int if max_int > 0 else raw_intensities

    def get_results(self) -> list[dict]:
        return [
            {
                "mode":         nu + 1,
                "frequency":    round(float(freq), 2),
                "intensity":    round(float(inten), 4),
                "is_imaginary": bool(freq < 0)
            }
            for nu, (freq, inten) in enumerate(
                zip(self.frequencies, self.intensities)
            )
        ]


# ─────────────────────────────────────────────────────────────
#  对外服务函数
# ─────────────────────────────────────────────────────────────

def get_ir_data(path: Union[str, Path]) -> dict:
    """解析入口，供路由层或脚本直接调用。"""
    parser      = VaspIRParser(path)
    raw_results = parser.parse()

    formatted_peaks = [
        {
            "mode":         res["mode"],
            "frequency":    round(abs(res["frequency"]), 1),
            "intensity":    round(res["intensity"], 3),
            "is_imaginary": res["is_imaginary"]
        }
        for res in raw_results
    ]
    active_peaks = [
        p for p in formatted_peaks
        if not p["is_imaginary"] and p["intensity"] > 0.01
    ]

    return {
        "total_modes":   len(formatted_peaks),
        "active_peaks":  len(active_peaks),
        "max_frequency": max((p["frequency"] for p in active_peaks), default=0.0),
        "peaks":         formatted_peaks,
        "source":        str(parser.outcar_path)
    }