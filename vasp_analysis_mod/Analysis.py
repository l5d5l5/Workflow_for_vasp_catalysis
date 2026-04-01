# -*- coding: utf-8 -*-

# ============================================================
# 1. models.py 读取信息，不同的文件和后端调用方法
# ============================================================
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

@dataclass
class ApiResponse:
    """
    统一API返回格式。
    Java端通过stdout捕获JSON， 通过stderr捕获日志信息。
    """
    success: bool
    code: int
    message: str
    data: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)
    
    @classmethod
    def ok(cls, data: Dict[str, Any], message: str = "Success") -> "ApiResponse":
        return cls(success=True, code=200, message=message, data=data)
    
    @classmethod
    def error(cls, message:str, code:int=500, data:Optional[dict] = None) -> "ApiResponse":
        return cls(success=False, code=code, message=message, data=data or {})
    
    @classmethod
    def not_found(cls, message:str) -> "ApiResponse":
        return cls(success=False, code=404, message=message, data={})
    
    @classmethod
    def bad_request(cls, message:str) -> "ApiResponse":
        return cls(success=False, code=400, message=message, data={})

@dataclass
class RelaxResult:
    """结构优化任务的分析结果"""
    converged: bool
    final_energy_eV: Optional[float]
    fermi_level_eV: Optional[float]
    total_electrons: Optional[float]
    ionic_steps: int
    electronic_steps_per_ionic: List[int]
    total_magnetization: Optional[float]
    last_dE_eV: Optional[float]
    site_magnetization: List[Dict[str, Any]]
    warnings: List[str]

# ============================================================
# 2. base.py
# ============================================================
import logging
import sys
import os
from pathlib import Path
import numpy as np
from pymatgen.io.vasp import Vasprun, Outcar, Oszicar

class VaspAnalysisBase:
    """
    VASP 后处理基类。

    日志策略（Java 友好）：
      - 所有日志 → stderr（Java 通过 Process.getErrorStream() 捕获）
      - 分析结果 → stdout（Java 通过 Process.getInputStream() 捕获）
      - 零文件 IO（除非 save_data=True 时写 CSV）
      - 日志级别可通过环境变量 VASP_LOG_LEVEL 动态控制
    """

    def __init__(
        self,
        work_dir: Union[str, Path],
        save_data: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        log_level: int = logging.WARNING,
    ):
        self.work_dir   = Path(work_dir).resolve()
        self.save_data  = save_data
        self.output_dir = Path(output_dir).resolve() if output_dir else self.work_dir

        if not self.work_dir.exists():
            raise FileNotFoundError(f"work_dir does not exist: {self.work_dir}")

        if self.save_data:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._vasprun: Optional[Vasprun] = None
        self._outcar:  Optional[Outcar]  = None
        self._oszicar: Optional[Oszicar] = None
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, default_level: int) -> logging.Logger:
        env_level = os.environ.get("VASP_LOG_LEVEL", "").upper()
        if env_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            default_level = getattr(logging, env_level)

        name   = f"vasp.{self.work_dir.name}"
        logger = logging.getLogger(name)
        logger.setLevel(default_level)
        logger.propagate = False

        if logger.hasHandlers():
            logger.handlers.clear()

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
        logger.addHandler(handler)
        return logger

    def _log(self, msg: str, level: str = "info"):
        getattr(self.logger, level, self.logger.info)(msg)

    def _get_oszicar(self) -> Optional[Oszicar]:
        if self._oszicar is None and (self.work_dir / "OSZICAR").exists():
            try:
                self._oszicar = Oszicar(str(self.work_dir / "OSZICAR"))
            except Exception as e:
                self._log(f"OSZICAR parse failed: {e}", "warning")
        return self._oszicar

    def _get_outcar(self) -> Optional[Outcar]:
        if self._outcar is None and (self.work_dir / "OUTCAR").exists():
            try:
                self._outcar = Outcar(str(self.work_dir / "OUTCAR"))
            except Exception as e:
                self._log(f"OUTCAR parse failed: {e}", "warning")
        return self._outcar

    def _get_vasprun(self, strict: bool = False) -> Optional[Vasprun]:
        if self._vasprun is None:
            path = self.work_dir / "vasprun.xml"
            if path.exists():
                try:
                    self._vasprun = Vasprun(str(path), parse_potcar_file=False)
                except Exception as e:
                    if strict: raise ValueError(f"vasprun.xml parse failed: {e}")
                    self._log(f"vasprun.xml parse failed: {e}", "warning")
            elif strict:
                raise FileNotFoundError(f"vasprun.xml not found in {self.work_dir}")
        return self._vasprun

    def save_to_csv(self, df, filename: str) -> Optional[Path]:
        """保存 DataFrame 为 CSV，仅在 save_data=True 时生效"""
        if not self.save_data:
            return None
        import pandas as pd
        path = self.output_dir / filename
        try:
            df.to_csv(path, index=False, float_format="%.6f")
            self._log(f"CSV saved → {path.name}", "debug")
            return path
        except OSError as e:
            self._log(f"Cannot save {filename}: {e}", "warning")
            return None

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.electronic_structure.core import Spin, Orbital, OrbitalType
class StructureAnalysis(VaspAnalysisBase):
    """
    用于前端第一步：快速获取 VASP 结构信息（原子总数、包含的元素种类）。
    优先级：CONTCAR > POSCAR
    """
    def get_info(self) -> ApiResponse:
        try:
            struct_path = None
            for fname in ["CONTCAR", "POSCAR"]:
                p = self.work_dir / fname
                if p.exists() and p.stat().st_size > 0:
                    struct_path = p
                    break
            
            if not struct_path:
                return ApiResponse.not_found("No valid CONTCAR or POSCAR found in work_dir.")

            struct = Structure.from_file(str(struct_path))
            # 提取元素种类（保持原有顺序并去重）
            elements = list(dict.fromkeys([specie.symbol for specie in struct.species]))

            return ApiResponse.ok(
                message="Structure loaded successfully",
                data={
                    "totalAtoms": len(struct),
                    "elements": elements
                }
            )
        except Exception as e:
            self._log(f"Structure parse failed: {e}", "error")
            return ApiResponse.error(f"Failed to parse structure: {e}")

# ============================================================
# 3. relax.py
# ============================================================
import pandas as pd

class RelaxAnalysis(VaspAnalysisBase):
    """
    结构优化任务分析器。

    数据读取优先级（轻量 → 重量）：
      能量   : OSZICAR > OUTCAR > vasprun.xml
      费米能 : OUTCAR  > vasprun.xml
      磁矩   : OUTCAR
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fast_outcar_data: Optional[Dict[str, Any]] = None
        
    def analyze(self, get_site_mag: bool = False) -> ApiResponse:
        """
        Parameters
        ----------
        get_site_mag : bool, default False
            是否解析每个原子的分波磁矩。正常情况下不需要，开启会增加少量解析时间。
        """
        try:
            # 1. 极速解析 OUTCAR，传入控制开关
            outcar_data = self._parse_outcar_fast(get_site_mag=get_site_mag)
            
            # 2. 解析 OSZICAR (获取能量和步数)
            osz = self._get_oszicar()
            ionic_steps = osz.ionic_steps if osz else []
            elec_steps = [len(step) for step in osz.electronic_steps] if osz else []
            
            final_energy = float(ionic_steps[-1].get("E0", np.nan)) if ionic_steps else None
            last_dE = float(ionic_steps[-1].get("dE", np.nan)) if ionic_steps else None

            result = RelaxResult(
                converged=outcar_data["converged"],
                final_energy_eV=final_energy,
                fermi_level_eV=outcar_data["efermi"],
                total_electrons=outcar_data["nelect"],
                ionic_steps=len(ionic_steps),
                electronic_steps_per_ionic=elec_steps,
                total_magnetization=outcar_data["total_mag"],
                last_dE_eV=last_dE,
                site_magnetization=outcar_data["site_mag"],
                warnings=self._collect_warnings(outcar_data["converged"], last_dE),
            )
            return ApiResponse.ok(
                data=asdict(result),
                message=f"Relax analysis complete: {'converged' if outcar_data['converged'] else 'NOT converged'}"
            )
        except Exception as e:
            self._log(f"RelaxAnalysis failed: {e}", "error")
            return ApiResponse.error(str(e))


    def _parse_outcar_fast(self, get_site_mag: bool) -> Dict[str, Any]:
        import re
        """核心加速逻辑：单遍遍历 OUTCAR"""
        if self._fast_outcar_data is not None:
            return self._fast_outcar_data

        data = {
            "converged": False,
            "efermi": None,
            "nelect": None,
            "total_mag": None,
            "site_mag": []  # 默认返回空列表
        }

        outcar_path = self.work_dir / "OUTCAR"
        if not outcar_path.exists():
            self._log("OUTCAR not found, some properties will be missing.", "warning")
            self._fast_outcar_data = data
            return data

        in_mag_block = False
        mag_lines = []
        mag_block_pattern = re.compile(r"^\s*magnetization \(x\)")
        mag_line_pattern = re.compile(r"^\s*\d+\s+[-.\d]+\s+[-.\d]+")
        tot_line_pattern = re.compile(r"^\s*tot\s+[-.\d]+")

        with open(outcar_path, "r", errors="ignore") as f:
            for line in f:
                # 1. 收敛标志
                if "reached required accuracy" in line:
                    data["converged"] = True
                # 2. 费米能级
                elif "E-fermi" in line:
                    try: data["efermi"] = float(line.split()[2])
                    except: pass
                # 3. 总电子数
                elif "NELECT" in line and data["nelect"] is None:
                    try: data["nelect"] = float(line.split()[2])
                    except: pass
                # 4. 总磁矩
                elif "number of electron " in line and "magnetization" in line:
                    try: data["total_mag"] = float(line.split()[-1])
                    except: pass
                
                # 5. 分波磁矩块 (仅在 get_site_mag=True 时才进行字符串匹配和收集)
                elif get_site_mag and mag_block_pattern.match(line):
                    in_mag_block = True
                    mag_lines = []  # 遇到新的块，清空旧数据
                elif in_mag_block:
                    if tot_line_pattern.match(line):
                        in_mag_block = False  # 遇到 tot 行，完美结束
                    elif mag_line_pattern.match(line):
                        mag_lines.append(line.strip()) # 纯数据行，收集

        # 处理最后一次捕获到的分波磁矩数据
        if get_site_mag and mag_lines:
            site_mag = []
            for ln in mag_lines:
                parts = ln.split()
                # 兼容 s, p, d (长度 5) 和 s, p, d, f (长度 6)
                if len(parts) >= 5:
                    try:
                        mag_dict = {
                            "atom_index": int(parts[0]) - 1,
                            "s": float(parts[1]),
                            "p": float(parts[2]),
                            "d": float(parts[3]),
                        }
                        if len(parts) >= 6:
                            mag_dict["f"] = float(parts[4])
                            mag_dict["tot"] = float(parts[5])
                        else:
                            mag_dict["tot"] = float(parts[4])
                        site_mag.append(mag_dict)
                    except ValueError:
                        pass
            data["site_mag"] = site_mag

        self._fast_outcar_data = data
        return data
    
    def get_convergence_data(self) -> Optional[pd.DataFrame]:
        """
        获取结构优化的收敛过程数据，用于前端绘图。
        返回包含 step, energy, dE, dE_abs, electronic_steps 的 DataFrame。
        """
        import pandas as pd
        import numpy as np
        
        osz = self._get_oszicar()
        if not osz or not osz.ionic_steps:
            self._log("OSZICAR is empty or invalid.", "warning")
            return None

        ionic_steps = osz.ionic_steps
        
        # 提取数据
        energies = [step.get("E0", np.nan) for step in ionic_steps]
        dE_list  = [step.get("dE", np.nan) for step in ionic_steps]
        elec_steps = [len(s) for s in osz.electronic_steps]
        steps_idx  = list(range(1, len(ionic_steps) + 1))
        
        # 计算 dE 的绝对值，方便前端画对数图
        dE_abs = [abs(x) if x is not None else np.nan for x in dE_list]

        # 组装成 DataFrame
        df = pd.DataFrame({
            "step": steps_idx,
            "energy_eV": energies,
            "dE_eV": dE_list,
            "dE_abs_eV": dE_abs,
            "electronic_steps": elec_steps
        })
        
        return df


    def _collect_warnings(self, converged: bool, last_dE: Optional[float]) -> List[str]:
        warns = []
        if not converged:
            warns.append("Calculation did NOT converge.")
        if last_dE is not None and abs(last_dE) > 1e-3 and converged:
            warns.append(f"Last dE = {last_dE:.2e} eV > 1e-3 eV, convergence marginal.")
        return warns
    
# ============================================================
# 4. dos.py —— 从 DOSCAR 直接读取（不依赖 vasprun.xml）
# ============================================================

class DoscarParser:
    """
    DOSCAR 文件原生解析器。

    DOSCAR 文件结构：
      第 1 行  : NIONS  NKPTS  ISPIN  ...
      第 2-5 行: 注释/系统信息
      第 6 行  : Emax  Emin  NEDOS  Efermi  weight
      第 7 ~ 7+NEDOS 行: TDOS 数据
        - 非自旋: energy  dos  integrated_dos
        - 自旋  : energy  dos_up  dos_down  idos_up  idos_down
      之后每个原子重复:
        第 1 行 : Emax  Emin  NEDOS  Efermi  weight
        后 NEDOS 行: PDOS 数据（列数取决于 LORBIT）
    """
    # LORBIT = 10 (仅 s, p, d)
    PDOS_COLS_L_NONCOL   = ["s", "p", "d"]
    PDOS_COLS_L_COL      = ["s_up", "s_down", "p_up", "p_down", "d_up", "d_down"]

    # LORBIT = 11 (s, p, d 细分)
    PDOS_COLS_SPD_NONCOL = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2"]
    PDOS_COLS_SPD_COL    = [f"{o}_{s}" for o in PDOS_COLS_SPD_NONCOL for s in ("up", "down")]
    
    # LORBIT = 11 (包含 f 轨道)
    PDOS_COLS_F_NONCOL   = PDOS_COLS_SPD_NONCOL + ["fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2y2", "fx3y2"]
    PDOS_COLS_F_COL      = [f"{o}_{s}" for o in PDOS_COLS_F_NONCOL for s in ("up", "down")]    


    def __init__(self, doscar_path: Union[str, Path]):
        self.path = Path(doscar_path)
        if not self.path.exists():
            raise FileNotFoundError(f"DOSCAR not found: {self.path}")

        # 解析结果（懒加载）
        self._nions:  Optional[int]   = None
        self._nedos:  Optional[int]   = None
        self._efermi: Optional[float] = None
        self._ispin:  Optional[int]   = None
        self._has_pdos: bool = False
        self._tdos:   Optional[np.ndarray] = None   # shape: (NEDOS, ncols)
        self._pdos:   Optional[List[np.ndarray]] = None  # list of (NEDOS, ncols)
        self._energies: Optional[np.ndarray] = None

        self._parse_header()

    def _parse_header(self):
        """解析 DOSCAR 前 6 行，获取关键参数"""
        with open(self.path, "r") as f:
            lines = [f.readline() for _ in range(6)]
            
        if len(lines) < 6 or not lines[5].strip():
            raise ValueError("DOSCAR is empty or severely truncated at header.")

        # 第 1 行: NIONS NKPTS ISPIN ...
        header_parts = lines[0].split()
        self._nions = int(header_parts[0])
        self._has_pdos = (len(header_parts) >= 3 and int(header_parts[2]) == 1)

        # 第 6 行: Emax Emin NEDOS Efermi weight
        dos_header = lines[5].split()
        self._nedos  = int(dos_header[2])
        self._efermi = float(dos_header[3])

    def _parse_all(self):
        """核心解析逻辑，包含完整性校验"""
        try:
            # 1. 读取 TDOS
            tdos_df = pd.read_csv(
                self.path, skiprows=6, nrows=self._nedos,
                sep=r'\s+', header=None, engine='c'
            )
            if len(tdos_df) < self._nedos:
                raise ValueError("DOSCAR is truncated in TDOS section.")
                
            self._tdos = tdos_df.values
            self._energies = self._tdos[:, 0]
            self._ispin = 2 if self._tdos.shape[1] == 5 else 1

            # 2. 读取 PDOS
            self._pdos = []
            if self._has_pdos:
                skip_lines = 6 + self._nedos
                total_pdos_lines = self._nions * (self._nedos + 1)
                with open(self.path, "r") as f:
                    # 跳过文件头、TDOS块，以及第一个PDOS块的表头行
                    for _ in range(skip_lines + 1): 
                        f.readline()
                    # 读取第一行真实的 PDOS 数据来获取列数 (比如 19 列)
                    first_data_line = f.readline()
                    pdos_ncols = len(first_data_line.split())
                
                # 使用 names=range(pdos_ncols) 强制指定列数
                pdos_raw_df = pd.read_csv(
                    self.path, skiprows=skip_lines, nrows=total_pdos_lines,
                    delim_whitespace=True, header=None, engine='c',
                    names=range(pdos_ncols)  # <--- 关键修复在这里
                )
                
                # 健壮性检查：文件是否不完整
                if len(pdos_raw_df) < total_pdos_lines:
                    raise ValueError(f"DOSCAR is truncated in PDOS section. Expected {total_pdos_lines} lines, got {len(pdos_raw_df)}.")
                    
                raw_values = pdos_raw_df.values
                block_size = self._nedos + 1
                for i in range(self._nions):
                    start_idx = i * block_size + 1
                    end_idx = start_idx + self._nedos
                    # 切片时完美跳过了包含 NaN 的表头行 (start_idx 已经 +1)
                    self._pdos.append(raw_values[start_idx:end_idx, :])
                    
        except Exception as e:
            raise ValueError(f"Failed to parse DOSCAR data: {e}")

    def _parse_all_fallback(self):
        """安全回退模式：使用原生的 readlines()"""
        with open(self.path, "r") as f:
            all_lines = f.readlines()
        cursor = 6 + self._nedos
        self._pdos = []
        for _ in range(self._nions):
            cursor += 1
            pdos_lines = all_lines[cursor : cursor + self._nedos]
            self._pdos.append(np.array([[float(x) for x in ln.split()] for ln in pdos_lines]))
            cursor += self._nedos    

    @property
    def nions(self) -> int: return self._nions
    @property
    def nedos(self) -> int: return self._nedos
    @property
    def efermi(self) -> float:
        if self._tdos is None: self._parse_all() 
        return self._efermi
    @property
    def ispin(self) -> int:
        if self._tdos is None: self._parse_all()
        return self._ispin
    @property
    def energies(self) -> np.ndarray:
        if self._tdos is None: self._parse_all()
        return self._energies
    @property
    def tdos(self) -> np.ndarray:
        if self._tdos is None: self._parse_all()
        return self._tdos
    @property
    def pdos(self) -> List[np.ndarray]:
        if self._pdos is None: self._parse_all()
        return self._pdos

    def get_pdos_col_names(self) -> List[str]:
        if not self.pdos: return []
        ncols = self.pdos[0].shape[1] - 1
        if self.ispin == 2:
            if ncols == 6: return self.PDOS_COLS_L_COL
            elif ncols == 18: return self.PDOS_COLS_SPD_COL
            elif ncols == 32: return self.PDOS_COLS_F_COL
        else:
            if ncols == 3: return self.PDOS_COLS_L_NONCOL
            elif ncols == 9: return self.PDOS_COLS_SPD_NONCOL
            elif ncols == 16: return self.PDOS_COLS_F_NONCOL
        return [f"col_{i}" for i in range(ncols)]

class DosAnalysis(VaspAnalysisBase):
    """
    精简且健壮的 DOS 分析器 (专为 Web API 打造)
    """
    # 轨道分组映射
    SPD_GROUP = {
        "s": ["s"], "p": ["py", "pz", "px"], "d": ["dxy", "dyz", "dz2", "dxz", "dx2-y2"],
        "f": ["fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2y2", "fx3y2"]
    }
    
    # 严格的前端到后端轨道名称映射字典
    FRONTEND_TO_VASP_ORB = {
        "s": "s", "p_y": "py", "p_z": "pz", "p_x": "px",
        "d_xy": "dxy", "d_yz": "dyz", "d_z2": "dz2", "d_xz": "dxz", "d_x2-y2": "dx2-y2",
        "f_{y(3x^2-y^2)}": "fy3x2", "f_xyz": "fxyz", "f_{yz^2}": "fyz2", 
        "f_{z^3}": "fz3", "f_{xz^2}": "fxz2", "f_{z(x^2-y^2)}": "fzx2y2", "f_{x(x^2-3y^2)}": "fx3y2"
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._doscar_parser: Optional[DoscarParser] = None
        self._structure = None
        self._site_elements: List[str] = []

    @property
    def parser(self) -> DoscarParser:
        if self._doscar_parser is None:
            self._doscar_parser = DoscarParser(self.work_dir / "DOSCAR")
        return self._doscar_parser

    @property
    def site_elements(self) -> List[str]:
        if not self._site_elements:
            from pymatgen.io.vasp import Poscar
            for fname in ("CONTCAR", "POSCAR"):
                path = self.work_dir / fname
                if path.exists():
                    try:
                        struct = Poscar.from_file(str(path)).structure
                        # 健壮性校验：结构原子数必须等于 DOSCAR 原子数
                        if len(struct) != self.parser.nions:
                            raise ValueError(f"{fname} has {len(struct)} atoms, but DOSCAR has {self.parser.nions}. Files mismatch!")
                        self._site_elements = [str(site.specie.symbol) for site in struct]
                        break
                    except Exception as e:
                        self._log(f"{fname} parse failed: {e}", "warning")
            
            if not self._site_elements:
                raise FileNotFoundError(f"No valid matching CONTCAR/POSCAR in {self.work_dir}")
        return self._site_elements

    def _aggregate_pdos(self, site_indices: List[int], sub_orbitals: List[str]) -> Dict[str, np.ndarray]:
        """高性能底层聚合引擎"""
        if not self.parser.pdos: return {}
        cols = self.parser.get_pdos_col_names()
        nedos = self.parser.nedos
        
        result = {"up": np.zeros(nedos)}
        if self.parser.ispin == 2: result["down"] = np.zeros(nedos)

        has_data = False
        for site_idx in site_indices:
            pdos_data = self.parser.pdos[site_idx]
            for sub_orb in sub_orbitals:
                if self.parser.ispin == 2:
                    col_up, col_dn = f"{sub_orb}_up", f"{sub_orb}_down"
                    if col_up in cols and col_dn in cols:
                        result["up"] += pdos_data[:, cols.index(col_up) + 1]
                        result["down"] += pdos_data[:, cols.index(col_dn) + 1]
                        has_data = True
                else:
                    if sub_orb in cols:
                        result["up"] += pdos_data[:, cols.index(sub_orb) + 1]
                        has_data = True
        return result if has_data else {}

    def _calculate_single_stat(self, energies: np.ndarray, dos: np.ndarray) -> dict:
        """计算单条 DOS 曲线（1D数组）的统计学特征"""
        m0 = np.trapz(dos, energies) # 0阶矩 (积分面积/电子填充数)
        
        if m0 < 1e-5: # 避免除以 0
            return {"center": 0.0, "width": 0.0, "skewness": 0.0, "kurtosis": 0.0, "filling": 0.0}
            
        center = np.trapz(energies * dos, energies) / m0
        variance = np.trapz(((energies - center)**2) * dos, energies) / m0
        width = np.sqrt(max(variance, 0.0))
        
        if width > 1e-5:
            skewness = np.trapz(((energies - center)**3) * dos, energies) / (m0 * (width**3))
            kurtosis = np.trapz(((energies - center)**4) * dos, energies) / (m0 * (width**4))
        else:
            skewness, kurtosis = 0.0, 0.0
            
        return {
            "center": float(center),
            "width": float(width),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "filling": float(m0)
        }

    def _calculate_band_stats(self, energies: np.ndarray, dos_up: np.ndarray, dos_down: np.ndarray = None) -> dict:
        """处理自旋，分别计算 up, down 和 total 的统计信息"""
        stats = {}
        
        # 1. 计算 Spin Up 的统计信息
        stats["up"] = self._calculate_single_stat(energies, dos_up)
        
        if dos_down is not None:
            # 2. 计算 Spin Down 的统计信息
            # 【关键修复】：VASP 的 DOSCAR 中 spin down 是负值，必须取绝对值才能代表真实的态密度分布！
            abs_down = np.abs(dos_down)
            stats["down"] = self._calculate_single_stat(energies, abs_down)
            
            # 3. 计算 Total (Up + |Down|) 的统计信息
            total_dos = dos_up + abs_down
            stats["total"] = self._calculate_single_stat(energies, total_dos)
        else:
            # 无自旋极化时，total 就是 up
            stats["total"] = stats["up"]
            
        return stats

    def analyze_multi_curves(self, curves: List[Dict[str, Any]], erange: List[float] = [-5.0, 3.0], show_tdos: bool = True) -> ApiResponse:
        """
        处理前端传来的多条自定义 DOS 曲线请求，并计算统计信息。
        【完全基于原生的高性能 DoscarParser 和 _aggregate_pdos 引擎】
        """
        try:
            # 1. 触发懒加载，确保 DOSCAR 已解析
            _ = self.parser.tdos 
            
            # 🚀 终极修复：直接计算对齐后的能量轴，不依赖 _energies() 方法，防止该方法意外丢失
            energies = self.parser.energies - self.efermi
            
            mask = (energies >= erange[0]) & (energies <= erange[1])
            energies_filtered = energies[mask]
            
            # 🚀 新增：专门用于统计学计算的占据态掩码 (E < 0)
            occ_mask = energies_filtered < 0
            energies_occ = energies_filtered[occ_mask]
            
            result_data = {
                "is_spin_polarized": self.is_spin,
                "energies": energies_filtered.tolist(),
                "fermi_level": self.efermi,
                "curves": []
            }
            
            # 2. 处理 TDOS
            if show_tdos:
                tdos = self.parser.tdos
                tdos_up = tdos[:, 1][mask]
                tdos_down = tdos[:, 2][mask] if self.is_spin else None
                
                # 🚀 提取占据态数据用于计算统计信息
                tdos_up_occ = tdos_up[occ_mask]
                tdos_down_occ = tdos_down[occ_mask] if tdos_down is not None else None
                
                result_data["curves"].append({
                    "id": "tdos",
                    "name": "Total DOS",
                    "data_up": tdos_up.tolist(),
                    "data_down": tdos_down.tolist() if tdos_down is not None else None,
                    "stats": self._calculate_band_stats(energies_occ, tdos_up_occ, tdos_down_occ)
                })
                
            cols = self.parser.get_pdos_col_names()
            
            # 3. 处理自定义曲线
            for curve in curves:
                curve_id = curve.get("id", "unknown")
                orbital_raw = curve.get("orbital", "all").lower()
                
                if orbital_raw in ["all", "s", "p", "d", "f"]:
                    orb_mapped = orbital_raw
                else:
                    orb_mapped = orbital_raw.replace("_", "")
                
                site_indices = []
                curve_name = ""
                
                curve_type = curve.get("type")
                if curve_type == "element" or "element" in curve:
                    element = curve.get("element")
                    site_indices = [i for i, sym in enumerate(self.site_elements) if sym == element]
                    curve_name = f"{element} ({orbital_raw})"
                elif curve_type == "site" or "site" in curve:
                    site_idx = int(curve.get("site")) - 1 
                    if not self._valid_site(site_idx):
                        continue
                    site_indices = [site_idx]
                    curve_name = f"Site {site_idx + 1} {self.site_elements[site_idx]} ({orbital_raw})"
                
                if not site_indices:
                    continue
                    
                target_cols = []
                if orb_mapped == "all":
                    base_names = []
                    for c in cols:
                        base = c.replace("_up", "").replace("_down", "")
                        if base not in base_names: 
                            base_names.append(base)
                    target_cols = base_names
                elif orb_mapped in ["s", "p", "d", "f"]:
                    if (self.is_spin and f"{orb_mapped}_up" in cols) or (not self.is_spin and orb_mapped in cols):
                        target_cols = [orb_mapped]
                    else:
                        target_cols = self.SPD_GROUP.get(orb_mapped, [])
                else:
                    target_cols = [orb_mapped]
                    
                agg = self._aggregate_pdos(site_indices, target_cols)
                
                data_up = agg.get("up", np.zeros_like(energies))[mask]
                data_down = agg.get("down", np.zeros_like(energies))[mask] if self.is_spin else None
                
                if "up" not in agg:
                    self._log(f"No data found for {curve_name} in DOSCAR (check LORBIT)", "warning")
                
                # 🚀 提取占据态数据用于计算统计信息
                data_up_occ = data_up[occ_mask]
                data_down_occ = data_down[occ_mask] if data_down is not None else None
                stats = self._calculate_band_stats(energies_occ, data_up_occ, data_down_occ)
                
                result_data["curves"].append({
                    "id": curve_id,
                    "name": curve_name,
                    "data_up": data_up.tolist(),
                    "data_down": data_down.tolist() if data_down is not None else None,
                    "stats": stats
                })

            return ApiResponse.ok(data=result_data, message="Multi-curve DOS extracted successfully.")
        except Exception as e:
            self._log(f"analyze_multi_curves failed: {e}", "error")
            return ApiResponse.error(str(e))
        
# ============================================================
# 5. cohp.py —— COHP/ICOHP 成键分析
# ============================================================
from pymatgen.io.lobster import Cohpcar, Icohplist
from pymatgen.electronic_structure.core import Spin

class CohpAnalysis(VaspAnalysisBase):
    """
    COHP / ICOHP 成键分析器（基于 LOBSTER 输出）。

    所需文件（位于 work_dir）：
      - COHPCAR.lobster   → COHP 曲线数据
      - ICOHPLIST.lobster → 积分 ICOHP 键对汇总
    """

    def __init__(
        self,
        work_dir: Union[str, Path],
        save_data: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        log_level: int = logging.WARNING,
    ):
        super().__init__(work_dir, save_data, output_dir, log_level)
        self._cohpcar:   Optional[Cohpcar]   = None
        self._icohplist: Optional[Icohplist] = None

    # ── 懒加载 ────────────────────────────────────────────

    @property
    def cohpcar(self) -> Optional[Cohpcar]:
        if self._cohpcar is None:
            path = self.work_dir / "COHPCAR.lobster"
            if path.exists():
                try:
                    self._cohpcar = Cohpcar(filename=str(path))
                except Exception as e:
                    self._log(f"COHPCAR parse failed: {e}", "warning")
            else:
                self._log("COHPCAR.lobster not found", "warning")
        return self._cohpcar

    @property
    def icohplist(self) -> Optional[Icohplist]:
        if self._icohplist is None:
            path = self.work_dir / "ICOHPLIST.lobster"
            if path.exists():
                try:
                    self._icohplist = Icohplist(filename=str(path))
                except Exception as e:
                    self._log(f"ICOHPLIST parse failed: {e}", "warning")
            else:
                self._log("ICOHPLIST.lobster not found", "warning")
        return self._icohplist

    # ── 公开分析入口 ──────────────────────────────────────

    def analyze(
        self,
        n_top_bonds: int = 10,
        erange: Optional[List[float]] = None,
        bond_labels: Optional[List[str]] = None,
    ) -> ApiResponse:
        """
        执行完整 COHP/ICOHP 分析。

        Parameters
        ----------
        n_top_bonds  : 按 |ICOHP| 排序后返回前 N 个键对
        erange       : COHP 曲线能量截取范围 [Emin, Emax]，None 则返回全部
        bond_labels  : 指定输出的键对标签，None 则输出全部
        """
        try:
            icohp_df = self.get_icohp_summary(n_top=n_top_bonds, save=self.save_data)
            cohp_df  = self.get_cohp_curves(
                bond_labels=bond_labels, erange=erange, save=self.save_data
            )
            data = {
                "icohp_summary": icohp_df.to_dict(orient="records") if icohp_df is not None else [],
                "cohp_curves":   cohp_df.to_dict(orient="records")  if cohp_df  is not None else [],
                "n_bonds":       len(icohp_df) if icohp_df is not None else 0,
            }
            return ApiResponse.ok(data=data, message="COHP analysis complete")
        except Exception as e:
            self._log(f"CohpAnalysis failed: {e}", "error")
            return ApiResponse.error(str(e))

    # ── ICOHP 汇总 ────────────────────────────────────────
    def get_icohp_summary(
        self,
        n_top: int = 10,
        element_pair: Optional[tuple] = None,
        save: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        提取 ICOHP 汇总表，按 |ICOHP| 降序排列。
        """
        from pymatgen.electronic_structure.core import Spin
        import pandas as pd
        import numpy as np
        
        if self.icohplist is None:
            return None
            
        # ── 终极修复：直接从文件解析原子名，绕过 pymatgen 的版本差异 ──
        atom_map = {}
        icohp_path = self.work_dir / "ICOHPLIST.lobster"
        if icohp_path.exists():
            with open(icohp_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    # 如果这一行是以数字开头（代表 COHP#），则提取原子名
                    if len(parts) >= 4 and parts[0].isdigit():
                        bond_idx = parts[0]
                        atom_map[bond_idx] = (parts[1], parts[2]) # (atomMU, atomNU)

        icohp_obj = self.icohplist        
        keys      = list(icohp_obj.icohplist.keys())
        rows      = []

        for label in keys:
            bond = icohp_obj.icohplist[label]

            # 直接从我们自己解析的字典里拿原子名
            atom1, atom2 = atom_map.get(label, ("", ""))

            # ── 元素对过滤 ──────────────────────────────────────────
            if element_pair is not None:
                # 提取纯元素符号 (例如 "Pd64" -> "Pd")
                el1 = "".join(c for c in atom1 if c.isalpha())
                el2 = "".join(c for c in atom2 if c.isalpha())
                
                # 如果不匹配指定的元素对，则跳过
                if not el1 or not el2 or set(element_pair) != {el1, el2}:
                    continue

            # ── 读取 ICOHP 值 ───────────────────────────────────────
            icohp_dict = bond.get("icohp", {})
            icohp_up   = icohp_dict.get(Spin.up,   None)
            icohp_down = icohp_dict.get(Spin.down, None)

            if icohp_up is not None and icohp_down is not None:
                icohp_total = icohp_up + icohp_down
            elif icohp_up is not None:
                icohp_total = icohp_up
            else:
                icohp_total = np.nan

            rows.append({
                "bond_label": label,
                "atom1":      atom1,
                "atom2":      atom2,
                "length_Ang": bond.get("length", np.nan),
                "icohp_eV":   icohp_total,
            })

        # 防止过滤后为空导致 DataFrame 报错
        if not rows:
            self._log(f"未找到匹配的键 (元素对 {element_pair})", "warning")
            return pd.DataFrame(columns=["bond_label", "atom1", "atom2", "length_Ang", "icohp_eV"])

        df = (
            pd.DataFrame(rows)
            .assign(abs_icohp=lambda x: x["icohp_eV"].abs())
            .sort_values("abs_icohp", ascending=False, na_position="last")
            .drop(columns=["abs_icohp"])
            .head(n_top)
            .reset_index(drop=True)
        )

        if save:
            self.save_to_csv(df, "icohp_summary.csv")

        return df

    # ── COHP 曲线 ─────────────────────────────────────────

    def get_cohp_curves(
        self,
        bond_labels: Optional[List[str]] = None,
        erange: Optional[List[float]] = None,
        save: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        提取 COHP 曲线数据。

        Parameters
        ----------
        bond_labels : 指定键对标签，None 则返回全部
        erange      : [Emin, Emax] 能量截取范围

        Returns
        -------
        DataFrame 列：energy_eV, {bond_label}_up, {bond_label}_down?
        """
        if self.cohpcar is None:
            return None

        energies  = np.array(self.cohpcar.energies)
        cohp_data = self.cohpcar.cohp_data

        mask = (
            (energies >= erange[0]) & (energies <= erange[1])
            if erange else np.ones(len(energies), dtype=bool)
        )

        data: Dict[str, Any] = {"energy_eV": energies[mask]}
        targets = bond_labels or list(cohp_data.keys())

        for label in targets:
            bond = cohp_data.get(label)
            if bond is None:
                self._log(f"Bond label '{label}' not found in COHPCAR", "warning")
                continue
            cohp = bond.get("COHP", {})
            if Spin.up in cohp:
                data[f"{label}_up"]   = np.array(cohp[Spin.up])[mask]
            if Spin.down in cohp:
                data[f"{label}_down"] = np.array(cohp[Spin.down])[mask]

        df = pd.DataFrame(data)
        if save:
            self.save_to_csv(df, "cohp_curves.csv")
        return df

    def get_integrated_cohp(
        self,
        bond_label: str,
        erange: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        对指定键对的 COHP 进行数值积分。

        Returns
        -------
        Dict: icohp_up, icohp_down (自旋时), icohp_total
        """
        if self.cohpcar is None:
            return {}

        energies = np.array(self.cohpcar.energies)
        bond     = self.cohpcar.cohp_data.get(bond_label)
        if bond is None:
            self._log(f"Bond label '{bond_label}' not found", "warning")
            return {}

        mask = (
            (energies >= erange[0]) & (energies <= erange[1])
            if erange else np.ones(len(energies), dtype=bool)
        )

        cohp   = bond.get("COHP", {})
        result: Dict[str, float] = {}
        total  = 0.0

        for spin, key in [(Spin.up, "icohp_up"), (Spin.down, "icohp_down")]:
            if spin in cohp:
                val = float(np.trapz(np.array(cohp[spin])[mask], energies[mask]))
                result[key] = val
                total += val

        result["icohp_total"] = total
        return result
    
    def get_orbital_cohp_curves(
        self,
        bond_label: str,
        erange: Optional[List[float]] = None,
        save: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        提取指定键的分轨道 COHP 曲线数据。

        Parameters
        ----------
        bond_label : 指定键对标签 (例如 "1")
        erange     : [Emin, Emax] 能量截取范围

        Returns
        -------
        DataFrame 列：energy_eV, {orb_label}_up, {orb_label}_down?
        """
        from pymatgen.electronic_structure.core import Spin
        import numpy as np
        import pandas as pd
        
        if self.cohpcar is None:
            return None

        # 检查是否包含分轨道数据
        if not hasattr(self.cohpcar, "orb_res_cohp") or not self.cohpcar.orb_res_cohp:
            self._log("No orbital-resolved COHP data found. Check your LOBSTER settings.", "warning")
            return None

        if bond_label not in self.cohpcar.orb_res_cohp:
            self._log(f"Bond label '{bond_label}' not found in orbital-resolved data.", "warning")
            return None

        energies = np.array(self.cohpcar.energies)
        mask = (
            (energies >= erange[0]) & (energies <= erange[1])
            if erange else np.ones(len(energies), dtype=bool)
        )

        data: Dict[str, Any] = {"energy_eV": energies[mask]}
        orb_data = self.cohpcar.orb_res_cohp[bond_label]

        # 遍历该键的所有轨道组合 (例如 "4s-2p_x", "3d_xy-2p_y" 等)
        for orb_label, orb_cohp in orb_data.items():
            cohp_vals = orb_cohp.get("COHP", {})
            if Spin.up in cohp_vals:
                data[f"{orb_label}_up"] = np.array(cohp_vals[Spin.up])[mask]
            if Spin.down in cohp_vals:
                data[f"{orb_label}_down"] = np.array(cohp_vals[Spin.down])[mask]

        df = pd.DataFrame(data)
        if save:
            self.save_to_csv(df, f"cohp_orb_{bond_label}.csv")
        return df


# ============================================================
# 6. dispatcher.py —— 统一调度入口（Java 调用此处）
# ============================================================

from pymatgen.electronic_structure.core import Spin


_TASK_REGISTRY: Dict[str, type] = {
    "relax": RelaxAnalysis,
    "dos":   DosAnalysis,
    "cohp":  CohpAnalysis,
    "structure": StructureAnalysis,  
}

def dispatch(task_type: str, work_dir: str, **kwargs) -> str:
    task_type = task_type.lower().strip()
    allowed_tasks = list(_TASK_REGISTRY.keys()) + ["dos_multi"]
    
    if task_type not in allowed_tasks:
        return ApiResponse.bad_request(f"Unknown task_type '{task_type}'").to_json()

    try:
        if task_type == "dos_multi":
            analyzer = DosAnalysis(work_dir=work_dir)
            return analyzer.analyze_multi_curves(**kwargs).to_json()
            
        elif task_type == "structure":
            analyzer = StructureAnalysis(work_dir=work_dir)
            return analyzer.get_info().to_json() # <--- 专门处理前端第一步
            
        else:
            analyzer = _TASK_REGISTRY[task_type](work_dir=work_dir)
            return analyzer.analyze(**kwargs).to_json()
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ApiResponse.error(str(e)).to_json()

def register_task(name: str, analyzer_cls: type):
    """
    注册自定义分析任务（扩展点，不修改已有代码）。

    示例：
        from my_module import NebAnalysis
        register_task("neb", NebAnalysis)
        dispatch("neb", "/path/to/neb")
    """
    if not hasattr(analyzer_cls, "analyze"):
        raise TypeError(f"{analyzer_cls.__name__} must implement analyze() method.")
    _TASK_REGISTRY[name] = analyzer_cls


# ============================================================
# 7. __init__.py
# ============================================================
__all__ = [
    "dispatch",
    "register_task",
    "RelaxAnalysis",
    "DosAnalysis",
    "CohpAnalysis",
    "StructureAnalysis",
    "ApiResponse",
]