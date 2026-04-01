# -*- coding: utf-8 -*-

###基本实现了JAVA前后端的使用思路，但是代码相对于来说比较冗余，版本二是在此基础上的修改


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

        # 第 1 行: NIONS NKPTS ISPIN ...
        header_parts = lines[0].split()
        self._nions = int(header_parts[0])
        self._has_pdos = (len(header_parts) >= 3 and int(header_parts[2]) == 1)

        # 第 6 行: Emax Emin NEDOS Efermi weight
        dos_header = lines[5].split()
        self._nedos  = int(dos_header[2])
        self._efermi = float(dos_header[3])

    def _parse_all(self):
        """完整解析 DOSCAR, 借鉴 vasppy 使用 pandas 加速读取"""
        # 1. 读取 TDOS
        tdos_df = pd.read_csv(
            self.path, 
            skiprows=6, 
            nrows=self._nedos,
            delim_whitespace=True, 
            header=None,
            engine='c' # 使用 C 引擎加速
        )
        self._tdos = tdos_df.values
        self._energies = self._tdos[:, 0]
        self._ispin = 2 if self._tdos.shape[1] == 5 else 1

        # 2. 读取 PDOS (如果存在)
        self._pdos = []
        if self._has_pdos:
            # VASP DOSCAR 结构：TDOS 后，每个原子有 1行 header + NEDOS 行数据
            # 为了极速读取，我们一次性跳过 TDOS，读取后面所有行，然后用 numpy 切片
            skip_lines = 6 + self._nedos
            total_pdos_lines = self._nions * (self._nedos + 1)
            
            try:
                # 一次性读入所有 PDOS 文本块
                pdos_raw_df = pd.read_csv(
                    self.path,
                    skiprows=skip_lines,
                    nrows=total_pdos_lines,
                    delim_whitespace=True,
                    header=None,
                    engine='c'
                )
                raw_values = pdos_raw_df.values
                
                # 剔除每个原子前面的 header 行 (Emax Emin NEDOS Efermi weight)
                # 每一块的长度是 nedos + 1
                block_size = self._nedos + 1
                for i in range(self._nions):
                    start_idx = i * block_size + 1 # +1 跳过 header
                    end_idx = start_idx + self._nedos
                    self._pdos.append(raw_values[start_idx:end_idx, :])
            except Exception as e:
                # 如果 Pandas 快速读取失败（比如文件格式不标准），回退到安全模式
                import logging
                logging.warning(f"Pandas fast read failed, fallback to standard read: {e}")
                self._parse_all_fallback()

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
        """
        根据 PDOS 数据列数自动推断轨道列名。
        返回列表不含第一列 energy。
        """
        if not self.pdos: return []
        ncols = self.pdos[0].shape[1] - 1  # 减去 energy 列
        ##根据列数判断轨道类型和自旋情况，返回对应的列名列表
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
    DOS 分析器。

    数据来源：
      - DOSCAR    : 能量轴、TDOS、PDOS（必须）
      - CONTCAR / POSCAR : 结构信息，用于元素/位点映射（必须）
      - OUTCAR    : 费米能级（可选，优先于 DOSCAR 中的值）

    所需文件：
      work_dir/
        ├── DOSCAR    ← 主数据源
        ├── CONTCAR   ← 结构（优先）
        └── POSCAR    ← 结构（备选）
    """

    # 轨道分组映射（用于 SPD 汇总）
    SPD_GROUP = {
        "s":   ["s"],
        "p":   ["py", "pz", "px"],
        "d":   ["dxy", "dyz", "dz2", "dxz", "dx2-y2"],
        "f":   ["fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2y2", "fx3y2"],
        "t2g": ["dxy", "dyz", "dxz"],
        "eg":  ["dz2", "dx2-y2"]
    }
    FRONTEND_TO_VASP_ORB = {
    "s": "s", "p_y": "py", "p_z": "pz", "p_x": "px",
    "d_xy": "dxy", "d_yz": "dyz", "d_z2": "dz2", "d_xz": "dxz", "d_x2-y2": "dx2-y2",
    "f_{y(3x^2-y^2)}": "fy3x2", "f_xyz": "fxyz", "f_{yz^2}": "fyz2", 
    "f_{z^3}": "fz3", "f_{xz^2}": "fxz2", "f_{z(x^2-y^2)}": "fzx2y2", "f_{x(x^2-3y^2)}": "fx3y2"
    }

    def __init__(self, work_dir: Union[str, Path], save_data: bool = False,
        output_dir: Optional[Union[str, Path]] = None, log_level: int = logging.WARNING,
    ):
        super().__init__(work_dir, save_data, output_dir, log_level)
        self.save_data = save_data
        self._doscar_parser: Optional[DoscarParser] = None
        self._structure = None
        self._align_offset: float = 0.0
        self._align_mode: str = "fermi"

    @property
    def parser(self) -> DoscarParser:
        """DOSCAR 解析器"""
        if self._doscar_parser is None:
            doscar_path = self.work_dir / "DOSCAR"
            if not doscar_path.exists():
                raise FileNotFoundError(f"DOSCAR not found in {self.work_dir}")
            self._doscar_parser = DoscarParser(doscar_path)
        return self._doscar_parser

    @property
    def structure(self):
        """加载结构（CONTCAR 优先，其次 POSCAR）"""
        if self._structure is None:
            from pymatgen.io.vasp import Poscar
            for fname in ("CONTCAR", "POSCAR"):
                path = self.work_dir / fname
                if path.exists():
                    try:
                        self._structure = Poscar.from_file(str(path)).structure
                        self._log(f"Structure loaded from {fname}", "debug")
                        break
                    except Exception as e:
                        self._log(f"{fname} parse failed: {e}", "warning")
            if self._structure is None:
                raise FileNotFoundError(f"No valid CONTCAR/POSCAR in {self.work_dir}")
        return self._structure

    @property
    def is_spin(self) -> bool: return self.parser.ispin == 2

    @property
    def efermi(self) -> float: return self.parser.efermi

    @property
    def site_elements(self) -> List[str]:
        """按位点顺序返回元素符号列表，与 DOSCAR 中 PDOS 块顺序一致"""
        return [str(site.specie.symbol) for site in self.structure]

    #能量对齐
    def set_alignment(self, mode: str = "fermi", value: float = 0.0,) -> "DosAnalysis":
        """
        设置能量零点。

        Parameters
        ----------
        mode  : "fermi"  → 以费米能级为零点（默认）
                "manual" → 使用自定义参考能量
        value : manual 模式下的参考能量 (eV)
        """
        if mode == "fermi":
            self._align_mode   = "fermi"
            self._align_offset = self.efermi
        elif mode == "manual":
            self._align_mode   = "manual"
            self._align_offset = float(value)
        else:
            raise ValueError(f"Unknown alignment mode: '{mode}'")
        self._log(f"Alignment: mode={mode}, offset={self._align_offset:.4f} eV", "debug")
        return self

    def _energies(self) -> np.ndarray:
        """返回对齐后的能量轴"""
        # 如果是对齐到费米能级，动态获取 efermi 并相减
        if self._align_mode == "fermi":
            return self.parser.energies - self.efermi
        # 如果是 manual 模式，则使用自定义的 offset
        return self.parser.energies - self._align_offset

    def _valid_site(self, idx: int) -> bool:
        if idx < 0 or idx >= self.parser.nions:
            self._log(f"Invalid site index {idx}", "error")
            return False
        return True   
    
    def _aggregate_pdos(self, site_indices: List[int], sub_orbitals: List[str]) -> Dict[str, np.ndarray]:
        """高性能底层聚合引擎，替代低效的 for 循环 sum"""
        if not self.parser.pdos: return {}
        cols = self.parser.get_pdos_col_names()
        nedos = self.parser.nedos
        
        result = {"up": np.zeros(nedos)}
        if self.is_spin: result["down"] = np.zeros(nedos)

        has_data = False
        for site_idx in site_indices:
            pdos_data = self.parser.pdos[site_idx]
            for sub_orb in sub_orbitals:
                if self.is_spin:
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
    
    def _parse_site_input(self, site_str: Any, element: str) -> List[int]:
        """解析前端传来的 site 参数，返回对应的原子索引列表(0-based)"""
        site_str = str(site_str).strip()
        
        # 如果为空或 "All"，返回该元素的所有位点
        if not site_str or site_str.lower() == "all":
            if element:
                return [i for i, sym in enumerate(self.site_elements) if sym == element]
            else:
                return list(range(self.parser.nions))
                
        # 如果指定了具体数字（前端是 1-based）
        elif site_str.isdigit():
            idx = int(site_str) - 1  # 转换为后端 0-based 索引
            if self._valid_site(idx):
                # 校验该位点的元素是否匹配
                if element and self.site_elements[idx] != element:
                    self._log(f"Warning: Site {site_str} is actually {self.site_elements[idx]}, not {element}", "warning")
                    return []
                return [idx]
        return []
    
    # ── 核心数据获取接口 (返回 DataFrame) ──
    def get_total_dos(self, save: bool = False) -> pd.DataFrame:
        tdos = self.parser.tdos
        data = {"energy_eV": self._energies(), "tdos_up": tdos[:, 1], "idos_up": tdos[:, 2 if not self.is_spin else 3]}
        if self.is_spin:
            data["tdos_down"] = tdos[:, 2]
            data["idos_down"] = tdos[:, 4]
        df = pd.DataFrame(data)
        if save: self.save_to_csv(df, "total_dos.csv")
        return df

    def get_site_dos(self, site_index: int, save: bool = False) -> Optional[pd.DataFrame]:
        if not self._valid_site(site_index): return None
        pdos_data = self.parser.pdos[site_index]
        cols = self.parser.get_pdos_col_names()
        
        data = {"energy_eV": self._energies()}
        if self.is_spin:
            up_idx = [i + 1 for i, c in enumerate(cols) if c.endswith("_up")]
            dn_idx = [i + 1 for i, c in enumerate(cols) if c.endswith("_down")]
            data["dos_up"] = pdos_data[:, up_idx].sum(axis=1)
            data["dos_down"] = pdos_data[:, dn_idx].sum(axis=1)
        else:
            data["dos"] = pdos_data[:, 1:].sum(axis=1)
            
        df = pd.DataFrame(data)
        if save: self.save_to_csv(df, f"site_{site_index}_{self.site_elements[site_index]}_dos.csv")
        return df

    def get_site_spd_dos(self, site_index: int, save: bool = False) -> Optional[pd.DataFrame]:
        if not self._valid_site(site_index): return None
        data = {"energy_eV": self._energies()}
        cols = self.parser.get_pdos_col_names()

        for orb_name in ["s", "p", "d", "f"]:
            target_cols = [orb_name] if ((self.is_spin and f"{orb_name}_up" in cols) or (not self.is_spin and orb_name in cols)) else self.SPD_GROUP.get(orb_name, [])
            agg = self._aggregate_pdos([site_index], target_cols)
            if "up" in agg: data[f"{orb_name}_up" if self.is_spin else orb_name] = agg["up"]
            if "down" in agg: data[f"{orb_name}_down"] = agg["down"]

        if len(data) == 1: return None
        df = pd.DataFrame(data)
        if save: self.save_to_csv(df, f"site_{site_index}_{self.site_elements[site_index]}_spd_dos.csv")
        return df

    def get_site_orbital_dos(self, site_index: int, orbital: str, save: bool = False) -> Optional[pd.DataFrame]:
        if not self._valid_site(site_index): return None
        data = {"energy_eV": self._energies()}
        agg = self._aggregate_pdos([site_index], [orbital])
        
        if "up" in agg: data[f"{orbital}_up" if self.is_spin else orbital] = agg["up"]
        if "down" in agg: data[f"{orbital}_down"] = agg["down"]
        
        if len(data) == 1: return None
        df = pd.DataFrame(data)
        if save: self.save_to_csv(df, f"site_{site_index}_{self.site_elements[site_index]}_{orbital}_dos.csv")
        return df

    def get_site_t2g_eg_dos(self, site_index: int, save: bool = False) -> Optional[pd.DataFrame]:
        if not self._valid_site(site_index): return None
        data = {"energy_eV": self._energies()}
        
        for grp in ["t2g", "eg"]:
            agg = self._aggregate_pdos([site_index], self.SPD_GROUP[grp])
            if "up" in agg: data[f"{grp}_up" if self.is_spin else grp] = agg["up"]
            if "down" in agg: data[f"{grp}_down"] = agg["down"]
            
        if len(data) == 1: return None
        df = pd.DataFrame(data)
        if save: self.save_to_csv(df, f"site_{site_index}_{self.site_elements[site_index]}_t2g_eg_dos.csv")
        return df

    def get_element_spd_dos(self, element: Optional[str] = None, save: bool = False) -> Optional[pd.DataFrame]:
        data = {"energy_eV": self._energies()}
        targets = [element] if element else list(dict.fromkeys(self.site_elements))
        cols = self.parser.get_pdos_col_names()

        for el in targets:
            site_indices = [i for i, sym in enumerate(self.site_elements) if sym == el]
            if not site_indices: continue
            
            for orb_name in ["s", "p", "d", "f"]:
                target_cols = [orb_name] if ((self.is_spin and f"{orb_name}_up" in cols) or (not self.is_spin and orb_name in cols)) else self.SPD_GROUP.get(orb_name, [])
                agg = self._aggregate_pdos(site_indices, target_cols)
                if "up" in agg: data[f"{el}_{orb_name}_up" if self.is_spin else f"{el}_{orb_name}"] = agg["up"]
                if "down" in agg: data[f"{el}_{orb_name}_down"] = agg["down"]

        if len(data) == 1: return None
        df = pd.DataFrame(data)
        if save: self.save_to_csv(df, f"element_spd_dos_{element}.csv" if element else "element_spd_dos.csv")
        return df

    def get_orbital_statistics(self, element: Optional[str] = None, site_index: Optional[int] = None, orbital: str = "d", erange: Optional[List[float]] = None) -> Dict[str, Any]:
        if erange is None: erange = [-10.0, 10.0]
        
        if site_index is not None:
            df = self.get_site_spd_dos(site_index)
            orb_col, prefix = orbital, f"site{site_index}_{orbital}"
        elif element:
            df = self.get_element_spd_dos(element)
            orb_col, prefix = f"{element}_{orbital}", f"{element}_{orbital}"
        else: return {}

        if df is None or df.empty: return {}

        energies = df["energy_eV"].values
        mask = (energies >= erange[0]) & (energies <= erange[1])
        e_sel = energies[mask]
        stats = {}

        spin_keys = [(f"{orb_col}_up", f"{orb_col}_down")] if self.is_spin else [(orb_col, None)]

        for up_col, dn_col in spin_keys:
            for col, spin_name in [(up_col, "up" if self.is_spin else "total"), (dn_col, "down")]:
                if not col or col not in df.columns: continue
                d_sel = df[col].values[mask]
                if len(e_sel) < 5 or np.allclose(d_sel, 0): continue
                
                filling = float(np.trapz(d_sel, e_sel))
                if filling < 1e-9: continue
                
                center = float(np.trapz(d_sel * e_sel, e_sel) / filling)
                var = float(np.trapz(d_sel * (e_sel - center) ** 2, e_sel) / filling)
                width = float(np.sqrt(max(var, 0)))
                
                # 强制转换为原生 float，保证 JSON 序列化安全
                stats[f"{prefix}_{spin_name}_center"] = center
                stats[f"{prefix}_{spin_name}_width"] = width
                stats[f"{prefix}_{spin_name}_filling"] = filling

        return stats

    # ── Web API 入口 ──
    def analyze(self, elements: Optional[List[str]] = None, orbital: str = "d", erange: Optional[List[float]] = None) -> Dict[str, Any]: # 如果有 ApiResponse 请替换返回类型
        """
        执行完整分析并返回纯 JSON 兼容的字典数据。
        可以直接作为 Web API 的 Response 返回给前端。
        """
        try:
            tdos_df = self.get_total_dos(save=self.save_data)
            espd_df = self.get_element_spd_dos(save=self.save_data)

            targets = elements or list(dict.fromkeys(self.site_elements))
            orb_stats = {el: self.get_orbital_statistics(element=el, orbital=orbital, erange=erange) for el in targets}

            data = {
                "is_spin_polarized": self.is_spin,
                "fermi_level_eV":    float(self.efermi),
                "alignment_mode":    self._align_mode,
                "alignment_offset":  float(self._align_offset),
                "n_ions":            self.parser.nions,
                "nedos":             self.parser.nedos,
                # orient="records" 生成 [{x:1, y:2}, {x:2, y:3}] 格式，前端图表库最喜欢的格式
                "total_dos":         tdos_df.to_dict(orient="records") if tdos_df is not None else [],
                "element_spd_dos":   espd_df.to_dict(orient="records") if espd_df is not None else [],
                "orbital_statistics": {k: v for k, v in orb_stats.items() if v} # 过滤掉空的统计
            }
            # return ApiResponse.ok(data=data, message="DOS analysis complete")
            return {"status": "success", "data": data}
        except Exception as e:
            self._log(f"DosAnalysis failed: {e}", "error")
            # return ApiResponse.error(str(e))
            return {"status": "error", "message": str(e)}
    
    def _calculate_single_stat(self, energies: np.ndarray, dos: np.ndarray) -> Dict[str, float]:
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

    def _calculate_band_stats(self, energies: np.ndarray, dos_up: np.ndarray, dos_down: Optional[np.ndarray] = None) -> Dict[str, Any]:
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
            
            # 使用您已经对齐过费米能级的能量轴
            energies = self._energies() 
            mask = (energies >= erange[0]) & (energies <= erange[1])
            energies_filtered = energies[mask]
            
            # 因为 energies 已经减去了 efermi，所以 E < 0 就是费米能级以下的占据态
            occ_mask = energies_filtered < 0
            energies_occ = energies_filtered[occ_mask]
            
            result_data = {
                "energies": energies_filtered.tolist(),
                "fermi_level": self.efermi,
                "curves": []
            }
            
            # 2. 处理 TDOS
            if show_tdos:
                tdos = self.parser.tdos
                tdos_up = tdos[:, 1][mask]
                tdos_down = tdos[:, 2][mask] if self.is_spin else None
                

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
                curve_type = curve.get("type")
                orbital_raw = curve.get("orbital", "all").lower()
                
                # 映射前端轨道名称到您的 DOSCAR 列名 (例如 p_z -> pz, d_x2-y2 -> dx2-y2)
                if orbital_raw in ["all", "s", "p", "d", "f"]:
                    orb_mapped = orbital_raw
                else:
                    orb_mapped = orbital_raw.replace("_", "")
                
                # 获取需要聚合的原子索引列表 (site_indices)
                site_indices = []
                curve_name = ""
                if curve_type == "element":
                    element = curve.get("element")
                    # 完美复用您原有的 self.site_elements
                    site_indices = [i for i, sym in enumerate(self.site_elements) if sym == element]
                    curve_name = f"{element} ({orbital_raw})"
                elif curve_type == "site":
                    site_idx = int(curve.get("site")) - 1 # 前端 1-based 转 后端 0-based
                    if not self._valid_site(site_idx):
                        continue
                    site_indices = [site_idx]
                    curve_name = f"Site {site_idx + 1} {self.site_elements[site_idx]} ({orbital_raw})"
                
                if not site_indices:
                    continue
                    
                # 确定需要聚合的目标列名
                target_cols = []
                if orb_mapped == "all":
                    # 提取所有基础列名 (去除 _up/_down 后缀)
                    base_names = []
                    for c in cols:
                        base = c.replace("_up", "").replace("_down", "")
                        if base not in base_names: 
                            base_names.append(base)
                    target_cols = base_names
                elif orb_mapped in ["s", "p", "d", "f"]:
                    # 完美复用您处理 LORBIT=10/11 的兼容逻辑
                    if (self.is_spin and f"{orb_mapped}_up" in cols) or (not self.is_spin and orb_mapped in cols):
                        target_cols = [orb_mapped]
                    else:
                        target_cols = self.SPD_GROUP.get(orb_mapped, [])
                else:
                    target_cols = [orb_mapped]
                    
                # 【核心】调用您的高性能底层聚合引擎！
                agg = self._aggregate_pdos(site_indices, target_cols)
                
                # 提取数据并截断到指定能量范围
                data_up = agg.get("up", np.zeros_like(energies))[mask]
                data_down = agg.get("down", np.zeros_like(energies))[mask] if self.is_spin else None
                
                if "up" not in agg:
                    self._log(f"No data found for {curve_name} in DOSCAR (check LORBIT)", "warning")
                

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

def dispatch(
    task_type: str,
    work_dir: Optional[Union[str, Path]] = None,
    source: Optional[str] = None,
    source_type: str = "path",
    save_data: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> str:
    task_type = task_type.lower().strip()
    
    if source:
        if source_type == "queue":
            work_dir = Path("/data/vasp/jobs") / source
        else:
            work_dir = Path(source)
            
    if not work_dir:
        return ApiResponse.bad_request("Either work_dir or source must be provided.").to_json()

    allowed_tasks = list(_TASK_REGISTRY.keys()) + ["dos_multi"]
    if task_type not in allowed_tasks:
        return ApiResponse.bad_request(f"Unknown task_type '{task_type}'. Supported: {allowed_tasks}").to_json()

    try:
        if task_type == "dos_multi":
            analyzer = DosAnalysis(work_dir=work_dir, save_data=save_data, output_dir=output_dir)
            return analyzer.analyze_multi_curves(**kwargs).to_json()
        else:
            analyzer = _TASK_REGISTRY[task_type](work_dir=work_dir, save_data=save_data, output_dir=output_dir)
            if task_type == "structure":
                return analyzer.get_info().to_json()
            else:
                return analyzer.analyze(**kwargs).to_json()

    except Exception as e:
        import traceback
        traceback.print_exc()
        return ApiResponse.error(f"Internal error: {e}").to_json()

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