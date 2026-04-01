import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Literal, Any

# Pymatgen imports
from pymatgen.io.vasp import Vasprun, Outcar, Oszicar
from pymatgen.core.periodic_table import Element
from pymatgen.io.lobster import Cohpcar, Icohplist
from pymatgen.electronic_structure.core import Spin, Orbital, OrbitalType
try:
    from pymatgen.analysis.local_env import CrystalNN
    HAS_CRYSTALNN = True
except ImportError:
    HAS_CRYSTALNN = False

class VaspAnalysis:
    """
    VASP 后处理基类：智能分级读取数据。
    策略：优先读取轻量级 OUTCAR/OSZICAR，仅在必要时或无其他文件时读取 vasprun.xml。
    """
    def __init__(
            self,
            work_dir: Optional[Union[str, Path]] = None,
            vasprun_file: Optional[Union[str, Path]] = None,
            save_data: bool = True,
            output_dir: Optional[Union[str, Path]] = None
    ):
        self.work_dir = Path(work_dir) if work_dir else None
        
        # 自动推导路径
        self.vasprun_path = Path(vasprun_file) if vasprun_file else None
        if not self.vasprun_path and self.work_dir:
            self.vasprun_path = self.work_dir / "vasprun.xml"
        if self.work_dir is None and self.vasprun_path:
            self.work_dir = self.vasprun_path.parent
        
        self.save_data = save_data
        self.output_dir = Path(output_dir) if output_dir else self.work_dir
        
        # 缓存 (Lazy Loading)
        self.logger = None
        self._vasprun = None
        self._outcar = None
        self._oszicar = None
        
        self._validate_input()
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger_name = str(self.output_dir.absolute()) if self.output_dir else "Vasp_analysis_dir"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if logger.hasHandlers():
            logger.handlers.clear()
        # 格式化器 (仅保留消息本身，保持 report 的表格美观)
        formatter = logging.Formatter('%(message)s')

        # 1. 控制台
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # 2. 文件 Handler
        if self.save_data and self.output_dir:
            try:
                log_file = self.output_dir / "vasp_analysis.log"
                fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except OSError as e:
                print(f"[Warning] 无法创建日志文件 {log_file}: {e}")

        return logger

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)        

    def _validate_input(self):
        if not self.work_dir and not self.vasprun_path:
             raise ValueError("必须提供 work_dir 或 vasprun_file 其中之一。")
        if self.save_data and not self.output_dir.exists():
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except OSError as e:
                raise ValueError(f"无法创建输出目录 {self.output_dir}: {e}")
    
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """辅助方法：保存 DataFrame 到 CSV"""
        if self.save_data and self.output_dir:
            path = self.output_dir / filename
            # 使用 float_format 保证精度但不过长
            try:
                df.to_csv(path, index=False, float_format="%.6f")
                self._log(f"[Saved] {filename}")
            except OSError as e:
                self._log(f"[Warning] 无法保存 {filename} 到 {path}: {e}")

    def _get_oszicar(self) -> Optional[Oszicar]:
        if self._oszicar is None and self.work_dir:
            path = self.work_dir / "OSZICAR"
            if path.exists():
                try:
                    self._oszicar = Oszicar(str(path))
                except Exception as e:
                    self._log(f"[Warning] OSZICAR parsing failed: {e}")
        return self._oszicar

    def _get_outcar(self) -> Optional[Outcar]:
        if self._outcar is None and self.work_dir:
            path = self.work_dir / "OUTCAR"
            if path.exists():
                try:
                    self._outcar = Outcar(str(path))
                except Exception as e:
                    self._log(f"[Warning] OUTCAR parsing failed: {e}")
        return self._outcar

    def _get_vasprun(self, strict: bool = False) -> Optional[Vasprun]:
        """读取 vasprun.xml Args: strict (bool): 如果为 True 且文件不存在，抛出异常。"""
        if self._vasprun is None:
            if self.vasprun_path and self.vasprun_path.exists():
                try:
                    self._vasprun = Vasprun(str(self.vasprun_path), parse_potcar_file=False)
                except Exception as e:
                    if strict: raise ValueError(f"vasprun.xml parsing failed: {e}")
                    self._log(f"[Warning] vasprun.xml parsing failed: {e}")
            elif strict:
                raise FileNotFoundError("vasprun.xml not found.")
        return self._vasprun

    def _get_status(self) -> Tuple[bool, str]:
        """用于快速判断是否收敛"""
        if not self.work_dir: return False, "No work_dir"
        outcar_p = self.work_dir / "OUTCAR"
        oszicar_p = self.work_dir / "OSZICAR"
        
        converged = False
        last_line = "N/A"

        if outcar_p.exists():
            try:
                with open(outcar_p, 'r', errors='ignore') as f:
                    if "accuracy" in f.read(): converged = True
            except Exception: pass
        
        if oszicar_p.exists():
            try:
                with open(oszicar_p, 'rb') as f:
                    try: f.seek(-2048, 2)
                    except OSError: f.seek(0)
                    lines = f.readlines()
                    for line in reversed(lines):
                        l_str = line.decode(errors='ignore').strip()
                        if "F=" in l_str:
                            last_line = l_str
                            break
            except Exception: pass            
        return converged, last_line

    @property
    def final_energy(self):
        """最终能量"""
        osz = self._get_oszicar()
        if osz and osz.ionic_steps: return osz.ionic_steps[-1]["E0"]
        out = self._get_outcar()
        if out and hasattr(out, "final_energy"): return out.final_energy
        vr = self._get_vasprun(strict=False)
        if vr: return vr.final_energy            
        raise ValueError("无法从 OSZICAR/OUTCAR/vasprun.xml 获取最终能量。")

    @property
    def efermi(self):
        """费米能级"""
        out = self._get_outcar()
        if out: return out.efermi
        vr = self._get_vasprun()
        if vr: return vr.efermi
        return np.nan
    
    @property
    def nelect(self) -> float:
        """体系总电子数"""
        out = self._get_outcar()
        if out: return out.nelect
        vr = self._get_vasprun()
        if vr: return vr.parameters["nelect", np.nan]
        return np.nan
    
    @property
    def magnetization(self):
        """读取磁性信息"""
        out = self._get_outcar()
        if not out: return {}
        res = {}
        if hasattr(out, "total_mag"):
            res["total"] = out.total_mag
        if out.magnetization:
            mag_data = out.magnetization
            if isinstance(mag_data, tuple) and len(mag_data) > 0:
                res["sites"] = mag_data[0]
            else:
                res["sites"] = mag_data
        return res
    
    @property
    def get_site_magnetization(self):
        """获取位点的磁性信息，分为s,p,d,tot分量"""
        out = self._get_outcar()
        if not out or not out.magnetization:
            return pd.DataFrame()
        mag_info = self.magnetization.get('sites', [])
        if not mag_info:
            return pd.DataFrame()
        try: 
            df = pd.DataFrame(mag_info)
            if not df.empty:
                df.insert(0, "Atom_index", range(len(df)))
            return df
        except Exception as e: 
            self._log(f"[Warning] magnetization parsing failed: {e}")
            return pd.DataFrame()

    @property
    def ionic_steps_count(self):
        """离子步数量"""
        osz = self._get_oszicar()
        if osz: return len(osz.ionic_steps)
        return 0

    @property
    def electronic_steps_history(self) -> List[int]:
        """ 返回每一步离子步包含的电子步数，例如 [12, 10, 8, 4] """
        osz = self._get_oszicar()
        if osz:
            return [len(step) for step in osz.electronic_steps]
        return []

    def summary(self):
        """打印计算概览"""
        folder_name = self.work_dir.name if self.work_dir else "Unknown"
        self._log(f"--- VASP Analysis Summary: {folder_name} ---")
        sources = []
        if self._get_outcar(): sources.append("OUTCAR")
        if self._get_oszicar(): sources.append("OSZICAR")
        if self._get_vasprun(): sources.append("vasprun.xml")
        self._log(f"Data Sources:         {', '.join(sources)}")
        is_converged, last_line = self._get_status()
        self._log(f"Convergence:          {'Yes' if is_converged else 'No'}")
        self._log(f"Last Line:            {last_line}")
        osz = self._get_oszicar()
        if osz and osz.ionic_steps:
            last_step = osz.ionic_steps[-1]
        if 'mag' in last_step:
            mag = last_step['mag']
            self._log(f"Total Magnetization:  {mag:.4f} mu_B")
        if 'dE' in last_step:
            dE = last_step['dE']
            self._log(f"dE(last step)     {dE:.1E} eV")
            if abs(dE) > 1e-3 and is_converged:
                self._log(f"[Warning] dE(last step) > 1e-3 eV, convergence may not be reached.")
        
        self._log(f"Energy (E0):          {self.final_energy:.4f} eV")
        self._log(f"Fermi Level:          {self.efermi:.4f} eV")
        self._log(f"Total Electrons:      {self.nelect:.1f}")
        self._log(f"Ionic Steps:          {self.ionic_steps_count}")
        self._log(f"User Time:            {self._get_outcar().run_stats.get('User time (sec)', 'N/A')} sec")
        self._log("----------------------------------------------")

class DOSAnalysis(VaspAnalysis):
    """
    VASP DOS 分析与可视化增强版。
    功能：提取 PDOS, 计算 d-band center, 不同的对齐方案, 读取信息更加多样化。
    """
    ORBITAL_MAP = {
        's': OrbitalType.s, 'p': OrbitalType.p, 'd': OrbitalType.d, 'f': OrbitalType.f,
    }
    # 使用安全的映射，避免硬编码索引带来的潜在错误
    SUB_ORBITAL_MAP = {
        "s": Orbital.s,
        "p_y": Orbital.py, "p_z": Orbital.pz, "p_x": Orbital.px,
        "d_xy": Orbital.dxy, "d_yz": Orbital.dyz, "d_z2": Orbital.dz2,
        "d_xz": Orbital.dxz, "d_x2-y2": Orbital.dx2, "f_{y(3x^2-y^2)}": Orbital.f_3,
        "f_xyz": Orbital.f_2, "f_{yz^2}": Orbital.f_1, "f_{z^3}": Orbital.f0,
        "f_{xz^2}": Orbital.f1, "f_{z(x^2-y^2)}": Orbital.f2, "f_{x(x^2-3y^2)}": Orbital.f3,
    }

    def __init__(self, work_dir: Union[str, Path], save_data: bool = False, output_dir: Optional[Path] = None):
        super().__init__(work_dir=work_dir, save_data=save_data, output_dir=output_dir)
        
        # 懒加载缓存
        self._dos = None
        self._structure = None
        self._symbols = None
        
        # Smart Align 设置
        self._align_mode = "fermi" # 'fermi', 'vacuum', or float value
        self._align_offset = 0.0

    @property
    def dos(self):
        """懒加载 DOS 对象"""
        if self._dos is None:
            vr = self._get_vasprun(strict=True)
            self._dos = vr.complete_dos
        return self._dos

    @property
    def structure(self):
        if self._structure is None:
            vr = self._get_vasprun(strict=True)
            self._structure = vr.final_structure
        return self._structure

    @property
    def symbols(self):
        if self._symbols is None:
            #获取体系中所有的元素符号
            self._symbols = list(set(self.dos.structure.symbol_set))
        return self._symbols

    @property
    def is_spin(self):
        #确定是否是自旋极化的
        return len(self.dos.densities) > 1

    # ================= Smart Align 功能 =================
    def set_energy_alignment(self, mode: Literal['fermi', 'vacuum', 'manual'] = 'fermi', value: float = 0.0):
        """
        设置能量对齐参照点。
        Args:
            mode: 
                - 'fermi': E = E_raw - E_fermi (默认)
                - 'vacuum' / 'manual': E = E_raw - value
            value: 参照能量值 (仅在 manual/vacuum 模式下有效)
        """
        if mode == 'fermi':
            self._align_mode = 'fermi'
            self._align_offset = self.dos.efermi
            self._log("[Info] Alignment set to Fermi Level.")
        elif mode in ['vacuum', 'manual']:
            self._align_mode = mode
            self._align_offset = value
            self._log(f"[Info] Alignment set to Manual/Vacuum level: {value} eV")
        else:
            raise ValueError("Unsupported alignment mode.")
        
    def _get_safe_densities(self, dos_obj) -> Dict[Spin, np.ndarray]:
        """安全提取密度，处理 None 情况"""
        if dos_obj is None: return {}
        return getattr(dos_obj, 'densities', {})
    
    def _get_energies(self):
        """获取经过对齐处理后的能量轴"""
        if self._align_mode == 'fermi':
            return self.dos.energies - self.dos.efermi
        else:
            return self.dos.energies - self._align_offset   

    def _is_valid_dos(self, densities_dict):
        """检查DOS是不是全是0"""
        if densities_dict is None: return False
        spins = [Spin.up, Spin.down] if self.is_spin else [Spin.up]
        for spin in spins:
            if spin in densities_dict and not np.allclose(densities_dict[spin], 0):
                return True
        return False

    def parse_tolal_dos(self, save: bool = False, filename: str = "total_dos.csv") -> Optional[pd.DataFrame]:
        """解析总 DOS (TDOS)"""
        energies = self._get_energies()
        data = {"Energy": energies, "TDOS_up": self.dos.densities[Spin.up]}
        if self.is_spin:
            data["TDOS_down"] = self.dos.densities[Spin.down]
        
        df = pd.DataFrame(data)
        if save: 
            self.save_to_csv(df, filename)
            self._log(f"[Info] Total DOS saved to {filename}")
        return df

    def parse_element_spd_dos(self, save:bool = False, filename: str = "element_spd_dos.csv") -> Optional[pd.DataFrame]:
        """解析并导出具体的轨道投影 (如 d_xy, d_z2)，对同种元素求和。"""
        # 1. 检查 PDOS 是否存在
        if not hasattr(self.dos, 'pdos') or not self.dos.pdos:
            self._log("[Warning] No PDOS data found. Ensure LORBIT >= 10 in INCAR.")
            return None

        energies = self._get_energies()
        # 使用 List of Dicts 构建 DataFrame，性能优于逐行 Append
        data = {"Energy": energies}
        
        for el in self.symbols:
            e = Element(el)
            spd_dos = self.dos.get_element_spd_dos(e)
            
            for orb_label, orb_type in self.ORBITAL_MAP.items():
                if orb_type in spd_dos:
                    dens = self._get_safe_densities(spd_dos[orb_type])
                    if not dens: continue
                    
                    # 检查是否有非零数据
                    if np.allclose(dens[Spin.up], 0): continue
                    
                    data[f"{el}_{orb_label}_up"] = dens[Spin.up]
                    if self.is_spin:
                        data[f"{el}_{orb_label}_down"] = dens[Spin.down]
        
        df = pd.DataFrame(data)
        if save: 
            self.save_to_csv(df, filename)
            self._log(f"[Info] Element SP-DOS saved to {filename}")
        return df
    # ================= 获取位点信息 =================
            
    def _validate_site_index(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self.structure):
            self._log(f"[Error] Invalid site index: {idx}")
            return False
        return True

    def get_site_info(self, site_index: int) -> Dict:
        """获取位点的配位环境 (CN) 和局域磁矩"""
        if not self._validate_site_index(site_index): return {}
        site = self.structure[site_index]
        info = {"Index": site_index, "Element": site.specie.symbol}

        # CrystalNN
        if HAS_CRYSTALNN:
            try:
                cnn = CrystalNN()
                nn_info = cnn.get_nn_info(self.structure, site_index)
                info["CN"] = len(nn_info)
            except Exception: info["CN"] = "Error"
        else:
            info["CN"] = "NoModule"
        return info

    def parse_site_dos(self, site_index: int, save: bool = True, filename: Optional[str] = None) -> pd.DataFrame:
        """解析指定位点的总 DOS"""
        if not self._validate_site_index(site_index): return pd.DataFrame()
        
        energies = self._get_energies()
        site_dos = self.dos.get_site_dos(self.structure[site_index])
        
        data = {"Energy": energies, "DOS_up": site_dos.densities[Spin.up]}
        if self.is_spin: data["DOS_down"] = site_dos.densities[Spin.down]
        
        df = pd.DataFrame(data)
        
        # Log Site Info
        info = self.get_site_info(site_index)
        self._log(f"[Site Analysis] Site {site_index} ({info['Element']}), CN={info.get('CN')}, Mag={info.get('Mag', 0):.2f}")

        if save:
            fname = filename if filename else f"site_{site_index}_{info['Element']}_dos.csv"
            self.save_to_csv(df, fname)
            self._log(f"[Info] Site {site_index} DOS saved to {fname}")
        return df

    def parse_site_spd_dos(self, site_index: int, save: bool = True, filename: Optional[str] = None) -> pd.DataFrame:
        """解析指定位点的 s/p/d/f DOS"""
        if not self._validate_site_index(site_index): return pd.DataFrame()
        
        energies = self._get_energies()
        data = {"Energy": energies}
        spd_dos = self.dos.get_site_spd_dos(self.structure[site_index])
        
        has_data = False
        for orb_type, dos_obj in spd_dos.items():
            dens = self._get_safe_densities(dos_obj)
            if np.allclose(dens.get(Spin.up, 0), 0): continue
            
            has_data = True
            label = getattr(orb_type, 'name', str(orb_type))
            data[f"{label}_up"] = dens[Spin.up]
            if self.is_spin: data[f"{label}_down"] = dens[Spin.down]

        if not has_data: return pd.DataFrame()
        df = pd.DataFrame(data)
        if save:
            el = self.structure[site_index].specie.symbol
            fname = filename if filename else f"site_{site_index}_{el}_spd_dos.csv"
            self.save_to_csv(df, fname)
            self._log(f"[Info] Site {site_index} SP-DOS saved to {fname}")
        return df

    def parse_site_orbital_dos(self, site_index: int, orbital: str, save: bool = True, filename: Optional[str] = None) -> pd.DataFrame:
        """解析指定位点和具体子轨道 (如 d_xy) 的 DOS，并保存为 CSV 文件。"""
        
        if site_index < 0 or site_index >= len(self.structure.sites):
            raise ValueError("Invalid site index")
            
        site = self.structure[site_index]
        orb_enum = self.SUB_ORBITAL_MAP.get(orbital)
        if orb_enum is None:
            raise ValueError(f"Orbital '{orbital}' not recognized.")
            
        orbital_data = self.dos.pdos.get(site, {}).get(orb_enum)
        
        if orbital_data is None:
            self._log(f"Warning: DOS data for site {site_index}, orbital {orbital} is missing or LORBIT too low.")
            return None

        # 自适应获取密度
        densities = orbital_data.densities if hasattr(orbital_data, 'densities') else orbital_data
        
        if not self._is_valid_dos(densities):
            self._log(f"Warning: DOS data for site {site_index}, orbital {orbital} is invalid or all zeros.")
            return None
            
        data = {"Energy": self._get_energies()}
        
        col_key_base = f"site{site_index}_{orbital}"
        
        if self.is_spin:
            if Spin.up in densities:
                data[f"{col_key_base}_up"] = densities[Spin.up]
            if Spin.down in densities:
                data[f"{col_key_base}_down"] = densities[Spin.down]
        else:
            if Spin.up in densities:
                data[col_key_base] = densities[Spin.up]

        df = pd.DataFrame(data)
        if save:
            if filename is None:
                filename = f"site_{site_index}_{orbital}_dos.csv"
            self.save_to_csv(df, filename)
            self._log(f"[Info] Site {site_index} {orbital} DOS saved to {filename}")

        return df

    def parse_site_t2g_eg_dos(self, site_index: int, save: bool = True, filename: Optional[str] = None) -> pd.DataFrame:
        """解析指定位点 (site) 的 t2g 和 eg 轨道 DOS，并保存为 CSV 文件。"""
        
        if site_index < 0 or site_index >= len(self.structure.sites):
            raise ValueError("Invalid site index")
        
        site = self.structure[site_index]

        try:
            t2g_eg_dos = self.dos.get_site_t2g_eg_resolved_dos(site)
        except Exception as e:
            self._log(f"Warning: Failed to get t2g-eg DOS for site {site_index}. Error: {e}")
            return None
        
        # 简化有效性检查
        has_valid = any(
            self._is_valid_dos(dos_obj.densities) 
            for dos_obj in t2g_eg_dos.values()
        )
        if not has_valid:
            self._log(f"Warning: t2g-eg DOS for site {site_index} is all zeros.")
            return None

        data = {"Energy(eV)": self._get_energies()} 

        for label, dos_obj in t2g_eg_dos.items():
            densities = dos_obj.densities
            col_key_base = str(label).lower() # 使用 t2g 或 eg 作为基准列名

            if self.is_spin:
                if Spin.up in densities:
                    data[f"{col_key_base}_up"] = densities[Spin.up]
                if Spin.down in densities:
                    data[f"{col_key_base}_down"] = densities[Spin.down]
            else:
                if Spin.up in densities:
                    data[col_key_base] = densities[Spin.up]

        df = pd.DataFrame(data)
        if save:
            if filename is None:
                filename = f"site_{site_index}_t2g_eg_dos.csv"
            self.save_to_csv(df, filename)
            self._log(f"[Info] Site {site_index} t2g-eg DOS saved to {filename}")
        return df
    # ================= 4. 科学计算 (Statistics) =================

    def get_orbital_statistics(self, element: Optional[str] = None, site: Optional[int] = None, 
                            orbital: str = "d", erange: List[float] = [-10, 5]) -> Dict[str, float]:
        """
        计算轨道统计量: Center, Width, Skewness, Kurtosis, Filling。
        计算 UP/DOWN 值，并统一添加平均值 (Average) 组合值。
        """
        # ------------------ 1. 数据获取和基础计算 ------------------
        orb_type = self.ORBITAL_MAP.get(orbital)
        if not orb_type: raise ValueError(f"Unknown orbital: {orbital}")

        target_dos = None
        if site is not None:
            if not self._validate_site_index(site): return {}
            spd = self.dos.get_site_spd_dos(self.structure[site])
            target_dos = spd.get(orb_type)
            prefix = f"site{site}_{orbital}"
        elif element:
            spd = self.dos.get_element_spd_dos(Element(element))
            target_dos = spd.get(orb_type)
            prefix = f"{element}_{orbital}"
        else:
            raise ValueError("Provide 'element' or 'site'.")

        if not target_dos: return {}

        energies = self._get_energies()
        stats: Dict[str, float] = {}
        spins = [Spin.up, Spin.down] if self.is_spin else [Spin.up]

        # 临时存储 UP/DOWN 值，用于后续计算组合值
        temp_spin_data: Dict[str, Dict[str, float]] = {}

        for spin in spins:
            s_name = spin.name
            dens = target_dos.densities[spin]
            mask = (energies >= erange[0]) & (energies <= erange[1])
            e_sel = energies[mask]
            d_sel = dens[mask]

            if len(e_sel) < 5 or np.all(d_sel == 0): continue

            # [注意] 使用 np.trapezoid 
            filling = np.trapezoid(d_sel, e_sel)
            if filling < 1e-9: continue

            center = np.trapezoid(d_sel * e_sel, e_sel) / filling
            var = np.trapezoid(d_sel * (e_sel - center)**2, e_sel) / filling
            width = np.sqrt(var)
            skew = np.trapezoid(d_sel * (e_sel - center)**3, e_sel) / filling / (width**3 + 1e-9)
            kurt = np.trapezoid(d_sel * (e_sel - center)**4, e_sel) / filling / (width**4 + 1e-9)
            
            mask_occ = mask & (energies <= 0)
            occ_filling = np.trapezoid(dens[mask_occ], energies[mask_occ])
            
            # 存储到 stats 字典和临时数据中
            current_stats = {
                'center': float(center), 'width': float(width), 'skewness': float(skew),
                'kurtosis': float(kurt), 'filling': float(filling), 'occ_filling': float(occ_filling)
            }
            
            for key, val in current_stats.items():
                stats[f"{prefix}_{s_name}_{key}"] = val
            
            temp_spin_data[s_name] = current_stats
        
        # ------------------ 2. 统一计算平均值和格式化打印 ------------------

        # 仅在存在自旋数据时进行组合计算和打印
        if self.is_spin and 'up' in temp_spin_data and 'down' in temp_spin_data:
            
            # 统一计算平均值 (Average)
            STATISTICS_TO_AVERAGE = ['center', 'width', 'skewness', 'kurtosis', 'filling', 'occ_filling']
            combine_label = "AVG" # 统一标签
            precision = 6 # 打印精度
            
            print(f"\n--- Orbital Statistics ({prefix.upper()}) ---")
            
            for stat in STATISTICS_TO_AVERAGE:
                # 明确从临时数据中获取 UP/DOWN 值，确保正确性
                val_up = temp_spin_data['up'].get(stat, 0.0)
                val_down = temp_spin_data['down'].get(stat, 0.0)
                
                # 统一计算平均值 (UP + DOWN) / 2
                combined_val = (val_up + val_down) / 2

                # 存储组合值到返回字典
                stats[f"{prefix}_{combine_label.lower()}_{stat}"] = combined_val
                
                # 格式化打印块
                field_width = precision + 3
                print(f"[{stat.upper()}]")
                print(f"  UP:    {val_up:>{field_width}.{precision}f}")
                print(f"  DOWN:  {val_down:>{field_width}.{precision}f}")
                print(f"  {combine_label}: {combined_val:>{field_width}.{precision}f}")
                print("-" * 20)
            
            print("---------------------------------------")
        
        # ------------------ 3. 返回完整字典 ------------------
        return stats

    # ================= 5. 可视化 (Plotting) =================
    def _get_npg_colors(self):

        return [
            "#E64B35", # Red
            "#4DBBD5", # Blue
            "#00A087", # Green
            "#3C5488", # Dark Blue
            "#F39B7F", # Orange
            "#8491B4", # Grey-Blue
            "#91D1C2", # Light Green
            "#DC0000", # Dark Red
            "#7E6148", # Brown
            "#B09C85"  # Beige
        ]

    def plot_dos(self, 
                 elements: Optional[List[str]] = None, 
                 orbitals: Optional[List[str]] = None,
                 site_index: Optional[int] = None, 
                 xlim: List[float] = [-6, 6], 
                 ylim: Optional[List[float]] = None,
                 save_filename: str = "dos_plot_nature.pdf", # 推荐保存为 PDF 矢量图
                 fill: bool = True,
                 colors: Optional[List[str]] = None,
                 figsize: Tuple[float, float] = (5, 4)): # 适合单栏发表的尺寸
        """
        绘制 Nature/Science 风格的 DOS 图。
        特点：Arial字体、NPG配色、内刻度、半透明填充、上下自旋对称。
        """
        import matplotlib.ticker as ticker
        from itertools import cycle

        # 1. 定义发表级样式上下文
        nature_style = {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 10,
            'axes.linewidth': 1.0,
            'axes.titlesize': 10,
            'axes.labelsize': 12,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.top': True,
            'ytick.right': True,
            'legend.fontsize': 9,
            'legend.frameon': False,
            'lines.linewidth': 1.2,
            'savefig.bbox': 'tight',
            'savefig.dpi': 600
        }

        with plt.rc_context(nature_style):
            fig, ax = plt.subplots(figsize=figsize)
            energies = self._get_energies()

            # 准备颜色生成器
            color_palette = colors if colors else self._get_npg_colors()
            color_cycle = cycle(color_palette)

            if site_index is not None:
                site_dos = self.dos.get_site_dos(self.structure[site_index])
                tdos_up = site_dos.densities[Spin.up]
                tdos_down = -site_dos.densities[Spin.down] if self.is_spin else None
                label_prefix = f"Site {site_index}"
            else:
                tdos_up = self.dos.densities[Spin.up]
                tdos_down = -self.dos.densities[Spin.down] if self.is_spin else None
                label_prefix = "Total"

            # 绘制 Total (灰色填充，无边框或细边框)
            total_color = "#404040" # 深灰色
            ax.plot(energies, tdos_up, color=total_color, lw=0.8, label=label_prefix, alpha=0.8, zorder=1)
            ax.fill_between(energies, 0, tdos_up, color='gray', alpha=0.15, lw=0, zorder=0)
            
            if self.is_spin and tdos_down is not None:
                ax.plot(energies, tdos_down, color=total_color, lw=0.8, alpha=0.8, zorder=1)
                ax.fill_between(energies, 0, tdos_down, color='gray', alpha=0.15, lw=0, zorder=0)

            targets = elements if elements else []
            # 逻辑：如果没指定元素但指定了 site_index + orbitals，自动推导元素
            if not targets and site_index is not None and orbitals:
                targets = [self.structure[site_index].specie.symbol]

            for el_str in targets:
                el = Element(el_str)
                # 确定要扫描的原子位点
                sites_to_scan = [self.structure[site_index]] if site_index is not None else [s for s in self.structure.sites if s.specie == el]
                
                orb_list = orbitals if orbitals else []
                
                # Case 1: 仅指定元素 (画该元素总 DOS)
                if not orb_list: 
                    dens_up = np.zeros_like(energies)
                    dens_down = np.zeros_like(energies)
                    for s in sites_to_scan:
                        sd = self.dos.get_site_dos(s)
                        dens_up += sd.densities[Spin.up]
                        if self.is_spin: dens_down += -sd.densities[Spin.down]
                    
                    c = next(color_cycle)
                    lbl = f"{el_str}"
                    self._plot_single_trace(ax, energies, dens_up, dens_down, c, lbl, fill)

                # Case 2: 指定具体轨道
                else: 
                    for orb_str in orb_list:
                        dens_up = np.zeros_like(energies)
                        dens_down = np.zeros_like(energies)
                        found = False
                        
                        target_sub = self.SUB_ORBITAL_MAP.get(orb_str)
                        target_broad = self.ORBITAL_MAP.get(orb_str)

                        for s in sites_to_scan:
                            try:
                                if target_sub: 
                                    sd = self.dos.get_site_orbital_dos(s, target_sub)
                                elif target_broad:
                                    sd = self.dos.get_site_spd_dos(s).get(target_broad)
                                else: continue

                                if sd:
                                    dens_up += sd.densities[Spin.up]
                                    if self.is_spin: dens_down += -sd.densities[Spin.down]
                                    found = True
                            except: pass
                        
                        if found:
                            c = next(color_cycle)
                            lbl = self._format_label_latex(el_str, orb_str)
                            self._plot_single_trace(ax, energies, dens_up, dens_down, c, lbl, fill)
            
            # 0线处理
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7) # Fermi level
            ax.axhline(0, color='black', linewidth=0.5) # Zero DOS

            # 坐标范围
            ax.set_xlim(xlim)
            if ylim: 
                ax.set_ylim(ylim)
            else:
                mask = (energies >= xlim[0]) & (energies <= xlim[1])
                max_dos = np.max(tdos_up[mask])
                if self.is_spin:
                    min_dos = np.min(tdos_down[mask])
                    ax.set_ylim([min_dos * 1.1, max_dos * 1.1])
                else:
                    ax.set_ylim([0, max_dos * 1.1])

            # 标签
            ref_label = "E_F" if self._align_mode == 'fermi' else "E_{vac}"
            ax.set_xlabel(f"Energy vs. ${ref_label}$ (eV)")
            ax.set_ylabel("DOS (states/eV)")

            # 次级刻度
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

            # 图例 (尽量放在右上角，无框)
            ax.legend(loc='best', ncol=1, handlelength=1.5)

            plt.tight_layout()
            
            # 保存
            out_path = self.output_dir / save_filename
            plt.savefig(out_path) # rc_context 中已经定义了 dpi=600 和 tight
            self._log(f"[Plot] Saved publication-quality plot to {out_path}")
            plt.close(fig)

    def _plot_single_trace(self, ax, x, y_up, y_down, color, label, fill):
        """内部绘图辅助函数：绘制单条曲线（包含自旋），增加了错误处理"""        
        try:
            # Spin Up
            ax.plot(x, y_up, color=color, label=label, lw=1.2, zorder=2)
            if fill:
                ax.fill_between(x, 0, y_up, color=color, alpha=0.3, lw=0, zorder=2)
            
            # Spin Down (Mirror)
            if self.is_spin:
                ax.plot(x, y_down, color=color, lw=1.2, zorder=2) # 不加 label 防止图例重复
                if fill:
                    ax.fill_between(x, 0, y_down, color=color, alpha=0.3, lw=0, zorder=2)
                    
        except AttributeError as e:
            self._log(
                f"[Critical Error] Failed to plot DOS trace. The object 'ax' is of type {type(ax)} and lacks the required 'plot' or 'fill_between' attribute. "
                f"Please ensure 'ax' is a Matplotlib Axes object. Original error: {e}"
            )
        except Exception as e:
            self._log(f"[Critical Error] An unexpected error occurred during plotting: {e}")

    def _format_label_latex(self, element: str, orbital: str) -> str:
        """ LaTeX 标签"""
        
        if not isinstance(element, str) or not isinstance(orbital, str):
            self._log(f"[Warning] Invalid type passed to _format_label_latex: element={type(element)}, orbital={type(orbital)}")
            return f"{str(element)}-{str(orbital)}"

        if len(orbital) == 1:
            return f"{element} ${orbital}$"
        
        # 2. 处理 d_xy 等子轨道 (包含下划线)
        if "_" in orbital:
            # 使用字符串的 split 方法
            parts = orbital.split("_", 1)
            base = parts[0]
            sub = parts[1] if len(parts) > 1 else "" # 确保即使没有子串也不会报错
            
            sub = sub.replace("x2-y2", "x^2-y^2")
            sub = sub.replace("x2", "x^2")
            sub = sub.replace("y2", "y^2")
            sub = sub.replace("z2", "z^2")
            
            return f"{element} ${base}_{{{sub}}}$"
        
        return f"{element}-{orbital}"
    
class CohpAnalysis(VaspAnalysis):
    """用于分析LOBSTER COHPCAR文件中的分轨道类"""
    def __init__(
        self,
        work_dir: Optional[Union[str, Path]] = None,
        cohpcar_path: Optional[Union[str, Path]] = None,
        icohplist_path: Optional[Union[str, Path]] = None,
        save_data: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(
            work_dir=work_dir,
            save_data=save_data,
            output_dir=output_dir,
        )
        self.cohpcar_path: Optional[Path] = Path(cohpcar_path) if cohpcar_path else None
        self.icohplist_path: Optional[Path] = Path(icohplist_path) if icohplist_path else None
        if not self.cohpcar_path and self.work_dir:
            self.cohpcar_path = self.work_dir / "COHPCAR.lobster"
        if not self.icohplist_path and self.work_dir:
            self.icohplist_path = self.work_dir / "ICOHPLIST.lobster"
        self.cohp_obj: Optional[Cohpcar] = None
        self.icohplist_obj: Optional[Icohplist] = None
        self.orb_res_cohp: Optional[Dict[str, Dict[str, Any]]] = None
        self.available_bond_labels: List[str] = []

        # 4. 加载 LOBSTER 数据
        self._load_lobster_data()

    def _load_lobster_data(self):
        """加载 LOBSTER COHPCAR 文件"""
        if self.cohpcar_path and self.cohpcar_path.exists():
                    try:
                        self.cohp_obj = Cohpcar(filename=self.cohpcar_path)
                        self.orb_res_cohp = self.cohp_obj.orb_res_cohp
                        
                        if not self.orb_res_cohp:
                            self._log("[Warning] COHPCAR 文件中不包含分轨道的 COHP 数据。")
                            return
                        
                        # 获取并排序所有可用的键标签
                        self.available_bond_labels = sorted(
                            list(self.orb_res_cohp.keys()), 
                            key=lambda x: int(x) if x.isdigit() else x
                        )
                        self._log(f"[Info] 成功加载 COHPCAR.lobster 数据。共 {len(self.available_bond_labels)} 个键。")

                    except Exception as e:
                        self._log(f"[Error] COHPCAR.lobster 解析失败: {e}")
                        
        else:
            self._log(f"[Warning] 找不到 COHPCAR.lobster 文件: {self.cohpcar_path}")
        
        if self.icohplist_path and self.icohplist_path.exists():
            try:
                self.icohplist_obj = Icohplist(filename=self.icohplist_path)
                self._log(f"[Info] 成功加载 ICOHPLIST.lobster 数据")
            except Exception as e:
                self._log(f"[Error] ICOHPLIST.lobster 解析失败: {e}")
        else:
            self._log(f"[Warning] 找不到 ICOHPLIST.lobster 文件: {self.icohplist_path}")              
    
    def _get_default_labels(self, bond_label: Optional[str], orbital_label: Optional[str]) -> Tuple[str, str]:
        """根据用户输入确定最终的键标签和轨道标签，若未指定则选择第一个。"""
        if not self.orb_res_cohp or not self.available_bond_labels:
            raise ValueError("LOBSTER 数据未加载或不包含轨道分辨信息。")

        # 1. 确定键标签 (Bond Label)
        if bond_label is None:
            final_bond_label = self.available_bond_labels[0]
        elif bond_label not in self.available_bond_labels:
            raise ValueError(f"键 '{bond_label}' 不存在。请选择从 {self.available_bond_labels} 中选择。")
        else:
            final_bond_label = bond_label
        # 2. 确定轨道标签 (Orbital Label)
        available_orbs = list(self.orb_res_cohp[final_bond_label].keys())
        if not available_orbs:
            raise ValueError(f"键 '{final_bond_label}' 下没有找到任何轨道相互作用。")
            
        if orbital_label is None:
            final_orbital_label = available_orbs[0]
        elif orbital_label not in available_orbs:
            raise ValueError(f"轨道 '{orbital_label}' 不存在于键 '{final_bond_label}'。请选择从 {available_orbs} 中选择。")
        else:
            final_orbital_label = orbital_label
        
        return final_bond_label, final_orbital_label
    
    def get_orbital_bond_info(self, bond_label: str):
        """获取特定键的所有轨道标签"""
        if bond_label in self.orb_res_cohp:
            self._log(f"[Info] 键 '{bond_label}'轨道相互作用: {list(self.orb_res_cohp[bond_label].keys())}")
            return list(self.orb_res_cohp[bond_label].keys())
        return []
    
    def get_orbital_data(self, bond_label: str, orbital_label: str) -> Optional[Dict[str, Any]]:
        """获取特定键和特定轨道的分 COHP 和 ICOHP 数组 (内部方法)。"""
        if (self.orb_res_cohp and 
            bond_label in self.orb_res_cohp and 
            orbital_label in self.orb_res_cohp[bond_label]):
            return self.orb_res_cohp[bond_label][orbital_label]
        return None

    def save_orbital_data_to_csv(self, 
                                    bond_label: Optional[str] = None, 
                                    orbital_label: Optional[str] = None, 
                                    data_type: str = "COHP", 
                                    spin: Spin = Spin.up,
                                    output_filename: Optional[str] = None):
            """
            将特定轨道相互作用的 COHP/ICOHP 数据保存为 CSV 文件。
            CSV 格式为: 两列，第一列是能量 (E-Ef)，第二列是 COHP/ICOHP 值。
            
            Args:
                bond_label (Optional[str]): 键的编号，例如 '1'。
                orbital_label (Optional[str]): 分轨道的标签，例如 '6s-2s'。
                data_type (str): 要保存的数据类型, 必须是 'COHP' 或 'ICOHP'。
                spin (Spin): 要保存的自旋分量 (Spin.up 或 Spin.down)。
                output_filename (Optional[str]): 输出 CSV 文件的名称。
            """
            if data_type not in ["COHP", "ICOHP"]:
                raise ValueError("data_type 必须是 'COHP' 或 'ICOHP'。")

            if self.cohp_obj is not None and spin == Spin.down and not self.cohp_obj.is_spin_polarized:
                raise ValueError(f"当前 COHPCAR.lobster 文件未启用自旋极化，{spin.name} 应为up 。")
            # 确定最终的键和轨道标签
            final_bond_label, final_orbital_label = self._get_default_labels(bond_label, orbital_label)
            
            orb_data1 = self.orb_res_cohp[final_bond_label][final_orbital_label]
            sites = orb_data1.get("sites")
            site_info_short = "N/A"
            if sites:
                site_info_short = f"{sites[0]}_{sites[1]}"
            self._log(f"--- 准备 CSV 导出 --- 键: '{final_bond_label}', 轨道: '{final_orbital_label}', 类型: '{data_type}', 自旋: {spin.name}, 位点{site_info_short}")
            # 1. 获取数据
            orb_data = self.get_orbital_data(final_bond_label, final_orbital_label)
            data_dict = orb_data.get(data_type)
            data_values = data_dict.get(spin)
            
            if data_values is None:
                self._log(f"[Warning] 找不到自旋 {spin.name} 的 {data_type} 数据。")
                return

            # 2. 准备 Pandas DataFrame (两列格式) pymatgen已经转换过了
            energy_relative_to_fermi = self.cohp_obj.energies
            
            column_name = f'{data_type} ({final_orbital_label}, {spin.name})'
            
            data_to_save = {
                'Energy (E-Ef)': energy_relative_to_fermi,
                column_name: data_values
            }
            # 不使用转置，直接创建 DataFrame
            df = pd.DataFrame(data_to_save)

            # 3. 导出 CSV
            if output_filename is None:
                # 自动生成文件名，替换特殊字符确保文件名有效
                filename_base = f"Bond{final_bond_label}_{site_info_short}_{final_orbital_label}_{data_type}_{spin.name}".replace("/", "_").replace("(", "").replace(")", "").replace(" ", "")
                output_filename = f"{filename_base}.csv"
            
            # 导出时，设置 index=False 以避免将 Pandas 索引作为额外的列写入
            self.save_to_csv(df=df, filename=output_filename) 
            
    def plot_orbital_cohp(self, 
                          bond_label: Optional[str] = None, 
                          orbital_label: Optional[str] = None, 
                          data_type: str = "COHP", 
                          spin: Spin = Spin.up,
                          xlim: Tuple[float, float] = (-20, 20)):
        """
        绘制特定轨道相互作用的 COHP/ICOHP 曲线。
        
        Args:
            bond_label (Optional[str]): 键的编号，例如 '1'。
            orbital_label (Optional[str]): 分轨道的标签，例如 '6s-2s'。
            data_type (str): 要绘制的数据类型, 必须是 'COHP' 或 'ICOHP'。
            spin (Spin): 要绘制的自旋分量 (Spin.up 或 Spin.down)。
            xlim (Tuple[float, float]): 绘图的 x 轴范围 (E - E_f)。
        """
        if data_type not in ["COHP", "ICOHP"]:
            raise ValueError("data_type 必须是 'COHP' 或 'ICOHP'。")

        # 确定最终的键和轨道标签
        final_bond_label, final_orbital_label = self._get_default_labels(bond_label, orbital_label)
        self._log(f"--- 准备绘图 --- 键: '{final_bond_label}', 轨道: '{final_orbital_label}', 类型: '{data_type}', 自旋: {spin.name}")

        orb_data1 = self.orb_res_cohp[final_bond_label][final_orbital_label]
        sites = orb_data1.get("sites")
        site_info_short = "N/A"
        if sites:
            site_info_short = f"{sites[0]}_{sites[1]}"

        # 1. 获取数据
        orb_data = self.get_orbital_data(final_bond_label, final_orbital_label)
        data_dict = orb_data.get(data_type)
        data_values = data_dict.get(spin)
        
        if data_values is None:
            self._log(f"[Warning] 找不到自旋 {spin.name} 的 {data_type} 数据。")
            return

        # 2. 准备绘图数据
        energy_relative_to_fermi = self.cohp_obj.energies
        
        plt.figure(figsize=(10, 6))

        # 绘制费米能级线 (E - E_Fermi = 0)
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.0, label="$E_f$")
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 3. 绘制曲线
        plot_label = f'{data_type}: {final_orbital_label} ({spin.name})'
        
        if data_type == "COHP":
            # 绘制 -COHP 曲线 (COHP取负，使键合贡献向上)
            plt.plot(energy_relative_to_fermi, -data_values, label=plot_label, linewidth=2.0)
            plt.ylabel("$-COHP$ (a.u. or eV$^{-1}$/cell)", fontsize=12)
        else: # ICOHP
            plt.plot(energy_relative_to_fermi, data_values, label=plot_label, linewidth=2.0)
            plt.ylabel("$ICOHP$ (eV)", fontsize=12)

        # 4. 设置图表属性
        plt.xlabel("Energy Relative to $E_f$ (eV)", fontsize=12)
        plt.title(f"{data_type} for Bond {final_bond_label}_{site_info_short}", fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xlim(xlim) 
        plt.show()

    def save_total_bond_data_to_csv(self, 
                                     bond_label: Optional[str] = None, 
                                     data_type: str = "COHP", 
                                     spin: Spin = Spin.up,
                                     output_filename: Optional[str] = None):
        """
        将特定键的总 COHP/ICOHP 数据保存为 CSV 文件 (非轨道分辨)。
        CSV 格式为: 两列，第一列是能量 (E-Ef)，第二列是 COHP/ICOHP 值。

        Args:
            bond_label (Optional[str]): 键的编号，例如 '1'。
            data_type (str): 要保存的数据类型, 必须是 'COHP' 或 'ICOHP'。
            spin (Spin): 要保存的自旋分量 (Spin.up 或 Spin.down)。
            output_filename (Optional[str]): 输出 CSV 文件的名称。
        """
        if data_type not in ["COHP", "ICOHP"]:
            raise ValueError("data_type 必须是 'COHP' 或 'ICOHP'。")

        # 确定最终的键标签 (使用 None 确保只进行键标签的检查和默认值选择)
        final_bond_label, _ = self._get_default_labels(bond_label, None)
        
        # 1. 获取总键数据
        total_bond_data = self.cohp_obj.cohp_data.get(final_bond_label)
        
        # 2. 提取 Sites 信息 (从 total_bond_data 中获取)
        sites = total_bond_data.get("sites")
        site_info_short = "N/A"
        if sites:
            # 提取 1-based index
            index1 = sites[0]
            index2 = sites[1]
            site_info_short = f"{index1}-{index2}"
            
        self._log(f"--- 准备 CSV 导出 (总键) --- 键: '{final_bond_label}', 类型: '{data_type}', 自旋: {spin.name}, 位点{site_info_short}")

        # 3. 获取 COHP/ICOHP 数据
        data_dict = total_bond_data.get(data_type)
        data_values = data_dict.get(spin)
        
        if data_values is None:
            self._log(f"[Warning] 找不到键 '{final_bond_label}' 自旋 {spin.name} 的 {data_type} 数据。")
            return

        # 4. 准备 Pandas DataFrame (两列格式)
        energy_relative_to_fermi = self.cohp_obj.energies
        
        column_name = f'Total {data_type} (Sites {site_info_short}, {spin.name})'
        
        data_to_save = {
            'Energy (E-Ef)': energy_relative_to_fermi,
            column_name: data_values
        }

        df = pd.DataFrame(data_to_save)

        # 5. 导出 CSV
        if output_filename is None:
            sanitized_site_info = site_info_short.replace("-", "_")
            filename_base = f"TotalBond{final_bond_label}_Sites{sanitized_site_info}_{data_type}_{spin.name}".replace("/", "_").replace("(", "").replace(")", "").replace(" ", "")
            output_filename = f"{filename_base}.csv"
    
        self.save_to_csv(df=df, filename=output_filename)
  
    def plot_total_bond_cohp(self, 
                             bond_label: Optional[str] = None, 
                             data_type: str = "COHP", 
                             spin: Spin = Spin.up,
                             xlim: Tuple[float, float] = (-10, 5)):
        """
        绘制特定键的总 COHP/ICOHP 曲线。
        
        Args:
            bond_label (Optional[str]): 键的编号，例如 '1'。如果为 None，则选择第一个键。
            data_type (str): 要绘制的数据类型, 必须是 'COHP' 或 'ICOHP'。
            spin (Spin): 要绘制的自旋分量 (Spin.up 或 Spin.down)。
            xlim (Tuple[float, float]): 绘图的 x 轴范围 (E - E_f)。
        """
        if data_type not in ["COHP", "ICOHP"]:
            raise ValueError("data_type 必须是 'COHP' 或 'ICOHP'。")

        # 1. 确定最终键标签
        final_bond_label, _ = self._get_default_labels(bond_label, None)
        self._log(f"--- 准备绘图 (总键) --- 键: '{final_bond_label}', 类型: '{data_type}', 自旋: {spin.name}")

        # 2. 获取总键数据字典
        total_bond_data = self.cohp_obj.cohp_data.get(final_bond_label)
        if total_bond_data is None:
            self._log(f"[Error] 找不到键 '{final_bond_label}' 的总键数据。")
            return
            
        # 3. 提取 Sites 和 COHP/ICOHP 数据
        sites = total_bond_data.get("sites")
        data_values = total_bond_data.get(data_type).get(spin)
        
        if data_values is None:
            self._log(f"[Warning] 找不到键 '{final_bond_label}' 自旋 {spin.name} 的 {data_type} 数据。")
            return

        # 4. 提取简洁的键原子序号信息 (Index1-Index2)
        site_info_short = "N/A"
        if sites:
            index1 = sites[0]
            index2 = sites[1]
            site_info_short = f"{index1}-{index2}"
            
        # 5. 准备绘图
        energy_relative_to_fermi = self.cohp_obj.energies
        
        plt.figure(figsize=(10, 6))

        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.0, label="$E_f$")
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 6. 绘制曲线和设置图表属性
        plot_label = f'Total {data_type} ({site_info_short}, {spin.name})'
        
        if data_type == "COHP":
            # 绘制 -COHP 曲线
            plt.plot(energy_relative_to_fermi, -data_values, label=plot_label, linewidth=2.0)
            plt.ylabel("$-COHP$ (a.u. or eV$^{-1}$/cell)", fontsize=12)
        else:
            plt.plot(energy_relative_to_fermi, data_values, label=plot_label, linewidth=2.0)
            plt.ylabel("$ICOHP$ (eV)", fontsize=12)

        plt.xlabel("Energy Relative to $E_f$ (eV)", fontsize=12)
        
        title = f"Total {data_type} for Bond {final_bond_label} (Sites: {site_info_short})"
        plt.title(title, fontsize=14)
        
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xlim(xlim) 
        plt.show()

    def get_icohp_summary(self, bond_label: Optional[str] = None):
        """
        获取 ICOHPLIST.lobster 文件中的键-轨道对信息。
        Returns:
            Dict[str, Dict[str, Any]]: 键-轨道对的 ICOHPLIST 数据。
        """
        if not self.icohplist_obj:
            self._log("[Error] ICOHPLIST.lobster 文件未加载。请先加载 ICOHPLIST.lobster 文件。")
            return None
        try:
            final_bond_label, _ = self._get_default_labels(bond_label, None)
        except ValueError as e:
            self._log(f"[Error] 无效的键标签 '{bond_label}': {e}")
            return None
        
        icohp_data = self.icohplist_obj.icohplist.get(final_bond_label)
        
        if icohp_data is None:
            raise ValueError(f"[Error] 找不到键 '{final_bond_label}' 的 ICOHPLIST 数据。")
        icohp_dict = icohp_data.get("icohp")
        summary = {
            "label": final_bond_label,
            "icohp_average": icohp_dict,  # 这是一个 {Spin.up: float, Spin.down: float} 字典
            "length": icohp_data.get("length"),
            "number_of_bonds": icohp_data.get("number_of_bonds"),
            "is_spin_polarized": self.icohplist_obj.is_spin_polarized
        }
        self._log(f"  平均 ICOHP (Spin Up): {summary['icohp_average'].get(Spin.up):.4f} eV")
        if summary['is_spin_polarized'] and summary['icohp_average'].get(Spin.down) is not None:
            self._log(f"  平均 ICOHP (Spin Down): {summary['icohp_average'].get(Spin.down):.4f} eV")
            
        return summary