# -*- coding: utf-8 -*-
"""
Simplified VASP Analysis Framework
"""

import json
import math
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pymatgen.io.vasp import Outcar, Oszicar, Poscar
from pymatgen.io.lobster import Icohplist
from .parse import FastCohpcar as Cohpcar
from .parse import DoscarParser
from pymatgen.electronic_structure.core import Spin

# ============================================================
# 1. 数据模型
# ============================================================
@dataclass
class ApiResponse:
    """统一API返回格式"""
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
    def error(cls, message: str, code: int = 500, data: Optional[dict] = None) -> "ApiResponse":
        return cls(success=False, code=code, message=message, data=data or {})

    @classmethod
    def not_found(cls, message:str) -> "ApiResponse":
        return cls(success=False, code=404, message=message, data={})
    
    @classmethod
    def bad_request(cls, message:str) -> "ApiResponse":
        return cls(success=False, code=400, message=message, data={})

# ============================================================
# 2. DOS 分析
# ============================================================
class DosAnalysis:
    """DOS分析器 - 保持所有计算方法不变"""
    
    def __init__(self, work_dir: Union[str, Path], save_data: bool = False):
        self.work_dir = Path(work_dir).resolve()
        self.save_data = save_data
        self.parser = DoscarParser(self.work_dir / "DOSCAR")
        self._elements = []
        self._parse_structure()
        
    
    def _parse_structure(self):
        """解析结构，获取元素列表，用于按元素提取"""
        poscar_path = self.work_dir / "POSCAR"
        outcar_path = self.work_dir / "OUTCAR"
        
        if poscar_path.exists():
            structure = Poscar.from_file(str(poscar_path)).structure
            self._elements = [site.specie.symbol for site in structure]
        elif outcar_path.exists():
            try:
                outcar = Outcar(str(outcar_path))
                if outcar.structure is not None:
                    self._elements = [site.specie.symbol for site in outcar.structure]
            except Exception:
                pass
    
    def get_structure_info(self) -> ApiResponse:
        """供前端 loadStructure 调用的接口"""
        if not self._elements:
            return ApiResponse.error("Structure file (POSCAR) not found.")
        
        unique_elements = list(dict.fromkeys(self._elements)) # 保持顺序去重
        return ApiResponse.ok({
            "totalAtoms": len(self._elements),
            "elements": unique_elements
        })
    
    def analyze(self, curves: List[Dict], erange: List[float] = [-10, 5], 
                show_tdos: bool = False, **kwargs) -> ApiResponse:
        """
        执行多曲线 DOS 分析 (无损全量版)
        """
        try:
            energies = self.parser.energies - self.parser.efermi
            ispin = self.parser.ispin
            
            # 1. 确定能量截取区间 (直接使用真实能量网格)
            mask = (energies >= erange[0]) & (energies <= erange[1])
            raw_e = energies[mask]
            
            result_data = {
                "energy": raw_e.tolist(),  # 直接返回真实的能量数组
                "ispin": ispin,
                "curves": [],
                "tdos": None
            }
            
            # 2. 处理 TDOS (无损截取)
            if show_tdos:
                tdos_raw = self.parser.tdos[mask]
                tdos_data = {"up": tdos_raw[:, 1].tolist()}
                if ispin == 2:
                    tdos_data["down"] = tdos_raw[:, 2].tolist()
                result_data["tdos"] = tdos_data

            # 3. 处理每一条曲线
            for curve_req in curves:
                mode = curve_req.get("mode", "element")
                orbital = curve_req.get("orbital", "d")
                
                target_indices = []
                if mode == "element":
                    target_el = curve_req.get("element")
                    target_indices = [i for i, el in enumerate(self._elements) if el == target_el]
                elif mode == "site":
                    site_idx = int(curve_req.get("site", 1)) - 1
                    if 0 <= site_idx < len(self._elements):
                        target_indices = [site_idx]
                
                if not target_indices:
                    continue 
                
                # 提取并累加 PDOS
                pdos_up_raw = np.zeros(len(raw_e))
                pdos_down_raw = np.zeros(len(raw_e)) if ispin == 2 else None
                
                for idx in target_indices:
                    atom_pdos = self.parser.pdos[idx][mask]
                    cols = self._get_orbital_columns(orbital, ispin)
                    
                    pdos_up_raw += np.sum(atom_pdos[:, cols["up"]], axis=1)
                    if ispin == 2 and cols["down"]:
                        pdos_down_raw += np.sum(atom_pdos[:, cols["down"]], axis=1)

                # 计算统计描述符
                stats = self._calculate_descriptors(raw_e, pdos_up_raw)
                
                if ispin == 2 and pdos_down_raw is not None:
                    stats_down = self._calculate_descriptors(raw_e, pdos_down_raw)
                    for k, v in stats_down.items():
                        stats[f"{k}_down"] = v
                    mask_occ = raw_e <= 0
                    if np.any(mask_occ):
                        mag_moment = np.trapz(pdos_up_raw[mask_occ] - pdos_down_raw[mask_occ], raw_e[mask_occ])
                        stats["magnetic_moment"] = round(float(mag_moment), 4)
                    else:
                        stats["magnetic_moment"] = 0.0    
                # 组装结果
                curve_result = {
                    "id": curve_req.get("id", "unknown"),
                    "label": curve_req.get("label", f"{mode}-{orbital}"),
                    "color": curve_req.get("color", "#333"),
                    "dos_up": pdos_up_raw.tolist(),
                    "stats": stats
                }
                
                if ispin == 2:
                    curve_result["dos_down"] = pdos_down_raw.tolist()
                    
                result_data["curves"].append(curve_result)

            return ApiResponse.ok(data=result_data, message="DOS analysis complete")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ApiResponse.error(f"DOS analysis failed: {e}")

    def _get_orbital_columns(self, orbital: str, ispin: int) -> Dict[str, List[int]]:
        """
        获取指定轨道在 DOSCAR PDOS 矩阵中的列索引 (0-based, 第0列是能量)
        兼容 LORBIT=11 的标准 VASP 输出
        """
        # 定义基础索引 (假设 ISPIN=1)
        # s=1, py=2, pz=3, px=4, dxy=5, dyz=6, dz2=7, dxz=8, dx2-y2=9
        orb_map_1 = {
            "s": [1],
            "p_y": [2], "p_z": [3], "p_x": [4], "p": [2, 3, 4],
            "d_xy": [5], "d_yz": [6], "d_z2": [7], "d_xz": [8], "d_x2-y2": [9], "d": [5, 6, 7, 8, 9]
        }
        
        if ispin == 1:
            return {"up": orb_map_1.get(orbital, [1]), "down": []}
        else:
            # ISPIN=2 时，每一项分裂为 up 和 down
            # s_up=1, s_dn=2, py_up=3, py_dn=4 ...
            base_cols = orb_map_1.get(orbital, [1])
            up_cols = [c * 2 - 1 for c in base_cols]
            down_cols = [c * 2 for c in base_cols]
            return {"up": up_cols, "down": down_cols}

    def _calculate_descriptors(self, energy: np.ndarray, dos: np.ndarray) -> Dict[str, float]:
        """计算催化高级描述符 (矩分析)"""
        try:
            # 保证 DOS 为正值用于积分权重
            dos_abs = np.abs(dos)
            
            # 0阶矩 (总面积)
            m0 = np.trapz(dos_abs, energy)
            if m0 < 1e-6:
                return {"center": 0, "width": 0, "skewness": 0, "kurtosis": 0, "filling": 0}
            
            # 1阶矩 (d带中心)
            center = np.trapz(energy * dos_abs, energy) / m0
            
            # 2阶矩 (方差) -> 宽度
            variance = np.trapz(((energy - center) ** 2) * dos_abs, energy) / m0
            width = np.sqrt(variance) if variance > 0 else 0
            
            # 3阶矩 -> 偏度 (Skewness)
            m3 = np.trapz(((energy - center) ** 3) * dos_abs, energy) / m0
            skewness = m3 / (width ** 3) if width > 0 else 0
            
            # 4阶矩 -> 峰度 (Kurtosis)
            m4 = np.trapz(((energy - center) ** 4) * dos_abs, energy) / m0
            kurtosis = m4 / (width ** 4) if width > 0 else 0
            
            # 计算 Filling (费米能级 0 eV 以下的电子数占比)
            mask_occ = energy <= 0
            filled_states = np.trapz(dos_abs[mask_occ], energy[mask_occ]) if np.any(mask_occ) else 0
            filling = filled_states / m0
            
            return {
                "center": round(float(center), 4),
                "width": round(float(width), 4),
                "skewness": round(float(skewness), 4),
                "kurtosis": round(float(kurtosis), 4),
                "filling": round(float(filling), 4),
                "filled_states": round(float(filled_states), 4) 
            }
        except Exception:
            return {"center": 0, "width": 0, "skewness": 0, "kurtosis": 0, "filling": 0, "filled_states": 0}

# ============================================================
# 3. 结构优化分析
# ============================================================
class RelaxAnalysis:
    """结构优化分析器 - 极速单遍解析版"""
    
    def __init__(self, work_dir: Union[str, Path], save_data: bool = False):
        self.work_dir = Path(work_dir).resolve()
        self.save_data = save_data
        self._fast_outcar_data: Optional[Dict[str, Any]] = None
        
        if not self.work_dir.exists():
            raise FileNotFoundError(f"Work directory not found: {self.work_dir}")
    
    def analyze(self, get_site_mag: bool = False, **kwargs) -> ApiResponse:
        """
        执行结构优化分析
        :param get_site_mag: 是否解析每个原子的分波磁矩
        """
        try:
            # 1. 极速解析 OUTCAR
            outcar_data = self._parse_outcar_fast(get_site_mag=get_site_mag)
            
            # 2. 解析 OSZICAR 获取能量和步数历史
            energy_history = []
            de_history = []
            oszicar_path = self.work_dir / "OSZICAR"
            
            if oszicar_path.exists():
                try:
                    osz = Oszicar(str(oszicar_path))
                    if osz.ionic_steps:
                        energy_history = [float(step.get("E0", 0)) for step in osz.ionic_steps]
                        de_history = [float(step.get("dE", 0)) for step in osz.ionic_steps]
                except Exception as e:
                    pass # 忽略 OSZICAR 解析错误，或者记录日志
            
            final_energy = energy_history[-1] if energy_history else None
            last_dE = de_history[-1] if de_history else None
            
            force_history = outcar_data.get("force_history", [])
            final_force = force_history[-1] if force_history else None
            
            struct_analyzer = StructureAnalysis(self.work_dir)
            initial_structure = struct_analyzer.get_structure_data("POSCAR")
            final_structure = struct_analyzer.get_structure_data("CONTCAR")
            
            # 3. 组装前端 UI 需要的数据结构
            data = {
                "converged": outcar_data["converged"],
                "final_energy_eV": final_energy,
                "fermi_level_eV": outcar_data["efermi"],
                "ionic_steps": len(energy_history),
                "total_electrons": outcar_data["nelect"],
                "total_magnetization": outcar_data["total_mag"] if outcar_data["total_mag"] is not None else 0.0,
                "energy_history": energy_history, #能量收敛历史
                "de_history": de_history, #能量差历史
                "force_history": force_history, #力收敛历史
                "final_force": final_force, #最终力
                "site_magnetization": outcar_data["site_mag"],
                "warnings": self._collect_warnings(outcar_data["converged"], last_dE, final_force),
                
                "initial_structure": initial_structure,
                "final_structure": final_structure
            }
            
            message = "Optimization converged" if outcar_data["converged"] else "Optimization NOT converged"
            return ApiResponse.ok(data=data, message=message)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ApiResponse.error(f"Relax analysis failed: {e}")

    def _parse_outcar_fast(self, get_site_mag: bool) -> Dict[str, Any]:
        """核心加速逻辑：单遍遍历 OUTCAR，摒弃 pymatgen 的全量慢速解析"""
        import re
        
        if self._fast_outcar_data is not None:
            return self._fast_outcar_data

        data = {
            "converged": False,
            "efermi": None,
            "nelect": None,
            "total_mag": None,
            "site_mag": [],
            "force_history": []
        }

        outcar_path = self.work_dir / "OUTCAR"
        if not outcar_path.exists():
            self._fast_outcar_data = data
            return data

        in_mag_block = False
        mag_lines = []
        in_force_block = False
        current_max_force = 0.0
        # 预编译正则表达式以提高匹配速度
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
                # 4.5 力收敛历史 (每次 ionic step 的最后一个力值)
                elif "TOTAL-FORCE (eV/Angst)" in line:
                    in_force_block = True
                    current_max_force = 0.0
                elif in_force_block:
                    if "total drift" in line:
                        in_force_block = False
                        data["force_history"].append(current_max_force)
                    else:
                        parts = line.split()
                        if len(parts) == 6:
                            try:
                                fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                                # 计算受力向量的模长 (F = sqrt(fx^2 + fy^2 + fz^2))
                                force_norm = math.sqrt(fx**2 + fy**2 + fz**2)
                                if force_norm > current_max_force:
                                    current_max_force = force_norm
                            except ValueError:
                                pass                                         
                # 5. 分波磁矩块 (仅在 get_site_mag=True 时才进行字符串匹配和收集)
                elif get_site_mag and mag_block_pattern.match(line):
                    in_mag_block = True
                    mag_lines = []  # 遇到新的块，清空旧数据（只保留最后一步的磁矩）
                elif in_mag_block:
                    if tot_line_pattern.match(line):
                        in_mag_block = False  # 遇到 tot 行，完美结束当前块
                    elif mag_line_pattern.match(line):
                        mag_lines.append(line.strip()) # 纯数据行，收集

        # 处理最后一次捕获到的分波磁矩数据
        if get_site_mag and mag_lines:
            site_mag = []
            for ln in mag_lines:
                parts = ln.split()
                if len(parts) >= 5:
                    try:
                        # 保持 atom_index 为 1-indexed，以匹配前端 UI 显示
                        mag_dict = {
                            "atom_index": int(parts[0]),
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

    def _collect_warnings(self, converged: bool, last_dE: Optional[float], final_force: Optional[float] ) -> List[str]:
        """收集警告信息"""
        warns = []
        if not converged:
            warns.append("Calculation did NOT converge.")
        if last_dE is not None and abs(last_dE) > 1e-3 and converged:
            warns.append(f"Last dE = {last_dE:.2e} eV > 1e-3 eV, convergence marginal.")
        if final_force is not None and final_force > 1.0 and converged:
            warns.append(f"Max force on atoms is {final_force:.3f} eV/Å > 0.1 eV/Å. Check EDIFFG setting.")        
        return warns

# ============================================================
# 4. 结构信息分析器
# ============================================================
class StructureAnalysis:
    """结构信息快速提取"""
    
    def __init__(self, work_dir: Union[str, Path], **kwargs):
        self.work_dir = Path(work_dir).resolve()
        
        if not self.work_dir.exists():
            raise FileNotFoundError(f"Work directory not found: {self.work_dir}")
    
    def get_structure_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """获取指定结构文件(POSCAR/CONTCAR)的详细信息和原生文本供前端渲染"""
        file_path = self.work_dir / filename
        if not file_path.exists() or file_path.stat().st_size == 0:
            return None
            
        try:
            from pymatgen.core import Structure
            struct = Structure.from_file(str(file_path))
            
            # 读取原生 VASP 文本
            with open(file_path, 'r', encoding='utf-8') as f:
                vasp_text = f.read()
                
            return {
                "formula": struct.composition.reduced_formula,
                "elements": list(dict.fromkeys([site.specie.symbol for site in struct])),
                "totalAtoms": len(struct),
                "volume": float(struct.volume),
                "lattice": {
                    "a": float(struct.lattice.a),
                    "b": float(struct.lattice.b),
                    "c": float(struct.lattice.c),
                    "alpha": float(struct.lattice.alpha),
                    "beta": float(struct.lattice.beta),
                    "gamma": float(struct.lattice.gamma),
                },
                "vasp_text": vasp_text
            }
        except Exception:
            return None

    def get_info(self) -> ApiResponse:
        """获取结构基本信息"""
        try:
            data = self.get_structure_data("CONTCAR") or self.get_structure_data("POSCAR")
            if data is None:
                return ApiResponse.error("No valid structure file found")
            return ApiResponse.ok(data=data, message="Structure info extracted successfully")
        except Exception as e:
            return ApiResponse.error(f"Failed to parse structure: {e}")

# ============================================================
# 5. COHP分析
# ============================================================
class CohpAnalysis:
    """
    COHP / ICOHP 成键分析器（基于 LOBSTER 输出）。
    """
    def __init__(self, work_dir: Union[str, Path], save_data: bool = False):
        self.work_dir = Path(work_dir).resolve()
        self.save_data = save_data
        self._cohpcar: Optional[Any] = None
        self._icohplist: Optional[Any] = None

    def _log(self, msg: str, level: str = "warning"):
        print(f"[{level.upper()}] CohpAnalysis: {msg}")

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        if self.save_data:
            out_path = self.work_dir / filename
            df.to_csv(out_path, index=False)

    @property
    def cohpcar(self) -> Optional[Any]:
        if self._cohpcar is None:
            path = self.work_dir / "COHPCAR.lobster"
            if path.exists():
                try:
                    self._cohpcar = Cohpcar(filename=str(path))
                except Exception as e:
                    self._log(f"COHPCAR parse failed: {e}", "warning")
        return self._cohpcar

    @property
    def icohplist(self) -> Optional[Any]:
        if self._icohplist is None:
            path = self.work_dir / "ICOHPLIST.lobster"
            if path.exists():
                try:
                    self._icohplist = Icohplist(filename=str(path))
                except Exception as e:
                    self._log(f"ICOHPLIST parse failed: {e}", "warning")
        return self._icohplist

    def analyze(
        self,
        n_top_bonds: int = 20,  # 默认展示前20个键
        erange: Optional[List[float]] = None,
        bond_labels: Optional[List[str]] = None,
        include_orbitals: bool = False,
        # 新增筛选参数
        filter_type: Optional[str] = None,  # "index" 或 "element_pair"
        filter_value: Optional[Union[List[int], tuple]] = None,  # index列表或元素对
        step: str = "summary"  # "summary" 或 "curves"
    ) -> Any:
        """
        主分析方法，支持分步加载和筛选
        
        Args:
            n_top_bonds: 展示前N个键
            erange: 能量范围
            bond_labels: 要分析的键标签
            include_orbitals: 是否包含轨道数据
            filter_type: 筛选类型 ("index" 或 "element_pair")
            filter_value: 筛选值
            step: 执行步骤 ("summary" - 仅返回ICOHP列表, "curves" - 返回曲线数据)
        """
        try:
            ispin = 2 if (self.cohpcar and getattr(self.cohpcar, "is_spin_polarized", False)) else 1

            # Step 1: 获取ICOHP摘要（支持筛选）
            if step == "summary" or step == "both":
                icohp_df = self._get_filtered_icohp_summary(
                    n_top=n_top_bonds,
                    filter_type=filter_type,
                    filter_value=filter_value,
                    save=self.save_data
                )
                
                if step == "summary":
                    # 仅返回摘要数据
                    data = {
                        "ispin": ispin,
                        "icohp_summary": icohp_df.replace({np.nan: None}).to_dict(orient="records") if icohp_df is not None else [],
                        "n_bonds": len(icohp_df) if icohp_df is not None else 0,
                        "step": "summary"
                    }
                    return ApiResponse.ok(data=data, message="ICOHP summary loaded")
            
            # Step 2: 获取曲线数据（用户选择特定键后）
            if step == "curves" or step == "both":
                if not bond_labels:
                    return ApiResponse.error("bond_labels required for curves analysis")
                
                cohp_df = self.get_cohp_curves(
                    bond_labels=bond_labels,
                    erange=erange,
                    save=self.save_data,
                    include_orbitals=include_orbitals
                )
                
                data = {
                    "ispin": ispin,
                    "cohp_curves": cohp_df.replace({np.nan: None}).to_dict(orient="records") if cohp_df is not None else [],
                    "bond_labels": bond_labels,
                    "include_orbitals": include_orbitals,
                    "step": "curves"
                }
                
                # 如果是both模式，也包含摘要
                if step == "both":
                    icohp_df = self._get_filtered_icohp_summary(
                        n_top=n_top_bonds,
                        filter_type=filter_type,
                        filter_value=filter_value,
                        save=False
                    )
                    data["icohp_summary"] = icohp_df.replace({np.nan: None}).to_dict(orient="records") if icohp_df is not None else []
                    data["n_bonds"] = len(icohp_df) if icohp_df is not None else 0
                
                return ApiResponse.ok(data=data, message="COHP curves loaded")
            
            return ApiResponse.error("Invalid step parameter")
            
        except Exception as e:
            traceback.print_exc()
            self._log(f"CohpAnalysis failed: {e}", "error")
            return ApiResponse.error(str(e))

    def _get_filtered_icohp_summary(
        self,
        n_top: int = 20,
        filter_type: Optional[str] = None,
        filter_value: Optional[Union[List[int], tuple]] = None,
        save: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        获取经过筛选的ICOHP摘要
        
        Args:
            n_top: 返回前N个键
            filter_type: "index" - 按键索引筛选, "element_pair" - 按元素对筛选
            filter_value: 筛选值
            save: 是否保存到CSV
        """
        if self.icohplist is None:
            return None
        
        # 获取原子映射信息
        atom_map = {}
        icohp_path = self.work_dir / "ICOHPLIST.lobster"
        if icohp_path.exists():
            with open(icohp_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 4 and parts[0].isdigit():
                        bond_idx = parts[0]
                        atom_map[bond_idx] = (parts[1], parts[2])
        
        icohp_obj = self.icohplist
        keys = list(icohp_obj.icohplist.keys())
        rows = []
        
        # 寻找费米能级索引
        zero_idx = 0
        if self.cohpcar is not None:
            energies = np.array(self.cohpcar.energies)
            zero_idx = np.abs(energies).argmin()
        
        for label in keys:
            # 按索引筛选
            if filter_type == "index" and filter_value:
                try:
                    label_idx = int(label)
                    if label_idx not in filter_value:
                        continue
                except ValueError:
                    continue
            
            bond = icohp_obj.icohplist[label]
            atom1, atom2 = atom_map.get(label, ("", ""))
            
            # 按元素对筛选
            if filter_type == "element_pair" and filter_value:
                el1 = "".join(c for c in atom1 if c.isalpha())
                el2 = "".join(c for c in atom2 if c.isalpha())
                if not el1 or not el2:
                    continue
                if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                    if set(filter_value) != {el1, el2}:
                        continue
            
            # 提取ICOHP值
            icohp_dict = bond.get("icohp", {})
            icohp_up = icohp_dict.get(Spin.up, None)
            icohp_down = icohp_dict.get(Spin.down, None)
            
            icohp_up_val = icohp_up if icohp_up is not None else np.nan
            icohp_down_val = icohp_down if icohp_down is not None else np.nan
            
            if icohp_up is not None and icohp_down is not None:
                icohp_total = (icohp_up + icohp_down) / 2
            elif icohp_up is not None:
                icohp_total = icohp_up
            else:
                icohp_total = None
            
            # 提取分轨道标量信息
            orbitals_dict = {}
            if self.cohpcar and hasattr(self.cohpcar, "orb_res_cohp") and self.cohpcar.orb_res_cohp:
                if label in self.cohpcar.orb_res_cohp:
                    orb_data = self.cohpcar.orb_res_cohp[label]
                    for orb_label, orb_info in orb_data.items():
                        icohp_arrs = orb_info.get("ICOHP", {})
                        orb_up_val = icohp_arrs[Spin.up][zero_idx] if Spin.up in icohp_arrs else 0.0
                        orb_down_val = icohp_arrs[Spin.down][zero_idx] if Spin.down in icohp_arrs else 0.0
                        orbitals_dict[orb_label] = {
                            "icohp_up": float(orb_up_val),
                            "icohp_down": float(orb_down_val)
                        }
            
            rows.append({
                "bond_label": label,
                "bond_index": int(label) if label.isdigit() else -1,  # 添加索引字段
                "atom1": atom1,
                "atom2": atom2,
                "element_pair": f"{el1}-{el2}" if 'el1' in locals() and 'el2' in locals() else "",  # 元素对
                "length_Ang": bond.get("length", None),
                "icohp_up": icohp_up_val,
                "icohp_down": icohp_down_val,
                "icohp_total": icohp_total,
                "orbitals": orbitals_dict,
            })
        
        if not rows:
            return pd.DataFrame(columns=["bond_label", "bond_index", "atom1", "atom2", "element_pair", 
                                        "length_Ang", "icohp_up", "icohp_down", "icohp_total", "orbitals"])
        
        df = (
            pd.DataFrame(rows)
            .assign(abs_icohp=lambda x: x["icohp_total"].abs())
            .sort_values("abs_icohp", ascending=False, na_position="last")
            .drop(columns=["abs_icohp"])
            .head(n_top)
            .reset_index(drop=True)
        )
        
        if save:
            self.save_to_csv(df, "icohp_summary.csv")
        
        return df

    def get_cohp_curves(
        self,
        bond_labels: Optional[List[str]] = None,
        erange: Optional[List[float]] = None,
        save: bool = False,
        include_orbitals: bool = False,
    ) -> Optional[pd.DataFrame]:
        """获取COHP曲线数据"""
        if self.cohpcar is None:
            return None

        energies = np.array(self.cohpcar.energies)
        cohp_data = self.cohpcar.cohp_data

        mask = (energies >= erange[0]) & (energies <= erange[1]) if erange else np.ones(len(energies), dtype=bool)

        data: Dict[str, Any] = {"energy_eV": energies[mask]}
        targets = bond_labels or list(cohp_data.keys())

        for label in targets:
            bond = cohp_data.get(label)
            if bond is None:
                continue
            
            cohp = bond.get("COHP", {})
            if Spin.up in cohp:
                data[f"{label}_up"] = np.array(cohp[Spin.up])[mask]
            if Spin.down in cohp:
                data[f"{label}_down"] = np.array(cohp[Spin.down])[mask]

            # 仅当开启时提取轨道曲线
            if include_orbitals and hasattr(self.cohpcar, "orb_res_cohp") and self.cohpcar.orb_res_cohp:
                orb_data = self.cohpcar.orb_res_cohp.get(label, {})
                for orb_label, orb_info in orb_data.items():
                    orb_cohp = orb_info.get("COHP", {})
                    if Spin.up in orb_cohp:
                        data[f"{label}_{orb_label}_up"] = np.array(orb_cohp[Spin.up])[mask]
                    if Spin.down in orb_cohp:
                        data[f"{label}_{orb_label}_down"] = np.array(orb_cohp[Spin.down])[mask]

        df = pd.DataFrame(data)
        if save:
            filename = "cohp_curves_with_orbitals.csv" if include_orbitals else "cohp_curves.csv"
            self.save_to_csv(df, filename)
        
        return df

    def export_single_bond_data(
        self,
        bond_label: str,
        erange: Optional[List[float]] = None,
        include_orbitals: bool = True,
        export_format: str = "csv"  # "csv" 或 "json"
    ) -> Any:
        """
        导出单个键的数据（包括可选的分轨道数据）
        
        Args:
            bond_label: 键标签
            erange: 能量范围
            include_orbitals: 是否包含轨道数据
            export_format: 导出格式
        """
        try:
            if not bond_label:
                return ApiResponse.error("bond_label is required")
            
            df = self.get_cohp_curves(
                bond_labels=[str(bond_label)],
                erange=erange,
                include_orbitals=include_orbitals
            )
            
            if df is None or df.empty:
                return ApiResponse.error(f"未找到键 {bond_label} 的数据")
            
            # 根据格式返回数据
            if export_format == "csv":
                csv_data = df.to_dict(orient="list")
                message = f"Extracted {'orbital ' if include_orbitals else ''}data for bond {bond_label}"
            else:
                csv_data = df.to_dict(orient="records")
                message = f"Extracted {'orbital ' if include_orbitals else ''}data for bond {bond_label}"
            
            return ApiResponse.ok(
                data={
                    "bond_label": bond_label,
                    "include_orbitals": include_orbitals,
                    "data": csv_data
                },
                message=message
            )
            
        except Exception as e:
            traceback.print_exc()
            return ApiResponse.error(str(e))

    def export_multiple_bonds_data(
        self,
        bond_labels: List[str],
        erange: Optional[List[float]] = None,
        include_orbitals: bool = False,
        separate_files: bool = False
    ) -> Any:
        """
        导出多个键的数据
        
        Args:
            bond_labels: 键标签列表
            erange: 能量范围
            include_orbitals: 是否包含轨道数据
            separate_files: 是否分别导出每个键
        """
        try:
            if not bond_labels:
                return ApiResponse.error("bond_labels is required")
            
            if separate_files:
                # 分别导出每个键
                results = {}
                for label in bond_labels:
                    df = self.get_cohp_curves(
                        bond_labels=[label],
                        erange=erange,
                        include_orbitals=include_orbitals
                    )
                    if df is not None and not df.empty:
                        results[label] = df.to_dict(orient="list")
                
                return ApiResponse.ok(
                    data={"bonds": results, "separate": True},
                    message=f"Exported {len(results)} bonds separately"
                )
            else:
                # 合并导出
                df = self.get_cohp_curves(
                    bond_labels=bond_labels,
                    erange=erange,
                    include_orbitals=include_orbitals
                )
                
                if df is None or df.empty:
                    return ApiResponse.error("No data found for specified bonds")
                
                return ApiResponse.ok(
                    data={"combined": df.to_dict(orient="list"), "separate": False},
                    message=f"Exported {len(bond_labels)} bonds combined"
                )
                
        except Exception as e:
            traceback.print_exc()
            return ApiResponse.error(str(e))

# ============================================================
# 6. 统一调度器
# ============================================================
class VaspAnalysisDispatcher:
    """
    VASP 分析任务统一调度器 (基于注册表模式)
    """
    
    TASK_REGISTRY = {
        "dos": DosAnalysis,
        "relax": RelaxAnalysis,
        "structure_info": StructureAnalysis,
        "cohp": CohpAnalysis,
        "cohp_summary": CohpAnalysis,  
        "cohp_curves": CohpAnalysis,  
        "cohp_export": CohpAnalysis,   
    }
    
    @classmethod
    def dispatch(cls, task_type: str, work_dir: Union[str, Path], **kwargs) -> str:
        task_type = task_type.lower().strip()
        
        if task_type not in cls.TASK_REGISTRY:
            return ApiResponse.error(
                f"Unknown task type: '{task_type}'. Available tasks: {list(cls.TASK_REGISTRY.keys())}"
            ).to_json()
        
        try:
            analyzer_class = cls.TASK_REGISTRY[task_type]
            
            init_kwargs = {}
            if "save_data" in kwargs:
                init_kwargs["save_data"] = kwargs.pop("save_data")
                
            analyzer = analyzer_class(work_dir=work_dir, **init_kwargs)
            
            # 针对不同任务类型调用不同的方法
            if task_type == "structure_info":
                result = analyzer.get_info()
                
            elif task_type == "cohp_summary":
                # 仅获取ICOHP摘要
                kwargs["step"] = "summary"
                result = analyzer.analyze(**kwargs)
                
            elif task_type == "cohp_curves":
                # 获取COHP曲线
                kwargs["step"] = "curves"
                result = analyzer.analyze(**kwargs)
                
            elif task_type == "cohp_export":
                # 导出功能
                if kwargs.get("export_type") == "single":
                    result = analyzer.export_single_bond_data(
                        bond_label=kwargs.get("bond_label"),
                        erange=kwargs.get("erange"),
                        include_orbitals=kwargs.get("include_orbitals", False),
                        export_format=kwargs.get("export_format", "csv")
                    )
                else:
                    result = analyzer.export_multiple_bonds_data(
                        bond_labels=kwargs.get("bond_labels", []),
                        erange=kwargs.get("erange"),
                        include_orbitals=kwargs.get("include_orbitals", False),
                        separate_files=kwargs.get("separate_files", False)
                    )
                    
            elif task_type == "cohp":
                # 默认COHP分析（向后兼容）
                kwargs.setdefault("step", "both")
                result = analyzer.analyze(**kwargs)
                
            else:
                # 默认处理 (dos, relax)
                result = analyzer.analyze(**kwargs)
            
            return result.to_json()
        
        except Exception as e:
            traceback.print_exc()
            return ApiResponse.error(f"Analysis task '{task_type}' failed: {str(e)}").to_json()