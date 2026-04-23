# -*- coding: utf-8 -*-
"""
PBS/SLURM 脚本模板渲染。

================================================================================
参数分类说明
================================================================================

【前端可配置参数】（优先级最高，用户显式传入）
  cores             - 核数（如不传则由模式默认值决定）
  walltime          - 计算时间，小时（如不传则由模式默认值决定）
  queue             - 计算队列（如不传则使用集群默认值）
  folders           - 计算文件夹路径
  functional        - 泛函（PBE, BEEF, SCAN等）
  calc_category     - 计算类别（CalcCategory枚举），用于自动生成默认资源配置
  output_filename   - 脚本文件名
  custom_context    - 自定义上下文（可覆盖任何参数，最终兜底）

【模式自动生成参数】（根据 calc_category 自动推断，当用户未显式传入时使用）
  WALLTIME          - 由 _CATEGORY_CONFIG[calc_category]["typical_walltime"] 提供
  CORES             - 由 _CATEGORY_CONFIG[calc_category]["typical_cores"] 提供
  COMPILER          - 由 _CATEGORY_CONFIG[calc_category]["compiler"] 提供
  CLEANUP_CMD       - 由 _CATEGORY_CONFIG[calc_category]["cleanup"] 决定

【内部使用参数】（系统自动管理，对用户透明）
  _CATEGORY_CONFIG  - 计算类别到脚本配置的映射
  _FUNCTIONAL_TYPE_MAP - 泛函到TYPE1的映射
  vdw处理           - BEEF泛函自动复制vdw_kernel.bindat

================================================================================
参数优先级（从低到高）
================================================================================

  集群全局默认值 (cluster_defaults)
       ↓
  calc_category 模式自动生成值
       ↓
  用户显式传入值 (cores / walltime / queue)
       ↓
  custom_context 完全覆盖（最终兜底）

================================================================================
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union
from enum import Enum

logger = logging.getLogger(__name__)


class CalcCategory(Enum):
    """
    计算类别枚举 - 用于自动脚本生成
    
    用户无需手动指定 is_lobster/is_static 等布尔标志，
    只需指定计算类别，系统自动推断正确的脚本配置。
    """
    # 基础结构优化（可清理中间文件）
    RELAX = "relax"
    
    # 单点/静态计算（需保留电子结构文件）
    STATIC = "static"
    
    # 过渡态搜索（VTST）
    NEB = "neb"
    DIMER = "dimer"
    
    # 特殊分析（依赖波函数）
    LOBSTER = "lobster"
    NBO = "nbo"
    
    # 其他性质计算
    FREQ = "freq"
    NMR = "nmr"
    MD = "md"


class CategoryConfig(TypedDict):
    """每种计算类别的脚本生成配置。"""
    cleanup: bool
    need_wavecharge: bool
    need_vdw: bool
    compiler: str
    typical_walltime: int
    typical_cores: int
    requires_wavecar: bool


# 计算类别 → 脚本配置映射
# 包含每种模式的"自动生成默认值"，当用户未显式传入时使用
_CATEGORY_CONFIG: Dict[CalcCategory, CategoryConfig] = {
    CalcCategory.RELAX: {
        "cleanup": True,
        "need_wavecharge": False,
        "need_vdw": False,
        "compiler": "2020u1",
        "typical_walltime": 124,       # 前端未传 walltime 时的模式默认值
        "typical_cores": 72,           # 前端未传 cores 时的模式默认值
        "requires_wavecar": False,
    },
    CalcCategory.STATIC: {
        "cleanup": False,
        "need_wavecharge": True,
        "need_vdw": False,
        "compiler": "2020u1",
        "typical_walltime": 48,
        "typical_cores": 72,
        "requires_wavecar": False,
    },
    CalcCategory.NEB: {
        "cleanup": True,
        "need_wavecharge": False,
        "need_vdw": False,
        "compiler": "2020u1",
        "typical_walltime": 200,
        "typical_cores": 144,          # NEB 通常需要更多核（每image × 几个核）
        "requires_wavecar": False,
    },
    CalcCategory.DIMER: {
        "cleanup": True,
        "need_wavecharge": False,
        "need_vdw": False,
        "compiler": "2020u1",
        "typical_walltime": 100,
        "typical_cores": 72,
        "requires_wavecar": False,
    },
    CalcCategory.LOBSTER: {
        "cleanup": False,
        "need_wavecharge": True,
        "need_vdw": False,
        "compiler": "2020u2",          # Lobster 需要特定编译器版本
        "typical_walltime": 360,
        "typical_cores": 72,
        "requires_wavecar": True,
    },
    CalcCategory.NBO: {
        "cleanup": False,
        "need_wavecharge": True,
        "need_vdw": False,
        "compiler": "2020u2",
        "typical_walltime": 200,
        "typical_cores": 72,
        "requires_wavecar": True,
    },
    CalcCategory.FREQ: {
        "cleanup": True,
        "need_wavecharge": False,
        "need_vdw": False,
        "compiler": "2020u1",
        "typical_walltime": 100,
        "typical_cores": 72,
        "requires_wavecar": False,
    },
    CalcCategory.NMR: {
        "cleanup": True,
        "need_wavecharge": False,
        "need_vdw": False,
        "compiler": "2020u1",
        "typical_walltime": 150,
        "typical_cores": 72,
        "requires_wavecar": False,
    },
    CalcCategory.MD: {
        "cleanup": False,
        "need_wavecharge": True,
        "need_vdw": False,
        "compiler": "2020u1",
        "typical_walltime": 200,  # 默认 200 小时；长时间 MD 需续跑机制
        "typical_cores": 72,
        "requires_wavecar": True,
    },
}

# 计算类型值（CalcType.value）→ 计算类别映射（供外部调用方使用）
CALC_TYPE_TO_CATEGORY: Dict[str, CalcCategory] = {
    "bulk_relax":            CalcCategory.RELAX,
    "slab_relax":            CalcCategory.RELAX,
    "static_sp":             CalcCategory.STATIC,
    "static_dos":            CalcCategory.STATIC,
    "static_charge_density": CalcCategory.STATIC,
    "static_elf":            CalcCategory.STATIC,
    "neb":                   CalcCategory.NEB,
    "dimer":                 CalcCategory.DIMER,
    "frequency":             CalcCategory.FREQ,
    "frequency_ir":          CalcCategory.FREQ,
    "lobster":               CalcCategory.LOBSTER,
    "nmr_cs":                CalcCategory.NMR,
    "nmr_efg":               CalcCategory.NMR,
    "nbo":                   CalcCategory.NBO,
    "md_nvt":                CalcCategory.MD,
    "md_npt":                CalcCategory.MD,
}

# 泛函 → TYPE1 映射（用户无需记忆）
_FUNCTIONAL_TYPE_MAP: Dict[str, str] = {
    "BEEFVTST": "beefvtst",
    "BEEF": "beef",
    "VTST": "vtst",
    "SCAN": "scan",
    "HSE": "hse",
    "PBE": "org",
    "PBE0": "pbe0",
    "LDA": "lda",
}


class Script:
    """通用计算脚本生成"""
    
    def __init__(
        self,
        template_path: Optional[Union[str, Path]] = None,
        cluster_defaults: Optional[Dict[str, Any]] = None,
        vdw_path: Optional[Union[str, Path]] = None,
    ):
        """
        :param template_path: 模板文件路径
        :param cluster_defaults: 当前超算集群的全局默认参数字典（优先级最低的基础值）
        :param vdw_path: vdw_kernel.bindat 在当前集群上的绝对路径；不传时尝试读取 FLOW_VDW_KERNEL
        """
        raw_vdw = str(vdw_path).strip() if vdw_path is not None else os.environ.get("FLOW_VDW_KERNEL", "").strip()
        self.vdw_path = Path(raw_vdw).expanduser() if raw_vdw else None
        
        # 1. 确定模板文件路径
        if template_path:
            self.template_path = Path(template_path)
        else:
            self.template_path = Path(__file__).resolve().parent / "scripts" / "script"

        # 2. 设置集群的"全局基础参数"（最低优先级 Fallback）
        #    当模式默认值和用户显式传入值都不存在时，才回落到这里
        self.base_context: Dict[str, Any] = {
            "CORES": 72,
            "QUEUE": "low",
            "WALLTIME": 124,
            "COMPILER": "2020u1",
            "TYPE1": "org",
            "CLEANUP_CMD": "rm REPORT CHG* DOSCAR EIGENVAL IBZKPT PCDAT PROCAR WAVECAR XDATCAR vasprun.xml FORCECAR",
            "EXTRA_CMD": "",
        }
        if cluster_defaults:
            self.base_context.update(cluster_defaults)

    # -------------------------------------------------------------------------
    # 自动推断方法
    # -------------------------------------------------------------------------

    def infer_calc_category(self, folder: Union[str, Path]) -> CalcCategory:
        """
        从文件夹自动推断计算类别
        
        智能检测策略：
        1. 读取 INCAR，检查 IBRION/NSW 等参数
        2. 检查文件名模式（如 "neb", "dimer"）
        3. 检查是否包含 lobsterin/nbo.config 等特殊文件
        """
        folder = Path(folder)
        incar_path = folder / "INCAR"
        
        # 如果没有 INCAR，尝试从文件夹名推断
        if not incar_path.exists():
            folder_name = folder.name.lower()
            if "neb" in folder_name:
                return CalcCategory.NEB
            elif "dimer" in folder_name:
                return CalcCategory.DIMER
            elif "lobster" in folder_name:
                return CalcCategory.LOBSTER
            elif "nbo" in folder_name:
                return CalcCategory.NBO
            elif "md" in folder_name:
                return CalcCategory.MD
            elif "freq" in folder_name or "phonon" in folder_name:
                return CalcCategory.FREQ
            elif "nmr" in folder_name:
                return CalcCategory.NMR
            else:
                logger.warning(f"无法从 {folder} 推断计算类别，默认使用 RELAX")
                return CalcCategory.RELAX
        
        try:
            with open(incar_path, "r") as f:
                incar_content = f.read().upper()
            
            # NEB/Dimer 检测
            if "IMAGES" in incar_content or "ICHAIN" in incar_content:
                if "ICHAIN" in incar_content and "2" in incar_content:
                    return CalcCategory.DIMER
                return CalcCategory.NEB
            
            # MD 检测
            if "IBRION" in incar_content and "0" in incar_content:
                if "SMASS" in incar_content or "MDALGO" in incar_content:
                    return CalcCategory.MD
            
            # 特殊文件检测
            if (folder / "lobsterin").exists():
                return CalcCategory.LOBSTER
            if (folder / "nbo.config").exists():
                return CalcCategory.NBO
            if (folder / "FREQ").exists() or "IBRION" in incar_content and "5" in incar_content:
                return CalcCategory.FREQ
            if "LCHIMAG" in incar_content or "LEFG" in incar_content:
                return CalcCategory.NMR
            
            # 从 IBRION/NSW 推断
            import re
            ibrion_match = re.search(r'IBRION\s*=\s*(\d+)', incar_content)
            nsw_match = re.search(r'NSW\s*=\s*(\d+)', incar_content)
            
            ibrion = int(ibrion_match.group(1)) if ibrion_match else -1
            nsw = int(nsw_match.group(1)) if nsw_match else 0
            
            if nsw > 0 and ibrion in (1, 2, 3):
                return CalcCategory.RELAX
            else:
                return CalcCategory.STATIC
                
        except Exception as e:
            logger.warning(f"解析 INCAR 失败: {e}，默认使用 STATIC")
            return CalcCategory.STATIC
    
    def infer_functional(self, folder: Union[str, Path]) -> str:
        """
        从 INCAR/GGA 行自动推断泛函类型
        """
        folder = Path(folder)
        incar_path = folder / "INCAR"
        
        if not incar_path.exists():
            return "PBE"
        
        try:
            with open(incar_path, "r") as f:
                content = f.read().upper()
            
            import re
            gga_match = re.search(r'GGA\s*=\s*(\w+)', content)
            if gga_match:
                return gga_match.group(1)
            
            folder_name = folder.name.upper()
            for key in _FUNCTIONAL_TYPE_MAP.keys():
                if key in folder_name:
                    return key
            
            return "PBE"
            
        except Exception:
            return "PBE"
    
    # -------------------------------------------------------------------------
    # 内部辅助方法
    # -------------------------------------------------------------------------

    def _resolve_type1(self, functional: str) -> str:
        """将泛函名称解析为 TYPE1 值"""
        functional_upper = functional.upper()
        if functional_upper in _FUNCTIONAL_TYPE_MAP:
            return _FUNCTIONAL_TYPE_MAP[functional_upper]
        for key, value in _FUNCTIONAL_TYPE_MAP.items():
            if key in functional_upper:
                return value
        return "org"

    def _build_context(
        self,
        functional: str = "PBE",
        calc_category: Optional[CalcCategory] = None,
        # --- 前端显式传入的资源参数（优先级高于模式默认值）---
        cores: Optional[int] = None,
        walltime: Optional[int] = None,
        queue: Optional[str] = None,
        # --- 向后兼容的布尔标志（低优先级，仅在无 calc_category 时生效）---
        is_lobster: bool = False,
        is_static: bool = False,
        is_nbo: bool = False,
        # --- 最终兜底覆盖 ---
        custom_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        生成最终的渲染字典，严格按优先级叠加：
        
        集群全局默认值
          → calc_category 模式自动生成值（typical_walltime / typical_cores）
          → 用户显式传入值（cores / walltime / queue）
          → custom_context 完全覆盖
        """
        # === 第一层：集群全局默认值 ===
        context = self.base_context.copy()
        functional = functional.upper()
        context["TYPE1"] = self._resolve_type1(functional)

        # === 第二层：calc_category 模式自动生成值 ===
        if calc_category:
            cat_cfg = _CATEGORY_CONFIG.get(calc_category, _CATEGORY_CONFIG[CalcCategory.RELAX])

            # 资源默认值（仅在用户未显式传入时生效）
            context["WALLTIME"] = cat_cfg["typical_walltime"]
            context["CORES"] = cat_cfg["typical_cores"]
            context["COMPILER"] = cat_cfg["compiler"]

            # 清理策略
            if cat_cfg["cleanup"]:
                context["CLEANUP_CMD"] = (
                    "rm REPORT CHG* DOSCAR EIGENVAL IBZKPT PCDAT PROCAR "
                    "WAVECAR XDATCAR vasprun.xml FORCECAR"
                )
            else:
                context["CLEANUP_CMD"] = ""

            # 特殊后处理命令（Lobster / NBO）
            context["EXTRA_CMD"] = self._build_extra_cmd(
                calc_category,
                # 此时 cores 尚未覆盖，先用当前上下文值做占位，后面若用户传了更大值会更新
                cores_for_cmd=cores if cores is not None else cat_cfg["typical_cores"],
            )

        else:
            # 向后兼容：使用传统布尔标志
            if is_lobster:
                context["WALLTIME"] = 360
                context["COMPILER"] = "2020u2"
                context["CLEANUP_CMD"] = ""
                _cores = cores if cores is not None else context["CORES"]
                context["EXTRA_CMD"] = f"export OMP_NUM_THREADS={_cores}\nlobster1"
            elif is_static:
                context["CLEANUP_CMD"] = ""
            elif is_nbo:
                context["CLEANUP_CMD"] = ""
                _cores = cores if cores is not None else context["CORES"]
                context["EXTRA_CMD"] = (
                    f"export OMP_NUM_THREADS={_cores}\n"
                    "export OMP_STACKSIZE=10GB\n"
                    "projection.exe basis.inp wavefunction.dat NBO.out >> $LOG_FILE 2>&1\n"
                    "nbo.exe NBO.out nbo.chk >> $LOG_FILE 2>&1\n"
                    "rm REPORT CHG* DOSCAR EIGENVAL IBZKPT PCDAT PROCAR WAVECAR XDATCAR vasprun.xml FORCECAR\n"
                )

        # === 第三层：前端显式传入的资源参数（覆盖模式默认值）===
        if cores is not None:
            context["CORES"] = cores
            # 同步更新 EXTRA_CMD 中依赖 cores 的部分（Lobster/NBO 需要 OMP_NUM_THREADS）
            if calc_category in (CalcCategory.LOBSTER, CalcCategory.NBO):
                context["EXTRA_CMD"] = self._build_extra_cmd(calc_category, cores_for_cmd=cores)
        if walltime is not None:
            context["WALLTIME"] = walltime
        if queue is not None:
            context["QUEUE"] = queue

        # === 第四层：custom_context 完全覆盖（最终兜底）===
        if custom_context:
            context.update(custom_context)

        return context

    def _build_extra_cmd(
        self,
        calc_category: CalcCategory,
        cores_for_cmd: int = 72,
    ) -> str:
        """
        根据计算类别生成 EXTRA_CMD 字符串。
        cores_for_cmd 用于需要 OMP_NUM_THREADS 的计算类型。
        """
        if calc_category == CalcCategory.LOBSTER:
            return f"export OMP_NUM_THREADS={cores_for_cmd}\nlobster1"
        elif calc_category == CalcCategory.NBO:
            return (
                f"export OMP_NUM_THREADS={cores_for_cmd}\n"
                "export OMP_STACKSIZE=10GB\n"
                "projection.exe basis.inp wavefunction.dat NBO.out >> $LOG_FILE 2>&1\n"
                "nbo.exe NBO.out nbo.chk >> $LOG_FILE 2>&1\n"
                "rm REPORT CHG* DOSCAR EIGENVAL IBZKPT PCDAT PROCAR WAVECAR XDATCAR vasprun.xml FORCECAR\n"
            )
        return ""

    def _parse_folders(self, folders: Union[List[str], str, Path]) -> List[Path]:
        """统一将输入的 folders 转换为 Path 对象列表"""
        if isinstance(folders, (str, Path)):
            return [Path(folders)]
        return [Path(f) for f in folders]

    def _copy_vdw_kernel(self, target_folders: List[Path]):
        """自动为 BEEF 泛函复制 vdw_kernel.bindat"""
        if self.vdw_path is None:
            logger.warning("未配置 vdw_kernel.bindat 路径；VASP 将耗时自行生成。")
            return
        if not self.vdw_path.exists():
            logger.warning(
                "找不到 vdw_kernel.bindat (%s)，VASP 将耗时自行生成。", self.vdw_path
            )
            return

        count = 0
        for dst_folder in target_folders:
            dst_file = dst_folder / "vdw_kernel.bindat"
            if not dst_file.exists():
                # Delegate file I/O to script_writer — all filesystem writes
                # live there; lazy import avoids a circular dependency at module load.
                from .script_writer import ScriptWriter as _SW
                _SW._copy_file(self.vdw_path, dst_file)
                count += 1
                
        if count > 0:
            logger.info("自动将 vdw_kernel.bindat 复制到 %d 个 BEEF 计算文件夹中！", count)

    # -------------------------------------------------------------------------
    # 公开渲染方法
    # -------------------------------------------------------------------------

    def render_script(
        self,
        folders: Union[List[str], str, Path],
        functional: str = "PBE",
        calc_category: Optional[CalcCategory] = None,
        # --- 前端显式传入的资源参数 ---
        cores: Optional[int] = None,
        walltime: Optional[int] = None,
        queue: Optional[str] = None,
        # --- 向后兼容标志 ---
        is_lobster: bool = False,
        is_static: bool = False,
        is_nbo: bool = False,
        # ---
        output_filename: str = "script",
        custom_context: Optional[Dict[str, Any]] = None,
        make_executable: bool = True,
    ) -> List[str]:
        """
        为指定的文件夹渲染并生成 PBS/SLURM 脚本。

        参数优先级（从低到高）：
          集群全局默认值 → calc_category 模式自动值 → cores/walltime/queue → custom_context

        Args:
            folders:         目标计算文件夹（单个路径或路径列表）
            functional:      泛函名称（"PBE", "BEEF", "SCAN" 等）
            calc_category:   计算类别，用于自动决定资源默认值和清理策略
            cores:           CPU 核数（前端显式传入，覆盖模式默认值）
            walltime:        墙钟时间，单位小时（前端显式传入，覆盖模式默认值）
            queue:           计算队列名（前端显式传入）
            is_lobster:      是否 Lobster 分析（向后兼容，推荐改用 calc_category）
            is_static:       是否静态计算（向后兼容，推荐改用 calc_category）
            is_nbo:          是否 NBO 分析（向后兼容，推荐改用 calc_category）
            output_filename: 生成的脚本文件名
            custom_context:  额外的模板变量，覆盖所有自动推断值
            make_executable: 是否设置可执行权限

        Returns:
            生成的脚本文件路径列表
        """
        if not self.template_path.exists():
            raise FileNotFoundError(f"找不到模板文件: {self.template_path}")

        with open(self.template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        final_context = self._build_context(
            functional=functional,
            calc_category=calc_category,
            cores=cores,
            walltime=walltime,
            queue=queue,
            is_lobster=is_lobster,
            is_static=is_static,
            is_nbo=is_nbo,
            custom_context=custom_context,
        )
        target_folders = self._parse_folders(folders)
        generated_paths: List[str] = []

        # Lazy import — avoids circular dependency at module load
        # (script_writer.py imports from script.py at the top level).
        from .script_writer import ScriptWriter as _SW

        for dst_folder in target_folders:
            _SW._ensure_dir(dst_folder)

            # JOB_NAME 使用文件夹名
            current_context = {"JOB_NAME": dst_folder.name}
            current_context.update(final_context)

            rendered = template_content
            for key, value in current_context.items():
                rendered = rendered.replace(f"{{{{{key}}}}}", str(value))

            file_path = dst_folder / output_filename
            _SW._write_script_file(file_path, rendered, make_executable=make_executable)

            generated_paths.append(str(file_path))

        logger.info("成功渲染并写入 %d 个 %s！", len(target_folders), output_filename)

        # BEEF 泛函自动复制 vdw_kernel.bindat
        if "BEEF" in functional.upper():
            self._copy_vdw_kernel(target_folders)

        return generated_paths

    def auto_render(
        self,
        folders: Union[List[str], str, Path],
        output_filename: str = "script",
        make_executable: bool = True,
        cores: Optional[int] = None,
        walltime: Optional[int] = None,
        queue: Optional[str] = None,
        **custom_context,
    ) -> List[str]:
        """
        自动模式：无需指定任何参数，系统自动检测并生成脚本。

        系统会自动：
        1. 从 INCAR 推断计算类别（→ 自动设定 WALLTIME/CORES/CLEANUP 等）
        2. 从 INCAR/GGA 推断泛函
        3. 根据计算类型决定清理策略
        4. 必要时复制 vdw_kernel.bindat

        用户可以通过 cores/walltime/queue 覆盖自动推断的资源配置：

        示例::

            # 最简单：只需指定文件夹
            script_maker.auto_render("calc/static")

            # 指定多个文件夹
            script_maker.auto_render(["calc/relax", "calc/static"])

            # 覆盖资源参数（前端传入）
            script_maker.auto_render("calc/neb", walltime=200, cores=144)

            # 同时覆盖模板变量
            script_maker.auto_render("calc/relax", cores=72, QUEUE="high")

        Args:
            folders:         目标计算文件夹
            output_filename: 生成的脚本文件名
            make_executable: 是否设置可执行权限
            cores:           前端传入的核数（覆盖模式默认值）
            walltime:        前端传入的计算时间，小时（覆盖模式默认值）
            queue:           前端传入的队列（覆盖模式默认值）
            **custom_context: 其他任意模板变量覆盖（最高优先级）

        Returns:
            生成的脚本文件路径列表
        """
        target_folders = self._parse_folders(folders)
        results: List[str] = []

        for folder in target_folders:
            calc_category = self.infer_calc_category(folder)
            functional = self.infer_functional(folder)

            logger.info(
                "自动检测: folder=%s, category=%s, functional=%s",
                folder.name, calc_category.value, functional,
            )

            script_paths = self.render_script(
                folders=[folder],
                functional=functional,
                calc_category=calc_category,
                cores=cores,
                walltime=walltime,
                queue=queue,
                output_filename=output_filename,
                make_executable=make_executable,
                custom_context=custom_context if custom_context else None,
            )
            results.extend(script_paths)

        return results

    # -------------------------------------------------------------------------
    # 工具方法：查询某模式的默认资源配置
    # -------------------------------------------------------------------------

    @staticmethod
    def get_category_defaults(calc_category: CalcCategory) -> Dict[str, Any]:
        """
        返回某个计算类别的默认资源配置（供前端展示参考值）。

        示例::

            defaults = Script.get_category_defaults(CalcCategory.NEB)
            print(defaults)
            # {'typical_walltime': 200, 'typical_cores': 144, 'compiler': '2020u1', ...}

        Args:
            calc_category: 计算类别枚举值

        Returns:
            包含 typical_walltime / typical_cores / compiler 等字段的字典
        """
        cfg = _CATEGORY_CONFIG.get(calc_category)
        if cfg is None:
            return {}
        return {
            "typical_walltime": cfg["typical_walltime"],
            "typical_cores": cfg["typical_cores"],
            "compiler": cfg["compiler"],
            "cleanup": cfg["cleanup"],
            "requires_wavecar": cfg["requires_wavecar"],
        }

    @staticmethod
    def list_category_defaults() -> Dict[str, Dict[str, Any]]:
        """
        返回所有计算类别的默认资源配置（供前端下拉菜单参考）。

        Returns:
            形如 {"relax": {...}, "static": {...}, ...} 的字典
        """
        return {
            cat.value: Script.get_category_defaults(cat)
            for cat in CalcCategory
        }
