"""PBS/SLURM 脚本模板渲染。"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class Script:
    """通用计算脚本生成"""
    
    def __init__(
        self,
        template_path: Optional[Union[str, Path]] = None,
        cluster_defaults: Optional[Dict[str, Any]] = None,
        vdw_path: Union[str, Path] = "/data2/home/luodh/Git-workflow/Workflow_for_vasp_catalysis/scripts/vdw_kernel.bindat"
    ):
        """
        :param template_path: 模板文件路径
        :param cluster_defaults: 当前超算集群的全局默认参数字典
        :param vdw_path: vdw_kernel.bindat 在当前集群上的绝对路径
        """
        self.vdw_path = Path(vdw_path)
        
        # 1. 确定模板文件路径
        if template_path:
            self.template_path = Path(template_path)
        else:
            self.template_path = Path(__file__).resolve().parent / "scripts" / "script"

        # 2. 设置集群的“全局基础参数” (Fallback)
        self.base_context = {
            "CORES": 72,
            "QUEUE": "low",
            "WALLTIME": 124,
            "COMPILER": "2020u1",
            "TYPE1": "org",
            "CLEANUP_CMD": "rm REPORT CHG* DOSCAR EIGENVAL IBZKPT PCDAT PROCAR WAVECAR XDATCAR vasprun.xml FORCECAR",
            "EXTRA_CMD": ""
        }
        if cluster_defaults:
            self.base_context.update(cluster_defaults)

    def _build_context(
        self, 
        functional: str = "PBE", 
        is_lobster: bool = False, 
        is_static: bool = False,
        is_nbo: bool = False,
        custom_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """内部方法：生成最终的渲染字典"""
        context = self.base_context.copy()
        functional = functional.upper()
        
        # 维度 1：处理泛函 (Functional)
        if "BEEFVTST" in functional:
            context["TYPE1"] = "beefvtst"
        elif "BEEF" in functional: 
            context["TYPE1"] = "beef"
        elif "VTST" in functional:
            context["TYPE1"] = "vtst"
        else:
            context["TYPE1"] = "org"
            
        # 维度 2：处理流程 (Lobster / Static)
        if is_lobster:
            context["WALLTIME"] = 360
            context["COMPILER"] = "2020u2"
            context["CLEANUP_CMD"] = ""  # Lobster 强依赖波函数等文件，不能删
            current_cores = custom_context.get("CORES", context["CORES"]) if custom_context else context["CORES"]
            context["EXTRA_CMD"] = f"export OMP_NUM_THREADS={current_cores}\nlobster1"
        elif is_static:
            # 单点计算 (Static/NoSCF) 需要保留电子结构性质文件 (DOSCAR, WAVECAR, CHGCAR 等)
            context["CLEANUP_CMD"] = ""
        elif is_nbo:
            context["CLEANUP_CMD"] = ""  # Lobster 强依赖波函数等文件，不能删
            current_cores = custom_context.get("CORES", context["CORES"]) if custom_context else context["CORES"]
            context["EXTRA_CMD"] = (
                f"export OMP_NUM_THREADS={current_cores}\n"
                "export OMP_STACKSIZE=10GB\n"
                "projection.exe basis.inp wavefunction.dat NBO.out >> $LOG_FILE 2>&1\n"
                "nbo.exe NBO.out nbo.chk >> $LOG_FILE 2>&1\n"
                "rm REPORT CHG* DOSCAR EIGENVAL IBZKPT PCDAT PROCAR WAVECAR XDATCAR vasprun.xml FORCECAR\n"
            )
        # 维度 3：用户自定义覆盖
        if custom_context:
            context.update(custom_context)
            
        return context

    def _parse_folders(self, folders: Union[List[str], str, Path]) -> List[Path]:
        """内部方法：统一将输入的 folders 转换为 Path 对象列表"""
        if isinstance(folders, (str, Path)):
            return [Path(folders)]
        return [Path(f) for f in folders]

    def _copy_vdw_kernel(self, target_folders: List[Path]):
        """内部方法：自动为 BEEF 泛函复制 vdw_kernel.bindat"""
        if not self.vdw_path.exists():
                logger.warning(
                    "找不到 vdw_kernel.bindat (%s)，VASP 将耗时自行生成。", self.vdw_path
                )

        count = 0
        for dst_folder in target_folders:
            dst_file = dst_folder / "vdw_kernel.bindat"
            if not dst_file.exists():
                shutil.copy2(self.vdw_path, dst_file)
                count += 1
                
        if count > 0:
            logger.info("自动将 vdw_kernel.bindat 复制到 %d 个 BEEF 计算文件夹中！", count)

    def render_script(
        self, 
        folders: Union[List[str], str, Path],
        functional: str = "PBE",
        is_lobster: bool = False,
        is_static: bool = False,
        is_nbo: bool = False,
        output_filename: str = "script", 
        custom_context: Optional[Dict[str, Any]] = None, 
        make_executable: bool = True
    ):
        """为指定的文件夹渲染并生成 PBS 脚本，并自动处理依赖文件"""
        if not self.template_path.exists():
            raise FileNotFoundError(f"找不到模板文件: {self.template_path}")

        with open(self.template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        # 传入 is_static 参数
        final_context = self._build_context(functional, is_lobster, is_static, is_nbo, custom_context)
        target_folders = self._parse_folders(folders)

        # 1. 渲染并写入 PBS 脚本
        for dst_folder in target_folders:
            dst_folder.mkdir(parents=True, exist_ok=True)
            
            current_context = {"JOB_NAME": dst_folder.name}
            current_context.update(final_context)

            rendered_content = template_content
            for key, value in current_context.items():
                rendered_content = rendered_content.replace(f"{{{{{key}}}}}", str(value))

            file_path = dst_folder / output_filename
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(rendered_content)
                
            if make_executable:
                os.chmod(file_path, 0o755)
                
        logger.info("成功渲染并写入 %d 个 %s！", len(target_folders), output_filename)

        # 2. 核心改进：如果是 BEEF 泛函，自动触发 vdw 文件复制！
        if "BEEF" in functional.upper():
            self._copy_vdw_kernel(target_folders)
        return [str(p / output_filename) for p in target_folders]
    
