"""
mp_query_service.py
───────────────────────────────────────────────
Materials Project 结构查询服务（简化版）

特性：
  - 单输入框自动识别查询模式（formula / elements / chemsys / material_id）
  - 只获取前端展示所需字段：material_id, formula_pretty, structure, symmetry
  - Structure 转换：XYZ（3Dmol渲染）/ CIF / POSCAR（本地下载）
  - 保留 pymatgen Structure 对象，供后端进一步计算
  - 返回标准 dict，FastAPI 路由直接序列化
───────────────────────────────────────────────
"""

import re
import time
import logging
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ──────────────────────────────────────────────
# 日志
# ──────────────────────────────────────────────
logger = logging.getLogger("MPQueryService")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────
# MP 请求字段（最小化）
# ──────────────────────────────────────────────
_MP_FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "symmetry",
]

# ──────────────────────────────────────────────
# 重试装饰器
# ──────────────────────────────────────────────
def _retry(max_attempts: int = 3, backoff: float = 1.0):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        raise
                    wait = backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "[%s] 第%d次失败：%s，%.1fs后重试",
                        func.__name__, attempt, exc, wait
                    )
                    time.sleep(wait)
        return wrapper
    return decorator

# ──────────────────────────────────────────────
# TTL 缓存
# ──────────────────────────────────────────────
class _TTLCache:
    def __init__(self, ttl: int = 300, maxsize: int = 256):
        self._ttl     = ttl
        self._maxsize = maxsize
        self._store: Dict[str, Any] = {}
        self._ts:    Dict[str, float] = {}
        self._lock   = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                return None
            if time.time() - self._ts[key] > self._ttl:
                del self._store[key], self._ts[key]
                return None
            return self._store[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self._maxsize:
                oldest = min(self._ts, key=lambda k: self._ts[k])
                del self._store[oldest], self._ts[oldest]
            self._store[key] = value
            self._ts[key]    = time.time()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._ts.clear()

# ──────────────────────────────────────────────
# 查询模式自动识别
# ──────────────────────────────────────────────
_ELEMENT_RE = re.compile(r'^[A-Z][a-z]?$')

def detect_query_mode(raw: str) -> Dict[str, Any]:
    """
    根据用户输入自动识别查询模式。

    规则（按优先级）：
      1. mp-\d+                    → material_id  （如 mp-126）
      2. 含"-" 且全为元素符号       → chemsys      （如 Fe-O）
      3. 含"," 或空格分隔的元素符号  → elements     （如 "Fe, O"）
      4. 其余                      → formula      （如 Pt, Fe2O3）

    返回 dict 可直接解包传入 MPRester.summary.search(**kwargs)
    其中 "_mode" 键仅供日志使用，调用前需 pop。
    """
    query = raw.strip()
    if not query:
        raise ValueError("查询内容不能为空")

    # 规则 1：material_id
    if re.match(r'^mp-\d+$', query, re.IGNORECASE):
        return {"_mode": "material_id", "material_ids": [query.lower()]}

    # 规则 2：化学体系（连字符分隔元素）
    if "-" in query:
        parts = [p.strip() for p in query.split("-")]
        if all(_ELEMENT_RE.match(p) for p in parts) and len(parts) >= 2:
            return {
                "_mode": "chemsys",
                "chemsys": "-".join(p.capitalize() for p in parts),
            }

    # 规则 3：元素列表（逗号或空格分隔）
    if "," in query or (
        " " in query.strip()
        and re.match(r'^([A-Za-z]{1,2}[\s,]+)+[A-Za-z]{1,2}$', query.strip())
    ):
        parts = [p.strip() for p in re.split(r'[,\s]+', query) if p.strip()]
        if all(_ELEMENT_RE.match(p.capitalize()) for p in parts):
            return {
                "_mode": "elements",
                "elements": [p.capitalize() for p in parts],
            }

    # 规则 4：化学式（默认）
    return {"_mode": "formula", "formula": query}

# ──────────────────────────────────────────────
# 结构转换工具函数（纯函数，无副作用）
# ──────────────────────────────────────────────

def get_conventional(struct: Structure) -> Structure:
    """转换为传统标准结构（conventional cell），失败时返回原结构。"""
    try:
        return SpacegroupAnalyzer(struct, symprec=1e-3).get_conventional_standard_structure()
    except Exception:
        logger.warning("SpacegroupAnalyzer 失败，使用原始结构")
        return struct


def structure_to_xyz(struct: Structure, comment: str = "") -> str:
    """
    Structure → XYZ 格式字符串（笛卡尔坐标）。
    3Dmol.js 前端直接解析此格式渲染三维结构。
    """
    lines = [str(len(struct)), comment or struct.composition.reduced_formula]
    for site in struct.sites:
        x, y, z = site.coords          # 笛卡尔坐标（Å）
        lines.append(f"{site.specie.symbol} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines)


def structure_to_cif(struct: Structure) -> str:
    """Structure → CIF 格式字符串（内存操作，不写磁盘）。"""
    try:
        return str(CifWriter(struct))
    except Exception as e:
        logger.warning("CIF 转换失败：%s", e)
        return ""


def structure_to_poscar(struct: Structure, comment: str = "") -> str:
    """Structure → POSCAR 格式字符串（内存操作，不写磁盘）。"""
    try:
        return Poscar(struct, comment=comment or struct.composition.reduced_formula).get_str()
    except Exception as e:
        logger.warning("POSCAR 转换失败：%s", e)
        return ""


def save_structure_to_disk(
    struct: Structure,
    save_dir: str,
    filename: str,
    fmt: str = "all",
) -> Dict[str, str]:
    """
    将 pymatgen Structure 保存到本地磁盘。

    参数：
      struct:   pymatgen Structure 对象
      save_dir: 保存目录（不存在时自动创建）
      filename: 文件名前缀（不含扩展名），如 "mp-126_Pt"
      fmt:      "xyz" | "cif" | "poscar" | "all"（默认保存三种格式）

    返回：
      已保存文件的路径字典，如：
      {"xyz": "/path/mp-126_Pt.xyz", "cif": "...", "poscar": "..."}
    """
    base = Path(save_dir)
    base.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, str] = {}
    fmt = fmt.lower()

    # XYZ
    if fmt in ("xyz", "all"):
        path = base / f"{filename}.xyz"
        path.write_text(structure_to_xyz(struct, filename), encoding="utf-8")
        saved["xyz"] = str(path)
        logger.info("已保存 XYZ → %s", path)

    # CIF
    if fmt in ("cif", "all"):
        path = base / f"{filename}.cif"
        path.write_text(structure_to_cif(struct), encoding="utf-8")
        saved["cif"] = str(path)
        logger.info("已保存 CIF → %s", path)

    # POSCAR
    if fmt in ("poscar", "all"):
        path = base / f"{filename}.POSCAR"
        path.write_text(structure_to_poscar(struct, filename), encoding="utf-8")
        saved["poscar"] = str(path)
        logger.info("已保存 POSCAR → %s", path)

    return saved

# ──────────────────────────────────────────────
# 单条结果构建
# ──────────────────────────────────────────────

def _build_result(doc: Any) -> Optional[Dict[str, Any]]:
    """
    将单条 MP summary doc 转为标准结果 dict。

    前端展示字段：
      material_id, formula, space_group, crystal_system,
      a, b, c, alpha, beta, gamma,
      xyz（3Dmol渲染）, cif（下载）, poscar（下载）

    后端计算字段：
      _structure：pymatgen Structure 对象（FastAPI 序列化时需排除此字段）
    """
    try:
        struct: Structure = doc.structure
        if struct is None:
            logger.warning("material_id=%s 无结构数据，跳过", doc.material_id)
            return None

        # 转换为传统标准结构
        conv    = get_conventional(struct)
        lat     = conv.lattice
        mid     = str(doc.material_id)
        formula = doc.formula_pretty or conv.composition.reduced_formula
        comment = f"{formula} | {mid}"

        # symmetry 由 MP 直接返回（SymmetryData 对象）
        sym            = doc.symmetry
        space_group    = getattr(sym, "symbol",         "?") if sym else "?"
        crystal_system = getattr(sym, "crystal_system", "?") if sym else "?"

        return {
            # ── 前端展示 ──
            "material_id":    mid,
            "formula":        formula,
            "space_group":    space_group,
            "crystal_system": crystal_system,
            "a":     round(lat.a,     4),
            "b":     round(lat.b,     4),
            "c":     round(lat.c,     4),
            "alpha": round(lat.alpha, 2),
            "beta":  round(lat.beta,  2),
            "gamma": round(lat.gamma, 2),
            # ── 结构文件字符串（前端渲染 & 下载）──
            "xyz":    structure_to_xyz(conv, comment),
            "cif":    structure_to_cif(conv),
            "poscar": structure_to_poscar(conv, comment),
            # ── 后端计算用（路由层序列化时排除）──
            "_structure": conv,
        }

    except Exception as e:
        logger.warning("构建结果失败 %s：%s", getattr(doc, "material_id", "?"), e)
        return None

# ──────────────────────────────────────────────
# MPQueryService 主类
# ──────────────────────────────────────────────

class MPQueryService:
    """
    Materials Project 结构查询服务。

    核心用法：
        svc = MPQueryService(api_key="your-mp-api-key")

        # 查询（自动识别模式）
        results = svc.query("Fe2O3")
        results = svc.query("Fe-O")
        results = svc.query("Fe, O")
        results = svc.query("mp-126")

        # 获取 pymatgen Structure 对象（供后端计算）
        struct = svc.get_structure("mp-126")

        # 保存到本地磁盘
        svc.save_to_disk("mp-126", save_dir="./structures", fmt="all")
    """

    def __init__(
        self,
        api_key: str,
        only_stable: bool = True,
        max_results: int = 20,
        cache_ttl: int = 300,
    ):
        """
        api_key:      MP API Key（在 materialsproject.org 个人页面获取）
        only_stable:  True = 只返回稳定结构（energy_above_hull == 0）
        max_results:  单次查询最大返回数量
        cache_ttl:    缓存有效期（秒），避免重复请求
        """
        self.api_key     = api_key
        self.only_stable = only_stable
        self.max_results = max_results
        self._cache      = _TTLCache(ttl=cache_ttl)

    @_retry(max_attempts=3, backoff=1.0)
    def _fetch(self, **kwargs) -> List[Any]:
        """调用 MP API，kwargs 透传给 summary.search()。"""
        with MPRester(self.api_key) as mpr:
            return list(mpr.materials.summary.search(
                fields=_MP_FIELDS,
                **kwargs,
            ))

    def query(self, raw_input: str) -> List[Dict[str, Any]]:
        """
        ★ 对外主查询入口 ★

        接受用户单输入框的任意字符串，自动识别模式并返回结构列表。
        结果中的 "_structure" 键为 pymatgen Structure 对象，
        FastAPI 路由序列化时应将其排除（见路由示例）。
        """
        detected  = detect_query_mode(raw_input)
        mode      = detected.pop("_mode")
        cache_key = f"{mode}::{raw_input.strip()}"

        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("[cache] %s", cache_key)
            return cached

        logger.info("[query] mode=%s  input=%r", mode, raw_input.strip())

        # 稳定性过滤（material_id 查询不加此限制）
        if self.only_stable and mode != "material_id":
            detected["energy_above_hull"] = (0, 0)

        docs = self._fetch(**detected)

        results = []
        for doc in docs[:self.max_results]:
            item = _build_result(doc)
            if item:
                results.append(item)

        if not results:
            logger.info("[query] 无结果：%r", raw_input)

        self._cache.set(cache_key, results)
        return results

    def get_structure(self, material_id: str) -> Optional[Structure]:
        """
        直接返回指定 material_id 的 pymatgen Structure 对象。
        供后端计算模块（结构优化、表面切割等）直接调用。
        """
        results = self.query(material_id)
        if not results:
            return None
        return results[0].get("_structure")

    def save_to_disk(
        self,
        material_id: str,
        save_dir: str = "./structures",
        fmt: str = "all",
    ) -> Dict[str, str]:
        """
        查询指定 material_id 并将结构保存到本地磁盘。

        fmt: "xyz" | "cif" | "poscar" | "all"
        返回已保存文件的路径字典。
        """
        results = self.query(material_id)
        if not results:
            logger.warning("未找到 %s，无法保存", material_id)
            return {}

        item     = results[0]
        struct   = item["_structure"]
        filename = f"{item['material_id']}_{item['formula']}"

        return save_structure_to_disk(struct, save_dir, filename, fmt)

    def clear_cache(self) -> None:
        """清空查询缓存。"""
        self._cache.clear()
        logger.info("缓存已清空")