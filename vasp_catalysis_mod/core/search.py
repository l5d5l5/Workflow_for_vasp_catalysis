"""
search.py
───────────────────────────────────────────────
Materials Project 结构查询服务（前端精简返回版）

特性：
  - 单输入框自动识别查询模式：formula / elements / chemsys / material_id
  - 自动从 .env 读取 MP_API_KEY
  - 查询结果最多返回 10 条
  - 前端只返回核心字段：
      material_id
      formula
      band_gap
      cif
  - 保留 CIF / XYZ / POSCAR 转换工具
  - 保留本地保存结构功能
───────────────────────────────────────────────
"""

import os
import re
import time
import logging
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
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
# 常量
# ──────────────────────────────────────────────

MAX_RETURN_LIMIT = 10

_MP_FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "band_gap",
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
                        "[%s] 第 %d 次失败：%s，%.1fs 后重试",
                        func.__name__,
                        attempt,
                        exc,
                        wait,
                    )
                    time.sleep(wait)

        return wrapper

    return decorator


# ──────────────────────────────────────────────
# TTL 缓存
# ──────────────────────────────────────────────

class _TTLCache:
    def __init__(self, ttl: int = 300, maxsize: int = 256):
        self._ttl = ttl
        self._maxsize = maxsize
        self._store: Dict[str, Any] = {}
        self._ts: Dict[str, float] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                return None

            if time.time() - self._ts[key] > self._ttl:
                del self._store[key]
                del self._ts[key]
                return None

            return self._store[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self._maxsize:
                oldest = min(self._ts, key=lambda k: self._ts[k])
                del self._store[oldest]
                del self._ts[oldest]

            self._store[key] = value
            self._ts[key] = time.time()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._ts.clear()


# ──────────────────────────────────────────────
# 查询模式自动识别
# ──────────────────────────────────────────────

_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")


def detect_query_mode(raw: str) -> Dict[str, Any]:
    """
    根据用户输入自动识别查询模式。

    规则：
      1. mp-数字              -> material_id，例如 mp-149
      2. 含 "-" 且全为元素符号 -> chemsys，例如 Fe-O
      3. 逗号或空格分隔元素    -> elements，例如 Fe, O
      4. 其余                 -> formula，例如 VO2、TiO2、Pt

    返回值可以直接传入 MPRester.materials.summary.search。
    其中 "_mode" 仅供日志和缓存使用，调用前会 pop 掉。
    """
    query = raw.strip()

    if not query:
        raise ValueError("查询内容不能为空")

    # 规则 1：material_id
    if re.fullmatch(r"mp-\d+", query, flags=re.IGNORECASE):
        return {
            "_mode": "material_id",
            "material_ids": [query.lower()],
        }

    # 规则 2：chemsys，例如 Fe-O、V-O
    if "-" in query:
        parts = [p.strip() for p in query.split("-") if p.strip()]

        if len(parts) >= 2 and all(_ELEMENT_RE.fullmatch(p.capitalize()) for p in parts):
            return {
                "_mode": "chemsys",
                "chemsys": "-".join(p.capitalize() for p in parts),
            }

    # 规则 3：元素列表，例如 Fe, O 或 Fe O
    if "," in query or " " in query:
        parts = [p.strip() for p in re.split(r"[,\s]+", query) if p.strip()]

        if len(parts) >= 1 and all(_ELEMENT_RE.fullmatch(p.capitalize()) for p in parts):
            return {
                "_mode": "elements",
                "elements": [p.capitalize() for p in parts],
            }

    # 规则 4：默认按化学式查询，例如 VO2、TiO2、Pt
    return {
        "_mode": "formula",
        "formula": query,
    }


# ──────────────────────────────────────────────
# 结构转换工具函数
# ──────────────────────────────────────────────

def get_conventional(struct: Structure) -> Structure:
    """
    转换为传统标准结构 conventional cell。

    注意：
      该函数会改变晶胞表达方式。
      查询接口默认不使用 conventional cell，而是返回 MP 原始 structure 的 CIF。
    """
    try:
        return SpacegroupAnalyzer(struct, symprec=1e-3).get_conventional_standard_structure()
    except Exception:
        logger.warning("SpacegroupAnalyzer 失败，使用原始结构")
        return struct


def structure_to_xyz(struct: Structure, comment: str = "") -> str:
    """
    Structure -> XYZ 格式字符串。

    可用于 3Dmol.js 简单渲染。
    """
    lines = [str(len(struct)), comment or struct.composition.reduced_formula]

    for site in struct.sites:
        x, y, z = site.coords
        lines.append(f"{site.specie.symbol} {x:.6f} {y:.6f} {z:.6f}")

    return "\n".join(lines)


def structure_to_cif(struct: Structure) -> str:
    """
    Structure -> CIF 格式字符串。
    """
    try:
        return str(CifWriter(struct))
    except Exception as e:
        logger.warning("CIF 转换失败：%s", e)
        return ""


def structure_to_poscar(struct: Structure, comment: str = "") -> str:
    """
    Structure -> POSCAR 格式字符串。
    """
    try:
        return Poscar(
            struct,
            comment=comment or struct.composition.reduced_formula,
        ).get_str()
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

    fmt:
      - xyz
      - cif
      - poscar
      - all
    """
    base = Path(save_dir)
    base.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, str] = {}
    fmt = fmt.lower()

    if fmt in ("xyz", "all"):
        path = base / f"{filename}.xyz"
        path.write_text(structure_to_xyz(struct, filename), encoding="utf-8")
        saved["xyz"] = str(path)
        logger.info("已保存 XYZ -> %s", path)

    if fmt in ("cif", "all"):
        path = base / f"{filename}.cif"
        path.write_text(structure_to_cif(struct), encoding="utf-8")
        saved["cif"] = str(path)
        logger.info("已保存 CIF -> %s", path)

    if fmt in ("poscar", "all"):
        path = base / f"{filename}.POSCAR"
        path.write_text(structure_to_poscar(struct, filename), encoding="utf-8")
        saved["poscar"] = str(path)
        logger.info("已保存 POSCAR -> %s", path)

    return saved


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _normalize_limit(limit: Optional[int], default: int = MAX_RETURN_LIMIT) -> int:
    """
    将 limit 限制在 1 到 10。
    """
    if limit is None:
        limit = default

    try:
        limit = int(limit)
    except Exception:
        limit = default

    if limit < 1:
        return 1

    return min(limit, MAX_RETURN_LIMIT)


def _get_summary_client(mpr: MPRester):
    """
    兼容新版和旧版 mp-api。

    新版推荐：
      mpr.materials.summary

    旧版兼容：
      mpr.summary
    """
    if hasattr(mpr, "materials") and hasattr(mpr.materials, "summary"):
        return mpr.materials.summary

    return mpr.summary


def _build_result(doc: Any, use_conventional: bool = False) -> Optional[Dict[str, Any]]:
    """
    将单条 MP summary doc 转为前端需要的精简结果。

    返回字段：
      - material_id
      - formula
      - band_gap
      - cif
    """
    try:
        struct: Optional[Structure] = getattr(doc, "structure", None)

        if struct is None:
            logger.warning("material_id=%s 无结构数据，跳过", getattr(doc, "material_id", "?"))
            return None

        if use_conventional:
            struct = get_conventional(struct)

        material_id = str(getattr(doc, "material_id", ""))
        formula = getattr(doc, "formula_pretty", None) or struct.composition.reduced_formula
        band_gap = getattr(doc, "band_gap", None)
        cif = structure_to_cif(struct)

        if not cif:
            logger.warning("material_id=%s CIF 为空，跳过", material_id)
            return None

        return {
            "material_id": material_id,
            "formula": formula,
            "band_gap": band_gap,
            "cif": cif,
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

    推荐用法：
        svc = MPQueryService()
        data = svc.search("VO2", limit=1)

    返回：
        {
            "query": "VO2",
            "limit": 1,
            "count": 1,
            "items": [
                {
                    "material_id": "mp-xxx",
                    "formula": "VO2",
                    "band_gap": 0.9781,
                    "cif": "..."
                }
            ]
        }
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        only_stable: bool = False,
        max_results: int = MAX_RETURN_LIMIT,
        cache_ttl: int = 300,
        use_conventional: bool = False,
    ):
        """
        api_key:
          MP API Key。默认从 .env 中读取 MP_API_KEY。

        only_stable:
          是否只返回稳定结构。
          默认 False，因为像你测试的 VO2 结果可能 is_stable=False。

        max_results:
          单次查询最大返回数量，强制不超过 10。

        cache_ttl:
          缓存有效期，单位秒。

        use_conventional:
          是否将结构转换为 conventional cell 后再输出 CIF。
          默认 False，保持 MP 原始结构。
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("MP_API_KEY")

        if not self.api_key:
            raise RuntimeError("MP_API_KEY is missing. Please set MP_API_KEY in .env")

        self.only_stable = bool(only_stable)
        self.max_results = _normalize_limit(max_results)
        self.use_conventional = bool(use_conventional)
        self._cache = _TTLCache(ttl=cache_ttl)

        logger.info(
            "MPQueryService initialized. only_stable=%s, max_results=%s, use_conventional=%s",
            self.only_stable,
            self.max_results,
            self.use_conventional,
        )

    @_retry(max_attempts=3, backoff=1.0)
    def _fetch(self, limit: int, **kwargs) -> List[Any]:
        """
        调用 MP API。

        limit 会通过 chunk_size 控制，避免一次拉太多数据。
        """
        actual_limit = _normalize_limit(limit, self.max_results)

        with MPRester(self.api_key) as mpr:
            summary = _get_summary_client(mpr)

            docs = summary.search(
                fields=_MP_FIELDS,
                num_chunks=1,
                chunk_size=actual_limit,
                **kwargs,
            )

            return list(docs)

    def query(self, raw_input: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        查询 MP 结构，返回 items 列表。

        为兼容旧代码，保留 query() 方法。
        """
        query_text = raw_input.strip()
        actual_limit = _normalize_limit(limit, self.max_results)

        detected = detect_query_mode(query_text)
        mode = detected.pop("_mode")

        cache_key = (
            f"{mode}::{query_text}::limit={actual_limit}"
            f"::stable={self.only_stable}"
            f"::conv={self.use_conventional}"
        )

        cached = self._cache.get(cache_key)

        if cached is not None:
            logger.info("[cache] %s", cache_key)
            return cached

        logger.info("[query] mode=%s input=%r limit=%d", mode, query_text, actual_limit)

        # 稳定性过滤。
        # 注意：VO2 这类材料可能存在非稳定结构。
        # 默认 only_stable=False，避免查不到你测试出来的结果。
        if self.only_stable and mode != "material_id":
            detected["energy_above_hull"] = (0, 0)

        docs = self._fetch(limit=actual_limit, **detected)

        results: List[Dict[str, Any]] = []

        for doc in docs:
            item = _build_result(doc, use_conventional=self.use_conventional)

            if item:
                results.append(item)

            if len(results) >= actual_limit:
                break

        if not results:
            logger.info("[query] 无结果：%r", query_text)

        self._cache.set(cache_key, results)
        return results

    def search(self, query: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        查询 MP 结构，返回 FastAPI 路由更适合直接包装的 dict。

        推荐 main.py 中调用这个方法。
        """
        actual_limit = _normalize_limit(limit, self.max_results)
        items = self.query(query, limit=actual_limit)

        return {
            "query": query.strip(),
            "limit": actual_limit,
            "count": len(items),
            "items": items,
        }

    @_retry(max_attempts=3, backoff=1.0)
    def get_structure(self, material_id: str) -> Optional[Structure]:
        """
        直接获取指定 material_id 的 pymatgen Structure 对象。

        供后端计算模块使用。
        """
        material_id = material_id.strip().lower()

        if not re.fullmatch(r"mp-\d+", material_id, flags=re.IGNORECASE):
            raise ValueError(f"Invalid material_id: {material_id}")

        with MPRester(self.api_key) as mpr:
            summary = _get_summary_client(mpr)

            docs = summary.search(
                material_ids=[material_id],
                fields=["material_id", "structure"],
                num_chunks=1,
                chunk_size=1,
            )

            docs = list(docs)

            if not docs:
                return None

            struct = getattr(docs[0], "structure", None)

            if struct is None:
                return None

            if self.use_conventional:
                struct = get_conventional(struct)

            return struct

    def save_to_disk(
        self,
        material_id: str,
        save_dir: str = "./structures",
        fmt: str = "all",
    ) -> Dict[str, str]:
        """
        查询指定 material_id 并将结构保存到本地磁盘。

        fmt:
          - xyz
          - cif
          - poscar
          - all
        """
        struct = self.get_structure(material_id)

        if struct is None:
            logger.warning("未找到 %s，无法保存", material_id)
            return {}

        formula = struct.composition.reduced_formula
        filename = f"{material_id}_{formula}"

        return save_structure_to_disk(struct, save_dir, filename, fmt)

    def clear_cache(self) -> None:
        """
        清空查询缓存。
        """
        self._cache.clear()
        logger.info("缓存已清空")
