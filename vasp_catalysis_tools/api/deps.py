"""
公共依赖：用户身份、文件路径、SQLite 连接
用户名通过 HTTP Header X-User 传入（后续对接 PBS 认证）
"""
import sqlite3
from pathlib import Path
from typing import Generator

from fastapi import Header, HTTPException

# ── 配置 ──────────────────────────────────────────────
# 用户结构根目录，每个用户有独立子目录
# 生产环境改为 /home 或集群挂载路径
USER_BASE_DIR = Path("/home")
APP_SUBDIR    = "catalyst_workbench"

# SQLite 数据库路径（全局共享一个，按 username 区分记录）
DB_PATH = Path(__file__).parent.parent / "data" / "workbench.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── 用户目录 ──────────────────────────────────────────
def get_user_dir(username: str) -> Path:
    """返回并创建用户的结构存储目录"""
    user_dir = USER_BASE_DIR / username / APP_SUBDIR
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

# ── FastAPI 依赖：从 Header 获取当前用户 ──────────────
def get_current_user(x_user: str = Header(..., description="PBS 集群用户名")) -> str:
    """
    从请求头 X-User 获取用户名。
    后续替换为 PBS 认证 token 校验。
    """
    if not x_user or not x_user.isidentifier():
        raise HTTPException(status_code=401, detail="无效的用户名（X-User header）")
    return x_user

# ── SQLite 连接 ───────────────────────────────────────
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI 依赖注入：获取 SQLite 连接"""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row   # 让结果可以按列名访问
    try:
        yield conn
    finally:
        conn.close()

# ── 初始化数据库表 ────────────────────────────────────
def init_db():
    """应用启动时调用，创建所需表"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_structures (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL,
            struct_id   TEXT    NOT NULL UNIQUE,   -- 如 "MY-0001"
            name        TEXT,
            formula     TEXT,
            reduced_formula TEXT,
            spacegroup  TEXT,
            file_path   TEXT    NOT NULL,           -- 绝对路径
            fmt         TEXT    DEFAULT 'poscar',
            source      TEXT    DEFAULT 'upload',   -- upload | mp | modified
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_user ON user_structures(username);
    """)
    conn.commit()
    conn.close()

init_db()