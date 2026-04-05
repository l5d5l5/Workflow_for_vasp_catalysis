"""
Catalyst Workbench — FastAPI 主入口
路由模块：
  - /bulk   : 晶体结构修改
  - /db     : Materials Project 检索
  - /ml     : ML 预测（ASE EMT）
  - /store  : 用户结构本地存储（SQLite + 用户文件夹）
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import bulk, db_search, ml, store

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Catalyst Workbench API",
    description="化学计算AI平台 — 催化剂结构生成与修改接口",
    version="1.0.0",
    openapi_tags=[
        {"name": "bulk",   "description": "晶体结构（Bulk）操作"},
        {"name": "db",     "description": "Materials Project 数据库检索"},
        {"name": "ml",     "description": "ML 结构预测（ASE EMT）"},
        {"name": "store",  "description": "用户结构本地存储"},
        {"name": "slab",   "description": "表面结构生成（预留）"},
        {"name": "ads",    "description": "吸附结构生成（预留）"},
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(bulk.router,      prefix="/bulk",  tags=["bulk"])
app.include_router(db_search.router, prefix="/db",    tags=["db"])
app.include_router(ml.router,        prefix="/ml",    tags=["ml"])
app.include_router(store.router,     prefix="/store", tags=["store"])

# ── 预留路由（待实现）──────────────────────────────
from fastapi import APIRouter
slab_router = APIRouter()
ads_router  = APIRouter()

@slab_router.post("/generate", summary="从 Bulk 切割生成 Slab（预留）")
def slab_generate(): 
    from fastapi import HTTPException
    raise HTTPException(501, "Slab 生成模块尚未实现")

@ads_router.post("/generate", summary="在 Slab 上生成吸附结构（预留）")
def ads_generate():
    from fastapi import HTTPException
    raise HTTPException(501, "吸附结构生成模块尚未实现")

app.include_router(slab_router, prefix="/slab", tags=["slab"])
app.include_router(ads_router,  prefix="/ads",  tags=["ads"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)