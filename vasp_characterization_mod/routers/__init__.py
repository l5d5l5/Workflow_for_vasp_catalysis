from fastapi import APIRouter
from routers import ir, xrd, nmr
# 后续新增：from routers import nmr, raman, xrd

router = APIRouter(prefix="/spectroscopy")
router.include_router(ir.router)
router.include_router(xrd.router)
router.include_router(nmr.router)

# 后续新增：XPS raman等技术
# router.include_router(raman.router)