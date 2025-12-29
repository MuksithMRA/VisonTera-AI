from app.utils import patch_torch_load
patch_torch_load()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.config import AppConfig, logger
from app.services.state import engine
from app.controllers.base_controller import router as base_router
from app.controllers.detection_controller import router as detection_router
from app.controllers.training_controller import router as training_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_model()
    logger.info("Application started")
    yield
    logger.info("Application shutdown")

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory=AppConfig.BASE_DIR / "static"), name="static")

app.include_router(base_router)
app.include_router(detection_router)
app.include_router(training_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
