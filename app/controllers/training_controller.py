from fastapi import APIRouter, BackgroundTasks
from app.services.state import pipeline
from app.services.training_service import run_training_task
from app.models.schemas import TrainingStartResponse, TrainingStatus

router = APIRouter()

@router.post("/api/retrain", response_model=TrainingStartResponse)
async def retrain_model(background_tasks: BackgroundTasks):
    if pipeline.is_training:
        return {"status": "error", "message": "Training is already in progress"}
    
    background_tasks.add_task(run_training_task)
    return {
        "status": "started", 
        "message": "Training pipeline started in background. Monitor logs or /api/training_status for progress."
    }

@router.get("/api/training_status", response_model=TrainingStatus)
async def get_training_status():
    return {
        "is_training": pipeline.is_training,
        "status": pipeline.status
    }
