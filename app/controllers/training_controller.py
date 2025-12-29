from fastapi import APIRouter, BackgroundTasks
from app.services.state import pipeline
from app.services.training_service import run_training_task
from app.models.schemas import TrainingStartResponse, TrainingStatus

router = APIRouter(tags=["Training"])

@router.post(
    "/api/retrain",
    response_model=TrainingStartResponse,
    summary="Start Model Retraining",
    description="""
Initiates a background training task to retrain the detection model.

**Process:**
1. Checks if training is already in progress
2. If not, starts a background training pipeline
3. Returns immediately while training continues in the background

**Monitor Progress:**
- Check `/api/training_status` for current training status
- View application logs for detailed progress

**Note:** Only one training session can run at a time.
"""
)
async def retrain_model(background_tasks: BackgroundTasks):
    if pipeline.is_training:
        return {"status": "error", "message": "Training is already in progress"}
    
    background_tasks.add_task(run_training_task)
    return {
        "status": "started", 
        "message": "Training pipeline started in background. Monitor logs or /api/training_status for progress."
    }

@router.get(
    "/api/training_status",
    response_model=TrainingStatus,
    summary="Get Training Status",
    description="Returns the current status of the model training pipeline including whether training is active and the current status message."
)
async def get_training_status():
    return {
        "is_training": pipeline.is_training,
        "status": pipeline.status
    }
