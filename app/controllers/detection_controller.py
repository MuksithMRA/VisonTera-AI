from fastapi import APIRouter, Query, BackgroundTasks, WebSocket
from fastapi.responses import StreamingResponse
import asyncio
import json
from app.services.state import engine
from app.config import logger
from app.models.schemas import StartResponse, StopResponse

router = APIRouter()

@router.post("/api/start", response_model=StartResponse)
async def start_detection(
    camera: int = Query(0),
    confidence: float = Query(0.5),
    show_coords: str = Query("1"),
    show_fps: str = Query("1"),
    box_color: str = Query("00FF88"),
    background_tasks: BackgroundTasks = None
):
    if engine.is_running:
        return {"status": "already_running"}

    engine.camera_index = camera
    engine.confidence = max(0.1, min(1.0, confidence))
    engine.show_coords = show_coords == "1"
    engine.show_fps = show_fps == "1"

    r = int(box_color[4:6], 16)
    g = int(box_color[2:4], 16)
    b = int(box_color[0:2], 16)
    engine.box_color = (b, g, r)

    engine.is_running = True
    
    try:
        engine.capture_task = asyncio.create_task(engine.capture_frames())
        engine.stats_task = asyncio.create_task(engine.broadcast_stats())
        logger.info(f"Started detection with camera {camera}")
    except Exception as e:
        engine.is_running = False
        logger.error(f"Failed to start tasks: {e}")
        return {"status": "error", "message": str(e)}

    return {"status": "started"}

@router.post("/api/stop", response_model=StopResponse)
async def stop_detection():
    engine.is_running = False
    await asyncio.sleep(0.1)
    
    if engine.capture_task and not engine.capture_task.done():
        engine.capture_task.cancel()
    if engine.stats_task and not engine.stats_task.done():
        engine.stats_task.cancel()
    
    if engine.cap:
        engine.cap.release()
        engine.cap = None
    
    await asyncio.sleep(0.2)
    return {"status": "stopped"}

@router.get("/video_feed")
async def video_feed():
    async def frame_generator():
        while engine.is_running:
            try:
                frame_bytes, _, _, _, _ = await asyncio.wait_for(engine.frame_queue.get(), timeout=2.0)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                       + frame_bytes + b'\r\n')
            except asyncio.TimeoutError:
                pass

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            try:
                stats = await asyncio.wait_for(engine.stats_queue.get(), timeout=1.0)
                await websocket.send_text(json.dumps(stats))
            except asyncio.TimeoutError:
                pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
