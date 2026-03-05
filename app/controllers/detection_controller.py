from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import asyncio
import json
from app.services.state import camera_manager
from app.services.api_client import api_client
from app.infrastructure.inference_engine import InferenceEngine
from app.config import logger
from app.models.schemas import (
    CameraStartRequest,
    CameraResponse,
    StopResponse,
    MultiCameraStatusResponse,
    DetectionStatus
)

router = APIRouter()


# ── Model Management Endpoints ──

@router.get("/api/models/detection", tags=["Models"])
async def list_detection_models():
    """List all available detection models for hot-swapping."""
    engine = InferenceEngine()
    models = engine.get_available_detection_models()
    return {"models": models, "current": engine.current_detection_model_path}


@router.post("/api/models/detection/switch", tags=["Models"])
async def switch_detection_model(request: dict):
    """Switch the active detection model at runtime."""
    model_path = request.get("model_path")
    if not model_path:
        return {"status": "error", "message": "model_path is required"}
    
    engine = InferenceEngine()
    result = engine.switch_detection_model(model_path)
    return result


# ── Cross-Camera Re-ID Endpoints ──

@router.get("/api/reid/counts", tags=["Re-ID"])
async def get_deduplicated_counts():
    """Get globally deduplicated person counts across all cameras."""
    engine = InferenceEngine()
    counts = engine.get_deduplicated_counts()
    return counts


@router.get("/api/reid/visible", tags=["Re-ID"])
async def get_currently_visible(max_age: float = 5.0):
    """Get deduplicated counts for persons seen in the last N seconds."""
    engine = InferenceEngine()
    visible = engine.get_currently_visible_persons(max_age=max_age)
    return visible


@router.get("/api/reid/stats", tags=["Re-ID"])
async def get_reid_stats():
    """Get Re-ID system diagnostic statistics."""
    engine = InferenceEngine()
    stats = engine.get_reid_stats()
    return stats


@router.get("/api/models/detection/current", tags=["Models"])
async def get_current_detection_model():
    """Get the currently active detection model."""
    engine = InferenceEngine()
    path = engine.current_detection_model_path
    return {
        "path": path,
        "name": path.split("\\")[-1] if path else None,
        "device": engine._device
    }

@router.get("/api/cameras", tags=["Detection"])
async def get_available_cameras():
    cameras = await camera_manager.get_available_cameras()
    return {"cameras": cameras}


@router.get("/api/cameras/backend", tags=["Detection"])
async def get_backend_cameras():
    """Get cameras registered in the VisionTera backend."""
    cameras = api_client.get_backend_cameras()
    return {"cameras": cameras}


@router.get("/api/cameras/backend/refresh", tags=["Detection"])
async def refresh_backend_cameras():
    """Refresh the list of cameras from the VisionTera backend."""
    cameras = await api_client.fetch_backend_cameras()
    return {"cameras": cameras, "count": len(cameras)}


@router.get("/api/cameras/active", tags=["Detection"])
async def get_active_cameras():
    statuses = camera_manager.get_all_camera_statuses()
    return MultiCameraStatusResponse(
        total_cameras=len(statuses),
        cameras=statuses
    )


@router.get("/api/camera/{camera_id}/status", tags=["Detection"])
async def get_camera_status(camera_id: str):
    status = camera_manager.get_camera_status(camera_id)
    if status is None:
        return {"status": "not_found", "camera_id": camera_id}
    return status.to_dict()


@router.post("/api/camera/start", response_model=CameraResponse, tags=["Detection"])
async def start_camera(request: CameraStartRequest):
    r = int(request.box_color[4:6], 16) if len(request.box_color) >= 6 else 0
    g = int(request.box_color[2:4], 16) if len(request.box_color) >= 4 else 255
    b = int(request.box_color[0:2], 16) if len(request.box_color) >= 2 else 136
    box_color = (b, g, r)

    if request.backend_camera_id:
        api_client.register_camera(request.camera_id, request.backend_camera_id)

    result = await camera_manager.start_camera(
        camera_id=request.camera_id,
        source=request.source,
        name=request.name,
        confidence=request.confidence,
        show_coords=request.show_coords,
        show_fps=request.show_fps,
        box_color=box_color,
        counting_line=request.counting_line
    )

    return CameraResponse(**result)


@router.post("/api/start", response_model=CameraResponse, tags=["Detection"])
async def start_detection_legacy(
    camera: str = Query("0", description="Camera index or RTSP URL"),
    confidence: float = Query(0.5, ge=0.1, le=1.0),
    show_coords: str = Query("1"),
    show_fps: str = Query("1"),
    box_color: str = Query("00FF88")
):
    r = int(box_color[4:6], 16) if len(box_color) >= 6 else 0
    g = int(box_color[2:4], 16) if len(box_color) >= 4 else 255
    b = int(box_color[0:2], 16) if len(box_color) >= 2 else 136
    box_color_tuple = (b, g, r)

    camera_id = f"legacy_{camera}"

    result = await camera_manager.start_camera(
        camera_id=camera_id,
        source=camera,
        name=f"Camera {camera}",
        confidence=confidence,
        show_coords=show_coords == "1",
        show_fps=show_fps == "1",
        box_color=box_color_tuple
    )

    return CameraResponse(**result)


@router.post("/api/camera/{camera_id}/stop", response_model=StopResponse, tags=["Detection"])
async def stop_camera(camera_id: str):
    api_client.unregister_camera(camera_id)
    result = await camera_manager.stop_camera(camera_id)
    return StopResponse(**result)


@router.post("/api/stop", response_model=StopResponse, tags=["Detection"])
async def stop_all_cameras():
    result = await camera_manager.stop_all_cameras()
    return StopResponse(**result)


@router.get("/api/status", response_model=DetectionStatus, tags=["Detection"])
async def get_status():
    return DetectionStatus(
        status="ok",
        running=camera_manager.active_camera_count > 0,
        active_cameras=camera_manager.active_camera_count,
        total_cameras=camera_manager.total_camera_count
    )


@router.get("/video_feed/{camera_id}", tags=["Detection"])
async def video_feed(camera_id: str):
    processor = camera_manager.get_processor(camera_id)
    if processor is None:
        return {"error": "Camera not found", "camera_id": camera_id}

    async def frame_generator():
        while processor.is_running:
            try:
                frame_bytes = await asyncio.wait_for(
                    processor.frame_queue.get(),
                    timeout=2.0
                )
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                    + frame_bytes + b'\r\n'
                )
            except asyncio.TimeoutError:
                pass

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/video_feed", tags=["Detection"])
async def video_feed_legacy():
    active_ids = camera_manager.get_active_camera_ids()
    if not active_ids:
        return {"error": "No active cameras"}

    camera_id = active_ids[0]
    return await video_feed(camera_id)


@router.websocket("/ws/stats/{camera_id}")
async def websocket_camera_stats(websocket: WebSocket, camera_id: str):
    await websocket.accept()

    processor = camera_manager.get_processor(camera_id)
    if processor is None:
        await websocket.send_text(json.dumps({"error": "Camera not found"}))
        await websocket.close()
        return

    try:
        while processor.is_running:
            stats = processor.get_latest_stats()
            if stats:
                await websocket.send_text(json.dumps(stats.to_dict()))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for camera {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket error for camera {camera_id}: {e}")
    finally:
        await websocket.close()


@router.websocket("/ws/stats")
async def websocket_all_stats(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            all_stats = camera_manager.get_all_stats()
            if all_stats:
                message = {
                    "type": "all_stats",
                    "cameras": all_stats
                }
                await websocket.send_text(json.dumps(message))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for all stats")
    except Exception as e:
        logger.error(f"WebSocket error for all stats: {e}")
    finally:
        await websocket.close()
