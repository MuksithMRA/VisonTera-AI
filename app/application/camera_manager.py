import asyncio
from typing import Dict, List, Optional
from app.domain.entities import CameraConfig, CameraStatus, CameraType, CameraState
from app.application.camera_processor import CameraProcessor
from app.infrastructure.inference_engine import InferenceEngine
from app.config import AppConfig, logger


class CameraManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._processors: Dict[str, CameraProcessor] = {}
        self._inference_engine = InferenceEngine()
        self._models_loaded = False

    def load_models(self, detection_path: str = None, gender_path: str = None) -> None:
        if self._models_loaded:
            return
        detection_path = detection_path or AppConfig.MODEL_PATH
        gender_path = gender_path or AppConfig.GENDER_MODEL_PATH
        self._inference_engine.load_models(detection_path, gender_path)
        self._models_loaded = True

    async def get_available_cameras(self) -> List[dict]:
        from app.services.api_client import api_client
        backend_cameras = await api_client.fetch_backend_cameras()
        
        cameras = []
        for cam in backend_cameras:
            camera_url = cam.get("url")
            if camera_url:
                logger.info(f"[CameraManager] Found camera: {camera_url}")
            cameras.append({
                "id": str(cam.get("id")),
                "name": cam.get("name") or f"Camera {cam.get('id')}",
                "type": CameraType.RTSP.value if camera_url and "rtsp" in camera_url else CameraType.LOCAL.value, 
                "site_name": cam.get("site_name"),
                "url": camera_url 
            })

        return cameras

    async def start_camera(
        self,
        camera_id: str,
        source: str,
        name: str = None,
        confidence: float = 0.5,
        show_coords: bool = True,
        show_fps: bool = True,
        box_color: tuple = (0, 255, 136),
        counting_line: List[List[float]] = None
    ) -> Dict:
        if camera_id in self._processors:
            processor = self._processors[camera_id]
            if processor.is_running:
                return {"status": "already_running", "camera_id": camera_id}

        camera_type = CameraType.RTSP if source.startswith("rtsp://") else CameraType.LOCAL

        # Convert list of lists to list of tuples if provided
        line_tuples = [tuple(p) for p in counting_line] if counting_line else None

        config = CameraConfig(
            camera_id=camera_id,
            source=source,
            name=name or f"Camera {camera_id}",
            camera_type=camera_type,
            frame_width=AppConfig.FRAME_WIDTH,
            frame_height=AppConfig.FRAME_HEIGHT,
            target_fps=AppConfig.TARGET_FPS,
            confidence_threshold=confidence,
            show_coords=show_coords,
            show_fps=show_fps,
            box_color=box_color,
            counting_line=line_tuples
        )

        processor = CameraProcessor(config, self._inference_engine)
        self._processors[camera_id] = processor

        success = await processor.start()

        if success:
            logger.info(f"Camera {camera_id} started successfully")
            return {"status": "started", "camera_id": camera_id}
        else:
            status = processor.get_status()
            return {"status": "error", "camera_id": camera_id, "message": status.error_message}

    async def stop_camera(self, camera_id: str) -> Dict:
        if camera_id not in self._processors:
            return {"status": "not_found", "camera_id": camera_id}

        processor = self._processors[camera_id]
        await processor.stop()
        del self._processors[camera_id]

        logger.info(f"Camera {camera_id} stopped")
        return {"status": "stopped", "camera_id": camera_id}

    async def stop_all_cameras(self) -> Dict:
        tasks = []
        camera_ids = list(self._processors.keys())

        for camera_id in camera_ids:
            tasks.append(self.stop_camera(camera_id))

        if tasks:
            await asyncio.gather(*tasks)

        return {"status": "all_stopped", "count": len(camera_ids)}

    def get_camera_status(self, camera_id: str) -> Optional[CameraStatus]:
        if camera_id not in self._processors:
            return None
        return self._processors[camera_id].get_status()

    def get_all_camera_statuses(self) -> List[Dict]:
        return [
            processor.get_status().to_dict()
            for processor in self._processors.values()
        ]

    def get_active_camera_ids(self) -> List[str]:
        return [
            camera_id
            for camera_id, processor in self._processors.items()
            if processor.is_running
        ]

    def get_processor(self, camera_id: str) -> Optional[CameraProcessor]:
        return self._processors.get(camera_id)

    def get_latest_frame(self, camera_id: str) -> Optional[bytes]:
        processor = self._processors.get(camera_id)
        if processor:
            return processor.get_latest_frame()
        return None

    def get_latest_stats(self, camera_id: str) -> Optional[Dict]:
        processor = self._processors.get(camera_id)
        if processor:
            stats = processor.get_latest_stats()
            if stats:
                return stats.to_dict()
        return None

    def get_all_stats(self) -> List[Dict]:
        all_stats = []
        for processor in self._processors.values():
            stats = processor.get_latest_stats()
            if stats:
                all_stats.append(stats.to_dict())
        return all_stats

    @property
    def active_camera_count(self) -> int:
        return sum(1 for p in self._processors.values() if p.is_running)

    @property
    def total_camera_count(self) -> int:
        return len(self._processors)
