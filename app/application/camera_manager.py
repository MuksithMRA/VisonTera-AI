import asyncio
import concurrent.futures
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
        self._batch_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._batch_loop_task: Optional[asyncio.Task] = None
        self._batch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="BatchInference"
        )

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
        counting_lines: List[List[List[float]]] = None
    ) -> Dict:
        if camera_id in self._processors:
            processor = self._processors[camera_id]
            if processor.is_running:
                return {"status": "already_running", "camera_id": camera_id}

        camera_type = CameraType.RTSP if source.startswith("rtsp://") else CameraType.LOCAL

        # Convert list of lines to list of list of tuples if provided
        lines_tuples = [[tuple(p) for p in line] for line in counting_lines] if counting_lines else None

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
            counting_lines=lines_tuples
        )

        processor = CameraProcessor(config, self._inference_engine)
        processor.set_batch_queue(self._batch_queue)
        self._processors[camera_id] = processor

        success = await processor.start()

        if success:
            if self._batch_loop_task is None or self._batch_loop_task.done():
                self._batch_loop_task = asyncio.create_task(
                    self._batch_inference_loop()
                )
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

        if self._batch_loop_task and not self._batch_loop_task.done():
            self._batch_loop_task.cancel()
            try:
                await self._batch_loop_task
            except asyncio.CancelledError:
                pass

        return {"status": "all_stopped", "count": len(camera_ids)}

    # ── Batched multi-camera inference loop ──────────────────────

    async def _batch_inference_loop(self) -> None:
        """Central loop that collects frames from all cameras and runs them
        through InferenceEngine.detect_persons_batch() for GPU-efficient
        batched inference."""
        loop = asyncio.get_running_loop()
        logger.info("Batch inference loop started")

        while True:
            batch = []
            try:
                # Wait for at least one camera to submit a frame
                try:
                    item = await asyncio.wait_for(
                        self._batch_queue.get(), timeout=0.1
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    continue

                # Short collection window to let other cameras submit too
                active = max(self.active_camera_count, 1)
                deadline = loop.time() + 0.005  # 5 ms
                while len(batch) < active:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._batch_queue.get(),
                            timeout=max(remaining, 0.001),
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                # Deduplicate: keep the latest frame per camera
                camera_frames = {}
                batch_futures: Dict[str, asyncio.Future] = {}
                for cam_id, frame, conf, future in batch:
                    if cam_id in batch_futures and not batch_futures[cam_id].done():
                        batch_futures[cam_id].set_result([])
                    camera_frames[cam_id] = (frame, conf)
                    batch_futures[cam_id] = future

                # Run batched inference in a dedicated thread
                try:
                    results = await loop.run_in_executor(
                        self._batch_executor,
                        self._inference_engine.detect_persons_batch,
                        camera_frames,
                    )
                except Exception as e:
                    logger.error(f"Batch inference failed: {e}", exc_info=True)
                    results = {}

                # Deliver results back to the waiting camera loops
                for cam_id, future in batch_futures.items():
                    if not future.done():
                        future.set_result(results.get(cam_id, []))

            except asyncio.CancelledError:
                for item in batch:
                    _, _, _, future = item
                    if not future.done():
                        future.set_result([])
                logger.info("Batch inference loop stopped")
                break
            except Exception as e:
                logger.error(f"Batch loop error: {e}", exc_info=True)
                for item in batch:
                    _, _, _, future = item
                    if not future.done():
                        future.set_result([])

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

    def update_counting_lines(self, camera_id: str, counting_lines: Optional[List[List[List[float]]]]) -> Dict:
        """Update the counting boundary lines on a running camera without restarting it."""
        processor = self._processors.get(camera_id)
        if not processor:
            return {"status": "not_found", "camera_id": camera_id}

        # Convert list-of-lists → list-of-tuples (same format as start_camera)
        lines_tuples = [[tuple(p) for p in line] for line in counting_lines] if counting_lines else None

        # Clone the current config and swap in the new line
        old = processor._config
        from dataclasses import replace as dc_replace
        new_config = dc_replace(old, counting_lines=lines_tuples)
        processor.update_config(new_config)

        # Reset cross-count so the new boundary starts fresh
        processor._cross_count = 0
        processor._line_counts = [0] * len(lines_tuples) if lines_tuples else []
        processor._track_paths = {}
        if hasattr(processor, '_counted_tracks'):
            processor._counted_tracks = set()

        logger.info(f"Counting lines updated for camera {camera_id}: {lines_tuples}")
        return {"status": "updated", "camera_id": camera_id}


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
