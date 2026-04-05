import cv2
import asyncio
import numpy as np
import concurrent.futures
from datetime import datetime
from typing import Optional, List
from app.domain.interfaces import ICameraProcessor
from app.domain.entities import (
    CameraConfig, CameraStatus, CameraState, ProcessingStats, Detection
)
from app.infrastructure.inference_engine import InferenceEngine
from app.infrastructure.camera_capture import CameraCapture
from app.services.api_client import api_client
from app.services.dataset_collector import DatasetCollector
from app.config import AppConfig, logger


class CameraProcessor(ICameraProcessor):
    def __init__(self, config: CameraConfig, inference_engine: InferenceEngine):
        self._config = config
        self._inference_engine = inference_engine
        self._capture = CameraCapture()
        self._dataset_collector = DatasetCollector()
        self._state = CameraState.IDLE
        self._error_message: Optional[str] = None
        self._frames_processed = 0
        self._last_frame_time: Optional[datetime] = None
        self._latest_frame: Optional[bytes] = None
        self._latest_stats: Optional[ProcessingStats] = None
        self._latest_detections: List[Detection] = []
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._intersection_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix='IntersectionExecutor')
        self._capture_task: Optional[asyncio.Task] = None
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
        self._stats_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._running = False
        self._fps = 0.0
        self._actual_width = config.frame_width
        self._actual_height = config.frame_height
        self._track_paths = {} # {track_id: last_point}
        self._cross_count = 0
        self._line_counts = [0] * len(config.counting_lines) if config.counting_lines else []
        self._counted_tracks = set()
        self._batch_queue: Optional[asyncio.Queue] = None

    def set_batch_queue(self, queue: asyncio.Queue) -> None:
        """Attach the shared batch-inference queue managed by CameraManager."""
        self._batch_queue = queue

    async def start(self) -> bool:
        if self._state == CameraState.RUNNING:
            return True

        self._state = CameraState.STARTING
        self._running = True
        self._error_message = None

        try:
            loop = asyncio.get_running_loop()
            opened = await loop.run_in_executor(
                self._executor,
                self._capture.open,
                self._config.source,
                self._config.frame_width,
                self._config.frame_height
            )

            if not opened:
                self._state = CameraState.ERROR
                self._error_message = f"Failed to open camera: {self._config.source}"
                logger.error(self._error_message)
                return False

            dims = self._capture.get_frame_dimensions()
            if dims[0] > 0 and dims[1] > 0:
                self._actual_width, self._actual_height = dims

            self._state = CameraState.RUNNING
            self._capture_task = asyncio.create_task(self._capture_loop())
            logger.info(f"Camera {self._config.camera_id} started: {self._actual_width}x{self._actual_height}")
            return True

        except Exception as e:
            self._state = CameraState.ERROR
            self._error_message = str(e)
            logger.error(f"Error starting camera {self._config.camera_id}: {e}")
            return False

    async def stop(self) -> None:
        if self._state == CameraState.IDLE:
            return

        self._state = CameraState.STOPPING
        self._running = False

        if self._capture_task and not self._capture_task.done():
            self._capture_task.cancel()
            try:
                await asyncio.wait_for(self._capture_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        self._capture.release()
        self._inference_engine.clear_track_history(self._config.camera_id)
        self._state = CameraState.IDLE
        logger.info(f"Camera {self._config.camera_id} stopped")

    async def _capture_loop(self) -> None:
        target_fps = self._config.target_fps
        frame_interval = 1.0 / target_fps
        failed_reads = 0
        max_failed_reads = AppConfig.MAX_FAILED_READS
        prev_time = datetime.now()

        try:
            while self._running:
                loop_start = datetime.now()
                loop = asyncio.get_running_loop()

                # 1. Capture frame (blocking I/O in thread)
                frame = await loop.run_in_executor(
                    self._executor, self._capture_frame
                )

                if frame is None:
                    failed_reads += 1
                    if failed_reads >= max_failed_reads:
                        await self._handle_reconnect()
                        failed_reads = 0
                    await asyncio.sleep(0.1)
                    continue

                failed_reads = 0

                # 2. Inference: batch queue (preferred) or direct fallback
                if self._batch_queue is not None:
                    result_future = loop.create_future()
                    await self._batch_queue.put((
                        self._config.camera_id,
                        frame,
                        self._config.confidence_threshold,
                        result_future,
                    ))
                    detections = await result_future
                else:
                    detections = await loop.run_in_executor(
                        self._executor, self._detect_direct, frame
                    )

                # 3. Post-process in thread (drawing, crossing, encoding)
                result = await loop.run_in_executor(
                    self._executor,
                    self._post_process_frame,
                    frame,
                    detections,
                )

                if result is None:
                    continue

                self._frames_processed += 1
                self._last_frame_time = datetime.now()

                current_time = datetime.now()
                time_diff = (current_time - prev_time).total_seconds()
                self._fps = 1.0 / time_diff if time_diff > 0 else 0
                prev_time = current_time

                self._latest_frame = result['frame_bytes']
                self._latest_detections = result['detections']
                self._update_stats()

                await self._push_stats_to_api()

                try:
                    if self._frame_queue.full():
                        self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(result['frame_bytes'])
                except Exception:
                    pass

                process_time = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, frame_interval - process_time)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info(f"Camera {self._config.camera_id} capture loop cancelled")
        except Exception as e:
            self._state = CameraState.ERROR
            self._error_message = str(e)
            logger.error(f"Error in camera {self._config.camera_id} capture loop: {e}")

    # ── Frame pipeline helpers ───────────────────────────────────

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Blocking read from the camera. Runs inside a thread executor."""
        ret, frame = self._capture.read()
        if not ret or frame is None:
            return None
        return frame

    def _detect_direct(self, frame: np.ndarray) -> List[Detection]:
        """Fallback: per-camera inference when no batch queue is attached."""
        return self._inference_engine.detect_persons(
            frame, self._config.confidence_threshold, self._config.camera_id
        )

    def _post_process_frame(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> Optional[dict]:
        """Dataset collection, boundary crossing, annotation, and JPEG encoding."""
        self._dataset_collector.process_frame(
            self._config.camera_id, frame, detections
        )

        # --- Boundary Crossing Detection ---
        if self._config.counting_lines and len(self._config.counting_lines) > 0:
            current_ids = set()
            new_crossings = 0
            tasks = []
            
            for det in detections:
                if getattr(det, 'is_employee', False):
                    continue
                track_id = det.track_id
                if track_id == -1: continue
                current_ids.add(track_id)
                
                if det.class_id == 1:
                    curr_pos = (det.bbox.center_x, det.bbox.center_y)
                else:
                    curr_pos = (det.bbox.center_x, det.bbox.y2)
                
                if track_id in self._track_paths:
                    prev_pos = self._track_paths[track_id]
                    if self._config.counting_lines:
                        for idx, line in enumerate(self._config.counting_lines):
                            if len(line) >= 2:
                                if (track_id, idx) not in self._counted_tracks:
                                    L1, L2 = line[0], line[1]
                                    tasks.append((prev_pos, curr_pos, L1, L2, idx, track_id))
                
                self._track_paths[track_id] = curr_pos
            
            if tasks:
                def check_intersection_wrapper(args):
                    prev, curr, l1, l2, idx, tid = args
                    return self._check_line_intersection(prev, curr, l1, l2), idx, tid
                
                results = list(self._intersection_executor.map(check_intersection_wrapper, tasks))
                
                for is_intersect, idx, tid in results:
                    if is_intersect:
                        if idx < len(self._line_counts):
                            self._line_counts[idx] += 1
                        self._cross_count += 1
                        new_crossings += 1
                        self._counted_tracks.add((tid, idx))
                
                if new_crossings > 0:
                    logger.info(f"Boundary crossed! Total: {self._cross_count}, Lines: {self._line_counts}")
            
            to_remove = [tid for tid in self._track_paths if tid not in current_ids]
            for tid in to_remove:
                del self._track_paths[tid]
                self._counted_tracks = {item for item in self._counted_tracks if item[0] != tid}

        annotated = self._draw_detections(frame, detections)

        if self._config.show_fps:
            cv2.putText(
                annotated,
                f"FPS: {self._fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 136),
                2
            )

        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return {
            'frame_bytes': buffer.tobytes(),
            'detections': detections
        }

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated = frame.copy()
        viz_mode = getattr(AppConfig, 'VISUALIZATION_MODE', 'box')

        # --- Draw Boundary Lines ---
        if self._config.counting_lines:
            for i, line in enumerate(self._config.counting_lines):
                if len(line) >= 2:
                    L1 = tuple(map(int, line[0]))
                    L2 = tuple(map(int, line[1]))
                    cv2.line(annotated, L1, L2, (0, 0, 255), 4) # Thicker Red Line
                    
                    # Position text in the middle of each line
                    mid_x = (L1[0] + L2[0]) // 2
                    mid_y = (L1[1] + L2[1]) // 2
                    line_count = self._line_counts[i] if i < len(self._line_counts) else 0
                    label = f"L{i+1}: {line_count}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated, (mid_x - 5, mid_y - h - 15), (mid_x + w + 5, mid_y - 5), (0, 0, 255), -1)
                    cv2.putText(annotated, label, (mid_x, mid_y - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for det in detections:
            bbox = det.bbox
            x1, y1 = int(bbox.x1), int(bbox.y1)
            x2, y2 = int(bbox.x2), int(bbox.y2)
            bottom_x, bottom_y = int(bbox.center_x), int(bbox.y2)
            
            # Head position estimation: center top of the box
            # If the detector is already hitting heads, use the center of the box.
            if det.class_id == 1:
                center_x, center_y = int(bbox.center_x), int(bbox.center_y)
            else:
                center_x, center_y = int(bbox.center_x), int(y1 + (y2 - y1) * 0.15)
            
            gender = det.gender or "Person"
            is_employee = getattr(det, 'is_employee', False)

            if is_employee:
                box_color = (128, 128, 128)
            elif gender == 'Male':
                box_color = (255, 150, 50)
            elif gender == 'Female':
                box_color = (180, 105, 255)
            else:
                box_color = self._config.box_color
                
            if viz_mode == 'head-dot':
                # --- HEAD DOT MODE ---
                if getattr(AppConfig, 'SHOW_GLOW_EFFECT', True):
                    overlay = annotated.copy()
                    cv2.circle(overlay, (center_x, center_y), 15, box_color, -1)
                    cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
                
                # Core Dot
                cv2.circle(annotated, (center_x, center_y), 6, (255, 255, 255), -1)
                cv2.circle(annotated, (center_x, center_y), 7, box_color, 2)
                
                # Minimalist ID: G{n} = Re-ID global id; T{n} = tracker-only (Re-ID pending / skipped)
                emp_tag = " EMP" if is_employee else ""
                if det.global_id >= 0:
                    label = f"G{det.global_id}{emp_tag}"
                else:
                    label = f"T{det.track_id}{emp_tag}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated, (center_x - 5, center_y - label_h - 15), 
                              (center_x + label_w + 5, center_y - 10), box_color, -1)
                cv2.putText(annotated, label, (center_x, center_y - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                # --- TRADITIONAL BOX MODE ---
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                cv2.rectangle(annotated, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 100, 50), 1)
                cv2.circle(annotated, (bottom_x, bottom_y), 8, (0, 212, 255), -1)
                cv2.circle(annotated, (bottom_x, bottom_y), 10, (255, 255, 255), 2)

                emp_tag = " [EMP]" if is_employee else ""
                id_part = f"G{det.global_id}" if det.global_id >= 0 else f"T{det.track_id}"
                label = f"ID:{id_part} {gender}{emp_tag} {det.confidence:.0%}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), box_color, -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                if self._config.show_coords:
                    coord_text = f"({bottom_x}, {bottom_y})"
                    cv2.putText(annotated, coord_text, (bottom_x + 15, bottom_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 2)

        return annotated

    def _update_stats(self) -> None:
        visitor_detections = [d for d in self._latest_detections if not getattr(d, 'is_employee', False)]
        male_count = sum(1 for d in visitor_detections if d.gender == 'Male')
        female_count = sum(1 for d in visitor_detections if d.gender == 'Female')

        # Get deduplicated counts from the Re-ID manager
        dedup_counts = self._inference_engine.get_currently_visible_persons(max_age=5.0)

        self._latest_stats = ProcessingStats(
            camera_id=self._config.camera_id,
            fps=self._fps,
            frame_width=self._actual_width,
            frame_height=self._actual_height,
            person_count=len(visitor_detections),
            total_detected=len(self._latest_detections),
            employees_excluded=len(self._latest_detections) - len(visitor_detections),
            male_count=male_count,
            female_count=female_count,
            detections=[d.to_dict() for d in visitor_detections],
            timestamp=datetime.now(),
            cross_count=self._cross_count,
            line_counts=self._line_counts.copy()
        )
        # Attach deduplicated counts to stats for frontend consumption
        self._latest_stats.deduplicated = dedup_counts

    async def _push_stats_to_api(self) -> None:
        if not self._latest_detections:
            return
        
        visitor_detections = [d for d in self._latest_detections if not getattr(d, 'is_employee', False)]
        detections_for_api = []
        for det in visitor_detections:
            detections_for_api.append({
                'id': det.track_id,
                'global_id': det.global_id,
                'gender': det.gender or 'Person',
                'bbox': {
                    'x1': det.bbox.x1,
                    'y1': det.bbox.y1,
                    'x2': det.bbox.x2,
                    'y2': det.bbox.y2
                }
            })
        
        await api_client.push_detections(
            camera_id=self._config.camera_id,
            detections=detections_for_api,
            cross_count=self._cross_count,
            line_counts=self._line_counts.copy()
        )

    async def _handle_reconnect(self) -> None:
        self._state = CameraState.RECONNECTING
        logger.warning(f"Camera {self._config.camera_id} reconnecting...")

        self._capture.release()
        await asyncio.sleep(AppConfig.RETRY_DELAY_SECONDS)

        loop = asyncio.get_running_loop()
        opened = await loop.run_in_executor(
            self._executor,
            self._capture.open,
            self._config.source,
            self._config.frame_width,
            self._config.frame_height
        )

        if opened:
            self._state = CameraState.RUNNING
            logger.info(f"Camera {self._config.camera_id} reconnected")
        else:
            self._state = CameraState.ERROR
            self._error_message = "Failed to reconnect"

    def get_status(self) -> CameraStatus:
        return CameraStatus(
            camera_id=self._config.camera_id,
            state=self._state,
            config=self._config,
            stats=self._latest_stats,
            error_message=self._error_message,
            frames_processed=self._frames_processed,
            last_frame_time=self._last_frame_time
        )

    def get_latest_frame(self) -> Optional[bytes]:
        return self._latest_frame

    def get_latest_stats(self) -> Optional[ProcessingStats]:
        return self._latest_stats

    def update_config(self, config: CameraConfig) -> None:
        self._config = config

    @property
    def frame_queue(self) -> asyncio.Queue:
        return self._frame_queue

    @property
    def is_running(self) -> bool:
        return self._state == CameraState.RUNNING

    def _check_line_intersection(self, p1, p2, l1, l2):
        """Standard line segment intersection algorithm."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        # Segment 1 is p1-p2, Segment 2 is l1-l2
        return ccw(p1,l1,l2) != ccw(p2,l1,l2) and ccw(p1,p2,l1) != ccw(p1,p2,l2)
