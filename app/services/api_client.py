import httpx
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from app.config import AppConfig, logger


class APIClient:
    """
    Singleton API client that manages a shared httpx.AsyncClient for efficient
    connection pooling across multiple camera threads.
    """
    _instance: Optional["APIClient"] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._client: Optional[httpx.AsyncClient] = None
        self._api_url = AppConfig.API_URL
        self._api_token = AppConfig.API_TOKEN
        self._timeout = httpx.Timeout(10.0, connect=5.0)
        self._limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0
        )
        self._push_interval = 1.0
        self._last_push_times: Dict[str, datetime] = {}
        self._push_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._backend_cameras: List[Dict[str, Any]] = []
        self._camera_id_map: Dict[str, int] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._client is None or self._client.is_closed:
            async with self._lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        timeout=self._timeout,
                        limits=self._limits,
                        headers={
                            "Authorization": f"Bearer {self._api_token}",
                            "Content-Type": "application/json",
                            "Accept-Language": "en"
                        }
                    )
                    logger.info("[API] HTTP client initialized with connection pooling")
        return self._client

    async def update_token(self, token: Optional[str]) -> None:
        """Update the API token and reset the client."""
        self._api_token = token
        async with self._lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
            self._client = None
        logger.info("[API] Token updated, client reset")
        await self.fetch_backend_cameras()

    async def start(self):
        """Start the API client background worker."""
        if self._is_running:
            return
        
        self._is_running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        await self.fetch_backend_cameras()
        logger.info("[API] API client worker started")

    async def fetch_backend_cameras(self) -> List[Dict[str, Any]]:
        """Fetch available cameras from the backend API."""
        if not self._api_url or not self._api_token:
            return []
        
        url = f"{self._api_url}/api/cameras"
        
        try:
            client = await self._get_client()
            response = await client.get(url)
            
            if response.status_code == 200:
                self._backend_cameras = response.json()
                logger.info(f"[API] Fetched {len(self._backend_cameras)} cameras from backend")
                for cam in self._backend_cameras:
                    logger.info(f"[API]   - Camera ID: {cam.get('id')}, Name: {cam.get('name')}, Site: {cam.get('site_name')}")
                return self._backend_cameras
            else:
                logger.warning(f"[API] Failed to fetch cameras: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"[API] Error fetching cameras: {e}")
            return []

    def get_backend_cameras(self) -> List[Dict[str, Any]]:
        """Get the cached list of backend cameras."""
        return self._backend_cameras

    def register_camera(self, local_camera_id: str, backend_camera_id: int) -> None:
        """Register a mapping between local camera ID and backend camera ID."""
        self._camera_id_map[local_camera_id] = backend_camera_id
        logger.info(f"[API] Registered local camera '{local_camera_id}' -> backend ID {backend_camera_id}")

    def unregister_camera(self, local_camera_id: str) -> None:
        """Remove a camera ID mapping."""
        if local_camera_id in self._camera_id_map:
            del self._camera_id_map[local_camera_id]
            logger.info(f"[API] Unregistered camera '{local_camera_id}'")

    def get_backend_camera_id(self, local_camera_id: str) -> Optional[int]:
        """Get the backend camera ID for a local camera ID."""
        return self._camera_id_map.get(local_camera_id)

    async def stop(self):
        """Stop the API client and cleanup resources."""
        self._is_running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.info("[API] HTTP client closed")

    async def _process_queue(self):
        """Background worker that processes the push queue."""
        while self._is_running:
            try:
                payload = await asyncio.wait_for(
                    self._push_queue.get(),
                    timeout=1.0
                )
                await self._do_push(payload)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[API] Queue processing error: {e}")

    async def _do_push(self, payload: Dict[str, Any]):
        """Execute the actual API push."""
        if not self._api_url or not self._api_token:
            return
        
        url = f"{self._api_url}/agent/stats"
        
        try:
            print(payload)
            client = await self._get_client()
            response = await client.post(url, json=payload)
            
            if response.status_code == 401:
                logger.warning("[API] Received 401 Unauthorized during push. Attempting refresh...")
                from app.services.auth_service import auth_service
                if await auth_service.refresh_access_token():
                    client = await self._get_client()
                    response = await client.post(url, json=payload)
            
            if response.status_code not in [200, 201]:
                logger.warning(f"[API] Push failed with status {response.status_code}: {response.text}")
            else:
                camera = payload.get('counts', {}).get('camera_id')
                count = payload.get('counts', {}).get('counter', 0)
                logger.info(f"[API] Push successful - Camera: {camera}, Detections: {count}")
        except httpx.TimeoutException:
            logger.warning("[API] Push request timed out")
        except httpx.RequestError as e:
            logger.warning(f"[API] Request error: {e}")
        except Exception as e:
            logger.error(f"[API] Unexpected error: {e}")

    def should_push(self, camera_id: str) -> bool:
        """Check if enough time has passed to allow another push for this camera."""
        camera_key = str(camera_id)
        now = datetime.now()
        last_push = self._last_push_times.get(camera_key, datetime.min)
        
        if (now - last_push).total_seconds() >= self._push_interval:
            self._last_push_times[camera_key] = now
            return True
        return False

    def _extract_camera_index(self, camera_id: str) -> int:
        """Extract numeric camera index from camera_id string like 'cam_123456_1' -> 1"""
        backend_id = self._camera_id_map.get(camera_id)
        if backend_id is not None:
            return backend_id
        
        try:
            if camera_id.isdigit():
                return int(camera_id)
            parts = camera_id.split('_')
            if len(parts) >= 3:
                return int(parts[-1])
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
            return hash(camera_id) % 1000
        except:
            return hash(camera_id) % 1000

    async def push_detections(
        self,
        camera_id: str,
        detections: List[Dict[str, Any]],
        cross_count: int = 0,
        force: bool = False
    ):
        """
        Queue detection data to be pushed to the API.
        
        Args:
            camera_id: Identifier for the camera
            detections: List of detection dictionaries
            cross_count: Cumulative number of boundary crossings
            force: If True, bypass the rate limit check
        """
        if not force and not self.should_push(camera_id):
            return
        
        camera_index = self._extract_camera_index(camera_id)
        now = datetime.now()
        now_iso = now.isoformat()
        
        boxes = []
        male_count = 0
        female_count = 0
        
        for d in detections:
            gender = d.get('gender', 'Person')
            if gender == 'Male':
                gender_code = 'M'
                male_count += 1
            elif gender == 'Female':
                gender_code = 'F'
                female_count += 1
            else:
                continue
            
            boxes.append({
                "camera_id": camera_index,
                "date": now_iso,
                "bbox_id": d.get('id', -1),
                "bbox_left": int(d['bbox']['x1']),
                "bbox_top": int(d['bbox']['y1']),
                "bbox_w": int(d['bbox']['x2'] - d['bbox']['x1']),
                "bbox_h": int(d['bbox']['y2'] - d['bbox']['y1']),
                "gender": gender_code
            })
        
        payload = {
            "boxes": boxes,
            "counts": {
                "camera_id": camera_index,
                "date": now_iso,
                "counter": len(detections),
                "male_counter": male_count,
                "female_counter": female_count,
                "cross_counter": cross_count
            }
        }
        
        try:
            self._push_queue.put_nowait(payload)
            logger.info(f"[API] Queued push for camera {camera_id} ({len(detections)} detections, {male_count}M/{female_count}F)")
        except asyncio.QueueFull:
            logger.warning("[API] Push queue is full, dropping oldest item")
            try:
                self._push_queue.get_nowait()
                self._push_queue.put_nowait(payload)
            except:
                pass

    async def push_immediate(
        self,
        camera_id: str,
        detections: List[Dict[str, Any]]
    ):
        """
        Immediately push detection data to the API (bypasses queue).
        Use sparingly for critical updates.
        """
        now = datetime.now()
        now_iso = now.isoformat()
        
        boxes = []
        male_count = 0
        female_count = 0
        
        for d in detections:
            gender = d.get('gender', 'Person')
            if gender == 'Male':
                gender_code = 'M'
                male_count += 1
            elif gender == 'Female':
                gender_code = 'F'
                female_count += 1
            else:
                continue
            
            boxes.append({
                "camera_id": camera_id,
                "date": now_iso,
                "bbox_id": d.get('id', -1),
                "bbox_left": int(d['bbox']['x1']),
                "bbox_top": int(d['bbox']['y1']),
                "bbox_w": int(d['bbox']['x2'] - d['bbox']['x1']),
                "bbox_h": int(d['bbox']['y2'] - d['bbox']['y1']),
                "gender": gender_code
            })
        
        payload = {
            "boxes": boxes,
            "counts": {
                "camera_id": camera_id,
                "date": now_iso,
                "counter": len(detections),
                "male_counter": male_count,
                "female_counter": female_count
            }
        }
        
        await self._do_push(payload)


api_client = APIClient()
