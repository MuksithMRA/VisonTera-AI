import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Optional, List
from pathlib import Path
from torchvision import models

from app.config import AppConfig, logger
from app.infrastructure.osnet_reid import osnet_x1_0, load_osnet_reid_checkpoint


def compute_reid_crop_quality(crop_bgr: np.ndarray, detection_confidence: float) -> float:
    """Heuristic quality in [0,1]: sharpness, size, detector confidence."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = min(lap_var / 300.0, 1.0)
    area_score = min((h * w) / float(96 * 48), 1.0)
    conf = float(np.clip(detection_confidence, 0.0, 1.0))
    return float(0.4 * blur_score + 0.3 * area_score + 0.3 * conf)


class GeMPooling(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p).flatten(1)


class BNNeck(nn.Module):
    def __init__(self, in_features: int, out_features: int = 512):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.bn.bias.requires_grad_(False)
        self.fc = nn.Linear(in_features, out_features, bias=False)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out")
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.fc(x)
        return x


class FastReIDExtractor(nn.Module):
    """Legacy ResNet50 + GeM + BNNeck (optional custom .pt weights)."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = GeMPooling(p=3.0)
        self.neck = BNNeck(in_features=2048, out_features=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.pool(features)
        embeddings = self.neck(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ReIDFeatureExtractor:
    INPUT_SIZE = (256, 128)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    EMBEDDING_DIM = 512

    def __init__(self, device: str = "cuda", use_half: bool = False):
        self._device = device
        self._use_half = use_half and device == "cuda"
        self._model: Optional[nn.Module] = None
        self._loaded = False
        self._backend = "none"

    def load(self, model_path: Optional[str] = None) -> bool:
        try:
            candidates: List[Path] = []
            resolved = self._resolve_weights_path(model_path)
            if resolved is not None:
                candidates.append(resolved)
            if model_path:
                mp = Path(model_path)
                if mp.exists() and mp.resolve() not in {c.resolve() for c in candidates}:
                    candidates.append(mp)
            for p in candidates:
                if self._try_load_osnet(p):
                    return True
            for p in candidates:
                if self._try_load_legacy(str(p)):
                    return True
            if self._try_load_legacy(None):
                return True
            logger.error(
                "Re-ID: no usable weights. Place OSNet Market-1501 weights at "
                f"{self._default_osnet_path()} or set REID_WEIGHTS_PATH."
            )
            self._loaded = False
            self._model = None
            self._backend = "none"
            return False
        except Exception as e:
            logger.error(f"Failed to load Re-ID model: {e}")
            self._loaded = False
            self._model = None
            self._backend = "none"
            return False

    def _default_osnet_path(self) -> Path:
        return (
            AppConfig.BASE_DIR
            / "infrastructure"
            / "models"
            / "reid"
            / AppConfig.REID_WEIGHTS_FILENAME
        )

    def _resolve_weights_path(self, model_path: Optional[str]) -> Optional[Path]:
        if model_path:
            p = Path(model_path)
            if p.exists():
                return p
        env = AppConfig.REID_WEIGHTS_PATH
        if env:
            p = Path(env)
            if p.exists():
                return p
        default = self._default_osnet_path()
        if default.exists():
            return default
        if AppConfig.REID_AUTO_DOWNLOAD_WEIGHTS:
            default.parent.mkdir(parents=True, exist_ok=True)
            if self._download_market1501_osnet(default):
                return default
        legacy = self._find_legacy_reid_weights()
        if legacy and legacy.exists():
            return legacy
        return None

    def _download_market1501_osnet(self, dest: Path) -> bool:
        try:
            import gdown
        except ImportError:
            logger.warning("Re-ID auto-download needs gdown: pip install gdown")
            return False
        url = f"https://drive.google.com/uc?id={AppConfig.REID_GDRIVE_MARKET1501_FILE_ID}"
        try:
            gdown.download(url, str(dest), quiet=False)
            return dest.exists() and dest.stat().st_size > 1000
        except Exception as e:
            logger.warning(f"Re-ID weight download failed: {e}")
            return False

    def _find_legacy_reid_weights(self) -> Optional[Path]:
        search_dirs = [
            AppConfig.BASE_DIR / "infrastructure" / "models" / "reid",
            AppConfig.BASE_DIR / "infrastructure" / "models",
        ]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for pattern in [
                "*reid*.pt",
                "*reid*.pth",
                "*sbs*.pt",
                "*sbs*.pth",
                "*re_id*.pt",
                "*re-id*.pt",
            ]:
                matches = list(search_dir.glob(pattern))
                if matches:
                    return max(matches, key=lambda p: p.stat().st_mtime)
        return None

    def _try_load_osnet(self, weights_path: Path) -> bool:
        model = osnet_x1_0(
            num_classes=AppConfig.REID_NUM_CLASSES,
            pretrained=False,
            loss="softmax",
        )
        n = load_osnet_reid_checkpoint(model, str(weights_path))
        if n < 10:
            logger.warning(
                f"Re-ID OSNet load matched only {n} keys from {weights_path}; trying legacy backbone."
            )
            return False
        model.eval()
        self._model = model.to(self._device)
        if self._use_half:
            self._model.half()
        self._backend = "osnet_market1501"
        self._loaded = True
        self._warmup()
        logger.info(
            f"Re-ID OSNet x1.0 loaded ({n} tensors) from {weights_path}, "
            f"device={self._device}, half={self._use_half}"
        )
        return True

    def _try_load_legacy(self, model_path: Optional[str]) -> bool:
        path = model_path or self._find_legacy_reid_weights()
        if path is None or not Path(path).exists():
            return False
        path = Path(path)
        self._model = FastReIDExtractor(embedding_dim=self.EMBEDDING_DIM)
        try:
            state_dict = torch.load(path, map_location="cpu")
            if isinstance(state_dict, dict):
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
            missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(
                    f"Legacy Re-ID loaded with {len(missing)} missing keys from {path}"
                )
        except Exception as e:
            logger.warning(f"Legacy Re-ID load failed ({path}): {e}")
            self._model = None
            return False
        self._model.to(self._device)
        if self._use_half:
            self._model.half()
        self._model.eval()
        self._backend = "legacy_resnet_gemn"
        self._loaded = True
        self._warmup()
        logger.info(f"Re-ID legacy backbone loaded from {path}")
        return True

    def _warmup(self) -> None:
        dummy = torch.randn(1, 3, *self.INPUT_SIZE).to(self._device)
        if self._use_half:
            dummy = dummy.half()
        with torch.no_grad():
            self._model(dummy)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend(self) -> str:
        return self._backend

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(crop_bgr, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        return torch.from_numpy(np.ascontiguousarray(img))

    def _embeddings_forward(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self._model(batch)
        if self._backend == "osnet_market1501":
            emb = F.normalize(out, p=2, dim=1)
        else:
            emb = out
        return emb

    def extract(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self._loaded or self._model is None:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 48 or w < 24:
            return None
        try:
            tensor = self._preprocess(crop_bgr).unsqueeze(0).to(self._device)
            if self._use_half:
                tensor = tensor.half()
            else:
                tensor = tensor.float()
            embedding = self._embeddings_forward(tensor)
            emb = embedding.cpu().numpy().flatten().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm < 1e-8 or np.isnan(norm):
                return None
            if self._backend != "osnet_market1501":
                emb = emb / norm
            return emb
        except Exception as e:
            logger.error(f"Re-ID feature extraction error: {e}")
            return None

    def extract_batch(self, crops_bgr: List[np.ndarray]) -> Optional[np.ndarray]:
        if not self._loaded or self._model is None or not crops_bgr:
            return None
        valid_crops, valid_indices = [], []
        for i, c in enumerate(crops_bgr):
            h, w = c.shape[:2]
            if h >= 48 and w >= 24:
                valid_crops.append(c)
                valid_indices.append(i)
        if not valid_crops:
            return None
        try:
            tensors = [self._preprocess(c) for c in valid_crops]
            batch = torch.stack(tensors).to(self._device)
            if self._use_half:
                batch = batch.half()
            else:
                batch = batch.float()
            embeddings = self._embeddings_forward(batch)
            embs = embeddings.cpu().numpy()
            result = np.zeros((len(crops_bgr), self.EMBEDDING_DIM), dtype=np.float32)
            result[:] = np.nan
            for j, orig_idx in enumerate(valid_indices):
                emb = embs[j].flatten().astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm > 1e-8 and not np.isnan(norm):
                    if self._backend == "osnet_market1501":
                        result[orig_idx] = emb
                    else:
                        result[orig_idx] = emb / norm
            return result
        except Exception as e:
            logger.error(f"Re-ID batch extraction error: {e}")
            return None
