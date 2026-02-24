"""
FastReID Strong Baseline (SBS) Re-ID Feature Extractor.

Architecture: ResNet50-IBN-a → GeM Pooling → BNNeck → FC(2048→512) → L2-norm
Produces 512-dimensional appearance embeddings for cross-camera person matching.

Supports:
  - Pre-trained Re-ID weights from infrastructure/models/reid/
  - ImageNet fallback when Re-ID weights are unavailable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Optional, List
from pathlib import Path
from torchvision import models
from app.config import logger


class GeMPooling(nn.Module):
    """Generalized Mean Pooling (GeM).
    
    Better than Global Average Pooling for Re-ID because it emphasises
    discriminative regions while still being differentiable.
    p=1 is equivalent to average pooling; p→∞ approaches max pooling.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p).flatten(1)


class BNNeck(nn.Module):
    """Batch Normalization Neck (BNNeck).
    
    From 'Bag of Tricks and A Strong Baseline for Deep Person Re-ID' (2019).
    Separates the ID loss space from the metric loss space by applying
    BatchNorm before the classification head.
    """

    def __init__(self, in_features: int, out_features: int = 512):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.bn.bias.requires_grad_(False)  # no shift, as per paper
        self.fc = nn.Linear(in_features, out_features, bias=False)

        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.fc(x)
        return x


class FastReIDExtractor(nn.Module):
    """FastReID Strong Baseline feature extractor.
    
    Architecture:
        ResNet50 (remove avgpool + fc)
        → GeM Pooling → 2048-d
        → BNNeck → FC → 512-d
        → L2 Normalisation
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()

        # ── Backbone: ResNet50 up to layer4 ──
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the original avgpool and fc layers
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

        # ── Pooling ──
        self.pool = GeMPooling(p=3.0)

        # ── Neck ──
        self.neck = BNNeck(in_features=2048, out_features=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)       # [B, 2048, H, W]
        pooled = self.pool(features)      # [B, 2048]
        embeddings = self.neck(pooled)    # [B, 512]
        # L2 normalise for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ReIDFeatureExtractor:
    """High-level interface for extracting Re-ID embeddings from person crops.
    
    Handles model loading, image preprocessing, and batch inference.
    Thread-safe via the InferenceEngine's existing lock.
    
    Usage:
        extractor = ReIDFeatureExtractor(device='cuda')
        extractor.load()
        embedding = extractor.extract(person_crop_bgr)   # → np.ndarray (512,)
        embeddings = extractor.extract_batch([crop1, crop2])  # → np.ndarray (N, 512)
    """

    # Standard Re-ID preprocessing (same as FastReID uses)
    INPUT_SIZE = (256, 128)  # (height, width) — taller than wide for standing people
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __init__(self, device: str = 'cuda', use_half: bool = False):
        self._device = device
        self._use_half = use_half and device == 'cuda'
        self._model: Optional[FastReIDExtractor] = None
        self._loaded = False

    def load(self, model_path: Optional[str] = None) -> bool:
        """Load the Re-ID model.
        
        Args:
            model_path: Path to Re-ID weights (.pt file).  
                         If None, searches infrastructure/models/reid/ for the latest,
                         falling back to ImageNet pre-trained weights.
        """
        try:
            self._model = FastReIDExtractor(embedding_dim=512)

            # ── Try to load Re-ID specific weights ──
            weights_loaded = False

            if model_path is None:
                model_path = self._find_reid_weights()

            if model_path and Path(model_path).exists():
                try:
                    state_dict = torch.load(model_path, map_location='cpu')
                    # Handle both plain state_dict and checkpoint-wrapped formats
                    if isinstance(state_dict, dict):
                        if 'model' in state_dict:
                            state_dict = state_dict['model']
                        elif 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                    
                    # Try loading — allow partial match for flexibility
                    missing, unexpected = self._model.load_state_dict(
                        state_dict, strict=False
                    )
                    if missing:
                        logger.warning(
                            f"Re-ID model loaded with {len(missing)} missing keys "
                            f"(using ImageNet init for those layers)"
                        )
                    weights_loaded = True
                    logger.info(f"Re-ID model loaded from: {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load Re-ID weights ({e}), using ImageNet backbone")

            if not weights_loaded:
                logger.info(
                    "Re-ID model initialised with ImageNet pre-trained backbone. "
                    "For best accuracy, place Re-ID weights in infrastructure/models/reid/"
                )

            self._model.to(self._device)
            if self._use_half:
                self._model.half()
            self._model.eval()
            self._loaded = True

            # Warmup inference
            dummy = torch.randn(1, 3, *self.INPUT_SIZE).to(self._device)
            if self._use_half:
                dummy = dummy.half()
            with torch.no_grad():
                self._model(dummy)
            logger.info(
                f"Re-ID extractor ready: device={self._device}, "
                f"half={self._use_half}, embedding=512-d"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load Re-ID model: {e}")
            self._loaded = False
            return False

    def _find_reid_weights(self) -> Optional[str]:
        """Search for Re-ID weights in the standard model directory."""
        search_dirs = [
            Path("infrastructure/models/reid"),
            Path("infrastructure/models"),
        ]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            # Look for files named *reid* or *sbs*
            for pattern in ["*reid*.pt", "*reid*.pth", "*sbs*.pt", "*sbs*.pth",
                            "*re_id*.pt", "*re-id*.pt"]:
                matches = list(search_dir.glob(pattern))
                if matches:
                    # Pick the most recently modified one
                    best = max(matches, key=lambda p: p.stat().st_mtime)
                    return str(best)
        return None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess a single BGR person crop into a normalised tensor."""
        # Resize to Re-ID input size (256 height × 128 width)
        img = cv2.resize(crop_bgr, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        return torch.from_numpy(np.ascontiguousarray(img))

    def extract(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract a 512-d feature embedding from a single person crop.
        
        Args:
            crop_bgr: Person crop in BGR format (OpenCV default).
            
        Returns:
            L2-normalised 512-d numpy array, or None on error.
        """
        if not self._loaded or self._model is None:
            return None

        try:
            tensor = self._preprocess(crop_bgr).unsqueeze(0).to(self._device)
            if self._use_half:
                tensor = tensor.half()
            else:
                tensor = tensor.float()

            with torch.no_grad():
                embedding = self._model(tensor)  # [1, 512]

            emb = embedding.cpu().numpy().flatten().astype(np.float32)  # (512,)
            
            # L2-normalise so cosine similarity = dot product
            norm = np.linalg.norm(emb)
            if norm < 1e-8 or np.isnan(norm):
                return None
            emb = emb / norm
            
            return emb

        except Exception as e:
            logger.error(f"Re-ID feature extraction error: {e}")
            return None

    def extract_batch(self, crops_bgr: List[np.ndarray]) -> Optional[np.ndarray]:
        """Extract embeddings for multiple person crops in a single batch.
        
        Args:
            crops_bgr: List of person crops in BGR format.
            
        Returns:
            (N, 512) numpy array, or None on error.
        """
        if not self._loaded or self._model is None or not crops_bgr:
            return None

        try:
            tensors = [self._preprocess(c) for c in crops_bgr]
            batch = torch.stack(tensors).to(self._device)
            if self._use_half:
                batch = batch.half()
            else:
                batch = batch.float()

            with torch.no_grad():
                embeddings = self._model(batch)  # [N, 512]

            return embeddings.cpu().numpy()

        except Exception as e:
            logger.error(f"Re-ID batch extraction error: {e}")
            return None
