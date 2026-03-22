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

        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.fc(x)
        return x


class FastReIDExtractor(nn.Module):
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

    def __init__(self, device: str = 'cuda', use_half: bool = False):
        self._device = device
        self._use_half = use_half and device == 'cuda'
        self._model: Optional[FastReIDExtractor] = None
        self._loaded = False

    def load(self, model_path: Optional[str] = None) -> bool:
        try:
            self._model = FastReIDExtractor(embedding_dim=512)

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
        search_dirs = [
            Path("infrastructure/models/reid"),
            Path("infrastructure/models"),
        ]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for pattern in ["*reid*.pt", "*reid*.pth", "*sbs*.pt", "*sbs*.pth",
                            "*re_id*.pt", "*re-id*.pt"]:
                matches = list(search_dir.glob(pattern))
                if matches:
                    best = max(matches, key=lambda p: p.stat().st_mtime)
                    return str(best)
        return None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(crop_bgr, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        return torch.from_numpy(np.ascontiguousarray(img))

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

            with torch.no_grad():
                embedding = self._model(tensor)

            emb = embedding.cpu().numpy().flatten().astype(np.float32)

            norm = np.linalg.norm(emb)
            if norm < 1e-8 or np.isnan(norm):
                return None
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

            with torch.no_grad():
                embeddings = self._model(batch)

            embs = embeddings.cpu().numpy()
            
            result = np.zeros((len(crops_bgr), 512), dtype=np.float32)
            result[:] = np.nan
            for j, orig_idx in enumerate(valid_indices):
                emb = embs[j].flatten().astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm > 1e-8 and not np.isnan(norm):
                    result[orig_idx] = emb / norm

            return result

        except Exception as e:
            logger.error(f"Re-ID batch extraction error: {e}")
            return None
