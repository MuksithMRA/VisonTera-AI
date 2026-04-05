"""
Cross-camera Re-ID tests.

Run from project root:
  pip install pytest
  pytest tests/test_reid.py -v

GPU and OSNet weights are optional; manager logic tests run on CPU only.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from app.infrastructure.cross_camera_reid import CrossCameraReIDManager


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / max(np.linalg.norm(v), 1e-8)


@pytest.fixture
def manager() -> CrossCameraReIDManager:
    # Fixed, permissive thresholds so tests do not depend on AppConfig defaults.
    return CrossCameraReIDManager(
        similarity_threshold=0.55,
        max_gallery_size=8,
        stale_timeout=3600.0,
        gallery_pollution_min_sim=0.25,
        top_k_gallery=3,
        gender_gate_enabled=True,
        adaptive_small_gallery_add=0.0,
        adaptive_large_gallery_sub=0.0,
        large_gallery_min_count=99,
        ambiguous_match_margin=0.001,
        viewpoint_diversity_min_sim=0.5,
        ema_alpha=0.1,
        transition_penalty=0.0,
        adjacent_bonus=0.0,
        same_cam_reentry_sec=0.0,
        same_cam_threshold_relax=0.0,
        camera_transitions={},
        adjacent_camera_pairs=[],
    )


def test_same_embedding_two_cameras_gets_same_global_id(manager: CrossCameraReIDManager) -> None:
    e = _unit_vec(1)
    g1 = manager.assign_global_id("cam_a", 1, e, gender="Male", quality_score=1.0)
    g2 = manager.assign_global_id("cam_b", 1, e.copy(), gender="Male", quality_score=1.0)
    assert g1 == g2
    stats = manager.get_stats()
    assert stats["total_new_persons"] == 1
    assert stats["total_cross_camera_matches"] >= 1


def test_orthogonal_embedding_new_identity(manager: CrossCameraReIDManager) -> None:
    e1 = _unit_vec(2)
    e2 = _unit_vec(3)
    e2 = e2 - float(np.dot(e2, e1)) * e1
    e2 = e2 / max(np.linalg.norm(e2), 1e-8)
    assert abs(float(np.dot(e1, e2))) < 0.05
    manager.assign_global_id("cam_a", 1, e1, gender=None, quality_score=1.0)
    g2 = manager.assign_global_id("cam_b", 1, e2, gender=None, quality_score=1.0)
    assert g2 == 2


def test_gender_gate_blocks_merge(manager: CrossCameraReIDManager) -> None:
    e = _unit_vec(4)
    manager.assign_global_id("cam_a", 1, e, gender="Male", quality_score=1.0)
    g2 = manager.assign_global_id("cam_b", 1, e.copy(), gender="Female", quality_score=1.0)
    assert g2 == 2


def test_track_reuse_updates_gallery(manager: CrossCameraReIDManager) -> None:
    e = _unit_vec(5)
    g1 = manager.assign_global_id("cam_a", 7, e, quality_score=1.0)
    e2 = e * 0.99 + _unit_vec(6) * 0.01
    e2 = e2 / max(np.linalg.norm(e2), 1e-8)
    g2 = manager.assign_global_id("cam_a", 7, e2, quality_score=0.9)
    assert g1 == g2
    assert manager.get_stats()["total_global_persons"] == 1


def test_cleanup_stale_removes_person(manager: CrossCameraReIDManager) -> None:
    short = CrossCameraReIDManager(
        similarity_threshold=0.55,
        max_gallery_size=8,
        stale_timeout=0.01,
        gallery_pollution_min_sim=0.2,
        gender_gate_enabled=False,
        ambiguous_match_margin=0.001,
        transition_penalty=0.0,
        adjacent_bonus=0.0,
        same_cam_reentry_sec=0.0,
        same_cam_threshold_relax=0.0,
        adaptive_small_gallery_add=0.0,
        adaptive_large_gallery_sub=0.0,
        large_gallery_min_count=99,
        viewpoint_diversity_min_sim=0.5,
        ema_alpha=0.1,
        camera_transitions={},
        adjacent_camera_pairs=[],
    )
    short.assign_global_id("cam_a", 1, _unit_vec(8), quality_score=1.0)
    time.sleep(0.05)
    n = short.cleanup_stale()
    assert n >= 1
    assert short.get_stats()["total_global_persons"] == 0


@pytest.mark.slow
def test_reid_extractor_same_crop_high_similarity() -> None:
    """Requires weights (OSNet download or legacy file); skipped if load fails."""
    import cv2

    from app.infrastructure.reid_extractor import ReIDFeatureExtractor

    weights_dir = Path(__file__).resolve().parents[1] / "infrastructure" / "models" / "reid"
    has_weights = any(weights_dir.glob("*.pth")) or any(weights_dir.glob("*reid*.pt"))

    ext = ReIDFeatureExtractor(device="cpu", use_half=False)
    if not ext.load():
        pytest.skip("Re-ID weights not available (place OSNet Market-1501 .pth in infrastructure/models/reid/)")

    fake_crop = np.random.default_rng(42).integers(0, 255, (200, 100, 3), dtype=np.uint8)
    emb1 = ext.extract(fake_crop)
    emb2 = ext.extract(fake_crop.copy())
    assert emb1 is not None and emb2 is not None
    sim = float(np.dot(emb1, emb2))
    assert sim > 0.85, f"Same crop should yield very high cosine similarity, got {sim}"


def test_noise_sensitivity_reference() -> None:
    """Sanity: tiny noise on a unit vector keeps cosine high (illustrates metric, not the OSNet model)."""
    base = _unit_vec(99)
    noisy = base + np.random.default_rng(1).standard_normal(512).astype(np.float32) * 0.01
    noisy = noisy / max(np.linalg.norm(noisy), 1e-8)
    assert float(np.dot(base, noisy)) > 0.92
