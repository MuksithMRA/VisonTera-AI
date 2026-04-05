import time
import threading
import numpy as np
from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass, field
from app.config import AppConfig, logger


@dataclass
class GalleryItem:
    embedding: np.ndarray
    quality: float = 1.0


@dataclass
class GlobalPerson:
    global_id: int
    gallery: List[GalleryItem] = field(default_factory=list)
    camera_tracks: Dict[str, int] = field(default_factory=dict)
    gender: Optional[str] = None
    is_employee: bool = False
    last_seen_time: float = 0.0
    last_seen_camera: Optional[str] = None
    last_seen_per_camera: Dict[str, float] = field(default_factory=dict)
    appearance_count: int = 0
    ema_centroid: Optional[np.ndarray] = None

    @property
    def feature_vectors(self) -> List[np.ndarray]:
        return [g.embedding for g in self.gallery]


class CrossCameraReIDManager:
    def __init__(
        self,
        similarity_threshold: float,
        max_gallery_size: int,
        stale_timeout: float,
        gallery_pollution_min_sim: float,
        top_k_gallery: int = None,
        gender_gate_enabled: bool = None,
        adaptive_small_gallery_add: float = None,
        adaptive_large_gallery_sub: float = None,
        large_gallery_min_count: int = None,
        ambiguous_match_margin: float = None,
        viewpoint_diversity_min_sim: float = None,
        ema_alpha: float = None,
        transition_penalty: float = None,
        adjacent_bonus: float = None,
        same_cam_reentry_sec: float = None,
        same_cam_threshold_relax: float = None,
        camera_transitions: Optional[Dict[str, Tuple[float, float]]] = None,
        adjacent_camera_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        self._lock = threading.RLock()
        self._global_persons: Dict[int, GlobalPerson] = {}
        self._next_global_id: int = 1
        self._track_to_global: Dict[Tuple[str, int], int] = {}
        self._current_active_tracks: Dict[str, set] = {}
        self._similarity_threshold = similarity_threshold
        self._max_gallery_size = max_gallery_size
        self._stale_timeout = stale_timeout
        self._gallery_pollution_min_sim = gallery_pollution_min_sim
        self._top_k = top_k_gallery if top_k_gallery is not None else AppConfig.REID_TOP_K_GALLERY_SIM
        self._gender_gate_enabled = (
            gender_gate_enabled
            if gender_gate_enabled is not None
            else AppConfig.REID_GENDER_GATE_ENABLED
        )
        self._threshold_small_add = (
            adaptive_small_gallery_add
            if adaptive_small_gallery_add is not None
            else AppConfig.REID_ADAPTIVE_THRESHOLD_SMALL_GALLERY_ADD
        )
        self._threshold_large_sub = (
            adaptive_large_gallery_sub
            if adaptive_large_gallery_sub is not None
            else AppConfig.REID_ADAPTIVE_THRESHOLD_LARGE_GALLERY_SUB
        )
        self._large_gallery_min = (
            large_gallery_min_count
            if large_gallery_min_count is not None
            else AppConfig.REID_LARGE_GALLERY_MIN_COUNT
        )
        self._ambiguous_margin = (
            ambiguous_match_margin
            if ambiguous_match_margin is not None
            else AppConfig.REID_AMBIGUOUS_MATCH_MARGIN
        )
        self._viewpoint_div_min_sim = (
            viewpoint_diversity_min_sim
            if viewpoint_diversity_min_sim is not None
            else AppConfig.REID_VIEWPOINT_DIVERSITY_MIN_SIM
        )
        self._ema_alpha = ema_alpha if ema_alpha is not None else AppConfig.REID_EMA_ALPHA
        self._transition_penalty = (
            transition_penalty
            if transition_penalty is not None
            else AppConfig.REID_TRANSITION_PENALTY
        )
        self._adjacent_bonus = (
            adjacent_bonus if adjacent_bonus is not None else AppConfig.REID_ADJACENT_BONUS
        )
        self._same_cam_reentry_sec = (
            same_cam_reentry_sec
            if same_cam_reentry_sec is not None
            else AppConfig.REID_SAME_CAM_REENTRY_SEC
        )
        self._same_cam_relax = (
            same_cam_threshold_relax
            if same_cam_threshold_relax is not None
            else AppConfig.REID_SAME_CAM_THRESHOLD_RELAX
        )
        raw_trans = camera_transitions if camera_transitions is not None else AppConfig.REID_CAMERA_TRANSITIONS
        self._camera_transitions: Dict[str, Tuple[float, float]] = dict(raw_trans or {})
        pairs_src = adjacent_camera_pairs if adjacent_camera_pairs is not None else AppConfig.REID_ADJACENT_CAMERA_PAIRS
        self._adjacent_pairs: Set[Tuple[str, str]] = set()
        for a, b in pairs_src or []:
            self._adjacent_pairs.add((str(a), str(b)))
            self._adjacent_pairs.add((str(b), str(a)))
        self._total_matches = 0
        self._total_new = 0

        logger.info(
            f"CrossCameraReIDManager initialised: threshold={similarity_threshold}, "
            f"gallery_size={max_gallery_size}, stale_timeout={stale_timeout}s, "
            f"top_k={self._top_k}, gender_gate={self._gender_gate_enabled}"
        )

    def assign_global_id(
        self,
        camera_id: str,
        local_track_id: int,
        feature_embedding: np.ndarray,
        gender: Optional[str] = None,
        is_employee: bool = False,
        quality_score: float = 1.0,
    ) -> int:
        with self._lock:
            key = (camera_id, local_track_id)
            now = time.time()

            if key in self._track_to_global:
                gid = self._track_to_global[key]
                if gid in self._global_persons:
                    person = self._global_persons[gid]
                    self._update_gallery(person, feature_embedding, quality_score)
                    self._touch_ema(person, feature_embedding, quality_score)
                    person.last_seen_time = now
                    person.last_seen_camera = camera_id
                    person.last_seen_per_camera[camera_id] = now
                    person.appearance_count += 1
                    if gender:
                        person.gender = gender
                    person.is_employee = person.is_employee or is_employee
                    return gid
                del self._track_to_global[key]

            candidates: List[Tuple[int, float]] = []

            for gid, person in self._global_persons.items():
                if camera_id in person.camera_tracks:
                    existing_track = person.camera_tracks[camera_id]
                    if existing_track != local_track_id:
                        active_on_cam = self._current_active_tracks.get(camera_id, set())
                        if existing_track in active_on_cam:
                            logger.debug(
                                f"Re-ID SKIP: cam={camera_id} track={local_track_id} "
                                f"vs global={gid} (same-cam active track={existing_track})"
                            )
                            continue
                        else:
                            logger.debug(
                                f"Re-ID RECALL: cam={camera_id} global={gid} "
                                f"lost track {existing_track}, associating with {local_track_id}"
                            )

                if self._gender_gate_enabled and gender and person.gender:
                    if gender in ("Male", "Female") and person.gender in ("Male", "Female"):
                        if gender != person.gender:
                            continue

                vecs = person.feature_vectors
                similarity = self._compute_similarity_topk(feature_embedding, vecs)
                temporal = self._temporal_adjustment(person, camera_id, now)
                adjusted = similarity + temporal

                gallery_len = len(vecs)
                thr = self._adaptive_threshold(gallery_len, person, camera_id, now)

                logger.debug(
                    f"Re-ID SIM: cam={camera_id} track={local_track_id} "
                    f"vs global={gid} cams={list(person.camera_tracks.keys())} "
                    f"sim={similarity:.4f} adj={adjusted:.4f} thr={thr:.4f}"
                )

                if adjusted > thr:
                    candidates.append((gid, adjusted))

            candidates.sort(key=lambda x: x[1], reverse=True)
            best_match_id: Optional[int] = None
            best_similarity: float = -1.0

            if candidates:
                best_gid, best_sim = candidates[0]
                if len(candidates) >= 2:
                    second_sim = candidates[1][1]
                    if (best_sim - second_sim) < self._ambiguous_margin:
                        logger.debug(
                            f"Re-ID AMBIGUOUS: top1={best_sim:.4f} top2={second_sim:.4f} "
                            f"margin<{self._ambiguous_margin} — new ID"
                        )
                        candidates = []
                    else:
                        best_match_id = best_gid
                        best_similarity = best_sim
                else:
                    best_match_id = best_gid
                    best_similarity = best_sim

            if best_match_id is not None:
                person = self._global_persons[best_match_id]
                old_track = person.camera_tracks.get(camera_id)
                if old_track is not None and old_track != local_track_id:
                    old_key = (camera_id, old_track)
                    self._track_to_global.pop(old_key, None)

                person.camera_tracks[camera_id] = local_track_id
                self._update_gallery(person, feature_embedding, quality_score)
                self._touch_ema(person, feature_embedding, quality_score)
                person.last_seen_time = now
                person.last_seen_camera = camera_id
                person.last_seen_per_camera[camera_id] = now
                person.appearance_count += 1
                if gender:
                    person.gender = gender
                person.is_employee = person.is_employee or is_employee
                self._track_to_global[key] = best_match_id
                self._total_matches += 1
                logger.info(
                    f"Re-ID MATCH: cam={camera_id} track={local_track_id} "
                    f"-> global={best_match_id} (adj_sim={best_similarity:.3f})"
                )

                orphan_gids = []
                for other_gid, other_person in self._global_persons.items():
                    if other_gid == best_match_id:
                        continue
                    if camera_id not in other_person.camera_tracks:
                        continue
                    other_track = other_person.camera_tracks[camera_id]
                    other_key = (camera_id, other_track)
                    if other_key not in self._track_to_global:
                        other_person.camera_tracks.pop(camera_id, None)
                        if not other_person.camera_tracks:
                            orphan_gids.append(other_gid)

                for orphan_gid in orphan_gids:
                    del self._global_persons[orphan_gid]
                    stale_keys = [k for k, v in self._track_to_global.items() if v == orphan_gid]
                    for k in stale_keys:
                        del self._track_to_global[k]
                    logger.info(
                        f"Re-ID ORPHAN REMOVED: global={orphan_gid} (superseded by global={best_match_id})"
                    )

                return best_match_id

            gid = self._next_global_id
            self._next_global_id += 1

            new_person = GlobalPerson(
                global_id=gid,
                gallery=[GalleryItem(feature_embedding.copy(), quality_score)],
                camera_tracks={camera_id: local_track_id},
                gender=gender,
                is_employee=is_employee,
                last_seen_time=now,
                last_seen_camera=camera_id,
                last_seen_per_camera={camera_id: now},
                appearance_count=1,
                ema_centroid=feature_embedding.copy(),
            )
            self._global_persons[gid] = new_person
            self._track_to_global[key] = gid
            self._total_new += 1
            logger.info(
                f"Re-ID NEW: cam={camera_id} track={local_track_id} -> global={gid}"
            )
            return gid

    def _adaptive_threshold(
        self, gallery_len: int, person: GlobalPerson, camera_id: str, now: float
    ) -> float:
        thr = self._similarity_threshold
        if gallery_len < 3:
            thr += self._threshold_small_add
        elif gallery_len >= self._large_gallery_min:
            thr -= self._threshold_large_sub
        t_cam = person.last_seen_per_camera.get(camera_id, 0.0)
        if t_cam > 0 and (now - t_cam) < self._same_cam_reentry_sec:
            thr -= self._same_cam_relax
        return thr

    def _temporal_adjustment(self, person: GlobalPerson, camera_id: str, now: float) -> float:
        adj = 0.0
        last_cam = person.last_seen_camera
        last_t = person.last_seen_time
        if last_cam is None or last_t <= 0:
            return adj
        elapsed = now - last_t
        if (last_cam, camera_id) in self._adjacent_pairs:
            adj += self._adjacent_bonus
        key_ab = f"{last_cam}|{camera_id}"
        if key_ab in self._camera_transitions:
            mn, mx = self._camera_transitions[key_ab]
            if elapsed < mn or elapsed > mx:
                adj -= self._transition_penalty
        return adj

    def _touch_ema(self, person: GlobalPerson, embedding: np.ndarray, quality: float) -> None:
        q = float(np.clip(quality, 0.1, 1.0))
        alpha = self._ema_alpha * q
        if person.ema_centroid is None:
            person.ema_centroid = embedding.copy()
            return
        person.ema_centroid = (1.0 - alpha) * person.ema_centroid + alpha * embedding
        n = np.linalg.norm(person.ema_centroid)
        if n > 1e-8:
            person.ema_centroid = person.ema_centroid / n

    def touch_person(self, global_id: int) -> None:
        with self._lock:
            if global_id in self._global_persons:
                self._global_persons[global_id].last_seen_time = time.time()

    def get_global_id(self, camera_id: str, local_track_id: int) -> Optional[int]:
        return self._track_to_global.get((camera_id, local_track_id))

    def update_active_tracks(self, camera_id: str, local_track_ids: List[int]) -> None:
        with self._lock:
            self._current_active_tracks[camera_id] = set(local_track_ids)

    def get_deduplicated_counts(self) -> Dict:
        with self._lock:
            persons = [p for p in self._global_persons.values() if not p.is_employee]
            return {
                "total": len(persons),
                "male": sum(1 for p in persons if p.gender == "Male"),
                "female": sum(1 for p in persons if p.gender == "Female"),
                "unknown": sum(
                    1 for p in persons if p.gender not in ("Male", "Female")
                ),
                "global_ids": [p.global_id for p in persons],
            }

    def get_currently_visible(self, max_age: float = 5.0) -> Dict:
        with self._lock:
            now = time.time()
            visible = [
                p
                for p in self._global_persons.values()
                if (now - p.last_seen_time) < max_age and not p.is_employee
            ]
            return {
                "total": len(visible),
                "male": sum(1 for p in visible if p.gender == "Male"),
                "female": sum(1 for p in visible if p.gender == "Female"),
                "unknown": sum(
                    1 for p in visible if p.gender not in ("Male", "Female")
                ),
                "global_ids": [p.global_id for p in visible],
            }

    def cleanup_stale(self) -> int:
        with self._lock:
            now = time.time()
            stale_gids = [
                gid
                for gid, p in self._global_persons.items()
                if (now - p.last_seen_time) > self._stale_timeout
            ]

            for gid in stale_gids:
                self._global_persons.pop(gid)
                keys_to_remove = [k for k, v in self._track_to_global.items() if v == gid]
                for k in keys_to_remove:
                    del self._track_to_global[k]

            if stale_gids:
                logger.info(f"Re-ID cleanup: removed {len(stale_gids)} stale persons")

            return len(stale_gids)

    def merge_duplicates(self) -> int:
        with self._lock:
            if len(self._global_persons) < 2:
                return 0

            gids = list(self._global_persons.keys())
            merges = 0
            absorbed = set()

            for i in range(len(gids)):
                gid_a = gids[i]
                if gid_a in absorbed or gid_a not in self._global_persons:
                    continue
                person_a = self._global_persons[gid_a]

                for j in range(i + 1, len(gids)):
                    gid_b = gids[j]
                    if gid_b in absorbed or gid_b not in self._global_persons:
                        continue
                    person_b = self._global_persons[gid_b]

                    cameras_a = set(person_a.camera_tracks.keys())
                    cameras_b = set(person_b.camera_tracks.keys())
                    if cameras_a & cameras_b:
                        continue

                    if self._gender_gate_enabled and person_a.gender and person_b.gender:
                        if person_a.gender in ("Male", "Female") and person_b.gender in (
                            "Male",
                            "Female",
                        ):
                            if person_a.gender != person_b.gender:
                                continue

                    ga_vecs = person_a.feature_vectors
                    gb_vecs = person_b.feature_vectors
                    if not ga_vecs or not gb_vecs:
                        continue
                    ga_mean = np.mean(np.stack(ga_vecs), axis=0)
                    ga_mean = ga_mean / max(np.linalg.norm(ga_mean), 1e-8)

                    similarity = self._compute_similarity_topk(ga_mean, gb_vecs)
                    if similarity > self._similarity_threshold:
                        for cam, track in person_b.camera_tracks.items():
                            person_a.camera_tracks[cam] = track
                            self._track_to_global[(cam, track)] = gid_a
                        person_a.is_employee = person_a.is_employee or person_b.is_employee

                        for item in person_b.gallery:
                            self._update_gallery(person_a, item.embedding, item.quality)

                        person_a.last_seen_time = max(
                            person_a.last_seen_time, person_b.last_seen_time
                        )
                        person_a.appearance_count += person_b.appearance_count
                        if person_b.gender and not person_a.gender:
                            person_a.gender = person_b.gender
                        for ck, ct in person_b.last_seen_per_camera.items():
                            prev = person_a.last_seen_per_camera.get(ck, 0.0)
                            person_a.last_seen_per_camera[ck] = max(prev, ct)

                        del self._global_persons[gid_b]
                        stale_keys = [k for k, v in self._track_to_global.items() if v == gid_b]
                        for k in stale_keys:
                            del self._track_to_global[k]

                        absorbed.add(gid_b)
                        merges += 1
                        self._total_matches += 1
                        logger.info(
                            f"Re-ID MERGE: global={gid_b} absorbed into "
                            f"global={gid_a} (sim={similarity:.3f})"
                        )

            return merges

    def clear_camera(self, camera_id: str) -> None:
        with self._lock:
            keys_to_remove = [k for k in self._track_to_global if k[0] == camera_id]
            gids_affected = set()
            for key in keys_to_remove:
                gid = self._track_to_global.pop(key)
                gids_affected.add(gid)

            for gid in gids_affected:
                if gid in self._global_persons:
                    person = self._global_persons[gid]
                    person.camera_tracks.pop(camera_id, None)
                    if not person.camera_tracks:
                        del self._global_persons[gid]

            logger.info(
                f"Re-ID: cleared camera {camera_id}, affected {len(gids_affected)} persons"
            )

    def clear_all(self) -> None:
        with self._lock:
            self._global_persons.clear()
            self._track_to_global.clear()
            self._next_global_id = 1
            self._total_matches = 0
            self._total_new = 0
            logger.info("Re-ID: all state cleared")

    def registry_integrity(self) -> Dict:
        with self._lock:
            dup_gids = {}
            for k, v in self._track_to_global.items():
                dup_gids.setdefault(v, []).append(k)
            multi_map = {gid: keys for gid, keys in dup_gids.items() if len(keys) > 1}
            orphan_tracks = []
            for gid, person in self._global_persons.items():
                for cam, tid in person.camera_tracks.items():
                    if self._track_to_global.get((cam, tid)) != gid:
                        orphan_tracks.append((cam, tid, gid))
            return {
                "registry_integrity_ok": len(multi_map) == 0 and len(orphan_tracks) == 0,
                "duplicate_mappings": len(multi_map),
                "orphaned_track_mappings": len(orphan_tracks),
            }

    def get_stats(self) -> Dict:
        with self._lock:
            base = {
                "total_global_persons": len(self._global_persons),
                "total_track_mappings": len(self._track_to_global),
                "total_cross_camera_matches": self._total_matches,
                "total_new_persons": self._total_new,
                "similarity_threshold": self._similarity_threshold,
                "next_global_id": self._next_global_id,
            }
            base.update(self.registry_integrity())
            return base

    def _compute_similarity_topk(
        self, query: np.ndarray, gallery_vecs: List[np.ndarray]
    ) -> float:
        if not gallery_vecs:
            return -1.0
        G = np.stack(gallery_vecs, axis=0)
        dots = G @ query
        k = min(self._top_k, len(dots))
        top = np.partition(dots, -k)[-k:]
        return float(np.mean(top))

    def _update_gallery(
        self, person: GlobalPerson, embedding: np.ndarray, quality: float
    ) -> None:
        vecs = person.feature_vectors
        if vecs:
            max_dot = max(float(np.dot(embedding, g)) for g in vecs)
            new_viewpoint = max_dot < self._viewpoint_div_min_sim
            if not new_viewpoint and max_dot < self._gallery_pollution_min_sim:
                return

        person.gallery.append(GalleryItem(embedding.copy(), float(quality)))
        while len(person.gallery) > self._max_gallery_size:
            idx_min = min(range(len(person.gallery)), key=lambda i: person.gallery[i].quality)
            person.gallery.pop(idx_min)
