import time
import threading
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from app.config import logger


@dataclass
class GlobalPerson:
    global_id: int
    feature_gallery: List[np.ndarray] = field(default_factory=list)
    camera_tracks: Dict[str, int] = field(default_factory=dict)
    gender: Optional[str] = None
    last_seen_time: float = 0.0
    last_seen_camera: Optional[str] = None
    appearance_count: int = 0


class CrossCameraReIDManager:
    def __init__(
        self,
        similarity_threshold: float,
        max_gallery_size: int,
        stale_timeout: float,
        gallery_pollution_min_sim: float,
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
        self._total_matches = 0
        self._total_new = 0

        logger.info(
            f"CrossCameraReIDManager initialised: threshold={similarity_threshold}, "
            f"gallery_size={max_gallery_size}, stale_timeout={stale_timeout}s"
        )

    def assign_global_id(
        self,
        camera_id: str,
        local_track_id: int,
        feature_embedding: np.ndarray,
        gender: Optional[str] = None,
    ) -> int:
        with self._lock:
            key = (camera_id, local_track_id)
            now = time.time()

            if key in self._track_to_global:
                gid = self._track_to_global[key]
                if gid in self._global_persons:
                    person = self._global_persons[gid]
                    self._update_gallery(person, feature_embedding)
                    person.last_seen_time = now
                    person.last_seen_camera = camera_id
                    person.appearance_count += 1
                    if gender:
                        person.gender = gender
                    return gid
                del self._track_to_global[key]

            best_match_id: Optional[int] = None
            best_similarity: float = -1.0

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

                similarity = self._compute_similarity(
                    feature_embedding, person.feature_gallery
                )

                logger.debug(
                    f"Re-ID SIM: cam={camera_id} track={local_track_id} "
                    f"vs global={gid} cams={list(person.camera_tracks.keys())} "
                    f"sim={similarity:.4f} threshold={self._similarity_threshold}"
                )

                if (
                    similarity > self._similarity_threshold
                    and similarity > best_similarity
                ):
                    best_similarity = similarity
                    best_match_id = gid

            if best_match_id is not None:
                person = self._global_persons[best_match_id]
                old_track = person.camera_tracks.get(camera_id)
                if old_track is not None and old_track != local_track_id:
                    old_key = (camera_id, old_track)
                    self._track_to_global.pop(old_key, None)

                person.camera_tracks[camera_id] = local_track_id
                self._update_gallery(person, feature_embedding)
                person.last_seen_time = now
                person.last_seen_camera = camera_id
                person.appearance_count += 1
                if gender:
                    person.gender = gender
                self._track_to_global[key] = best_match_id
                self._total_matches += 1
                logger.info(
                    f"Re-ID MATCH: cam={camera_id} track={local_track_id} "
                    f"-> global={best_match_id} (sim={best_similarity:.3f})"
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
                    logger.info(f"Re-ID ORPHAN REMOVED: global={orphan_gid} (superseded by global={best_match_id})")

                return best_match_id
            else:
                gid = self._next_global_id
                self._next_global_id += 1

                new_person = GlobalPerson(
                    global_id=gid,
                    feature_gallery=[feature_embedding.copy()],
                    camera_tracks={camera_id: local_track_id},
                    gender=gender,
                    last_seen_time=now,
                    last_seen_camera=camera_id,
                    appearance_count=1,
                )
                self._global_persons[gid] = new_person
                self._track_to_global[key] = gid
                self._total_new += 1
                logger.info(
                    f"Re-ID NEW: cam={camera_id} track={local_track_id} "
                    f"-> global={gid}"
                )
                return gid

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
            persons = list(self._global_persons.values())
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
                p for p in self._global_persons.values()
                if (now - p.last_seen_time) < max_age
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
                gid for gid, p in self._global_persons.items()
                if (now - p.last_seen_time) > self._stale_timeout
            ]

            for gid in stale_gids:
                person = self._global_persons.pop(gid)
                keys_to_remove = [
                    k for k, v in self._track_to_global.items() if v == gid
                ]
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

                    ga_mean = np.mean(person_a.feature_gallery, axis=0)
                    ga_mean = ga_mean / max(np.linalg.norm(ga_mean), 1e-8)

                    similarity = self._compute_similarity(ga_mean, person_b.feature_gallery)

                    if similarity > self._similarity_threshold:
                        for cam, track in person_b.camera_tracks.items():
                            person_a.camera_tracks[cam] = track
                            self._track_to_global[(cam, track)] = gid_a

                        for emb in person_b.feature_gallery:
                            self._update_gallery(person_a, emb)

                        person_a.last_seen_time = max(
                            person_a.last_seen_time, person_b.last_seen_time
                        )
                        person_a.appearance_count += person_b.appearance_count
                        if person_b.gender and not person_a.gender:
                            person_a.gender = person_b.gender

                        del self._global_persons[gid_b]
                        stale_keys = [
                            k for k, v in self._track_to_global.items()
                            if v == gid_b
                        ]
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
            keys_to_remove = [
                k for k in self._track_to_global if k[0] == camera_id
            ]
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
                f"Re-ID: cleared camera {camera_id}, "
                f"affected {len(gids_affected)} persons"
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

    def _compute_similarity(
        self, query: np.ndarray, gallery: List[np.ndarray]
    ) -> float:
        if not gallery:
            return -1.0

        gallery_mean = np.mean(gallery, axis=0)
        norm = np.linalg.norm(gallery_mean)
        if norm < 1e-8:
            return -1.0
        gallery_mean = gallery_mean / norm

        return float(np.dot(query, gallery_mean))

    def _update_gallery(self, person: GlobalPerson, embedding: np.ndarray) -> None:
        if person.feature_gallery:
            sim = self._compute_similarity(embedding, person.feature_gallery)
            if sim < self._gallery_pollution_min_sim:
                return

        person.feature_gallery.append(embedding.copy())
        if len(person.feature_gallery) > self._max_gallery_size:
            person.feature_gallery.pop(0)
