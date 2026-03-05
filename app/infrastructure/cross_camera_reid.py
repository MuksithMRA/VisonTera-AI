"""
Cross-Camera Person Re-Identification Manager.

Maintains a global gallery of known persons and resolves per-camera ByteTrack IDs
into globally unique person IDs. This ensures that the same person seen on
Camera A and Camera B is counted only once in the aggregate total.

Core Algorithm:
    1. Each person detection produces a (camera_id, local_track_id) pair.
    2. For each new pair, a Re-ID embedding (512-d vector) is extracted.
    3. The embedding is compared against the global gallery using cosine similarity.
    4. If the best match exceeds the threshold AND is from a DIFFERENT camera,
       the detection is assigned the same global ID.
    5. Otherwise, a new global ID is created.

Key Constraint — Same-Camera Skip Rule:
    Two different local_track_ids on the SAME camera are ALWAYS different people.
    ByteTrack already does within-camera deduplication, so we never merge them.
"""

import time
import threading
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from app.config import logger


@dataclass
class GlobalPerson:
    """Represents a single unique person across all cameras."""
    global_id: int
    feature_gallery: List[np.ndarray] = field(default_factory=list)
    # camera_id → local_track_id  (tracks this person across cameras)
    camera_tracks: Dict[str, int] = field(default_factory=dict)
    gender: Optional[str] = None
    last_seen_time: float = 0.0
    last_seen_camera: Optional[str] = None
    appearance_count: int = 0


class CrossCameraReIDManager:
    """Manages global person identities across multiple camera feeds.
    
    Thread-safe. Designed to be used as a singleton alongside InferenceEngine.
    
    Configuration:
        similarity_threshold: Minimum cosine similarity to consider a match (default 0.60).
                              Lower → more aggressive merging (risk of false merge).
                              Higher → more conservative (risk of duplicate counts).
        max_gallery_size:     Max embeddings stored per person (FIFO eviction).
        stale_timeout:        Seconds before a person is removed from the gallery.
                              Prevents unbounded gallery growth.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.60,
        max_gallery_size: int = 10,
        stale_timeout: float = 300.0,  # 5 minutes
    ):
        self._lock = threading.Lock()
        self._global_persons: Dict[int, GlobalPerson] = {}
        self._next_global_id: int = 1

        # (camera_id, local_track_id) → global_id
        self._track_to_global: Dict[Tuple[str, int], int] = {}
        
        # camera_id → set of local_track_ids seen in the CURRENT frame
        # (Used to determine if a same-camera track is truly "active" or just a memory)
        self._current_active_tracks: Dict[str, set] = {}

        # Configuration
        self._similarity_threshold = similarity_threshold
        self._max_gallery_size = max_gallery_size
        self._stale_timeout = stale_timeout

        # Stats
        self._total_matches = 0
        self._total_new = 0

        logger.info(
            f"CrossCameraReIDManager initialised: "
            f"threshold={similarity_threshold}, "
            f"gallery_size={max_gallery_size}, "
            f"stale_timeout={stale_timeout}s"
        )

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def assign_global_id(
        self,
        camera_id: str,
        local_track_id: int,
        feature_embedding: np.ndarray,
        gender: Optional[str] = None,
    ) -> int:
        """Resolve a per-camera track into a global unique person ID.
        
        Args:
            camera_id:         Unique camera identifier.
            local_track_id:    ByteTrack ID within this camera.
            feature_embedding: 512-d L2-normalised Re-ID embedding.
            gender:            Optional gender label for this person.
            
        Returns:
            Global person ID (int >= 1).
        """
        with self._lock:
            key = (camera_id, local_track_id)
            now = time.time()

            # -- Fast path: already mapped --
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
                # Stale mapping - cleanup and fall through
                del self._track_to_global[key]

            # -- Search for a match across ALL persons --
            best_match_id: Optional[int] = None
            best_similarity: float = -1.0

            for gid, person in self._global_persons.items():
                # Skip if this person already has a DIFFERENT active track
                # on the same camera (two people visible simultaneously)
                if camera_id in person.camera_tracks:
                    existing_track = person.camera_tracks[camera_id]
                    if existing_track != local_track_id:
                        # Check if this existing track was seen in the CURRENT frame
                        active_on_cam = self._current_active_tracks.get(camera_id, set())
                        if existing_track in active_on_cam:
                            # If it's still visible, we CANNOT merge (it's two different people)
                            logger.debug(
                                f"Re-ID SKIP: cam={camera_id} track={local_track_id} "
                                f"vs global={gid} (same-cam active track={existing_track})"
                            )
                            continue
                        else:
                            # It's NOT visible in the current frame. 
                            # We can potentially re-associate this person with the new track ID.
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
                # -- Match found --
                person = self._global_persons[best_match_id]
                
                # Clean up old track mapping for this camera if it changed
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
                
                # -- Orphan cleanup --
                # When this camera gets a new track that matches an existing global,
                # check if there are OTHER global persons that had this camera as
                # their only tracker with an old (now-inactive) track.
                # Those are orphans from initial ByteTrack IDs that weren't matched.
                orphan_gids = []
                for other_gid, other_person in self._global_persons.items():
                    if other_gid == best_match_id:
                        continue
                    if camera_id not in other_person.camera_tracks:
                        continue
                    other_track = other_person.camera_tracks[camera_id]
                    other_key = (camera_id, other_track)
                    # If the track mapping is gone (no longer active), this person
                    # is an orphan on this camera
                    if other_key not in self._track_to_global:
                        other_person.camera_tracks.pop(camera_id, None)
                        if not other_person.camera_tracks:
                            orphan_gids.append(other_gid)
                
                for orphan_gid in orphan_gids:
                    del self._global_persons[orphan_gid]
                    # Clean up any remaining mappings
                    stale_keys = [k for k, v in self._track_to_global.items() if v == orphan_gid]
                    for k in stale_keys:
                        del self._track_to_global[k]
                    logger.info(f"Re-ID ORPHAN REMOVED: global={orphan_gid} (superseded by global={best_match_id})")
                
                return best_match_id
            else:
                # -- No match: new person --
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
        """Update last_seen_time for a person (called from cached path)."""
        with self._lock:
            if global_id in self._global_persons:
                self._global_persons[global_id].last_seen_time = time.time()

    def get_global_id(self, camera_id: str, local_track_id: int) -> Optional[int]:
        """Look up the global ID for a known track (no embedding required)."""
        return self._track_to_global.get((camera_id, local_track_id))

    def update_active_tracks(self, camera_id: str, local_track_ids: List[int]) -> None:
        """Inform the manager which tracks are currently visible in the frame.
        
        This helps resolve same-camera re-associations when a tracking ID flips.
        """
        with self._lock:
            self._current_active_tracks[camera_id] = set(local_track_ids)

    def get_deduplicated_counts(self) -> Dict:
        """Return deduplicated person counts across all cameras.
        
        Returns:
            Dict with keys 'total', 'male', 'female', 'unknown', 'global_ids'.
        """
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
        """Return only persons seen within the last `max_age` seconds.
        
        This is useful for the real-time dashboard — persons who haven't been
        seen recently are excluded from the "live" count.
        """
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
        """Remove persons not seen for longer than stale_timeout.
        
        Returns:
            Number of persons removed.
        """
        with self._lock:
            now = time.time()
            stale_gids = [
                gid for gid, p in self._global_persons.items()
                if (now - p.last_seen_time) > self._stale_timeout
            ]

            for gid in stale_gids:
                person = self._global_persons.pop(gid)
                # Remove all track mappings for this person
                keys_to_remove = [
                    k for k, v in self._track_to_global.items() if v == gid
                ]
                for k in keys_to_remove:
                    del self._track_to_global[k]

            if stale_gids:
                logger.info(f"Re-ID cleanup: removed {len(stale_gids)} stale persons")

            return len(stale_gids)

    def merge_duplicates(self) -> int:
        """Periodically scan for global persons that should be merged.
        
        This handles the race condition where both cameras detect a new person
        simultaneously and create separate global IDs. The merge pass compares
        all pairs from different cameras and merges them if similarity is high.
        
        Returns:
            Number of merges performed.
        """
        with self._lock:
            if len(self._global_persons) < 2:
                return 0

            gids = list(self._global_persons.keys())
            merges = 0
            absorbed = set()  # gids that have been merged into another

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

                    # Only merge persons from DIFFERENT cameras
                    cameras_a = set(person_a.camera_tracks.keys())
                    cameras_b = set(person_b.camera_tracks.keys())
                    if cameras_a & cameras_b:
                        # They share a camera — skip (could be two different people)
                        continue

                    similarity = self._compute_similarity(
                        np.mean(person_a.feature_gallery, axis=0) / max(np.linalg.norm(np.mean(person_a.feature_gallery, axis=0)), 1e-8),
                        person_b.feature_gallery,
                    )

                    if similarity > self._similarity_threshold:
                        # Merge B into A (keep the one with lower gid)
                        # Transfer all tracks from B to A
                        for cam, track in person_b.camera_tracks.items():
                            person_a.camera_tracks[cam] = track
                            self._track_to_global[(cam, track)] = gid_a

                        # Merge galleries
                        for emb in person_b.feature_gallery:
                            self._update_gallery(person_a, emb)

                        # Update metadata
                        person_a.last_seen_time = max(
                            person_a.last_seen_time, person_b.last_seen_time
                        )
                        person_a.appearance_count += person_b.appearance_count
                        if person_b.gender and not person_a.gender:
                            person_a.gender = person_b.gender

                        # Remove B
                        del self._global_persons[gid_b]
                        # Clean up any remaining B mappings
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
        """Remove all tracks for a specific camera (called when camera is stopped)."""
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
                    # Remove person entirely if no longer tracked by any camera
                    if not person.camera_tracks:
                        del self._global_persons[gid]

            logger.info(
                f"Re-ID: cleared camera {camera_id}, "
                f"affected {len(gids_affected)} persons"
            )

    def clear_all(self) -> None:
        """Reset all Re-ID state."""
        with self._lock:
            self._global_persons.clear()
            self._track_to_global.clear()
            self._next_global_id = 1
            self._total_matches = 0
            self._total_new = 0
            logger.info("Re-ID: all state cleared")

    def get_stats(self) -> Dict:
        """Return Re-ID system statistics."""
        with self._lock:
            return {
                "total_global_persons": len(self._global_persons),
                "total_track_mappings": len(self._track_to_global),
                "total_cross_camera_matches": self._total_matches,
                "total_new_persons": self._total_new,
                "similarity_threshold": self._similarity_threshold,
                "next_global_id": self._next_global_id,
            }

    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────

    def _compute_similarity(
        self, query: np.ndarray, gallery: List[np.ndarray]
    ) -> float:
        """Compute cosine similarity between query and gallery mean embedding.
        
        Both query and gallery embeddings are assumed to be L2-normalised,
        so cosine similarity = dot product.
        """
        if not gallery:
            return -1.0

        # Average the gallery for a more robust representation
        gallery_mean = np.mean(gallery, axis=0)
        norm = np.linalg.norm(gallery_mean)
        if norm < 1e-8:
            return -1.0
        gallery_mean = gallery_mean / norm

        return float(np.dot(query, gallery_mean))

    def _update_gallery(self, person: GlobalPerson, embedding: np.ndarray) -> None:
        """Add an embedding to a person's gallery with pollution protection."""
        # Minimum similarity to existing gallery to be added
        # This prevents "pollution" from occluded/noisy crops during overlays
        if person.feature_gallery:
            sim = self._compute_similarity(embedding, person.feature_gallery)
            # If the new embedding is wildly different from the person's history,
            # ignore it — it's likely an occlusion or a wrong match.
            if sim < 0.40:  # Fairly loose, but blocks total noise
                return

        person.feature_gallery.append(embedding.copy())
        if len(person.feature_gallery) > self._max_gallery_size:
            person.feature_gallery.pop(0)
