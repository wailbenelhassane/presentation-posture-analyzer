"""
Landmark smoothing and multi-person tracking.

Provides:
  - Exponential Moving Average (EMA) filter for jitter reduction
  - Person tracker that matches detections across frames by proximity
  - Visibility-aware smoothing (low-visibility landmarks get more smoothing)
"""

import math
from typing import Optional

from .gestures import LM


class SmoothedLandmark:
    """A single smoothed landmark with x, y, z, visibility."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class LandmarkSmoother:
    """
    Exponential Moving Average (EMA) filter for a single person's landmarks.

    Lower alpha = more smoothing. Higher alpha = more responsive.
    Adapts per landmark based on visibility, and dampens large jumps.
    """

    def __init__(self, num_landmarks: int = 33, alpha: float = 0.4):
        self._alpha = alpha
        self._num = num_landmarks
        self._state: Optional[list[SmoothedLandmark]] = None

    def smooth(self, raw_landmarks) -> list[SmoothedLandmark]:
        if self._state is None:
            self._state = [
                SmoothedLandmark(lm.x, lm.y, lm.z, lm.visibility)
                for lm in raw_landmarks[:self._num]
            ]
            return list(self._state)

        result = []
        for i in range(min(self._num, len(raw_landmarks))):
            raw = raw_landmarks[i]
            prev = self._state[i]

            vis = max(0.0, min(1.0, raw.visibility))
            alpha = self._alpha * (0.5 + 0.5 * vis)

            # Dampen tracking jumps that are likely noise
            jump = math.sqrt((raw.x - prev.x) ** 2 + (raw.y - prev.y) ** 2)
            if jump > 0.15:
                alpha *= 0.3

            smoothed = SmoothedLandmark(
                x=prev.x + alpha * (raw.x - prev.x),
                y=prev.y + alpha * (raw.y - prev.y),
                z=prev.z + alpha * (raw.z - prev.z),
                visibility=prev.visibility + alpha * (raw.visibility - prev.visibility),
            )
            self._state[i] = smoothed
            result.append(smoothed)

        return result

    def reset(self):
        self._state = None


class SmoothedLandmarkList:
    """Wraps a list of SmoothedLandmark to match the .landmark[i] interface."""
    def __init__(self, landmarks: list[SmoothedLandmark]):
        self.landmark = landmarks


class PersonTracker:
    """
    Tracks multiple people across frames using shoulder-center proximity.

    Assigns a stable person_id (0..N-1) to each detection so that the same
    GestureDetector and LandmarkSmoother are reused per person.
    """

    MAX_PERSONS = 4
    MATCH_THRESHOLD = 0.25

    def __init__(self):
        self._last_centers: dict[int, tuple[float, float]] = {}
        self._next_id = 0
        self._smoothers: dict[int, LandmarkSmoother] = {}

    def match(self, pose_landmarks_list: list) -> list[tuple[int, object]]:
        """
        Match detected poses to tracked person IDs.

        Returns list of (person_id, landmarks) sorted by person_id.
        """
        new_centers = [self._shoulder_center(lms) for lms in pose_landmarks_list]
        used_ids: set[int] = set()
        assignments: list[tuple[int, object]] = []

        for det_idx, center in enumerate(new_centers):
            person_id = self._resolve_person(center, used_ids)
            if person_id is None:
                continue
            used_ids.add(person_id)
            self._last_centers[person_id] = center
            assignments.append((person_id, pose_landmarks_list[det_idx]))

        assignments.sort(key=lambda x: x[0])
        return assignments

    def get_smoother(self, person_id: int) -> LandmarkSmoother:
        if person_id not in self._smoothers:
            self._smoothers[person_id] = LandmarkSmoother(alpha=0.4)
        return self._smoothers[person_id]

    @property
    def num_tracked(self) -> int:
        return len(self._last_centers)

    def reset(self):
        self._last_centers.clear()
        self._next_id = 0
        for s in self._smoothers.values():
            s.reset()
        self._smoothers.clear()

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _shoulder_center(lms) -> tuple[float, float]:
        ls = lms[LM.LEFT_SHOULDER] if len(lms) > LM.LEFT_SHOULDER else None
        rs = lms[LM.RIGHT_SHOULDER] if len(lms) > LM.RIGHT_SHOULDER else None
        if ls and rs:
            return (ls.x + rs.x) / 2, (ls.y + rs.y) / 2
        return 0.5, 0.5

    def _resolve_person(
        self, center: tuple[float, float], used_ids: set[int]
    ) -> Optional[int]:
        """Find the closest existing person or register a new one."""
        cx, cy = center
        best_id: Optional[int] = None
        best_dist = self.MATCH_THRESHOLD

        for pid, (px, py) in self._last_centers.items():
            if pid in used_ids:
                continue
            dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_id = pid

        if best_id is None:
            if self._next_id >= self.MAX_PERSONS:
                return None
            best_id = self._next_id
            self._next_id += 1
            self._smoothers[best_id] = LandmarkSmoother(alpha=0.4)

        return best_id
