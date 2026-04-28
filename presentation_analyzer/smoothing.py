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

    Reduces jitter while preserving real movement.
    Lower alpha = more smoothing (slower response).
    Higher alpha = less smoothing (faster response, more jitter).
    """

    def __init__(self, num_landmarks: int = 33, alpha: float = 0.4):
        """
        Args:
            num_landmarks: Number of landmarks to track (33 for Pose).
            alpha: EMA weight for new data. 0.3-0.5 is a good range.
                   0.3 = very smooth, 0.5 = responsive.
        """
        self._alpha = alpha
        self._num = num_landmarks
        self._state: Optional[list[SmoothedLandmark]] = None
        self._initialized = False

    def smooth(self, raw_landmarks) -> list[SmoothedLandmark]:
        """
        Apply EMA smoothing to a frame's landmarks.

        Args:
            raw_landmarks: list of landmarks with .x, .y, .z, .visibility

        Returns:
            list of SmoothedLandmark with filtered values.
        """
        if not self._initialized:
            # First frame — initialize state from raw data
            self._state = []
            for lm in raw_landmarks[:self._num]:
                self._state.append(SmoothedLandmark(lm.x, lm.y, lm.z, lm.visibility))
            self._initialized = True
            return list(self._state)

        result = []
        for i in range(min(self._num, len(raw_landmarks))):
            raw = raw_landmarks[i]
            prev = self._state[i]

            # Adaptive alpha: use more smoothing for low-visibility landmarks
            # and less smoothing for high-visibility ones (they're more reliable)
            vis = max(0.0, min(1.0, raw.visibility))
            adaptive_alpha = self._alpha * (0.5 + 0.5 * vis)

            # Detect large jumps (likely tracking errors) and smooth more aggressively
            dx = raw.x - prev.x
            dy = raw.y - prev.y
            jump = math.sqrt(dx * dx + dy * dy)
            if jump > 0.15:  # large jump = probably noise
                adaptive_alpha *= 0.3

            smoothed = SmoothedLandmark(
                x=prev.x + adaptive_alpha * (raw.x - prev.x),
                y=prev.y + adaptive_alpha * (raw.y - prev.y),
                z=prev.z + adaptive_alpha * (raw.z - prev.z),
                visibility=prev.visibility + adaptive_alpha * (raw.visibility - prev.visibility),
            )
            self._state[i] = smoothed
            result.append(smoothed)

        return result

    def reset(self):
        self._state = None
        self._initialized = False


class SmoothedLandmarkList:
    """Wraps a list of SmoothedLandmark to match the .landmark[i] interface."""
    def __init__(self, landmarks: list[SmoothedLandmark]):
        self.landmark = landmarks


class PersonTracker:
    """
    Tracks multiple people across frames using shoulder-center proximity.

    Assigns a stable person_id (0..N-1) to each detection so that
    the same GestureDetector + LandmarkSmoother are reused per person.
    """

    MAX_PERSONS = 4
    MATCH_THRESHOLD = 0.25  # max normalised distance to consider same person

    def __init__(self):
        self._last_centers: dict[int, tuple[float, float]] = {}
        self._next_id = 0
        self._smoothers: dict[int, LandmarkSmoother] = {}

    def match(self, pose_landmarks_list: list) -> list[tuple[int, object]]:
        """
        Match detected poses to tracked person IDs.

        Args:
            pose_landmarks_list: list of landmark lists (one per detected person).

        Returns:
            list of (person_id, landmarks) tuples, sorted by person_id.
        """
        # Compute center of each detection (midpoint of shoulders)
        new_centers = []
        for lms in pose_landmarks_list:
            ls = lms[LM.LEFT_SHOULDER] if len(lms) > LM.LEFT_SHOULDER else None
            rs = lms[LM.RIGHT_SHOULDER] if len(lms) > LM.RIGHT_SHOULDER else None
            if ls and rs:
                cx = (ls.x + rs.x) / 2
                cy = (ls.y + rs.y) / 2
            else:
                cx, cy = 0.5, 0.5
            new_centers.append((cx, cy))

        # Greedy matching: for each detection, find closest existing person
        used_ids = set()
        assignments: list[tuple[int, object]] = []

        for det_idx, (cx, cy) in enumerate(new_centers):
            best_id = None
            best_dist = self.MATCH_THRESHOLD

            for pid, (px, py) in self._last_centers.items():
                if pid in used_ids:
                    continue
                dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_id = pid

            if best_id is None:
                # New person
                if self._next_id < self.MAX_PERSONS:
                    best_id = self._next_id
                    self._next_id += 1
                    self._smoothers[best_id] = LandmarkSmoother(alpha=0.4)
                else:
                    continue  # max persons reached, skip

            used_ids.add(best_id)
            self._last_centers[best_id] = (cx, cy)
            assignments.append((best_id, pose_landmarks_list[det_idx]))

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
