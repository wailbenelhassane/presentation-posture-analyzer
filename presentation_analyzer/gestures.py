"""
Gesture detection from MediaPipe pose landmarks.

Detects problematic presentation gestures:
- Crossed arms
- Hands in pockets
- Offensive gestures (middle finger via hand landmarks)
- Fidgeting / nervous movements
- Touching face or hair
- Arms too rigid (no gesturing)
- Slouching / poor posture
- Hands behind back
"""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Optional

import numpy as np


class Severity(Enum):
    LOW = 1
    MEDIUM = 3
    HIGH = 5
    CRITICAL = 15


@dataclass
class GestureEvent:
    """A detected gesture occurrence."""
    name: str
    severity: Severity
    confidence: float
    frame: int
    timestamp_sec: float
    description: str
    sustained_frames: int = 1
    highlight_landmarks: list[int] = field(default_factory=list)


class LM:
    """MediaPipe Pose landmark indices.

    Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    """
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _mid(a, b):
    return SimpleNamespace(x=(a.x + b.x) / 2, y=(a.y + b.y) / 2, z=(a.z + b.z) / 2)


def _point_to_segment_dist(p, seg_a, seg_b) -> float:
    """Shortest 2D distance from point p to segment (seg_a → seg_b)."""
    dx = seg_b.x - seg_a.x
    dy = seg_b.y - seg_a.y
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return _dist(p, seg_a)
    t = max(0.0, min(1.0, ((p.x - seg_a.x) * dx + (p.y - seg_a.y) * dy) / len_sq))
    return math.sqrt((p.x - seg_a.x - t * dx) ** 2 + (p.y - seg_a.y - t * dy) ** 2)


def _angle(a, b, c) -> float:
    """Angle at point b in the a–b–c triangle, in degrees."""
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2) + 1e-9
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2) + 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))))


# ── Visibility ────────────────────────────────────────────────────────────────

class VisibilityMode(Enum):
    FULL_BODY = "full_body"
    HALF_BODY = "half_body"
    UPPER_ONLY = "upper_only"
    UNKNOWN = "unknown"


# ── Detector ──────────────────────────────────────────────────────────────────

class GestureDetector:
    """
    Stateful per-person gesture detector.

    Keeps a rolling wrist-speed buffer to detect sustained poses and fidgeting.
    Call detect() once per processed frame.
    """

    FACE_TOUCH_DIST = 0.07
    BEHIND_BACK_Z = -0.18
    SLOUCH_ANGLE_THRESHOLD = 145
    FIDGET_SPEED_THRESHOLD = 0.035
    STATIC_ARMS_SPEED = 0.002
    SUSTAINED_FRAMES_MIN = 18

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._prev_wrists: Optional[list] = None
        self._speed_window = int(fps * 2)
        self._wrist_speeds: deque[float] = deque(maxlen=self._speed_window)
        self._sustained: dict[str, int] = {}

    # ── Public API ───────────────────────────────────────────────────────

    def detect(
        self,
        pose_landmarks,
        hand_landmarks_list: Optional[list] = None,
        frame_idx: int = 0,
    ) -> list[GestureEvent]:
        lm = pose_landmarks.landmark
        ts = frame_idx / self.fps
        vis = self._visibility(lm)
        events: list[GestureEvent] = []

        self._check_crossed_arms(lm, frame_idx, ts, events)
        self._check_face_touch(lm, frame_idx, ts, events)
        self._check_hands_behind_back(lm, frame_idx, ts, events)
        self._check_slouch(lm, frame_idx, ts, events)
        self._check_fidgeting(lm, frame_idx, ts, events)
        self._check_static_arms(frame_idx, ts, events)

        if vis in (VisibilityMode.FULL_BODY, VisibilityMode.HALF_BODY):
            self._check_hands_in_pockets(lm, frame_idx, ts, events)

        if hand_landmarks_list:
            self._check_offensive_gesture(hand_landmarks_list, frame_idx, ts, events)

        return events

    def detect_visibility(self, pose_landmarks) -> VisibilityMode:
        return self._visibility(pose_landmarks.landmark)

    def reset(self):
        self._prev_wrists = None
        self._wrist_speeds.clear()
        self._sustained.clear()

    # ── Visibility ───────────────────────────────────────────────────────

    def _visibility(self, lm) -> VisibilityMode:
        shoulder_vis = min(lm[LM.LEFT_SHOULDER].visibility, lm[LM.RIGHT_SHOULDER].visibility)
        hip_vis = min(lm[LM.LEFT_HIP].visibility, lm[LM.RIGHT_HIP].visibility)
        knee_vis = min(lm[LM.LEFT_KNEE].visibility, lm[LM.RIGHT_KNEE].visibility)

        if shoulder_vis < 0.5:
            return VisibilityMode.UNKNOWN
        if hip_vis < 0.5:
            return VisibilityMode.UPPER_ONLY
        if knee_vis < 0.5:
            return VisibilityMode.HALF_BODY
        return VisibilityMode.FULL_BODY

    # ── Sustained-frame tracking ─────────────────────────────────────────

    def _sustain(self, name: str, detected: bool) -> int:
        count = (self._sustained.get(name, 0) + 1) if detected else 0
        self._sustained[name] = count
        return count

    # ── Individual gesture detectors ─────────────────────────────────────

    def _check_crossed_arms(self, lm, frame, ts, events):
        lw, rw = lm[LM.LEFT_WRIST], lm[LM.RIGHT_WRIST]
        le, re = lm[LM.LEFT_ELBOW], lm[LM.RIGHT_ELBOW]
        ls, rs = lm[LM.LEFT_SHOULDER], lm[LM.RIGHT_SHOULDER]

        if min(lw.visibility, rw.visibility, le.visibility, re.visibility) < 0.5:
            self._sustain("crossed_arms", False)
            return

        torso_width = abs(rs.x - ls.x)
        margin = torso_width * 0.25

        lw_near_right_arm = min(
            _point_to_segment_dist(lw, rs, re),
            _point_to_segment_dist(lw, re, rw),
        )
        rw_near_left_arm = min(
            _point_to_segment_dist(rw, ls, le),
            _point_to_segment_dist(rw, le, lw),
        )

        n = self._sustain("crossed_arms", lw_near_right_arm < margin and rw_near_left_arm < margin)
        min_frames = int(self.fps * 2)
        if n < min_frames:
            return

        events.append(GestureEvent(
            name="crossed_arms",
            severity=Severity.MEDIUM,
            confidence=min(1.0, n / (min_frames * 2)),
            frame=frame, timestamp_sec=ts,
            description="Brazos cruzados — transmite actitud defensiva o cerrada",
            sustained_frames=n,
            highlight_landmarks=[LM.LEFT_WRIST, LM.RIGHT_WRIST, LM.LEFT_ELBOW, LM.RIGHT_ELBOW],
        ))

    def _check_hands_in_pockets(self, lm, frame, ts, events):
        lw, rw = lm[LM.LEFT_WRIST], lm[LM.RIGHT_WRIST]
        lh, rh = lm[LM.LEFT_HIP], lm[LM.RIGHT_HIP]
        le, re = lm[LM.LEFT_ELBOW], lm[LM.RIGHT_ELBOW]

        shoulder_y = (lm[LM.LEFT_SHOULDER].y + lm[LM.RIGHT_SHOULDER].y) / 2
        torso_height = abs((lh.y + rh.y) / 2 - shoulder_y)

        left_in = self._wrist_at_hip(lw, le, lh, torso_height)
        right_in = self._wrist_at_hip(rw, re, rh, torso_height)

        n = self._sustain("hands_in_pockets", left_in or right_in)
        min_frames = int(self.fps * 2)
        if n < min_frames:
            return

        both = left_in and right_in
        hl = ([LM.LEFT_WRIST] if left_in else []) + ([LM.RIGHT_WRIST] if right_in else [])
        events.append(GestureEvent(
            name="hands_in_pockets",
            severity=Severity.HIGH if both else Severity.MEDIUM,
            confidence=min(1.0, n / (min_frames * 2)),
            frame=frame, timestamp_sec=ts,
            description="Manos en los bolsillos — proyecta desinterés o inseguridad"
                        + (" (ambas manos)" if both else ""),
            sustained_frames=n,
            highlight_landmarks=hl,
        ))

    @staticmethod
    def _wrist_at_hip(wrist, elbow, hip, torso_height: float) -> bool:
        y_margin = torso_height * 0.15
        x_margin = torso_height * 0.25
        return (
            (abs(wrist.y - hip.y) < y_margin or wrist.y > hip.y)
            and abs(wrist.x - hip.x) < x_margin
            and wrist.visibility > 0.2
            and elbow.y < wrist.y - 0.02
        )

    def _check_face_touch(self, lm, frame, ts, events):
        nose = lm[LM.NOSE]
        lw, rw = lm[LM.LEFT_WRIST], lm[LM.RIGHT_WRIST]

        left_touch = _dist(lw, nose) < self.FACE_TOUCH_DIST
        right_touch = _dist(rw, nose) < self.FACE_TOUCH_DIST
        n = self._sustain("face_touch", left_touch or right_touch)
        if n < self.SUSTAINED_FRAMES_MIN:
            return

        hl = ([LM.LEFT_WRIST] if left_touch else []) + ([LM.RIGHT_WRIST] if right_touch else [])
        events.append(GestureEvent(
            name="face_touch",
            severity=Severity.LOW,
            confidence=min(1.0, n / self.SUSTAINED_FRAMES_MIN),
            frame=frame, timestamp_sec=ts,
            description="Tocarse la cara/pelo — señal de nerviosismo",
            sustained_frames=n,
            highlight_landmarks=[LM.NOSE] + hl,
        ))

    def _check_hands_behind_back(self, lm, frame, ts, events):
        lw, rw = lm[LM.LEFT_WRIST], lm[LM.RIGHT_WRIST]
        ls, rs = lm[LM.LEFT_SHOULDER], lm[LM.RIGHT_SHOULDER]

        left_behind = lw.z < ls.z + self.BEHIND_BACK_Z and lw.visibility < 0.4
        right_behind = rw.z < rs.z + self.BEHIND_BACK_Z and rw.visibility < 0.4
        n = self._sustain("hands_behind_back", left_behind and right_behind)
        if n < self.SUSTAINED_FRAMES_MIN:
            return

        events.append(GestureEvent(
            name="hands_behind_back",
            severity=Severity.MEDIUM,
            confidence=min(1.0, n / (self.SUSTAINED_FRAMES_MIN * 2)),
            frame=frame, timestamp_sec=ts,
            description="Manos detrás de la espalda — oculta lenguaje corporal",
            sustained_frames=n,
            highlight_landmarks=[LM.LEFT_WRIST, LM.RIGHT_WRIST],
        ))

    def _check_slouch(self, lm, frame, ts, events):
        ls, rs = lm[LM.LEFT_SHOULDER], lm[LM.RIGHT_SHOULDER]
        lh, rh = lm[LM.LEFT_HIP], lm[LM.RIGHT_HIP]

        if min(lh.visibility, rh.visibility) < 0.5:
            self._sustain("slouch", False)
            return

        angle = _angle(lm[LM.NOSE], _mid(ls, rs), _mid(lh, rh))
        n = self._sustain("slouch", angle < self.SLOUCH_ANGLE_THRESHOLD)
        if n < self.SUSTAINED_FRAMES_MIN * 3:
            return

        events.append(GestureEvent(
            name="slouch",
            severity=Severity.MEDIUM,
            confidence=min(1.0, n / (self.SUSTAINED_FRAMES_MIN * 4)),
            frame=frame, timestamp_sec=ts,
            description="Postura encorvada — reduce presencia y autoridad",
            sustained_frames=n,
            highlight_landmarks=[LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.NOSE],
        ))

    def _check_fidgeting(self, lm, frame, ts, events):
        lw, rw = lm[LM.LEFT_WRIST], lm[LM.RIGHT_WRIST]
        current = [(lw.x, lw.y), (rw.x, rw.y)]

        if self._prev_wrists is not None:
            speed = sum(
                math.sqrt((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2)
                for c, p in zip(current, self._prev_wrists)
            ) / 2
            self._wrist_speeds.append(speed)

            if len(self._wrist_speeds) >= self._speed_window // 2:
                avg_speed = float(np.mean(self._wrist_speeds))
                std_speed = float(np.std(self._wrist_speeds))
                is_fidget = (
                    std_speed > self.FIDGET_SPEED_THRESHOLD * 0.7
                    and avg_speed > self.FIDGET_SPEED_THRESHOLD * 0.5
                )
                n = self._sustain("fidgeting", is_fidget)
                if n >= self._speed_window // 2:
                    events.append(GestureEvent(
                        name="fidgeting",
                        severity=Severity.MEDIUM,
                        confidence=min(1.0, avg_speed / self.FIDGET_SPEED_THRESHOLD),
                        frame=frame, timestamp_sec=ts,
                        description="Movimiento nervioso excesivo de manos",
                        sustained_frames=n,
                        highlight_landmarks=[LM.LEFT_WRIST, LM.RIGHT_WRIST],
                    ))

        self._prev_wrists = current

    def _check_static_arms(self, frame, ts, events):
        if len(self._wrist_speeds) < self._speed_window:
            return

        avg_speed = float(np.mean(self._wrist_speeds))
        n = self._sustain("static_arms", avg_speed < self.STATIC_ARMS_SPEED)
        if n < int(self.fps * 8):
            return

        events.append(GestureEvent(
            name="static_arms",
            severity=Severity.LOW,
            confidence=min(1.0, n / (self.fps * 10)),
            frame=frame, timestamp_sec=ts,
            description="Brazos estáticos — gesticular refuerza el mensaje",
            sustained_frames=n,
            highlight_landmarks=[LM.LEFT_WRIST, LM.RIGHT_WRIST, LM.LEFT_ELBOW, LM.RIGHT_ELBOW],
        ))

    def _check_offensive_gesture(self, hand_landmarks_list, frame, ts, events):
        for hand_lm in hand_landmarks_list:
            lm = hand_lm.landmark
            middle_ext = self._finger_extended(lm, 12, 10)
            index_ext = self._finger_extended(lm, 8, 6)
            ring_ext = self._finger_extended(lm, 16, 14)
            pinky_ext = self._finger_extended(lm, 20, 18)

            if middle_ext and not index_ext and not ring_ext and not pinky_ext:
                events.append(GestureEvent(
                    name="offensive_gesture",
                    severity=Severity.CRITICAL,
                    confidence=0.85,
                    frame=frame, timestamp_sec=ts,
                    description="¡Gesto ofensivo detectado! — totalmente inapropiado",
                    sustained_frames=1,
                    highlight_landmarks=[LM.LEFT_WRIST, LM.RIGHT_WRIST],
                ))

    @staticmethod
    def _finger_extended(lm, tip_idx: int, pip_idx: int) -> bool:
        return lm[tip_idx].y < lm[pip_idx].y
