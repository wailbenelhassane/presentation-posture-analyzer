"""
Gesture detection from MediaPipe pose landmarks.

Detects problematic presentation gestures:
- Crossed arms
- Hands in pockets
- Offensive gestures (middle finger via hand landmarks)
- Fidgeting / nervous movements
- Touching face or hair
- Closed/tense fists
- Arms too rigid (no gesturing)
- Slouching / poor posture
- Hands behind back
"""

import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(Enum):
    """How severely a gesture impacts the score."""
    LOW = 1        # Minor issue (e.g., slightly rigid arms)
    MEDIUM = 3     # Noticeable problem (e.g., crossed arms)
    HIGH = 5       # Serious issue (e.g., hands in pockets for long)
    CRITICAL = 15  # Offensive or very damaging (e.g., middle finger)


@dataclass
class GestureEvent:
    """A detected gesture occurrence."""
    name: str
    severity: Severity
    confidence: float       # 0.0 - 1.0
    frame: int
    timestamp_sec: float
    description: str
    sustained_frames: int = 1
    highlight_landmarks: list = field(default_factory=list)  # pose landmark indices to circle in red


# ── MediaPipe Pose landmark indices ──────────────────────────────────────────
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
class LM:
    NOSE = 0
    LEFT_EYE_INNER = 1; LEFT_EYE = 2; LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4; RIGHT_EYE = 5; RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7; RIGHT_EAR = 8
    MOUTH_LEFT = 9; MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13; RIGHT_ELBOW = 14
    LEFT_WRIST = 15; RIGHT_WRIST = 16
    LEFT_PINKY = 17; RIGHT_PINKY = 18
    LEFT_INDEX = 19; RIGHT_INDEX = 20
    LEFT_THUMB = 21; RIGHT_THUMB = 22
    LEFT_HIP = 23; RIGHT_HIP = 24
    LEFT_KNEE = 25; RIGHT_KNEE = 26
    LEFT_ANKLE = 27; RIGHT_ANKLE = 28


def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _mid(a, b):
    """Return a simple namespace with midpoint coords."""
    class P:
        pass
    p = P()
    p.x = (a.x + b.x) / 2
    p.y = (a.y + b.y) / 2
    p.z = (a.z + b.z) / 2
    return p


def _angle(a, b, c) -> float:
    """Angle at point b formed by segments ba and bc, in degrees."""
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2) + 1e-9
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2) + 1e-9
    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


class VisibilityMode(Enum):
    """What portion of the body is visible."""
    FULL_BODY = "full_body"
    HALF_BODY = "half_body"      # waist up
    UPPER_ONLY = "upper_only"    # shoulders/head only
    UNKNOWN = "unknown"


class GestureDetector:
    """
    Stateful detector that tracks gestures across frames.

    Keeps a small rolling buffer to detect sustained poses and fidgeting.
    Memory footprint: ~50 KB for a 30 fps / 10-min video.
    """

    # ── tuneable thresholds ──────────────────────────────────────────────
    CROSSED_ARMS_WRIST_DIST = 0.08      # normalised; wrists close + crossed
    POCKET_Y_OFFSET = 0.03              # wrist below hip
    POCKET_X_CLOSENESS = 0.10           # wrist horizontally near hip
    FACE_TOUCH_DIST = 0.10              # wrist near nose
    BEHIND_BACK_Z = -0.12              # wrist z behind shoulder z
    SLOUCH_ANGLE_THRESHOLD = 155        # shoulder-hip vertical angle
    FIDGET_SPEED_THRESHOLD = 0.025      # per-frame normalised movement
    STATIC_ARMS_SPEED = 0.003           # too little movement
    SUSTAINED_FRAMES_MIN = 8            # frames before reporting sustained pose
    OFFENSIVE_FINGER_RATIO = 1.8        # middle finger extended vs others

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._prev_wrists: Optional[list] = None
        self._wrist_speeds: list[float] = []       # rolling window
        self._speed_window = int(fps * 2)           # 2-second window
        self._sustained: dict[str, int] = {}        # gesture_name -> frame_count

    # ── public API ───────────────────────────────────────────────────────
    def detect(
        self,
        pose_landmarks,
        hand_landmarks_list: Optional[list] = None,
        frame_idx: int = 0,
    ) -> list[GestureEvent]:
        """
        Run all detectors on a single frame.

        Args:
            pose_landmarks: MediaPipe Pose result (NormalizedLandmarkList).
            hand_landmarks_list: Optional list of hand NormalizedLandmarkList
                                 from MediaPipe Hands (for offensive gesture detection).
            frame_idx: Current frame number.

        Returns:
            List of GestureEvent detected this frame.
        """
        lm = pose_landmarks.landmark
        ts = frame_idx / self.fps
        vis = self._detect_visibility(lm)
        events: list[GestureEvent] = []

        # --- posture checks (always available with upper body) ---
        self._check_crossed_arms(lm, frame_idx, ts, events)
        self._check_face_touch(lm, frame_idx, ts, events)
        self._check_hands_behind_back(lm, frame_idx, ts, events)
        self._check_slouch(lm, frame_idx, ts, events)
        self._check_fidgeting(lm, frame_idx, ts, events)
        self._check_static_arms(lm, frame_idx, ts, events)

        # --- pocket detection only if hips visible ---
        if vis in (VisibilityMode.FULL_BODY, VisibilityMode.HALF_BODY):
            self._check_hands_in_pockets(lm, frame_idx, ts, events)

        # --- offensive gestures if hand landmarks provided ---
        if hand_landmarks_list:
            self._check_offensive_gesture(hand_landmarks_list, frame_idx, ts, events)

        return events

    def detect_visibility(self, pose_landmarks) -> VisibilityMode:
        return self._detect_visibility(pose_landmarks.landmark)

    # ── visibility detection ─────────────────────────────────────────────
    def _detect_visibility(self, lm) -> VisibilityMode:
        hip_vis = min(lm[LM.LEFT_HIP].visibility, lm[LM.RIGHT_HIP].visibility)
        knee_vis = min(lm[LM.LEFT_KNEE].visibility, lm[LM.RIGHT_KNEE].visibility)
        shoulder_vis = min(lm[LM.LEFT_SHOULDER].visibility, lm[LM.RIGHT_SHOULDER].visibility)

        if shoulder_vis < 0.5:
            return VisibilityMode.UNKNOWN
        if hip_vis < 0.5:
            return VisibilityMode.UPPER_ONLY
        if knee_vis < 0.5:
            return VisibilityMode.HALF_BODY
        return VisibilityMode.FULL_BODY

    # ── individual gesture detectors ─────────────────────────────────────

    def _sustain(self, name: str, detected: bool) -> int:
        """Track how many consecutive frames a gesture has been active."""
        if detected:
            self._sustained[name] = self._sustained.get(name, 0) + 1
        else:
            self._sustained[name] = 0
        return self._sustained[name]

    def _check_crossed_arms(self, lm, frame, ts, events):
        lw, rw = lm[LM.LEFT_WRIST], lm[LM.RIGHT_WRIST]
        le, re = lm[LM.LEFT_ELBOW], lm[LM.RIGHT_ELBOW]
        ls, rs = lm[LM.LEFT_SHOULDER], lm[LM.RIGHT_SHOULDER]

        wrists_crossed = lw.x > rw.x
        wrists_close = _dist(lw, rw) < self.CROSSED_ARMS_WRIST_DIST * 3

        left_angle = _angle(ls, le, lw)
        right_angle = _angle(rs, re, rw)
        elbows_bent = left_angle < 100 and right_angle < 100

        torso_mid_y = (ls.y + lm[LM.LEFT_HIP].y) / 2
        wrists_at_chest = (
            min(lw.y, rw.y) > ls.y - 0.05 and
            max(lw.y, rw.y) < lm[LM.LEFT_HIP].y + 0.05
        )

        detected = wrists_crossed and (wrists_close or elbows_bent) and wrists_at_chest
        n = self._sustain("crossed_arms", detected)

        if n >= self.SUSTAINED_FRAMES_MIN:
            confidence = min(1.0, n / (self.SUSTAINED_FRAMES_MIN * 3))
            events.append(GestureEvent(
                name="crossed_arms",
                severity=Severity.MEDIUM,
                confidence=confidence,
                frame=frame, timestamp_sec=ts,
                description="Brazos cruzados — transmite actitud defensiva o cerrada",
                sustained_frames=n,
                highlight_landmarks=[LM.LEFT_WRIST, LM.RIGHT_WRIST, LM.LEFT_ELBOW, LM.RIGHT_ELBOW],
            ))

    def _check_hands_in_pockets(self, lm, frame, ts, events):
        lw, rw = lm[LM.LEFT_WRIST], lm[LM.RIGHT_WRIST]
        lh, rh = lm[LM.LEFT_HIP], lm[LM.RIGHT_HIP]

        left_in = (
            lw.y > lh.y + self.POCKET_Y_OFFSET and
            abs(lw.x - lh.x) < self.POCKET_X_CLOSENESS and
            lw.visibility > 0.3
        )
        right_in = (
            rw.y > rh.y + self.POCKET_Y_OFFSET and
            abs(rw.x - rh.x) < self.POCKET_X_CLOSENESS and
            rw.visibility > 0.3
        )
        detected = left_in or right_in
        n = self._sustain("hands_in_pockets", detected)

        if n >= self.SUSTAINED_FRAMES_MIN:
            both = left_in and right_in
            sev = Severity.HIGH if both else Severity.MEDIUM
            hl = []
            if left_in:
                hl.append(LM.LEFT_WRIST)
            if right_in:
                hl.append(LM.RIGHT_WRIST)
            events.append(GestureEvent(
                name="hands_in_pockets",
                severity=sev,
                confidence=min(1.0, n / (self.SUSTAINED_FRAMES_MIN * 2)),
                frame=frame, timestamp_sec=ts,
                description="Manos en los bolsillos — proyecta desinterés o inseguridad"
                            + (" (ambas manos)" if both else ""),
                sustained_frames=n,
                highlight_landmarks=hl,
            ))

    def _check_face_touch(self, lm, frame, ts, events):
        nose = lm[LM.NOSE]
        lw, rw = lm[LM.LEFT_WRIST], lm[LM.RIGHT_WRIST]

        left_touch = _dist(lw, nose) < self.FACE_TOUCH_DIST
        right_touch = _dist(rw, nose) < self.FACE_TOUCH_DIST
        detected = left_touch or right_touch
        n = self._sustain("face_touch", detected)

        if n >= self.SUSTAINED_FRAMES_MIN // 2:
            hl = [LM.NOSE]
            if left_touch:
                hl.append(LM.LEFT_WRIST)
            if right_touch:
                hl.append(LM.RIGHT_WRIST)
            events.append(GestureEvent(
                name="face_touch",
                severity=Severity.LOW,
                confidence=min(1.0, n / self.SUSTAINED_FRAMES_MIN),
                frame=frame, timestamp_sec=ts,
                description="Tocarse la cara/pelo — señal de nerviosismo",
                sustained_frames=n,
                highlight_landmarks=hl,
            ))

    def _check_hands_behind_back(self, lm, frame, ts, events):
        lw, rw = lm[LM.LEFT_WRIST], lm[LM.RIGHT_WRIST]
        ls, rs = lm[LM.LEFT_SHOULDER], lm[LM.RIGHT_SHOULDER]

        left_behind = lw.z < ls.z + self.BEHIND_BACK_Z and lw.visibility < 0.4
        right_behind = rw.z < rs.z + self.BEHIND_BACK_Z and rw.visibility < 0.4
        detected = left_behind and right_behind
        n = self._sustain("hands_behind_back", detected)

        if n >= self.SUSTAINED_FRAMES_MIN:
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

        mid_shoulder = _mid(ls, rs)
        mid_hip = _mid(lh, rh)
        nose = lm[LM.NOSE]

        angle = _angle(nose, mid_shoulder, mid_hip)
        detected = angle < self.SLOUCH_ANGLE_THRESHOLD
        n = self._sustain("slouch", detected)

        if n >= self.SUSTAINED_FRAMES_MIN * 2:
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
            if len(self._wrist_speeds) > self._speed_window:
                self._wrist_speeds.pop(0)

            if len(self._wrist_speeds) >= self._speed_window // 2:
                avg_speed = np.mean(self._wrist_speeds)
                std_speed = np.std(self._wrist_speeds)

                is_fidget = (
                    std_speed > self.FIDGET_SPEED_THRESHOLD * 0.5 and
                    avg_speed > self.FIDGET_SPEED_THRESHOLD * 0.3
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

    def _check_static_arms(self, lm, frame, ts, events):
        """Detect overly rigid arms (no gesturing at all)."""
        if len(self._wrist_speeds) < self._speed_window:
            return

        avg_speed = np.mean(self._wrist_speeds[-self._speed_window:])
        detected = avg_speed < self.STATIC_ARMS_SPEED
        n = self._sustain("static_arms", detected)

        if n >= int(self.fps * 5):
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
        """
        Detect middle finger using MediaPipe Hand landmarks.
        Middle finger extended while others are curled.
        """
        for hand_lm in hand_landmarks_list:
            lm = hand_lm.landmark

            def is_extended(tip_idx, pip_idx):
                return lm[tip_idx].y < lm[pip_idx].y

            middle_ext = is_extended(12, 10)
            index_ext = is_extended(8, 6)
            ring_ext = is_extended(16, 14)
            pinky_ext = is_extended(20, 18)

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

    def reset(self):
        """Reset state between videos."""
        self._prev_wrists = None
        self._wrist_speeds.clear()
        self._sustained.clear()