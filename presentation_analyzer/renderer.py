"""
Video renderer — draws skeleton, red alert circles, and event labels.

Creates an annotated output video showing:
  - Pose skeleton in green/white
  - Red pulsing circles on problematic landmarks
  - Event label + severity banner at the top
  - Timestamp MM:SS overlay
"""

import cv2
import math
import numpy as np
from typing import Optional
from .gestures import GestureEvent, Severity, LM


# ── Skeleton connections (MediaPipe Pose) ────────────────────────────────
POSE_CONNECTIONS = [
    (LM.NOSE, LM.LEFT_EYE), (LM.NOSE, LM.RIGHT_EYE),
    (LM.LEFT_EYE, LM.LEFT_EAR), (LM.RIGHT_EYE, LM.RIGHT_EAR),
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP), (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW), (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.LEFT_WRIST, LM.LEFT_INDEX), (LM.LEFT_WRIST, LM.LEFT_PINKY),
    (LM.LEFT_WRIST, LM.LEFT_THUMB),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW), (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.RIGHT_WRIST, LM.RIGHT_INDEX), (LM.RIGHT_WRIST, LM.RIGHT_PINKY),
    (LM.RIGHT_WRIST, LM.RIGHT_THUMB),
    (LM.LEFT_HIP, LM.LEFT_KNEE), (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE), (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
]

SEVERITY_COLORS = {
    Severity.LOW:      (0, 200, 255),    # naranja
    Severity.MEDIUM:   (0, 100, 255),    # rojo-naranja
    Severity.HIGH:     (0, 0, 255),      # rojo
    Severity.CRITICAL: (0, 0, 200),      # rojo oscuro
}

SEVERITY_LABELS = {
    Severity.LOW:      "LEVE",
    Severity.MEDIUM:   "MEDIO",
    Severity.HIGH:     "ALTO",
    Severity.CRITICAL: "CRITICO",
}


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


class VideoRenderer:
    """
    Renders annotated video frames with skeleton and gesture highlights.
    """

    SKELETON_COLOR = (180, 220, 180)
    SKELETON_THICKNESS = 2
    JOINT_COLOR = (255, 255, 255)
    JOINT_RADIUS = 4
    ALERT_CIRCLE_RADIUS = 28
    ALERT_CIRCLE_THICKNESS = 3
    BANNER_HEIGHT = 50
    LABEL_FADE_FRAMES = 45

    def __init__(self, width: int, height: int, fps: float):
        self.w = width
        self.h = height
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._active_labels: list[dict] = []

    def open(self, output_path: str):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.w, self.h))
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open output video: {output_path}")

    def draw_frame(
        self,
        frame: np.ndarray,
        pose_landmarks,
        events: list[GestureEvent],
        frame_idx: int,
    ) -> np.ndarray:
        annotated = frame.copy()

        if pose_landmarks is not None:
            lm = pose_landmarks.landmark
            self._draw_skeleton(annotated, lm)
            self._draw_joints(annotated, lm)

            for ev in events:
                self._draw_alert_circles(annotated, lm, ev, frame_idx)

        for ev in events:
            self._active_labels.append({
                "text": ev.description,
                "name": ev.name,
                "severity": ev.severity,
                "ttl": self.LABEL_FADE_FRAMES,
            })

        self._draw_labels(annotated)
        self._draw_timestamp(annotated, frame_idx / self.fps)

        if self._writer:
            self._writer.write(annotated)

        return annotated

    def close(self):
        if self._writer:
            self._writer.release()
            self._writer = None

    # ── Drawing helpers ──────────────────────────────────────────────────

    def _lm_to_px(self, landmark) -> Optional[tuple]:
        if landmark.visibility < 0.3:
            return None
        return (int(landmark.x * self.w), int(landmark.y * self.h))

    def _draw_skeleton(self, frame, lm):
        for (a, b) in POSE_CONNECTIONS:
            if a >= len(lm) or b >= len(lm):
                continue
            pa = self._lm_to_px(lm[a])
            pb = self._lm_to_px(lm[b])
            if pa and pb:
                cv2.line(frame, pa, pb, self.SKELETON_COLOR, self.SKELETON_THICKNESS,
                         lineType=cv2.LINE_AA)

    def _draw_joints(self, frame, lm):
        for i in range(min(29, len(lm))):
            pt = self._lm_to_px(lm[i])
            if pt:
                cv2.circle(frame, pt, self.JOINT_RADIUS, self.JOINT_COLOR, -1,
                           lineType=cv2.LINE_AA)

    def _draw_alert_circles(self, frame, lm, event: GestureEvent, frame_idx: int):
        if not event.highlight_landmarks:
            return

        color = SEVERITY_COLORS.get(event.severity, (0, 0, 255))
        pulse = math.sin(frame_idx * 0.3) * 5
        radius = self.ALERT_CIRCLE_RADIUS + int(pulse)

        for lm_idx in event.highlight_landmarks:
            if lm_idx >= len(lm):
                continue
            pt = self._lm_to_px(lm[lm_idx])
            if pt is None:
                continue

            # Outer pulsing circle
            cv2.circle(frame, pt, radius, color, self.ALERT_CIRCLE_THICKNESS,
                       lineType=cv2.LINE_AA)

            # Semi-transparent fill for CRITICAL
            if event.severity == Severity.CRITICAL:
                overlay = frame.copy()
                cv2.circle(overlay, pt, radius, color, -1)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

            # Inner solid dot
            cv2.circle(frame, pt, 6, color, -1, lineType=cv2.LINE_AA)

    def _draw_labels(self, frame):
        still_active = []
        for label in self._active_labels:
            label["ttl"] -= 1
            if label["ttl"] > 0:
                still_active.append(label)
        self._active_labels = still_active

        if not self._active_labels:
            return

        current = max(self._active_labels, key=lambda l: (l["severity"].value, l["ttl"]))

        color = SEVERITY_COLORS.get(current["severity"], (0, 0, 255))
        sev_label = SEVERITY_LABELS.get(current["severity"], "")

        # Banner background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.BANNER_HEIGHT), (0, 0, 0), -1)
        alpha = min(0.75, current["ttl"] / self.LABEL_FADE_FRAMES)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Severity badge
        badge_text = f" {sev_label} "
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (10, 8), (10 + tw + 8, 8 + th + 10), color, -1)
        cv2.putText(frame, badge_text, (14, 8 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    lineType=cv2.LINE_AA)

        # Description
        text_x = 10 + tw + 20
        cv2.putText(frame, current["text"], (text_x, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    lineType=cv2.LINE_AA)

    def _draw_timestamp(self, frame, seconds: float):
        ts_text = format_timestamp(seconds)
        (tw, th), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = self.w - tw - 15
        y = self.h - 15
        cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, ts_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    lineType=cv2.LINE_AA)