"""
Video renderer — draws skeleton, alert circles, and event labels.

Supports multiple persons with different skeleton colors.
"""

import cv2
import math
import numpy as np
from typing import Optional

from .gestures import GestureEvent, Severity, LM


POSE_CONNECTIONS = [
    (LM.NOSE, LM.LEFT_EYE),   (LM.NOSE, LM.RIGHT_EYE),
    (LM.LEFT_EYE, LM.LEFT_EAR), (LM.RIGHT_EYE, LM.RIGHT_EAR),
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP),   (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW), (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.LEFT_WRIST, LM.LEFT_INDEX),    (LM.LEFT_WRIST, LM.LEFT_PINKY),
    (LM.LEFT_WRIST, LM.LEFT_THUMB),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW), (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.RIGHT_WRIST, LM.RIGHT_INDEX),    (LM.RIGHT_WRIST, LM.RIGHT_PINKY),
    (LM.RIGHT_WRIST, LM.RIGHT_THUMB),
    (LM.LEFT_HIP, LM.LEFT_KNEE),   (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE), (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
]

# BGR colors, one per person
PERSON_COLORS = [
    (180, 220, 180),  # soft green
    (220, 180, 120),  # soft blue
    (140, 200, 220),  # warm yellow
    (200, 160, 200),  # soft purple
]

SEVERITY_COLORS = {
    Severity.LOW:      (0, 200, 255),
    Severity.MEDIUM:   (0, 100, 255),
    Severity.HIGH:     (0, 0, 255),
    Severity.CRITICAL: (0, 0, 200),
}

SEVERITY_LABELS = {
    Severity.LOW:      "LEVE",
    Severity.MEDIUM:   "MEDIO",
    Severity.HIGH:     "ALTO",
    Severity.CRITICAL: "CRITICO",
}


def format_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _brighten(color: tuple, amount: int = 40) -> tuple:
    return tuple(min(255, c + amount) for c in color)


class VideoRenderer:

    SKELETON_THICKNESS = 2
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

    def draw_frame_multi(
        self,
        frame: np.ndarray,
        frame_data: list,
        frame_idx: int,
        events_override: Optional[list] = None,
    ) -> np.ndarray:
        """
        Draw skeletons and highlights for all detected persons.

        events_override: when set (even to []), replaces per-person events
                         (used for skipped frames to avoid duplicate labels).
        """
        annotated = frame.copy()
        all_events: list[GestureEvent] = []

        for person_id, events, landmarks in frame_data:
            if landmarks is None:
                continue

            lm = landmarks.landmark
            color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
            self._draw_skeleton(annotated, lm, color)
            self._draw_joints(annotated, lm, color)

            evts = events if events_override is None else events_override
            for ev in evts:
                self._draw_alert_circles(annotated, lm, ev, frame_idx)
            all_events.extend(evts)

        for ev in all_events:
            self._active_labels.append({
                "text":     ev.description,
                "severity": ev.severity,
                "ttl":      self.LABEL_FADE_FRAMES,
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

    def _draw_skeleton(self, frame, lm, color):
        for a, b in POSE_CONNECTIONS:
            if a >= len(lm) or b >= len(lm):
                continue
            pa = self._lm_to_px(lm[a])
            pb = self._lm_to_px(lm[b])
            if pa and pb:
                cv2.line(frame, pa, pb, color, self.SKELETON_THICKNESS, lineType=cv2.LINE_AA)

    def _draw_joints(self, frame, lm, color):
        joint_color = _brighten(color)
        for i in range(min(29, len(lm))):
            pt = self._lm_to_px(lm[i])
            if pt:
                cv2.circle(frame, pt, self.JOINT_RADIUS, joint_color, -1, lineType=cv2.LINE_AA)

    def _draw_alert_circles(self, frame, lm, event: GestureEvent, frame_idx: int):
        if not event.highlight_landmarks:
            return

        color = SEVERITY_COLORS.get(event.severity, (0, 0, 255))
        radius = self.ALERT_CIRCLE_RADIUS + int(math.sin(frame_idx * 0.3) * 5)

        for lm_idx in event.highlight_landmarks:
            if lm_idx >= len(lm):
                continue
            pt = self._lm_to_px(lm[lm_idx])
            if pt is None:
                continue
            cv2.circle(frame, pt, radius, color, self.ALERT_CIRCLE_THICKNESS, lineType=cv2.LINE_AA)
            if event.severity == Severity.CRITICAL:
                overlay = frame.copy()
                cv2.circle(overlay, pt, radius, color, -1)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.circle(frame, pt, 6, color, -1, lineType=cv2.LINE_AA)

    def _draw_labels(self, frame):
        for label in self._active_labels:
            label["ttl"] -= 1
        self._active_labels = [l for l in self._active_labels if l["ttl"] > 0]
        if not self._active_labels:
            return
        current = max(self._active_labels, key=lambda l: (l["severity"].value, l["ttl"]))
        self._render_label_banner(frame, current)

    def _render_label_banner(self, frame, label: dict):
        color = SEVERITY_COLORS.get(label["severity"], (0, 0, 255))
        sev_text = SEVERITY_LABELS.get(label["severity"], "")
        alpha = min(0.75, label["ttl"] / self.LABEL_FADE_FRAMES)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.BANNER_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        badge = f" {sev_text} "
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (10, 8), (10 + tw + 8, 8 + th + 10), color, -1)
        cv2.putText(frame, badge, (14, 8 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, label["text"], (10 + tw + 20, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    def _draw_timestamp(self, frame, seconds: float):
        ts_text = format_timestamp(seconds)
        (tw, th), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = self.w - tw - 15
        y = self.h - 15
        cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, ts_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
