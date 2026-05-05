"""
Main video analyzer — MediaPipe Tasks API (0.10.x+) with:

  - Multi-person tracking (up to 4 people)
  - Landmark smoothing (EMA filter per person)
  - Per-person gesture detection
  - Scaled scoring (100 pts per person)
  - Optional annotated video output
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import json
import urllib.request
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Callable

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from .gestures import GestureDetector
from .scoring import ScoringEngine, ScoreReport
from .renderer import VideoRenderer, format_timestamp
from .smoothing import PersonTracker, SmoothedLandmarkList


# ── Model download ──────────────────────────────────────────────────────────
_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
_POSE_MODELS = {
    0: f"{_MODEL_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    1: f"{_MODEL_BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    2: f"{_MODEL_BASE}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}
_HAND_MODEL = f"{_MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
_MODELS_DIR = Path.home() / ".cache" / "presentation_analyzer" / "models"

_POSE_COMPLEXITY_NAMES = {0: "lite", 1: "full", 2: "heavy"}
_PROGRESS_INTERVAL = 30  # frames between progress_callback calls


def _download_model(url: str, name: str) -> str:
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _MODELS_DIR / name
    if local_path.exists():
        return str(local_path)
    print(f"  Descargando modelo {name} (solo la primera vez)...")
    urllib.request.urlretrieve(url, local_path)
    print(f"  Modelo guardado en {local_path}")
    return str(local_path)


class _LandmarkListAdapter:
    """Wraps list[NormalizedLandmark] to provide the .landmark[i] interface."""
    def __init__(self, landmarks: list):
        self.landmark = landmarks


class PresentationAnalyzer:
    """
    Multi-person presentation body-language analyzer with smoothing.

    Usage:
        with PresentationAnalyzer() as analyzer:
            report = analyzer.analyze_video("input.mp4", output_video="result.mp4")
        print(f"{report.final_score}/{report.max_score} ({report.grade})")
    """

    def __init__(
        self,
        *,
        use_hands: bool = True,
        model_complexity: int = 1,
        max_persons: int = 4,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        process_every_n: int = 2,
    ):
        self._use_hands = use_hands
        self._model_complexity = model_complexity
        self._max_persons = min(max_persons, 4)
        self._min_det = min_detection_confidence
        self._min_track = min_tracking_confidence
        self._process_every_n = max(1, process_every_n)

        self._pose: Optional[PoseLandmarker] = None
        self._hands: Optional[HandLandmarker] = None

        self._tracker = PersonTracker()
        self._detectors: dict[int, GestureDetector] = {}
        self._scorer = ScoringEngine()
        self._session_active = False
        self._fps = 30.0
        self._last_frame_data: list = []
        self._peak_persons = 1

    # ── Full-video API ───────────────────────────────────────────────────

    def analyze_video(
        self,
        video_path: str,
        *,
        output_video: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        max_duration_sec: Optional[float] = None,
    ) -> ScoreReport:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(max_duration_sec * fps) if max_duration_sec else total_frames

        renderer = None
        if output_video:
            renderer = VideoRenderer(width, height, fps)
            renderer.open(output_video)

        self.begin_session(fps=fps)
        frame_idx = 0

        try:
            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self._process_every_n == 0:
                    self._last_frame_data = self.process_frame(frame, frame_idx)
                    if renderer:
                        renderer.draw_frame_multi(frame, self._last_frame_data, frame_idx)
                elif renderer:
                    renderer.draw_frame_multi(
                        frame, self._last_frame_data, frame_idx, events_override=[]
                    )

                frame_idx += 1
                if progress_callback and frame_idx % _PROGRESS_INTERVAL == 0:
                    progress_callback(min(1.0, frame_idx / max_frames))
        finally:
            cap.release()
            if renderer:
                renderer.close()

        return self.end_session()

    # ── Frame-by-frame API ───────────────────────────────────────────────

    def begin_session(self, fps: float = 30.0):
        self._ensure_models()
        self._fps = fps
        self._tracker = PersonTracker()
        self._detectors = {}
        self._scorer = ScoringEngine()
        self._scorer.set_fps(fps)
        self._session_active = True
        self._last_frame_data = []
        self._peak_persons = 1

    def process_frame(self, frame, frame_idx: int) -> list:
        """
        Process a single BGR frame.

        Returns list of (person_id, events, smoothed_landmarks) tuples.
        """
        if not self._session_active:
            raise RuntimeError("Call begin_session() first")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * 1000 / self._fps)

        pose_result = self._pose.detect_for_video(mp_image, timestamp_ms)
        if not pose_result.pose_landmarks:
            return []

        assignments = self._tracker.match(pose_result.pose_landmarks)
        if len(assignments) > self._peak_persons:
            self._peak_persons = len(assignments)
            self._scorer.set_num_persons(self._peak_persons)

        hand_results = self._detect_hands(mp_image, timestamp_ms)

        frame_data = []
        for person_id, raw_landmarks in assignments:
            smoother = self._tracker.get_smoother(person_id)
            if person_id not in self._detectors:
                self._detectors[person_id] = GestureDetector(fps=self._fps)

            smoothed = SmoothedLandmarkList(smoother.smooth(raw_landmarks))
            events = self._detectors[person_id].detect(
                smoothed,
                hand_landmarks_list=hand_results,
                frame_idx=frame_idx,
            )
            self._scorer.add_events(events, frame_idx, person_id=person_id)
            frame_data.append((person_id, events, smoothed))

        return frame_data

    def end_session(self) -> ScoreReport:
        self._session_active = False
        self._scorer.set_num_persons(self._peak_persons)
        return self._scorer.compute_report()

    def close(self):
        if self._pose:
            self._pose.close()
            self._pose = None
        if self._hands:
            self._hands.close()
            self._hands = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Model initialisation ─────────────────────────────────────────────

    def _ensure_models(self):
        if self._pose is None:
            self._pose = self._create_pose_model()
        if self._use_hands and self._hands is None:
            self._hands = self._create_hand_model()

    def _create_pose_model(self) -> PoseLandmarker:
        model_file = f"pose_landmarker_{_POSE_COMPLEXITY_NAMES[self._model_complexity]}.task"
        model_path = _download_model(_POSE_MODELS[self._model_complexity], model_file)
        return PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                min_pose_detection_confidence=self._min_det,
                min_tracking_confidence=self._min_track,
                num_poses=self._max_persons,
            )
        )

    def _create_hand_model(self) -> HandLandmarker:
        hand_path = _download_model(_HAND_MODEL, "hand_landmarker.task")
        return HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_path),
                running_mode=RunningMode.VIDEO,
                min_hand_detection_confidence=self._min_det,
                min_tracking_confidence=self._min_track,
                num_hands=2,
            )
        )

    def _detect_hands(self, mp_image, timestamp_ms: int) -> Optional[list]:
        if not (self._use_hands and self._hands):
            return None
        result = self._hands.detect_for_video(mp_image, timestamp_ms)
        if result.hand_landmarks:
            return [_LandmarkListAdapter(hl) for hl in result.hand_landmarks]
        return None


# ── Convenience helpers ──────────────────────────────────────────────────────

def report_to_dict(report: ScoreReport) -> dict:
    d = asdict(report)
    for gs in d.get("gesture_summaries", []):
        gs["first_seen_fmt"] = format_timestamp(gs["first_seen_sec"])
        gs["last_seen_fmt"] = format_timestamp(gs["last_seen_sec"])
        gs["timestamps_fmt"] = [format_timestamp(t) for t in gs.get("timestamps", [])]
    for item in d.get("timeline", []):
        item["time_fmt"] = format_timestamp(item["sec"])
    return d


def report_to_json(report: ScoreReport, indent: int = 2) -> str:
    return json.dumps(report_to_dict(report), indent=indent, ensure_ascii=False)


def print_report(report: ScoreReport):
    persons_str = f" ({report.num_persons} personas)" if report.num_persons > 1 else ""
    print(f"\n{'='*60}")
    print(f"  PUNTUACIÓN: {report.final_score}/{report.max_score}  ({report.grade}){persons_str}")
    print(f"  Duración: {format_timestamp(report.video_duration_sec)} | "
          f"Frames: {report.total_frames_analyzed}")
    print(f"{'='*60}\n")

    if report.gesture_summaries:
        print("  GESTOS DETECTADOS:")
        print(f"  {'-'*56}")
        for gs in report.gesture_summaries:
            first = format_timestamp(gs.first_seen_sec)
            last = format_timestamp(gs.last_seen_sec)
            times = first if gs.first_seen_sec == gs.last_seen_sec else f"{first} → {last}"
            print(f"  ! {gs.name:<22} -{gs.points_deducted:>5.1f} pts  "
                  f"({gs.total_occurrences}x, {gs.total_seconds}s)")
            print(f"    {gs.description}")
            print(f"    Momento: {times}")
        print()

    if report.timeline:
        print("  LÍNEA TEMPORAL DE ERRORES:")
        print(f"  {'-'*56}")
        for item in report.timeline[:20]:
            t = format_timestamp(item["sec"])
            pid = item.get("person_id", 0)
            pid_str = f" P{pid+1}" if report.num_persons > 1 else ""
            print(f"    {t}{pid_str}  [{item['severity']:<8}]  {item['gesture']}"
                  f"  (-{item['penalty']} pts)")
        if len(report.timeline) > 20:
            print(f"    ... y {len(report.timeline) - 20} eventos más")
        print()

    if report.recommendations:
        print("  RECOMENDACIONES:")
        print(f"  {'-'*56}")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        print()

    print(f"{'='*60}\n")
