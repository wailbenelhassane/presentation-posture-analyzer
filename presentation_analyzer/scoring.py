"""
Scoring engine for presentation body language analysis.

Converts raw gesture events into a structured score report.
Starts at 100 points per person and deducts based on gesture severity,
duration, and frequency.

Multi-person: if N people are tracked, max score = 100 × N.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .gestures import GestureEvent, Severity


@dataclass
class GestureSummary:
    """Aggregated stats about one gesture type across the video."""
    name: str
    total_occurrences: int
    total_frames: int
    total_seconds: float
    points_deducted: float
    peak_confidence: float
    first_seen_sec: float
    last_seen_sec: float
    description: str
    person_id: int = -1
    timestamps: list[float] = field(default_factory=list)


@dataclass
class ScoreReport:
    """Final analysis report."""
    final_score: float
    max_score: float
    grade: str
    num_persons: int
    gesture_summaries: list[GestureSummary]
    total_frames_analyzed: int
    video_duration_sec: float
    penalty_breakdown: dict[str, float]
    timeline: list[dict]
    recommendations: list[str]


class ScoringEngine:
    """
    Accumulates GestureEvents and produces a ScoreReport.

    v1.1 — More lenient scoring:
      - Penalty weights reduced (~60% of v1.0)
      - Duration multiplier uses sqrt (grows slower than log2)
      - Higher caps per gesture type
      - Confidence threshold: events with < 0.4 confidence are ignored

    Multi-person:
      - max_score = 100 × num_persons
      - Each person's gestures are penalised independently
      - Grade is based on percentage: final_score / max_score
    """

    MAX_SCORE_PER_PERSON = 100.0
    CAP_PER_GESTURE = 20.0
    CAP_OFFENSIVE = 40.0
    CONFIDENCE_THRESHOLD = 0.4
    DEDUP_FRAMES = 10

    PENALTY_WEIGHT = {
        Severity.LOW:      0.4,
        Severity.MEDIUM:   1.2,
        Severity.HIGH:     2.5,
        Severity.CRITICAL: 10.0,
    }

    GRADE_THRESHOLDS = [
        (95, "A+"), (90, "A"), (85, "A-"),
        (80, "B+"), (75, "B"), (70, "B-"),
        (65, "C+"), (60, "C"), (55, "C-"),
        (50, "D+"), (45, "D"), (40, "D-"),
        (0,  "F"),
    ]

    RECOMMENDATIONS = {
        "crossed_arms":     "Practica mantener los brazos relajados a los lados o usa gestos abiertos con las palmas visibles.",
        "hands_in_pockets": "Mantén las manos visibles. Usa un clicker o bolígrafo si necesitas ocupar las manos.",
        "face_touch":       "Evita tocarte la cara. Si notas el impulso, redirige la mano hacia un gesto con propósito.",
        "hands_behind_back":"Las manos deben estar visibles para el público. Úsalas para enfatizar tus puntos.",
        "slouch":           "Mantén los hombros hacia atrás y el pecho abierto. Una buena postura proyecta confianza.",
        "fidgeting":        "Practica la 'quietud con intención' — muévete solo cuando tenga propósito comunicativo.",
        "static_arms":      "Incorpora gestos naturales. Señalar, enumerar con los dedos y gestos abiertos mejoran la comunicación.",
        "offensive_gesture":"Se detectaron gestos ofensivos. Esto es completamente inaceptable en cualquier presentación.",
    }

    def __init__(self):
        self._events: list[tuple[GestureEvent, int]] = []  # (event, person_id)
        self._last_frame: dict[str, int] = {}              # "gesture:person_id" → last frame
        self._total_frames = 0
        self._fps = 30.0
        self._num_persons = 1

    def set_fps(self, fps: float):
        self._fps = fps

    def set_num_persons(self, n: int):
        self._num_persons = max(1, n)

    def add_events(self, events: list[GestureEvent], frame_idx: int, person_id: int = 0):
        """Add events from a single frame, with deduplication."""
        self._total_frames = max(self._total_frames, frame_idx + 1)
        for ev in events:
            if ev.confidence < self.CONFIDENCE_THRESHOLD:
                continue
            key = f"{ev.name}:{person_id}"
            if frame_idx - self._last_frame.get(key, -999) >= self.DEDUP_FRAMES:
                self._events.append((ev, person_id))
                self._last_frame[key] = frame_idx

    def compute_report(self) -> ScoreReport:
        """Generate the final score report from all accumulated events."""
        max_score = self.MAX_SCORE_PER_PERSON * self._num_persons
        grouped = self._group_by_gesture()
        summaries, penalties, timeline, total_deducted = self._build_summaries(grouped)

        final = max(0.0, max_score - total_deducted)
        pct = (final / max_score * 100) if max_score > 0 else 100.0
        duration = self._total_frames / self._fps if self._fps > 0 else 0.0

        return ScoreReport(
            final_score=round(final, 1),
            max_score=round(max_score, 1),
            grade=self._compute_grade(pct),
            num_persons=self._num_persons,
            gesture_summaries=sorted(summaries, key=lambda s: s.points_deducted, reverse=True),
            total_frames_analyzed=self._total_frames,
            video_duration_sec=round(duration, 1),
            penalty_breakdown=penalties,
            timeline=sorted(timeline, key=lambda t: t["sec"]),
            recommendations=self._build_recommendations(summaries),
        )

    def reset(self):
        self._events.clear()
        self._last_frame.clear()
        self._total_frames = 0
        self._num_persons = 1

    # ── Private helpers ──────────────────────────────────────────────────

    def _group_by_gesture(self) -> dict[str, list[tuple[GestureEvent, int]]]:
        grouped: dict[str, list[tuple[GestureEvent, int]]] = defaultdict(list)
        for ev, person_id in self._events:
            grouped[ev.name].append((ev, person_id))
        return grouped

    def _build_summaries(
        self,
        grouped: dict[str, list[tuple[GestureEvent, int]]],
    ) -> tuple[list[GestureSummary], dict[str, float], list[dict], float]:
        summaries: list[GestureSummary] = []
        penalties: dict[str, float] = {}
        timeline: list[dict] = []
        total_deducted = 0.0

        for name, entries in grouped.items():
            cap = (self.CAP_OFFENSIVE if name == "offensive_gesture" else self.CAP_PER_GESTURE)
            cap *= self._num_persons
            deducted = 0.0

            for ev, person_id in entries:
                penalty = self._compute_penalty(ev)
                deducted += penalty
                timeline.append({
                    "sec":       round(ev.timestamp_sec, 1),
                    "gesture":   name,
                    "severity":  ev.severity.name,
                    "penalty":   round(penalty, 2),
                    "person_id": person_id,
                })

            deducted = min(deducted, cap)
            penalties[name] = round(deducted, 2)
            total_deducted += deducted

            evts = [ev for ev, _ in entries]
            total_frames = sum(e.sustained_frames for e in evts)
            summaries.append(GestureSummary(
                name=name,
                total_occurrences=len(evts),
                total_frames=total_frames,
                total_seconds=round(total_frames / self._fps, 1),
                points_deducted=round(deducted, 2),
                peak_confidence=max(e.confidence for e in evts),
                first_seen_sec=round(min(e.timestamp_sec for e in evts), 1),
                last_seen_sec=round(max(e.timestamp_sec for e in evts), 1),
                description=evts[0].description,
                timestamps=[round(e.timestamp_sec, 1) for e in evts],
            ))

        return summaries, penalties, timeline, total_deducted

    def _compute_penalty(self, ev: GestureEvent) -> float:
        sustained_sec = ev.sustained_frames / self._fps
        weight = self.PENALTY_WEIGHT.get(ev.severity, 1.0)
        return weight * ev.confidence * max(1.0, math.sqrt(1 + sustained_sec))

    def _compute_grade(self, pct: float) -> str:
        return next(g for threshold, g in self.GRADE_THRESHOLDS if pct >= threshold)

    def _build_recommendations(self, summaries: list[GestureSummary]) -> list[str]:
        return [
            self.RECOMMENDATIONS[s.name]
            for s in summaries
            if s.name in self.RECOMMENDATIONS and s.points_deducted > 2
        ]
