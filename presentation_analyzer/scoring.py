"""
Scoring engine for presentation body language analysis.

Converts raw gesture events into a structured score report.
Starts at 100 points and deducts based on gesture severity,
duration, and frequency.
"""

from dataclasses import dataclass, field
from collections import defaultdict
from .gestures import GestureEvent, Severity


@dataclass
class GestureSummary:
    """Aggregated info about one gesture type across the video."""
    name: str
    total_occurrences: int
    total_frames: int
    total_seconds: float
    points_deducted: float
    peak_confidence: float
    first_seen_sec: float
    last_seen_sec: float
    description: str
    timestamps: list[float] = field(default_factory=list)


@dataclass
class ScoreReport:
    """Final analysis report."""
    final_score: float              # 0 – 100
    max_score: float                # always 100
    grade: str                      # A+ to F
    gesture_summaries: list[GestureSummary]
    total_frames_analyzed: int
    video_duration_sec: float
    penalty_breakdown: dict[str, float]
    timeline: list[dict]            # [{sec, gesture, severity}, ...]
    recommendations: list[str]


class ScoringEngine:
    """
    Accumulates GestureEvents across frames and produces a ScoreReport.

    Deduction formula per gesture occurrence:
        base_penalty = severity_value × confidence
        duration_multiplier = log2(1 + sustained_seconds) for sustained gestures
        final_penalty = base_penalty × duration_multiplier

    Caps:
        - Single gesture type capped at 30 points total deduction
        - Offensive gestures capped at 50 points
        - Total never below 0
    """

    MAX_SCORE = 100.0
    CAP_PER_GESTURE = 30.0
    CAP_OFFENSIVE = 50.0

    # Deduplication: ignore repeat events for same gesture within this window
    DEDUP_FRAMES = 5

    GRADE_THRESHOLDS = [
        (95, "A+"), (90, "A"), (85, "A-"),
        (80, "B+"), (75, "B"), (70, "B-"),
        (65, "C+"), (60, "C"), (55, "C-"),
        (50, "D+"), (45, "D"), (40, "D-"),
        (0, "F"),
    ]

    RECOMMENDATIONS = {
        "crossed_arms": "Practica mantener los brazos relajados a los lados o usa gestos abiertos con las palmas visibles.",
        "hands_in_pockets": "Mantén las manos visibles. Usa un clicker o bolígrafo si necesitas ocupar las manos.",
        "face_touch": "Evita tocarte la cara. Si notas el impulso, redirige la mano hacia un gesto con propósito.",
        "hands_behind_back": "Las manos deben estar visibles para el público. Úsalas para enfatizar tus puntos.",
        "slouch": "Mantén los hombros hacia atrás y el pecho abierto. Una buena postura proyecta confianza.",
        "fidgeting": "Practica la 'quietud con intención' — muévete solo cuando tenga propósito comunicativo.",
        "static_arms": "Incorpora gestos naturales. Señalar, enumerar con los dedos y gestos abiertos mejoran la comunicación.",
        "offensive_gesture": "Se detectaron gestos ofensivos. Esto es completamente inaceptable en cualquier presentación.",
    }

    def __init__(self):
        self._events: list[GestureEvent] = []
        self._last_frame: dict[str, int] = {}  # gesture -> last reported frame
        self._total_frames = 0
        self._fps = 30.0

    def set_fps(self, fps: float):
        self._fps = fps

    def add_events(self, events: list[GestureEvent], frame_idx: int):
        """Add events from a single frame, with deduplication."""
        self._total_frames = max(self._total_frames, frame_idx + 1)
        for ev in events:
            last = self._last_frame.get(ev.name, -999)
            if frame_idx - last >= self.DEDUP_FRAMES:
                self._events.append(ev)
                self._last_frame[ev.name] = frame_idx

    def compute_report(self) -> ScoreReport:
        """Generate the final score report."""
        import math

        # Group events by gesture name
        grouped: dict[str, list[GestureEvent]] = defaultdict(list)
        for ev in self._events:
            grouped[ev.name].append(ev)

        summaries: list[GestureSummary] = []
        penalties: dict[str, float] = {}
        timeline: list[dict] = []
        total_deducted = 0.0

        for name, evts in grouped.items():
            cap = self.CAP_OFFENSIVE if name == "offensive_gesture" else self.CAP_PER_GESTURE
            deducted = 0.0

            for ev in evts:
                sustained_sec = ev.sustained_frames / self._fps
                base = ev.severity.value * ev.confidence
                duration_mult = math.log2(1 + sustained_sec)
                penalty = base * max(1.0, duration_mult)
                deducted += penalty

                timeline.append({
                    "sec": round(ev.timestamp_sec, 1),
                    "gesture": name,
                    "severity": ev.severity.name,
                    "penalty": round(penalty, 2),
                })

            deducted = min(deducted, cap)
            penalties[name] = round(deducted, 2)
            total_deducted += deducted

            total_sustained = sum(e.sustained_frames for e in evts)
            summaries.append(GestureSummary(
                name=name,
                total_occurrences=len(evts),
                total_frames=total_sustained,
                total_seconds=round(total_sustained / self._fps, 1),
                points_deducted=round(deducted, 2),
                peak_confidence=max(e.confidence for e in evts),
                first_seen_sec=round(min(e.timestamp_sec for e in evts), 1),
                last_seen_sec=round(max(e.timestamp_sec for e in evts), 1),
                description=evts[0].description,
                timestamps=[round(e.timestamp_sec, 1) for e in evts],
            ))

        final = max(0.0, self.MAX_SCORE - total_deducted)
        grade = "A+"
        for threshold, g in self.GRADE_THRESHOLDS:
            if final >= threshold:
                grade = g
                break

        # Sort summaries by impact
        summaries.sort(key=lambda s: s.points_deducted, reverse=True)
        timeline.sort(key=lambda t: t["sec"])

        # Build recommendations (only for detected issues)
        recs = [
            self.RECOMMENDATIONS[s.name]
            for s in summaries
            if s.name in self.RECOMMENDATIONS and s.points_deducted > 1
        ]

        duration = self._total_frames / self._fps if self._fps > 0 else 0

        return ScoreReport(
            final_score=round(final, 1),
            max_score=self.MAX_SCORE,
            grade=grade,
            gesture_summaries=summaries,
            total_frames_analyzed=self._total_frames,
            video_duration_sec=round(duration, 1),
            penalty_breakdown=penalties,
            timeline=timeline,
            recommendations=recs,
        )

    def reset(self):
        self._events.clear()
        self._last_frame.clear()
        self._total_frames = 0