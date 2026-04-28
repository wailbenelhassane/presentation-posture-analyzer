"""
Scoring engine for presentation body language analysis.

Converts raw gesture events into a structured score report.
Starts at 100 points per person and deducts based on gesture severity,
duration, and frequency.

Multi-person: if N people are tracked, max score = 100 × N.
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
    person_id: int = -1  # -1 = aggregated across all persons
    timestamps: list[float] = field(default_factory=list)


@dataclass
class ScoreReport:
    """Final analysis report."""
    final_score: float              # 0 – max_score
    max_score: float                # 100 × num_persons
    grade: str                      # A+ to F
    num_persons: int                # how many people were tracked
    gesture_summaries: list[GestureSummary]
    total_frames_analyzed: int
    video_duration_sec: float
    penalty_breakdown: dict[str, float]
    timeline: list[dict]            # [{sec, gesture, severity, person_id}, ...]
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
    CAP_PER_GESTURE = 20.0          # was 30
    CAP_OFFENSIVE = 40.0            # was 50

    CONFIDENCE_THRESHOLD = 0.4      # ignore low-confidence detections

    # Penalty weight per severity (replaces raw enum values)
    PENALTY_WEIGHT = {
        Severity.LOW:      0.4,     # was 1
        Severity.MEDIUM:   1.2,     # was 3
        Severity.HIGH:     2.5,     # was 5
        Severity.CRITICAL: 10.0,    # was 15
    }

    DEDUP_FRAMES = 10               # was 5 — less frequent reporting

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
        self._last_frame: dict[str, int] = {}  # "gesture:person_id" -> last frame
        self._total_frames = 0
        self._fps = 30.0
        self._num_persons = 1

    def set_fps(self, fps: float):
        self._fps = fps

    def set_num_persons(self, n: int):
        """Set the number of persons being tracked (for max_score scaling)."""
        self._num_persons = max(1, n)

    def add_events(self, events: list[GestureEvent], frame_idx: int,
                   person_id: int = 0):
        """Add events from a single frame, with deduplication."""
        self._total_frames = max(self._total_frames, frame_idx + 1)
        for ev in events:
            # Skip low-confidence detections
            if ev.confidence < self.CONFIDENCE_THRESHOLD:
                continue

            key = f"{ev.name}:{person_id}"
            last = self._last_frame.get(key, -999)
            if frame_idx - last >= self.DEDUP_FRAMES:
                ev._person_id = person_id  # tag for report
                self._events.append(ev)
                self._last_frame[key] = frame_idx

    def compute_report(self) -> ScoreReport:
        """Generate the final score report."""
        import math

        max_score = self.MAX_SCORE_PER_PERSON * self._num_persons

        grouped: dict[str, list[GestureEvent]] = defaultdict(list)
        for ev in self._events:
            grouped[ev.name].append(ev)

        summaries: list[GestureSummary] = []
        penalties: dict[str, float] = {}
        timeline: list[dict] = []
        total_deducted = 0.0

        for name, evts in grouped.items():
            cap = self.CAP_OFFENSIVE if name == "offensive_gesture" else self.CAP_PER_GESTURE
            # Scale cap with number of persons
            cap *= self._num_persons
            deducted = 0.0

            for ev in evts:
                sustained_sec = ev.sustained_frames / self._fps
                weight = self.PENALTY_WEIGHT.get(ev.severity, 1.0)
                base = weight * ev.confidence

                # sqrt grows much slower than log2, making duration less punishing
                duration_mult = math.sqrt(1 + sustained_sec)
                penalty = base * max(1.0, duration_mult)
                deducted += penalty

                person_id = getattr(ev, "_person_id", 0)
                timeline.append({
                    "sec": round(ev.timestamp_sec, 1),
                    "gesture": name,
                    "severity": ev.severity.name,
                    "penalty": round(penalty, 2),
                    "person_id": person_id,
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

        final = max(0.0, max_score - total_deducted)

        # Grade based on percentage
        pct = (final / max_score * 100) if max_score > 0 else 100
        grade = "A+"
        for threshold, g in self.GRADE_THRESHOLDS:
            if pct >= threshold:
                grade = g
                break

        summaries.sort(key=lambda s: s.points_deducted, reverse=True)
        timeline.sort(key=lambda t: t["sec"])

        recs = [
            self.RECOMMENDATIONS[s.name]
            for s in summaries
            if s.name in self.RECOMMENDATIONS and s.points_deducted > 2
        ]

        duration = self._total_frames / self._fps if self._fps > 0 else 0

        return ScoreReport(
            final_score=round(final, 1),
            max_score=round(max_score, 1),
            grade=grade,
            num_persons=self._num_persons,
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
        self._num_persons = 1
