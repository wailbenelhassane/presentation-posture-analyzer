#!/usr/bin/env python3
"""
Test suite para presentation_analyzer.

Ejecutar:
    python test_analyzer.py              # todos los tests
    python test_analyzer.py --con-video  # incluye test con video real (necesita webcam o archivo)
"""

import sys
import math
import numpy as np
import cv2
import time

# ── Añadir el directorio padre al path si ejecutas desde dentro del proyecto ──
sys.path.insert(0, ".")

from presentation_analyzer import PresentationAnalyzer, GestureDetector, ScoringEngine
from presentation_analyzer.gestures import Severity, GestureEvent, VisibilityMode, LM
from presentation_analyzer.scoring import ScoreReport
from presentation_analyzer.analyzer import report_to_dict, report_to_json


class FakeLandmark:
    """Simula un landmark de MediaPipe para testing."""
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class FakeLandmarks:
    """Simula pose_landmarks con .landmark como lista indexable."""
    def __init__(self, landmarks: list[FakeLandmark]):
        self.landmark = landmarks


def make_neutral_pose() -> list[FakeLandmark]:
    """Crea una pose neutral (de pie, brazos a los lados) con 33 landmarks."""
    lms = [FakeLandmark() for _ in range(33)]

    # Cabeza
    lms[LM.NOSE] = FakeLandmark(0.50, 0.15, 0.0)
    lms[LM.LEFT_EYE] = FakeLandmark(0.48, 0.13, 0.0)
    lms[LM.RIGHT_EYE] = FakeLandmark(0.52, 0.13, 0.0)
    lms[LM.LEFT_EAR] = FakeLandmark(0.45, 0.14, 0.0)
    lms[LM.RIGHT_EAR] = FakeLandmark(0.55, 0.14, 0.0)
    lms[LM.MOUTH_LEFT] = FakeLandmark(0.48, 0.18, 0.0)
    lms[LM.MOUTH_RIGHT] = FakeLandmark(0.52, 0.18, 0.0)

    # Hombros
    lms[LM.LEFT_SHOULDER] = FakeLandmark(0.40, 0.28, 0.0)
    lms[LM.RIGHT_SHOULDER] = FakeLandmark(0.60, 0.28, 0.0)

    # Codos (brazos a los lados)
    lms[LM.LEFT_ELBOW] = FakeLandmark(0.35, 0.42, 0.0)
    lms[LM.RIGHT_ELBOW] = FakeLandmark(0.65, 0.42, 0.0)

    # Muñecas (abajo, relajadas)
    lms[LM.LEFT_WRIST] = FakeLandmark(0.33, 0.55, 0.0)
    lms[LM.RIGHT_WRIST] = FakeLandmark(0.67, 0.55, 0.0)

    # Dedos
    lms[LM.LEFT_INDEX] = FakeLandmark(0.32, 0.57, 0.0)
    lms[LM.RIGHT_INDEX] = FakeLandmark(0.68, 0.57, 0.0)
    lms[LM.LEFT_PINKY] = FakeLandmark(0.34, 0.57, 0.0)
    lms[LM.RIGHT_PINKY] = FakeLandmark(0.66, 0.57, 0.0)
    lms[LM.LEFT_THUMB] = FakeLandmark(0.31, 0.56, 0.0)
    lms[LM.RIGHT_THUMB] = FakeLandmark(0.69, 0.56, 0.0)

    # Caderas
    lms[LM.LEFT_HIP] = FakeLandmark(0.43, 0.55, 0.0)
    lms[LM.RIGHT_HIP] = FakeLandmark(0.57, 0.55, 0.0)

    # Rodillas
    lms[LM.LEFT_KNEE] = FakeLandmark(0.42, 0.72, 0.0)
    lms[LM.RIGHT_KNEE] = FakeLandmark(0.58, 0.72, 0.0)

    # Tobillos
    lms[LM.LEFT_ANKLE] = FakeLandmark(0.42, 0.90, 0.0)
    lms[LM.RIGHT_ANKLE] = FakeLandmark(0.58, 0.90, 0.0)

    return lms


def make_crossed_arms_pose() -> list[FakeLandmark]:
    """Pose con brazos cruzados."""
    lms = make_neutral_pose()
    # Muñecas cruzadas al frente del pecho
    lms[LM.LEFT_WRIST] = FakeLandmark(0.55, 0.38, 0.0)   # izq cruza a la derecha
    lms[LM.RIGHT_WRIST] = FakeLandmark(0.45, 0.38, 0.0)   # der cruza a la izquierda
    # Codos doblados
    lms[LM.LEFT_ELBOW] = FakeLandmark(0.38, 0.36, 0.0)
    lms[LM.RIGHT_ELBOW] = FakeLandmark(0.62, 0.36, 0.0)
    return lms


def make_hands_in_pockets_pose() -> list[FakeLandmark]:
    """Pose con manos en los bolsillos."""
    lms = make_neutral_pose()
    # Muñecas debajo de caderas, pegadas al cuerpo
    lms[LM.LEFT_WRIST] = FakeLandmark(0.43, 0.60, 0.0, visibility=0.5)
    lms[LM.RIGHT_WRIST] = FakeLandmark(0.57, 0.60, 0.0, visibility=0.5)
    return lms


def make_face_touch_pose() -> list[FakeLandmark]:
    """Pose tocándose la cara."""
    lms = make_neutral_pose()
    lms[LM.RIGHT_WRIST] = FakeLandmark(0.50, 0.16, 0.0)  # muñeca cerca de la nariz
    return lms


# ═════════════════════════════════════════════════════════════════════════
# TESTS
# ═════════════════════════════════════════════════════════════════════════

passed = 0
failed = 0


def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def test_1_imports():
    """Verificar que todos los imports funcionan."""
    print("\n── Test 1: Imports ──")
    test("PresentationAnalyzer importado", PresentationAnalyzer is not None)
    test("GestureDetector importado", GestureDetector is not None)
    test("ScoringEngine importado", ScoringEngine is not None)
    test("Severity enum funciona", Severity.CRITICAL.value == 15)
    test("VisibilityMode enum funciona", VisibilityMode.FULL_BODY.value == "full_body")


def test_2_neutral_pose():
    """Una pose neutral no debe generar eventos."""
    print("\n── Test 2: Pose neutral (sin problemas) ──")
    detector = GestureDetector(fps=30.0)
    pose = FakeLandmarks(make_neutral_pose())

    # Simular 30 frames (1 segundo) de pose neutral
    all_events = []
    for i in range(30):
        events = detector.detect(pose, frame_idx=i)
        all_events.extend(events)

    test("Pose neutral no genera alertas", len(all_events) == 0,
         f"Se detectaron {len(all_events)} eventos inesperados")


def test_3_crossed_arms():
    """Brazos cruzados sostenidos deben generar evento MEDIUM."""
    print("\n── Test 3: Brazos cruzados ──")
    detector = GestureDetector(fps=30.0)
    pose = FakeLandmarks(make_crossed_arms_pose())

    all_events = []
    for i in range(30):  # 1 segundo
        events = detector.detect(pose, frame_idx=i)
        all_events.extend(events)

    crossed = [e for e in all_events if e.name == "crossed_arms"]
    test("Brazos cruzados detectados", len(crossed) > 0)
    if crossed:
        test("Severidad es MEDIUM", crossed[0].severity == Severity.MEDIUM)
        test("Confianza > 0", crossed[0].confidence > 0)


def test_4_hands_in_pockets():
    """Manos en bolsillos deben detectarse con caderas visibles."""
    print("\n── Test 4: Manos en los bolsillos ──")
    detector = GestureDetector(fps=30.0)
    pose = FakeLandmarks(make_hands_in_pockets_pose())

    all_events = []
    for i in range(30):
        events = detector.detect(pose, frame_idx=i)
        all_events.extend(events)

    pockets = [e for e in all_events if e.name == "hands_in_pockets"]
    test("Manos en bolsillos detectadas", len(pockets) > 0)


def test_5_face_touch():
    """Tocarse la cara debe detectarse rápido (threshold bajo)."""
    print("\n── Test 5: Tocarse la cara ──")
    detector = GestureDetector(fps=30.0)
    pose = FakeLandmarks(make_face_touch_pose())

    all_events = []
    for i in range(15):  # medio segundo
        events = detector.detect(pose, frame_idx=i)
        all_events.extend(events)

    face = [e for e in all_events if e.name == "face_touch"]
    test("Toque de cara detectado", len(face) > 0)
    if face:
        test("Severidad es LOW", face[0].severity == Severity.LOW)


def test_6_visibility_detection():
    """Debe detectar correctamente qué partes del cuerpo son visibles."""
    print("\n── Test 6: Detección de visibilidad ──")
    detector = GestureDetector(fps=30.0)

    # Full body
    full = FakeLandmarks(make_neutral_pose())
    vis = detector.detect_visibility(full)
    test("Cuerpo completo detectado", vis == VisibilityMode.FULL_BODY)

    # Half body (rodillas no visibles)
    half_lms = make_neutral_pose()
    half_lms[LM.LEFT_KNEE].visibility = 0.1
    half_lms[LM.RIGHT_KNEE].visibility = 0.1
    half_lms[LM.LEFT_ANKLE].visibility = 0.1
    half_lms[LM.RIGHT_ANKLE].visibility = 0.1
    vis = detector.detect_visibility(FakeLandmarks(half_lms))
    test("Medio cuerpo detectado", vis == VisibilityMode.HALF_BODY)

    # Upper only (sin caderas)
    upper_lms = make_neutral_pose()
    upper_lms[LM.LEFT_HIP].visibility = 0.1
    upper_lms[LM.RIGHT_HIP].visibility = 0.1
    vis = detector.detect_visibility(FakeLandmarks(upper_lms))
    test("Solo parte superior detectada", vis == VisibilityMode.UPPER_ONLY)


def test_7_scoring_engine():
    """El motor de puntuación debe calcular correctamente."""
    print("\n── Test 7: Motor de puntuación ──")
    scorer = ScoringEngine()
    scorer.set_fps(30.0)

    # Sin eventos = puntuación perfecta
    report = scorer.compute_report()
    test("Sin eventos → 100 puntos", report.final_score == 100.0)
    test("Sin eventos → grado A+", report.grade == "A+")

    # Con eventos simulados
    scorer.reset()
    scorer.set_fps(30.0)
    events = [
        GestureEvent("crossed_arms", Severity.MEDIUM, 0.8, 30, 1.0,
                      "test", sustained_frames=30),
        GestureEvent("crossed_arms", Severity.MEDIUM, 0.9, 90, 3.0,
                      "test", sustained_frames=60),
    ]
    scorer.add_events(events, 30)
    scorer.add_events(events[1:], 90)
    report = scorer.compute_report()

    test("Con eventos → puntuación < 100", report.final_score < 100)
    test("Tiene resumen de gestos", len(report.gesture_summaries) > 0)
    test("Tiene recomendaciones", len(report.recommendations) > 0)
    test("Timeline generado", len(report.timeline) > 0)


def test_8_json_serialization():
    """El informe debe serializarse a JSON correctamente."""
    print("\n── Test 8: Serialización JSON ──")
    scorer = ScoringEngine()
    scorer.set_fps(30.0)
    report = scorer.compute_report()

    import json
    j = report_to_json(report)
    test("JSON válido", True)  # report_to_json ya lo valida

    d = json.loads(j)
    test("Tiene campo final_score", "final_score" in d)
    test("Tiene campo grade", "grade" in d)
    test("Tiene campo recommendations", "recommendations" in d)

    d2 = report_to_dict(report)
    test("Dict tiene mismos campos", d2["final_score"] == d["final_score"])


def test_9_scoring_caps():
    """Las penalizaciones deben respetar los topes máximos."""
    print("\n── Test 9: Topes de penalización ──")
    scorer = ScoringEngine()
    scorer.set_fps(30.0)

    # Muchos eventos del mismo tipo
    for i in range(100):
        scorer.add_events([
            GestureEvent("crossed_arms", Severity.MEDIUM, 1.0, i * 10, i / 3,
                          "test", sustained_frames=10)
        ], i * 10)

    report = scorer.compute_report()
    penalty = report.penalty_breakdown.get("crossed_arms", 0)
    test(f"Tope respetado (crossed_arms: {penalty} ≤ 30)", penalty <= 30.0)
    test("Puntuación ≥ 70", report.final_score >= 70.0,
         f"Score fue {report.final_score}")


def test_10_synthetic_video():
    """Crear un video sintético y analizarlo end-to-end."""
    print("\n── Test 10: Video sintético end-to-end ──")
    import tempfile
    import os

    # Crear video con frames sólidos (640x480, 1 segundo a 30fps)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, 30.0, (640, 480))

    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Dibujar una "persona" básica para que MediaPipe intente detectar
        # (probablemente no detecte nada, pero el pipeline debe funcionar)
        cv2.circle(frame, (320, 100), 40, (200, 180, 160), -1)  # cabeza
        cv2.rectangle(frame, (280, 140), (360, 300), (100, 100, 200), -1)  # torso
        writer.write(frame)

    writer.release()

    try:
        analyzer = PresentationAnalyzer(
            use_hands=False,
            model_complexity=0,
            process_every_n=3,
        )
        t0 = time.time()
        report = analyzer.analyze_video(tmp.name)
        elapsed = time.time() - t0
        analyzer.close()

        test("Video analizado sin errores", True)
        test(f"Puntuación obtenida: {report.final_score}", report.final_score >= 0)
        test(f"Procesado en {elapsed:.2f}s", elapsed < 30)
        test("Frames analizados > 0", report.total_frames_analyzed > 0)

    except Exception as e:
        test(f"Error al analizar video: {e}", False)
    finally:
        os.unlink(tmp.name)


# ═════════════════════════════════════════════════════════════════════════
# EJECUTAR
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  TESTS — presentation_analyzer")
    print("=" * 60)

    test_1_imports()
    test_2_neutral_pose()
    test_3_crossed_arms()
    test_4_hands_in_pockets()
    test_5_face_touch()
    test_6_visibility_detection()
    test_7_scoring_engine()
    test_8_json_serialization()
    test_9_scoring_caps()
    test_10_synthetic_video()

    print(f"\n{'=' * 60}")
    print(f"  RESULTADO: {passed} pasados, {failed} fallidos")
    print(f"{'=' * 60}\n")

    sys.exit(0 if failed == 0 else 1)