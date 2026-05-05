"""
Ejemplo de integración con una aplicación existente.

Muestra 3 formas de usar el módulo:
  1. Análisis completo de video (una línea)
  2. Frame-by-frame para streaming/real-time
  3. API REST con FastAPI (ver presentation_analyzer/api.py)
"""

import cv2

from presentation_analyzer import PresentationAnalyzer
from presentation_analyzer.analyzer import report_to_dict


# ═══════════════════════════════════════════════════════════════════════════
# 1. ANÁLISIS SIMPLE — una línea
# ═══════════════════════════════════════════════════════════════════════════

def analizar_video_simple(ruta_video: str) -> dict:
    """Analiza un video y devuelve el informe como dict."""
    with PresentationAnalyzer() as analyzer:
        report = analyzer.analyze_video(ruta_video)
    return report_to_dict(report)


# ═══════════════════════════════════════════════════════════════════════════
# 2. FRAME-BY-FRAME — para integrar con streaming o procesamiento propio
# ═══════════════════════════════════════════════════════════════════════════

def analizar_frame_a_frame(ruta_video: str) -> dict:
    """Procesamiento frame a frame con control total."""
    analyzer = PresentationAnalyzer(
        use_hands=True,
        model_complexity=0,  # lite para velocidad
        process_every_n=1,   # nosotros controlamos qué frames enviar
    )

    cap = cv2.VideoCapture(ruta_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    analyzer.begin_session(fps=fps)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 3 == 0:
            frame_data = analyzer.process_frame(frame, frame_idx)
            for _, person_events, _ in frame_data:
                for ev in person_events:
                    if ev.severity.value >= 10:
                        print(f"[{ev.timestamp_sec:.1f}s] {ev.description}")

        frame_idx += 1

    cap.release()
    report = analyzer.end_session()
    analyzer.close()
    return report_to_dict(report)


# ═══════════════════════════════════════════════════════════════════════════
# 3. API REST — ver presentation_analyzer/api.py
#
# Arrancar:
#     python run_api.py
#
# Endpoints disponibles:
#     GET  /health
#     POST /analyze            análisis síncrono
#     POST /analyze/async      lanza job en background
#     GET  /jobs/{job_id}      consulta estado del job
#     DELETE /jobs/{job_id}    limpia el job
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO DIRECTO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python integration_example.py <video.mp4>")
        sys.exit(1)

    resultado = analizar_video_simple(sys.argv[1])

    print(f"\nPuntuación: {resultado['final_score']}/100 ({resultado['grade']})")
    print(f"Duración: {resultado['video_duration_sec']}s")

    if resultado["gesture_summaries"]:
        print("\nProblemas detectados:")
        for g in resultado["gesture_summaries"]:
            print(f"  - {g['name']}: -{g['points_deducted']} pts ({g['description']})")

    if resultado["recommendations"]:
        print("\nRecomendaciones:")
        for i, r in enumerate(resultado["recommendations"], 1):
            print(f"  {i}. {r}")
