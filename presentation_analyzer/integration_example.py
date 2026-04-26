"""
Ejemplo de integración con una aplicación existente.

Muestra 3 formas de usar el módulo:
  1. Análisis completo de video (una línea)
  2. Frame-by-frame para streaming/real-time
  3. API REST con FastAPI (ejemplo mínimo)
"""

# ═══════════════════════════════════════════════════════════════════════════
# 1. ANÁLISIS SIMPLE — una línea
# ═══════════════════════════════════════════════════════════════════════════

from presentation_analyzer import PresentationAnalyzer
from presentation_analyzer.analyzer import report_to_dict, report_to_json

def analizar_video_simple(ruta_video: str) -> dict:
    """Analiza un video y devuelve el informe como dict."""
    with PresentationAnalyzer() as analyzer:
        report = analyzer.analyze_video(ruta_video)
    return report_to_dict(report)


# ═══════════════════════════════════════════════════════════════════════════
# 2. FRAME-BY-FRAME — para integrar con streaming o procesamiento propio
# ═══════════════════════════════════════════════════════════════════════════

import cv2

def analizar_frame_a_frame(ruta_video: str) -> dict:
    """Ejemplo de procesamiento frame a frame con control total."""
    analyzer = PresentationAnalyzer(
        use_hands=True,
        model_complexity=0,      # lite para velocidad
        process_every_n=1,       # nosotros controlamos qué frames enviar
    )

    cap = cv2.VideoCapture(ruta_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    analyzer.begin_session(fps=fps)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar solo 1 de cada 3 frames para ahorrar CPU
        if frame_idx % 3 == 0:
            events = analyzer.process_frame(frame, frame_idx)

            # Reaccionar en tiempo real a eventos críticos
            for ev in events:
                if ev.severity.value >= 10:
                    print(f"⚠️  [{ev.timestamp_sec:.1f}s] {ev.description}")

        frame_idx += 1

    cap.release()
    report = analyzer.end_session()
    analyzer.close()

    return report_to_dict(report)


# ═══════════════════════════════════════════════════════════════════════════
# 3. API REST — ejemplo con FastAPI (pip install fastapi uvicorn python-multipart)
# ═══════════════════════════════════════════════════════════════════════════

"""
Para ejecutar:
    uvicorn integration_example:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /analyze         — sube un video y recibe el informe
    POST /analyze/quick   — análisis rápido (modelo lite, sin manos)
"""

try:
    from fastapi import FastAPI, UploadFile, File, BackgroundTasks
    from fastapi.responses import JSONResponse
    import tempfile
    import os

    app = FastAPI(title="Presentation Body Language Analyzer")

    @app.post("/analyze")
    async def analyze_video(
        video: UploadFile = File(...),
        fast: bool = False,
        max_duration: float | None = None,
    ):
        # Guardar video temporal
        suffix = "." + (video.filename.split(".")[-1] if video.filename else "mp4")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await video.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            config = dict(
                use_hands=not fast,
                model_complexity=0 if fast else 1,
                process_every_n=3 if fast else 2,
            )
            with PresentationAnalyzer(**config) as analyzer:
                report = analyzer.analyze_video(
                    tmp_path,
                    max_duration_sec=max_duration,
                )
            return JSONResponse(content=report_to_dict(report))
        finally:
            os.unlink(tmp_path)

    @app.post("/analyze/quick")
    async def analyze_quick(video: UploadFile = File(...)):
        """Análisis rápido — 3x más veloz, sin detección de manos."""
        suffix = "." + (video.filename.split(".")[-1] if video.filename else "mp4")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await video.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            with PresentationAnalyzer(
                use_hands=False,
                model_complexity=0,
                process_every_n=4,
            ) as analyzer:
                report = analyzer.analyze_video(tmp_path, max_duration_sec=120)
            return JSONResponse(content=report_to_dict(report))
        finally:
            os.unlink(tmp_path)

except ImportError:
    # FastAPI no instalado — no pasa nada, los otros métodos siguen disponibles
    pass


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

    if resultado['gesture_summaries']:
        print("\nProblemas detectados:")
        for g in resultado['gesture_summaries']:
            print(f"  • {g['name']}: -{g['points_deducted']} pts ({g['description']})")

    if resultado['recommendations']:
        print("\nRecomendaciones:")
        for i, r in enumerate(resultado['recommendations'], 1):
            print(f"  {i}. {r}")