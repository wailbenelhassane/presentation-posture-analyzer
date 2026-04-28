#!/usr/bin/env python3
"""
CLI for presentation_analyzer v2.

Usage:
    python -m presentation_analyzer video.mp4
    python -m presentation_analyzer video.mp4 -v resultado.mp4
    python -m presentation_analyzer video.mp4 -v resultado.mp4 --max-persons 4
    python -m presentation_analyzer video.mp4 --json -o informe.json
"""

import argparse
import sys
from pathlib import Path

from .analyzer import PresentationAnalyzer, report_to_json, print_report


def progress_bar(progress: float):
    bar_len = 40
    filled = int(bar_len * progress)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = int(progress * 100)
    print(f"\r  Analizando... [{bar}] {pct}%", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Analiza lenguaje corporal en videos de presentaciones"
    )
    parser.add_argument("video", help="Ruta al archivo de video")
    parser.add_argument(
        "--output-video", "-v", type=str, default=None,
        help="Generar video anotado con esqueleto y marcas de errores"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Salida del informe en formato JSON"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Guardar informe JSON en archivo"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Modo rápido: modelo lite, procesa 1 de cada 3 frames"
    )
    parser.add_argument(
        "--no-hands", action="store_true",
        help="Desactivar detección de manos (más rápido, sin gestos ofensivos)"
    )
    parser.add_argument(
        "--max-persons", type=int, default=4,
        help="Máximo de personas a rastrear simultáneamente (1-4, default: 4)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=None,
        help="Analizar solo los primeros N segundos"
    )

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: No se encuentra el video '{args.video}'", file=sys.stderr)
        sys.exit(1)

    config = dict(
        use_hands=not args.no_hands,
        model_complexity=0 if args.fast else 1,
        process_every_n=3 if args.fast else 2,
        max_persons=min(4, max(1, args.max_persons)),
    )

    print(f"\n  📹 Analizando: {args.video}")
    print(f"  ⚙  Modo: {'rápido' if args.fast else 'estándar'} | "
          f"Manos: {'sí' if not args.no_hands else 'no'} | "
          f"Personas: máx {config['max_persons']}")
    if args.output_video:
        print(f"  🎬 Video resultado: {args.output_video}")
    print()

    with PresentationAnalyzer(**config) as analyzer:
        report = analyzer.analyze_video(
            args.video,
            output_video=args.output_video,
            progress_callback=progress_bar if not args.json else None,
            max_duration_sec=args.max_duration,
        )

    if not args.json:
        print()

    if args.json:
        output = report_to_json(report)
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"Informe guardado en: {args.output}")
        else:
            print(output)
    else:
        print_report(report)
        if args.output:
            Path(args.output).write_text(
                report_to_json(report), encoding="utf-8"
            )
            print(f"  💾 Informe JSON guardado en: {args.output}")
        if args.output_video:
            print(f"  🎬 Video anotado guardado en: {args.output_video}")


if __name__ == "__main__":
    main()
