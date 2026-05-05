"""
Launch the Voxly Presentation Analyzer REST API.

Usage:
    python run_api.py [--host HOST] [--port PORT] [--reload]

Examples:
    python run_api.py
    python run_api.py --port 9000
    python run_api.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Voxly API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev only)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    args = parser.parse_args()

    print(f"\n  Voxly Presentation Analyzer API")
    print(f"  Running on http://{args.host}:{args.port}")
    print(f"  Docs:      http://{args.host}:{args.port}/docs")
    print(f"  OpenAPI:   http://{args.host}:{args.port}/openapi.json\n")

    uvicorn.run(
        "presentation_analyzer.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
