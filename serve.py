"""Run the REST API with uvicorn.

Usage:
    python serve.py                # 127.0.0.1:8000, default workers
    python serve.py --host 0.0.0.0 --port 8080
    python serve.py --reload       # auto-reload on code changes (dev only)
"""

import argparse

import uvicorn


# Parses host/port/reload flags; defaults are local-only and prod-safe.
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the API server."""
    parser = argparse.ArgumentParser(description="Multi-Tool Research Agent REST API")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev only)")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="uvicorn log level",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: launch uvicorn with the FastAPI app from src.api.app."""
    args = parse_args()
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
