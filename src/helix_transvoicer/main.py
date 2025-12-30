"""
Helix Transvoicer - Main application entry point.

Launches both the backend API server and frontend UI.
"""

import argparse
import multiprocessing
import sys
from typing import Optional


def run_backend(host: str = "127.0.0.1", port: int = 8420) -> None:
    """Start the backend API server."""
    from helix_transvoicer.backend.main import run_server
    run_server(host=host, port=port)


def run_frontend(api_url: Optional[str] = None) -> None:
    """Start the frontend UI application."""
    from helix_transvoicer.frontend.main import main as frontend_main
    frontend_main(api_url=api_url)


def main() -> int:
    """Main entry point for Helix Transvoicer."""
    parser = argparse.ArgumentParser(
        prog="helix",
        description="Helix Transvoicer - Studio-grade voice conversion and TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  helix                    # Launch full application (UI + backend)
  helix --server-only      # Launch only the API server
  helix --ui-only          # Launch only the UI (requires external server)
  helix --port 8421        # Use custom port for API server
        """,
    )

    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Run only the backend API server",
    )
    parser.add_argument(
        "--ui-only",
        action="store_true",
        help="Run only the frontend UI",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="API server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8420,
        help="API server port (default: 8420)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="API URL for UI-only mode (default: http://host:port)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    args = parser.parse_args()

    api_url = args.api_url or f"http://{args.host}:{args.port}"

    if args.server_only:
        print(f"Starting Helix Transvoicer API server at {args.host}:{args.port}")
        run_backend(host=args.host, port=args.port)
        return 0

    if args.ui_only:
        print(f"Starting Helix Transvoicer UI (connecting to {api_url})")
        run_frontend(api_url=api_url)
        return 0

    # Run both backend and frontend
    print("Starting Helix Transvoicer...")
    print(f"  API Server: {args.host}:{args.port}")
    print(f"  UI connecting to: {api_url}")

    # Start backend in a separate process
    backend_process = multiprocessing.Process(
        target=run_backend,
        args=(args.host, args.port),
        daemon=True,
    )
    backend_process.start()

    # Give the server a moment to start
    import time
    time.sleep(1.5)

    # Run frontend in main process
    try:
        run_frontend(api_url=api_url)
    finally:
        backend_process.terminate()
        backend_process.join(timeout=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
