"""
Helix Transvoicer - Main application entry point.

Windows 11 optimized application launcher.
"""

import argparse
import multiprocessing
import os
import sys
from typing import Optional


def setup_windows_environment():
    """Configure Windows-specific environment settings."""
    if sys.platform != "win32":
        return

    # Enable ANSI colors in Windows terminal
    os.system("")

    # Set UTF-8 encoding
    if sys.stdout:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if sys.stderr:
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Windows multiprocessing fix
    multiprocessing.freeze_support()

    # Set DPI awareness for crisp UI on high-DPI displays
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    # Hide console window when running with pythonw
    if "pythonw" in sys.executable.lower():
        try:
            import ctypes
            ctypes.windll.user32.ShowWindow(
                ctypes.windll.kernel32.GetConsoleWindow(), 0
            )
        except Exception:
            pass


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
    # Windows setup
    setup_windows_environment()

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

Windows 11:
  Run setup.ps1 for first-time installation.
  Use run.bat to start the application.
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
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    args = parser.parse_args()

    # Set debug mode
    if args.debug:
        os.environ["HELIX_DEBUG"] = "true"

    api_url = args.api_url or f"http://{args.host}:{args.port}"

    if args.server_only:
        print(f"Starting Helix Transvoicer API server at {args.host}:{args.port}")
        print(f"API documentation: http://{args.host}:{args.port}/docs")
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

    # On Windows, use spawn method for multiprocessing
    if sys.platform == "win32":
        multiprocessing.set_start_method("spawn", force=True)

    # Start backend in a separate process
    backend_process = multiprocessing.Process(
        target=run_backend,
        args=(args.host, args.port),
        daemon=True,
    )
    backend_process.start()

    # Give the server a moment to start
    import time
    time.sleep(2.0)  # Slightly longer wait on Windows

    # Run frontend in main process
    try:
        run_frontend(api_url=api_url)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if backend_process.is_alive():
            backend_process.terminate()
            backend_process.join(timeout=3)
            if backend_process.is_alive():
                backend_process.kill()

    return 0


if __name__ == "__main__":
    sys.exit(main())
