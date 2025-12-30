"""
Helix Transvoicer Frontend - Application entry point.
"""

import sys
from typing import Optional

from helix_transvoicer.frontend.app import HelixApp


def main(api_url: Optional[str] = None) -> int:
    """Launch the Helix Transvoicer UI."""
    try:
        app = HelixApp(api_url=api_url or "http://127.0.0.1:8420")
        app.run()
        return 0
    except Exception as e:
        print(f"Error starting UI: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
