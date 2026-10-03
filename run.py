#!/usr/bin/env python3
"""Root-level runner for the Semantic Caching LLM project."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import main

if __name__ == "__main__":
    if "--ui" in sys.argv:
        from src.server import start_server
        start_server(port=8000, open_browser=True)
    else:
        main()
