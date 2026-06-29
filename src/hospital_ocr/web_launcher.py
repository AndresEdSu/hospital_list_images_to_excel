from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("web_app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",
        "--server.address=127.0.0.1",
        "--server.fileWatcherType=none",
        "--server.maxUploadSize=15",
        "--browser.gatherUsageStats=false",
        "--client.toolbarMode=minimal",
    ]
    raise SystemExit(streamlit_cli.main())
