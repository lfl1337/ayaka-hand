#!/usr/bin/env python3
"""Local sink for the opt-in live telemetry beacon (web/src/telemetry.ts).

Run on the controller's machine while a remote user drives the edge demo with
?live=1 in the URL; each beacon becomes one compact line on stdout and in
web/telemetry.log. Verification tooling only — stdlib, no deps, fail-open.

    python3 web/scripts/telemetry-server.py
"""
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST, PORT = "0.0.0.0", 4174
LOG_PATH = Path(__file__).resolve().parent.parent / "telemetry.log"


def format_line(payload: dict) -> str:
    """{t, event, ...data} -> 'HH:MM:SS event key=val key=val'."""
    ts = datetime.now().strftime("%H:%M:%S")
    event = payload.get("event", "?")
    pairs = [f"{k}={v}" for k, v in payload.items() if k not in ("t", "event")]
    return " ".join([ts, str(event), *pairs])


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        # Belt-and-braces: the beacon uses mode:'no-cors' so this isn't required,
        # but it keeps the sink usable from any other tool without surprises.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            line = format_line(json.loads(raw or b"{}"))
        except (ValueError, TypeError):
            line = f'{datetime.now().strftime("%H:%M:%S")} malformed {raw!r}'
        print(line, flush=True)
        try:
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass  # fail-open: never let the sink error out a running gate
        self.send_response(204)
        self._cors()
        self.end_headers()

    def log_message(self, *args) -> None:  # silence per-request stderr noise
        pass


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"telemetry sink on {HOST}:{PORT} -> {LOG_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
