from __future__ import annotations

import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path
from urllib.parse import quote


def launch(results_json: str | None = None, host: str = "127.0.0.1", port: int = 0, open_browser: bool = True) -> str:
    static_dir = Path(__file__).resolve().parent / "web"
    results_path = Path(results_json).expanduser().resolve() if results_json else None
    if results_path is not None and not results_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")
    if not (static_dir / "index.html").exists():
        raise FileNotFoundError(f"Built frontend not found under {static_dir}")

    # Give each served payload a stable, descriptive URL rather than the
    # ambiguous shared `/results.json`. This also prevents browser caches from
    # making a later triplet/cross-pair/listwise launch appear to show an old
    # file.
    payload_path = f"/payload/{quote(results_path.name, safe='')}" if results_path else None
    handler = functools.partial(
        _NashHandler,
        directory=str(static_dir),
        results_path=results_path,
        payload_path=payload_path,
    )
    server = _ReusableThreadingTCPServer((host, port), handler)
    server.daemon_threads = True
    actual_port = int(server.server_address[1])
    url = f"http://{host}:{actual_port}/?tab=batch"
    if payload_path is not None:
        url += f"&payload={quote(payload_path, safe='/')}"

    if open_browser:
        webbrowser.open(url)
    print(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return url


class _ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class _NashHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, results_path: Path | None, payload_path: str | None, **kwargs):
        self.results_path = results_path
        self.payload_path = payload_path
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == self.payload_path and self.results_path is not None:
            self._serve_payload(include_body=True)
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self.path.split("?", 1)[0] == self.payload_path and self.results_path is not None:
            self._serve_payload(include_body=False)
            return
        super().do_HEAD()

    def _serve_payload(self, include_body: bool) -> None:
        assert self.results_path is not None
        data = self.results_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        if include_body:
            self.wfile.write(data)
