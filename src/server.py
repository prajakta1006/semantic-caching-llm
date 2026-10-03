"""Lightweight Python Web Server for SemCache AI UI.

Provides pure Python stdlib REST API and static file serving for the web interface.
"""

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import sys
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = PROJECT_ROOT / "ui"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.pipeline import SemanticCachingPipeline

# Singleton pipeline instance
pipeline_instance = SemanticCachingPipeline()


class PipelineRequestHandler(SimpleHTTPRequestHandler):
    """Handles static files and API requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_POST(self):
        if self.path == "/api/query":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                query_text = data.get("query", "").strip()
                if not query_text:
                    self._send_json({"error": "Empty query"}, status=400)
                    return

                resp = pipeline_instance.process_query(query_text)
                
                payload = {
                    "query": resp.query,
                    "response_text": resp.response_text,
                    "source": resp.source,
                    "similarity_score": resp.similarity_score,
                    "decision": {
                        "route": resp.decision.route if resp.decision else "N/A",
                        "estimated_cost": resp.decision.estimated_cost if resp.decision else 0.0,
                        "confidence": resp.decision.confidence if resp.decision else 0.0,
                        "reason": resp.decision.reason if resp.decision else "",
                    } if resp.decision else None,
                    "latency_ms": resp.latency_ms,
                    "metadata": resp.metadata,
                }
                self._send_json(payload)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
        else:
            self.send_error(404, "Endpoint not found")

    def do_GET(self):
        if self.path == "/api/metrics":
            m = pipeline_instance.evaluator.get_metrics()
            payload = {
                "total_queries": m.total_queries,
                "cache_hits": m.cache_hits,
                "cache_misses": m.cache_misses,
                "hit_rate": m.hit_rate,
                "total_cost_saved": m.total_cost_saved,
                "avg_latency_ms": m.avg_latency_ms,
            }
            self._send_json(payload)
        elif self.path == "/api/config":
            payload = {
                "project_name": config.project_name,
                "version": config.version,
                "similarity_threshold": config.similarity_threshold,
                "embedding_model_name": config.embedding_model_name,
            }
            self._send_json(payload)
        else:
            super().do_GET()

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def start_server(port=8000, open_browser=True):
    server_address = ("", port)
    httpd = HTTPServer(server_address, PipelineRequestHandler)
    url = f"http://localhost:{port}"
    print("=" * 65)
    print(f"  SemCache AI Web UI Server running at: {url}")
    print(f"  Press Ctrl+C to stop the server.")
    print("=" * 65)

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


if __name__ == "__main__":
    start_server(port=8000, open_browser=False)
