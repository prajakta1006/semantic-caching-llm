"""Optional HTTP adapter for the semantic caching pipeline.

This module contains no cache, LLM, or evaluation algorithms.
It simply exposes the pipeline through a small HTTP API.
"""

import json
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


pipeline_instance = SemanticCachingPipeline()


class PipelineRequestHandler(
    SimpleHTTPRequestHandler
):
    """HTTP handler for the pipeline API and optional UI."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            directory=str(UI_DIR),
            **kwargs,
        )

    def do_POST(self):
        """Handle pipeline query requests."""

        if self.path != "/api/query":
            self.send_error(
                404,
                "Endpoint not found",
            )
            return

        content_length = int(
            self.headers.get(
                "Content-Length",
                0,
            )
        )

        body = self.rfile.read(
            content_length
        ).decode("utf-8")

        try:

            data = json.loads(body)

            query_text = data.get(
                "query",
                "",
            ).strip()

            if not query_text:

                self._send_json(
                    {"error": "Empty query"},
                    status=400,
                )

                return

            response = pipeline_instance.process_query(
                query_text
            )

            decision = None

            if response.decision:

                decision = {
                    "route": response.decision.route,
                    "estimated_cost": (
                        response.decision.estimated_cost
                    ),
                    "confidence": (
                        response.decision.confidence
                    ),
                    "reason": (
                        response.decision.reason
                    ),
                }

            payload = {
                "query": response.query,
                "response_text": response.response_text,
                "source": response.source,
                "similarity_score": (
                    response.similarity_score
                ),
                "decision": decision,
                "latency_ms": response.latency_ms,
                "metadata": response.metadata,
            }

            self._send_json(payload)

        except Exception as exc:

            self._send_json(
                {"error": str(exc)},
                status=500,
            )

    def do_GET(self):
        """Handle metrics/config/static-file requests."""

        if self.path == "/api/metrics":

            if pipeline_instance.evaluator is None:

                self._send_json(
                    {
                        "available": False,
                        "message": (
                            "Evaluation module is not "
                            "connected yet."
                        ),
                    }
                )

                return

            metrics = (
                pipeline_instance
                .evaluator
                .get_metrics()
            )

            self._send_json(
                {
                    "available": True,
                    "total_queries": (
                        metrics.total_queries
                    ),
                    "cache_hits": (
                        metrics.cache_hits
                    ),
                    "cache_misses": (
                        metrics.cache_misses
                    ),
                    "hit_rate": (
                        metrics.hit_rate
                    ),
                    "total_cost_saved": (
                        metrics.total_cost_saved
                    ),
                    "avg_latency_ms": (
                        metrics.avg_latency_ms
                    ),
                }
            )

            return

        if self.path == "/api/config":

            self._send_json(
                {
                    "project_name": (
                        config.project_name
                    ),
                    "version": config.version,
                    "similarity_threshold": (
                        config.similarity_threshold
                    ),
                    "embedding_model_name": (
                        config.embedding_model_name
                    ),
                    "llm_provider": (
                        config.llm_provider
                    ),
                    "default_llm_model": (
                        config.default_llm_model
                    ),
                }
            )

            return

        super().do_GET()

    def _send_json(
        self,
        data,
        status=200,
    ):
        """Send a JSON response."""

        body = json.dumps(
            data
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.end_headers()

        self.wfile.write(body)


def start_server(
    port=8000,
    open_browser=True,
):
    """Start the optional development server."""

    server_address = (
        "",
        port,
    )

    httpd = HTTPServer(
        server_address,
        PipelineRequestHandler,
    )

    url = (
        f"http://localhost:{port}"
    )

    print("=" * 65)

    print(
        f"  SemCache AI server running at: "
        f"{url}"
    )

    print(
        "  Press Ctrl+C to stop the server."
    )

    print("=" * 65)

    if open_browser:
        webbrowser.open(url)

    try:

        httpd.serve_forever()

    except KeyboardInterrupt:

        print("\nServer stopped.")

        httpd.server_close()


if __name__ == "__main__":
    start_server(
        port=8000,
        open_browser=False,
    )
