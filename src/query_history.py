
import json
from datetime import datetime
from pathlib import Path

HISTORY_FILE = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "console_query_history.json"
)


def save_query(query: str) -> None:
    """Append a console query to permanent history."""
    query = query.strip()

    if not query:
        return

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "query": query,
        "timestamp": datetime.now().astimezone().isoformat(),
    }

    with HISTORY_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
