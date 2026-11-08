
"""Optional integration test for MongoDB Atlas connectivity.

Normal unit-test runs do not require a live MongoDB connection.

To run this test explicitly:
    Set RUN_MONGODB_INTEGRATION=true in the environment.
    Ensure MONGODB_URI is configured in the project-root .env.
    Run: python -m pytest -q test_mongodb_connection.py
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv(
    "MONGODB_DATABASE",
    "semantic_cache_db",
)

RUN_INTEGRATION = (
    os.getenv("RUN_MONGODB_INTEGRATION", "").strip().lower()
    in ("true", "1", "yes")
)


@pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="MongoDB integration test is disabled by default.",
)
@pytest.mark.skipif(
    not MONGODB_URI,
    reason="MONGODB_URI is not configured.",
)
def test_mongodb_atlas_connection():
    """Verify that MongoDB Atlas is reachable."""

    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=10000,
    )

    try:
        client.admin.command("ping")

        database = client[DATABASE_NAME]

        assert database.name == DATABASE_NAME

    finally:
        client.close()
