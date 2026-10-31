
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env from the project root.
load_dotenv(Path(__file__).resolve().parent / ".env")

uri = os.getenv("MONGODB_URI")
database_name = os.getenv("MONGODB_DATABASE", "semantic_cache_db")

if not uri:
    raise RuntimeError(
        "MONGODB_URI is missing. Check your project-root .env file."
    )

client = MongoClient(
    uri,
    serverSelectionTimeoutMS=10000,
)

try:
    client.admin.command("ping")
    print("SUCCESS: Connected to MongoDB Atlas!")

    database = client[database_name]
    print(f"Database configured: {database.name}")
finally:
    client.close()
