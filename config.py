import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    # Environment variables still work when python-dotenv is not installed,
    # which keeps pure unit tests independent from application dependencies.
    pass

KB_PATH = BASE_DIR / "knowledge_base"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "1.35"))
UPLOAD_MAX_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", str(15 * 1024 * 1024)))
