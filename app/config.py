import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads the .env file and puts the key into os.environ

# Folder paths
BASE_DIR = Path(__file__).resolve().parent.parent   # the "Travel Desk" folder
DATA_DIR = BASE_DIR / "data"
POLICY_PATH = DATA_DIR / "Demo_Corporate_Travel_Policy.txt"
INDEX_DIR = BASE_DIR / "faiss_index"

# Model settings
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# RAG settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing. Add it to the .env file.")