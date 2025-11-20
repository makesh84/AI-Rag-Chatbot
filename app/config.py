import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3-70b-8192")

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment (.env)")
