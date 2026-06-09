"""SGCP-RAG: A RAG service using FastAPI and LangChain."""

import logging
import os

import dotenv

__version__ = "0.0.1"

if os.getenv("PYTHON_DOTENV_DISABLED") != "1":
    dotenv.load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
