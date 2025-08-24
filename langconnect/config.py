import json

from langchain_core.embeddings import Embeddings
from starlette.config import Config

env = Config()

IS_TESTING = env("IS_TESTING", cast=str, default="").lower() == "true"


def get_embeddings() -> Embeddings:
    """Get the embeddings instance based on the environment."""
    # from langchain_openai import OpenAIEmbeddings

    # return OpenAIEmbeddings(model="text-embedding-3-small")

    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = "neuml/pubmedbert-base-embeddings"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


DEFAULT_EMBEDDINGS = get_embeddings()
DEFAULT_COLLECTION_NAME = "default_collection"


# Database configuration
POSTGRES_HOST = env("POSTGRES_HOST", cast=str, default="localhost")
POSTGRES_PORT = env("POSTGRES_PORT", cast=int, default="5432")
POSTGRES_USER = env("POSTGRES_USER", cast=str, default="langchain")
POSTGRES_PASSWORD = env("POSTGRES_PASSWORD", cast=str, default="langchain")
POSTGRES_DB = env("POSTGRES_DB", cast=str, default="langchain_test")

# Read allowed origins from environment variable
ALLOW_ORIGINS_JSON = env("ALLOW_ORIGINS", cast=str, default="")

if ALLOW_ORIGINS_JSON:
    ALLOWED_ORIGINS = json.loads(ALLOW_ORIGINS_JSON.strip())
    print(f"ALLOW_ORIGINS environment variable set to: {ALLOW_ORIGINS_JSON}")
else:
    ALLOWED_ORIGINS = "http://localhost:3000"
    print("ALLOW_ORIGINS environment variable not set.")
