"""
Embedding engine. Supports Voyage (recommended), OpenAI, and Ollama (local).
"""

import os
from typing import Literal

import httpx
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIM = 1024


def _get_provider() -> Literal["voyage", "openai", "ollama"]:
    provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
    if provider:
        return provider  # type: ignore
    if os.environ.get("VOYAGE_API_KEY"):
        return "voyage"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts. Routes to configured provider."""
    provider = _get_provider()
    if provider == "voyage":
        return _voyage_embed(texts, "document")
    elif provider == "openai":
        return _openai_embed(texts)
    else:
        return _ollama_embed(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query (uses query-optimized mode for Voyage)."""
    provider = _get_provider()
    if provider == "voyage":
        return _voyage_embed([text], "query")[0]
    elif provider == "openai":
        return _openai_embed([text])[0]
    else:
        return _ollama_embed([text])[0]


def _voyage_embed(
    texts: list[str], input_type: str = "document"
) -> list[list[float]]:
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        raise RuntimeError("VOYAGE_API_KEY not set")

    all_embeddings: list[list[float]] = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "voyage-3", "input": batch, "input_type": input_type},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        all_embeddings.extend([d["embedding"] for d in data["data"]])

    return all_embeddings


def _openai_embed(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts, dimensions=EMBEDDING_DIM)
    return [d.embedding for d in resp.data]


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("EMBEDDING_MODEL", "mxbai-embed-large")
    embeddings = []
    for text in texts:
        resp = httpx.post(
            f"{url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        embeddings.append(resp.json()["embedding"])
    return embeddings
