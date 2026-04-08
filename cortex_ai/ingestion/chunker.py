"""Text chunking with token-aware splitting."""

import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = 256
CHUNK_OVERLAP = 25


def count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base encoding."""
    return len(_encoder.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into token-aware chunks with overlap.

    Returns list of {"text": str, "index": int, "tokens": int}.
    """
    if not text.strip():
        return []

    tokens = _encoder.encode(text)
    chunks = []
    start = 0
    index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _encoder.decode(chunk_tokens)
        chunks.append({
            "text": chunk_text,
            "index": index,
            "tokens": len(chunk_tokens),
        })
        index += 1
        start += chunk_size - overlap

    return chunks
