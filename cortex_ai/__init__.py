"""CORTEX — Synthetic cognition infrastructure for AI agents."""

__version__ = "2.4.0"

from cortex_ai.db.connection import get_db, init_database
from cortex_ai.search import search, recall
from cortex_ai.ingestion.ingest import ingest, ingest_file
from cortex_ai.dream.cycle import dream
from cortex_ai.hippocampus.dentate_gyrus import dg_encode, sparse_overlap

__all__ = [
    "get_db",
    "init_database",
    "search",
    "recall",
    "ingest",
    "ingest_file",
    "dream",
    "dg_encode",
    "sparse_overlap",
]
