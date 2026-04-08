"""Database connection and schema initialization."""

import os
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()


@contextmanager
def get_db() -> Generator[psycopg.Connection, None, None]:
    """Get a database connection."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    with psycopg.connect(url, row_factory=dict_row) as conn:
        yield conn


def init_database() -> None:
    """Initialize pgvector extension and run schema migrations."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Core tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    id SERIAL PRIMARY KEY,
                    external_id VARCHAR(64) NOT NULL UNIQUE,
                    name VARCHAR(255) NOT NULL,
                    owner_id VARCHAR(255),
                    config JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW() NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    id SERIAL PRIMARY KEY,
                    agent_id INTEGER NOT NULL REFERENCES agents(id),
                    content TEXT NOT NULL,
                    summary TEXT,
                    source TEXT,
                    source_type VARCHAR(64) DEFAULT 'markdown',
                    chunk_index INTEGER DEFAULT 0,
                    embedding vector(1024),
                    entities TEXT[] DEFAULT '{}',
                    semantic_tags TEXT[] DEFAULT '{}',
                    priority INTEGER DEFAULT 2,
                    resonance_score REAL DEFAULT 5.0,
                    novelty_score REAL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed_at TIMESTAMP DEFAULT NOW(),
                    last_recalled_at TIMESTAMP,
                    status VARCHAR(32) DEFAULT 'active',
                    valid_from TIMESTAMP,
                    valid_until TIMESTAMP,
                    superseded_by INTEGER,
                    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW() NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_synapses (
                    id SERIAL PRIMARY KEY,
                    memory_a INTEGER NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
                    memory_b INTEGER NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
                    connection_type VARCHAR(32) NOT NULL,
                    connection_strength REAL DEFAULT 0.5 NOT NULL,
                    activation_count INTEGER DEFAULT 0,
                    decay_rate REAL DEFAULT 0.01,
                    last_activated_at TIMESTAMP DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
                    UNIQUE(memory_a, memory_b, connection_type)
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS hippocampal_codes (
                    id SERIAL PRIMARY KEY,
                    memory_id INTEGER NOT NULL UNIQUE REFERENCES memory_nodes(id) ON DELETE CASCADE,
                    agent_id INTEGER NOT NULL REFERENCES agents(id),
                    sparse_indices INTEGER[] NOT NULL,
                    sparse_values REAL[] NOT NULL,
                    sparse_dim INTEGER DEFAULT 4096,
                    novelty_score REAL,
                    created_at TIMESTAMP DEFAULT NOW() NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS emotional_valence (
                    id SERIAL PRIMARY KEY,
                    memory_id INTEGER NOT NULL UNIQUE REFERENCES memory_nodes(id) ON DELETE CASCADE,
                    agent_id INTEGER NOT NULL REFERENCES agents(id),
                    valence REAL NOT NULL DEFAULT 0,
                    arousal REAL NOT NULL DEFAULT 0,
                    dominance REAL NOT NULL DEFAULT 0,
                    certainty REAL NOT NULL DEFAULT 0,
                    relevance REAL NOT NULL DEFAULT 0.3,
                    urgency REAL NOT NULL DEFAULT 0,
                    intensity REAL NOT NULL DEFAULT 0,
                    decay_resistance REAL NOT NULL DEFAULT 0,
                    recall_boost REAL NOT NULL DEFAULT 0,
                    dominant_dimension VARCHAR(32),
                    created_at TIMESTAMP DEFAULT NOW() NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS procedural_memories (
                    id SERIAL PRIMARY KEY,
                    agent_id INTEGER NOT NULL REFERENCES agents(id),
                    name VARCHAR(255) NOT NULL,
                    description TEXT NOT NULL,
                    procedural_type VARCHAR(32) NOT NULL,
                    trigger_context TEXT NOT NULL,
                    steps TEXT[] DEFAULT '{}',
                    embedding vector(1024),
                    proficiency VARCHAR(32) DEFAULT 'novice',
                    execution_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0,
                    domain_tags TEXT[] DEFAULT '{}',
                    source_memory_ids INTEGER[] DEFAULT '{}',
                    version INTEGER DEFAULT 1,
                    status VARCHAR(32) DEFAULT 'active',
                    last_executed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW() NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS dream_cycle_logs (
                    id SERIAL PRIMARY KEY,
                    agent_id INTEGER NOT NULL REFERENCES agents(id),
                    cycle_type VARCHAR(32) NOT NULL,
                    stats JSONB DEFAULT '{}',
                    insights_discovered JSONB DEFAULT '[]',
                    started_at TIMESTAMP DEFAULT NOW() NOT NULL,
                    completed_at TIMESTAMP
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS cognitive_artifacts (
                    id SERIAL PRIMARY KEY,
                    agent_id INTEGER NOT NULL REFERENCES agents(id),
                    session_id VARCHAR(128),
                    artifact_type VARCHAR(32) NOT NULL,
                    content JSONB NOT NULL,
                    embedding vector(1024),
                    resonance_score REAL DEFAULT 5.0,
                    created_at TIMESTAMP DEFAULT NOW() NOT NULL
                )
            """)

            # Performance indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mn_agent_status ON memory_nodes(agent_id, status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mn_agent_status_priority ON memory_nodes(agent_id, status, priority)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mn_resonance ON memory_nodes(resonance_score)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mn_valid_until ON memory_nodes(valid_until) WHERE valid_until IS NOT NULL")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_synapses_a ON memory_synapses(memory_a, connection_strength)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_synapses_b ON memory_synapses(memory_b, connection_strength)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hc_agent ON hippocampal_codes(agent_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hc_memory ON hippocampal_codes(memory_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ev_memory ON emotional_valence(memory_id)")

            # HNSW index for vector search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_mn_embedding_hnsw
                ON memory_nodes USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

        conn.commit()
        print("[cortex] Database initialized (pgvector + schema + indexes)")


def resolve_agent(agent_id: str = "default") -> int:
    """Resolve an agent by external ID, creating if needed."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM agents WHERE external_id = %s",
                (agent_id,),
            )
            row = cur.fetchone()
            if row:
                return row["id"]

            cur.execute(
                "INSERT INTO agents (external_id, name, owner_id) VALUES (%s, %s, %s) RETURNING id",
                (agent_id, agent_id.capitalize(), "default"),
            )
            row = cur.fetchone()
            conn.commit()
            return row["id"]
