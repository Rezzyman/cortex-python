# CORTEX (Python)

**Synthetic cognition infrastructure for AI agents.**

Python implementation of the core [CORTEX](https://github.com/Rezzyman/cortex) memory stack — the same ingestion pipeline, hippocampal encoding, dream cycle, procedural memory, and hybrid search that power the TypeScript flagship. Full CORTEX (TypeScript) holds #1 on LongMemEval (500/500) and LoCoMo (93.6% R@10); this Python port is wire-compatible with the same Postgres+pgvector schema so you can read and write the same memory store from Python agents. Zero LLM required for core operation.

```bash
pip install cortex-ai
```

## Quick Start

```python
from cortex_ai import init_database, ingest, search, recall, dream

# Initialize
init_database()

# Ingest
ingest("The quarterly review showed 40% growth in enterprise accounts.", agent_id="myagent")

# Search
results = search("enterprise growth", agent_id="myagent")
for r in results:
    print(f"[{r.score:.3f}] {r.content[:80]}...")

# Token-budget recall
context = recall("quarterly performance", agent_id="myagent", token_budget=2000)

# Dream cycle (nightly maintenance)
from cortex_ai.db.connection import resolve_agent
agent_num = resolve_agent("myagent")
dream(agent_num)
```

## CLI

```bash
cortex init                              # Initialize database
cortex ingest "Some important fact"      # Store a memory
cortex search "what happened"            # Search memories
cortex recall "project status" --budget 4000  # Budget-aware retrieval
cortex dream                             # Run dream cycle
cortex status                            # System health
```

## Architecture

cortex-python implements the CORTEX core memory stack in Python. The package is wire-compatible with the TypeScript flagship's Postgres+pgvector schema — both can read and write the same memory store. What's implemented in this Python release:

- **Ingestion pipeline**: chunking, entity extraction, embedding generation (Voyage / OpenAI / Ollama)
- **Hippocampal encoding**: Dentate Gyrus pattern separation (4096-dim sparse coding, 5% sparsity) and CA1 novelty detection with sparse gating
- **Dream cycle**: resonance decay and cluster consolidation
- **Procedural memory**: skill storage and execution tracking
- **5-factor hybrid search**: cosine + text match + recency + resonance + priority boost
- **Temporal validity**: `valid_from`, `valid_until`, `superseded_by` (schema-level)
- **CLI**: `cortex init`, `ingest`, `search`, `recall`, `dream`, `status`

Features in full CORTEX (TypeScript) that are **not yet in this Python release**:

- **CA3 pattern completion** — hippocampal recurrent pattern completion (cortex-python falls back to pure hybrid search)
- **Reconsolidation / labile window** — belief updating on recall (schema is present, worker module not yet ported)
- **Emotional valence (6-dimensional)** — valence scoring and decay resistance
- **Metacognition / proprioception / autonomous cognition** — reasoning threads, self-diagnostic, bias detection
- **MCP server** — Model Context Protocol integration (Python agents integrate via the direct API for now)
- **Social graph / empathy modules**

If you need those capabilities, use [full CORTEX](https://github.com/Rezzyman/cortex) (TypeScript). cortex-python is maintained as the first-class Python interface to the same memory store and will gain parity on the above over subsequent releases.

## Requirements

- Python 3.10+
- PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector)
- Embedding API key (Voyage recommended, OpenAI or Ollama supported)

## Configuration

```bash
cp .env.example .env
# Set DATABASE_URL and VOYAGE_API_KEY (or OPENAI_API_KEY)
```

## Development

```bash
git clone https://github.com/Rezzyman/cortex-python.git
cd cortex-python
pip install -e ".[dev]"
pytest
```

## Benchmarks

The full CORTEX (TypeScript) implementation — which shares the same storage schema as cortex-python — holds #1 on:

- **LongMemEval**: 500/500 (100% R@5)
- **LoCoMo**: 93.6% R@10 (beats MemPalace hybrid)

Both scored without LLM reranking. See [CORTEX Benchmarks](https://github.com/Rezzyman/cortex/blob/main/BENCHMARKS.md) for the full methodology and leaderboard.

cortex-python reads and writes the same memory store, but because it uses 5-factor hybrid search (vs the TypeScript flagship's 7-factor search with emotional valence and CA3 pattern completion), its retrieval scores against these benchmarks have not been independently certified. Use cortex-python for Python-native agent integration against an existing CORTEX memory store; use [full CORTEX](https://github.com/Rezzyman/cortex) for benchmark-certified retrieval.

## The CORTEX ecosystem

cortex-python is one of three compatible CORTEX projects. Pick the one that matches your situation:

| Project | Language | Storage | Audience | Install |
|---|---|---|---|---|
| **[cortex](https://github.com/Rezzyman/cortex)** | TypeScript (Node 22+) | PostgreSQL + pgvector | Production agent teams · benchmark-certified retrieval · MCP and REST · full subsystem stack | `git clone` + Docker Compose |
| **[cortex-lite](https://github.com/Rezzyman/cortex-lite)** | Python 3.10+ | SQLite (one file) | Individual developers · zero-config · local embeddings · 30-second quickstart | `pip install cortex-lite` |
| **[cortex-python](https://github.com/Rezzyman/cortex-python)** (this repo) | Python 3.10+ | PostgreSQL + pgvector (shares schema with cortex) | Python agent codebases that need to read and write the same memory store as a TypeScript CORTEX deployment | `pip install cortex-ai` |

**Use cortex-python** when your agent code is already Python and you need first-class read/write access to a CORTEX memory store. cortex-python implements the core subsystems (ingestion, hippocampal encoding, dream cycle, procedural memory, hybrid search) and is schema-compatible with the TypeScript flagship.

**Use full [cortex](https://github.com/Rezzyman/cortex)** when you need CA3 pattern completion, reconsolidation, emotional valence, autonomous cognition, metacognition, or the benchmark-certified 7-factor retrieval stack.

**Use [cortex-lite](https://github.com/Rezzyman/cortex-lite)** when you want the hybrid-search pattern running in under a minute with zero infrastructure.

---

## License

Apache 2.0.

Built by [Atanasio Juarez](https://github.com/Rezzyman) at [ATERNA.AI](https://aterna.ai).
