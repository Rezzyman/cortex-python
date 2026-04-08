# CORTEX (Python)

**Synthetic cognition infrastructure for AI agents.**

The Python implementation of [CORTEX](https://github.com/Rezzyman/cortex), the #1 ranked AI agent memory system on LongMemEval (500/500) and LoCoMo (93.6% R@10). Zero LLM required.

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

Same cognitive architecture as the TypeScript version:

- **Hippocampal encoding**: Dentate Gyrus pattern separation (4096-dim sparse coding, 5% sparsity), CA1 novelty detection with sparse gating
- **Dream cycles**: Resonance decay, adaptive pruning (percentile-based), cluster consolidation
- **Emotional valence**: 6-dimensional emotional vectors with decay resistance
- **Reconsolidation**: Labile window belief updates with temporal validity
- **7-factor hybrid search**: cosine + text + recency + resonance + priority
- **Temporal validity**: valid_from, valid_until, superseded_by

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

See [CORTEX Benchmarks](https://github.com/Rezzyman/cortex/blob/main/BENCHMARKS.md):
- **LongMemEval**: 500/500 (100% R@5, #1)
- **LoCoMo**: 93.6% R@10 (#1, beats MemPalace hybrid)

Both without LLM reranking. Both share the same database with the TypeScript version.

## License

Apache 2.0.

Built by [Atanasio Juarez](https://github.com/Rezzyman) at [ATERNA.AI](https://aterna.ai).
