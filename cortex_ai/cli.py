"""CORTEX CLI — cortex command-line interface."""

import json
import click

from cortex_ai.db.connection import init_database, resolve_agent


@click.group()
def main():
    """CORTEX -- Synthetic cognition infrastructure for AI agents."""
    pass


@main.command()
def init():
    """Initialize the database schema."""
    init_database()
    click.echo("CORTEX database initialized.")


@main.command()
@click.argument("query")
@click.option("--agent", default="default", help="Agent ID")
@click.option("--limit", default=10, help="Max results")
def search(query: str, agent: str, limit: int):
    """Search CORTEX memory."""
    from cortex_ai.search import search as do_search

    results = do_search(query, agent_id=agent, limit=limit)
    for r in results:
        src = (r.source or "unknown").split("/")[-1]
        click.echo(f"[{r.id}] ({r.score:.4f}) [{src}] {r.content[:120]}...")
        if r.entities:
            click.echo(f"    Entities: {', '.join(r.entities)}")


@main.command()
@click.argument("content")
@click.option("--agent", default="default", help="Agent ID")
@click.option("--source", default="cli", help="Source label")
@click.option("--priority", default=2, help="Priority (0=critical, 4=ephemeral)")
def ingest(content: str, agent: str, source: str, priority: int):
    """Ingest text into CORTEX memory."""
    from cortex_ai.ingestion.ingest import ingest as do_ingest

    result = do_ingest(content, agent_id=agent, source=source, priority=priority)
    click.echo(f"Ingested {result.chunks} chunks -> {len(result.memory_ids)} memories")


@main.command()
@click.argument("query")
@click.option("--agent", default="default", help="Agent ID")
@click.option("--budget", default=4000, help="Token budget")
def recall(query: str, agent: str, budget: int):
    """Token-budget-aware context retrieval."""
    from cortex_ai.search import recall as do_recall

    context = do_recall(query, agent_id=agent, token_budget=budget)
    click.echo(context)


@main.command()
@click.option("--agent", default="default", help="Agent ID")
@click.option("--type", "cycle_type", default="full", help="Cycle type (full, sws_only, rem_only)")
def dream(agent: str, cycle_type: str):
    """Run a dream cycle."""
    from cortex_ai.dream.cycle import dream as do_dream

    agent_num = resolve_agent(agent)
    stats = do_dream(agent_num, cycle_type=cycle_type)
    click.echo(json.dumps(stats.__dict__, indent=2))


@main.command()
@click.option("--agent", default="default", help="Agent ID")
def status(agent: str):
    """Show CORTEX status."""
    from cortex_ai.db.connection import get_db

    agent_num = resolve_agent(agent)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as cnt FROM memory_nodes WHERE agent_id = %s AND status = 'active'",
                (agent_num,),
            )
            active = cur.fetchone()["cnt"]

            cur.execute(
                "SELECT COUNT(*) as cnt FROM memory_synapses WHERE memory_a IN (SELECT id FROM memory_nodes WHERE agent_id = %s)",
                (agent_num,),
            )
            synapses = cur.fetchone()["cnt"]

            cur.execute(
                "SELECT AVG(resonance_score) as avg FROM memory_nodes WHERE agent_id = %s AND status = 'active'",
                (agent_num,),
            )
            avg_res = cur.fetchone()["avg"] or 0

    click.echo(f"Agent: {agent}")
    click.echo(f"Active memories: {active}")
    click.echo(f"Synapses: {synapses}")
    click.echo(f"Avg resonance: {avg_res:.2f}")


if __name__ == "__main__":
    main()
