"""Phase 1 smoke check.

Verifies the app container can reach both backing services before any of the
real RAG pipeline exists. Phases 2-6 replace this with the LangGraph workflow.
"""

import os
import sys

import psycopg
import requests

DATABASE_URL = os.environ["DATABASE_URL"]
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def check_postgres() -> None:
    # langchain-postgres uses a SQLAlchemy-style URL; psycopg wants a plain DSN.
    dsn = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            version = cur.fetchone()[0]
    print(f"postgres ok - pgvector {version}")


def check_ollama() -> None:
    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30)
    resp.raise_for_status()
    models = sorted(m["name"] for m in resp.json().get("models", []))
    print(f"ollama ok - models: {', '.join(models) or 'none'}")

    required = {"llama3.2:3b", "nomic-embed-text:latest"}
    missing = required - set(models)
    if missing:
        print(f"WARNING: expected models not pulled: {', '.join(sorted(missing))}")


def check_imports() -> None:
    import langchain  # noqa: F401
    import langgraph  # noqa: F401

    print("imports ok - langchain, langgraph")


def main() -> int:
    checks = [check_imports, check_postgres, check_ollama]
    failed = False
    for check in checks:
        try:
            check()
        except Exception as exc:
            print(f"FAIL {check.__name__}: {exc}")
            failed = True
    print("smoke check failed" if failed else "smoke check passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
