import os
from dataclasses import dataclass

FALSY = {"0", "false", "no", "off"}


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in FALSY


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag_db"
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_device: str = "cuda"
    rerank_max_length: int = 512
    rerank_enabled: bool = True
    citations_enabled: bool = True
    history_enabled: bool = True
    tracing_enabled: bool = False
    fetch_k: int = 20
    retrieve_k: int = 5
    num_ctx: int = 8192
    relevance_floor: float = 0.56
    relevance_ratio: float = 0.9
    rrf_k: int = 60
    tsvector_config: str = "english"
    embed_batch_size: int = 512
    embed_workers: int = 4
    embed_retries: int = 3
    api_host: str = "127.0.0.1"
    api_port: int = 8000


def get_settings() -> Settings:
    return Settings(
        database_url=os.environ.get("DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5432/rag_db"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        embed_model=os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text"),
        rerank_model=os.environ.get("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rerank_device=os.environ.get("RAG_RERANK_DEVICE", "cuda"),
        rerank_max_length=int(os.environ.get("RAG_RERANK_MAX_LENGTH", "512")),
        rerank_enabled=env_flag("RAG_RERANK", default=True),
        citations_enabled=env_flag("RAG_CITATIONS", default=True),
        history_enabled=env_flag("RAG_HISTORY", default=True),
        tracing_enabled=env_flag("RAG_TRACE", default=False),
        fetch_k=int(os.environ.get("RAG_FETCH_K", "20")),
        retrieve_k=int(os.environ.get("RAG_RETRIEVE_K", "5")),
        num_ctx=int(os.environ.get("RAG_NUM_CTX", "8192")),
        relevance_floor=float(os.environ.get("RAG_RELEVANCE_FLOOR", "0.56")),
        relevance_ratio=float(os.environ.get("RAG_RELEVANCE_RATIO", "0.9")),
        rrf_k=int(os.environ.get("RAG_RRF_K", "60")),
        tsvector_config=os.environ.get("RAG_TSVECTOR_CONFIG", "english"),
        embed_batch_size=int(os.environ.get("RAG_EMBED_BATCH", "512")),
        embed_workers=int(os.environ.get("RAG_EMBED_WORKERS", "4")),
        embed_retries=int(os.environ.get("RAG_EMBED_RETRIES", "3")),
        api_host=os.environ.get("RAG_API_HOST", "127.0.0.1"),
        api_port=int(os.environ.get("RAG_API_PORT", "8000")),
    )
