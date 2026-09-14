"""Application settings.

Model choice, index backend and every tuning knob resolve here and nowhere
else, so that swapping a model or a database is an environment change rather
than a code change.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class IndexBackend(StrEnum):
    PGVECTOR = "pgvector"
    LOCAL = "local"


class LLMProviderName(StrEnum):
    NEMOTRON = "nemotron"
    FAKE = "fake"


class EmbeddingProviderName(StrEnum):
    NVIDIA = "nvidia"
    LOCAL = "local"
    FAKE = "fake"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        # Our own settings are FINX_-prefixed; vendor keys such as NVIDIA_API_KEY
        # are read under their conventional names via explicit aliases below.
        env_prefix="FINX_",
    )

    # --- Application --------------------------------------------------------
    env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Vector index -------------------------------------------------------
    index: IndexBackend = IndexBackend.LOCAL
    database_url: str = "postgresql+psycopg://finx:finx@localhost:5432/finx"
    local_index_path: Path = REPO_ROOT / "data" / "finx_local.sqlite3"

    # --- LLM ----------------------------------------------------------------
    llm_provider: LLMProviderName = LLMProviderName.FAKE
    llm_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")

    # --- Embeddings ---------------------------------------------------------
    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.FAKE
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    embedding_dim: int = 1024

    # --- Retrieval / evidence gate -----------------------------------------
    retrieval_top_k: int = 24
    rerank_top_n: int = 8
    evidence_min_score: float = 0.35
    evidence_min_passages: int = 2

    # --- Parser -------------------------------------------------------------
    ocr_enabled: bool = True
    tesseract_cmd: str = ""

    # --- Corpus paths -------------------------------------------------------
    corpus_dir: Path = REPO_ROOT / "corpus"

    @property
    def rbi_corpus_dir(self) -> Path:
        return self.corpus_dir / "rbi"

    @property
    def sample_corpus_dir(self) -> Path:
        return self.corpus_dir / "samples"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
