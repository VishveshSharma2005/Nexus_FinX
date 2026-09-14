from __future__ import annotations

import pytest

from app.config import IndexBackend, Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings isolated from the developer's own .env and data directory."""
    return Settings(
        _env_file=None,
        env="test",
        index=IndexBackend.LOCAL,
        local_index_path=tmp_path / "test_index.sqlite3",
    )
