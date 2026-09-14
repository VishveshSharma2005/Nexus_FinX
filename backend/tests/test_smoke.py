"""Phase 0 smoke tests: the skeleton boots and the architectural seams hold."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

BACKEND_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_resolved_backends(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body["index_backend"] == "local"
    assert "llm_model" in body


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


# The two rules in core/contracts.py are only real if something enforces them.
# These tests fail the build the moment a concrete parser or a vendor SDK leaks
# into a layer that is supposed to depend on the interface.

VENDOR_SDKS = {"openai", "anthropic", "google", "cohere", "mistralai", "ollama", "httpx"}


def _python_files(relative: str) -> list[Path]:
    return sorted((BACKEND_ROOT / relative).rglob("*.py"))


def test_rag_layer_never_imports_a_vendor_sdk() -> None:
    offenders: list[str] = []
    for path in _python_files("app/rag"):
        for module in _imported_modules(path):
            if module.split(".")[0] in VENDOR_SDKS:
                offenders.append(f"{path.relative_to(BACKEND_ROOT)} imports {module}")
    assert not offenders, (
        "app.rag must depend on the LLMProvider interface, not a vendor SDK: "
        + "; ".join(offenders)
    )


def test_only_the_parser_package_imports_a_concrete_parser() -> None:
    offenders: list[str] = []
    for path in _python_files("app"):
        relative = path.relative_to(BACKEND_ROOT).as_posix()
        if relative.startswith("app/parsers/") or relative == "app/core/deps.py":
            continue
        for module in _imported_modules(path):
            if module.startswith("app.parsers."):
                offenders.append(f"{relative} imports {module}")
    assert not offenders, (
        "Depend on the DocumentParser interface, not a concrete parser: " + "; ".join(offenders)
    )
