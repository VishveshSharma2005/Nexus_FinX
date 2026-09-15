# FinX developer commands.
# Windows without GNU make: use the equivalent `./make.ps1 <target>`.

PY := .venv/Scripts/python.exe
ifeq ($(OS),)
PY := .venv/bin/python
endif

.PHONY: help setup dev api web test lint seed samples parse index search corpus compose-up compose-down clean

help:
	@echo "setup        Create .venv and install backend dependencies"
	@echo "dev          Run API (reload) — uses FINX_INDEX from .env"
	@echo "web          Run the Next.js frontend"
	@echo "test         Run the backend test suite"
	@echo "lint         Ruff check + format check"
	@echo "seed         Build a demo-ready database (synthetic data only)"
	@echo "samples      Generate the synthetic demo documents"
	@echo "parse        Parse one document: make parse FILE=path/to.pdf"
	@echo "index        Build the vector index from corpus + samples"
	@echo "search       Query the index: make search Q=\"lock-in period\""
	@echo "corpus       Fetch the public RBI corpus into corpus/rbi/"
	@echo "compose-up   Full stack via Docker (postgres+pgvector, api, web)"

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r backend/requirements.txt

dev api:
	$(PY) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend

web:
	cd frontend && npm run dev

test:
	$(PY) -m pytest backend/tests -q

lint:
	$(PY) -m ruff check backend
	$(PY) -m ruff format --check backend

seed:
	$(PY) scripts/seed_demo.py

samples:
	$(PY) scripts/make_sample_documents.py

parse:
	$(PY) scripts/parse_document.py $(FILE) --summary

index:
	$(PY) scripts/build_index.py --reset

search:
	$(PY) scripts/build_index.py --samples-only --search "$(Q)"

corpus:
	$(PY) scripts/fetch_rbi_corpus.py

compose-up:
	docker compose up --build

compose-down:
	docker compose down -v

clean:
	rm -rf .venv data .pytest_cache .ruff_cache
