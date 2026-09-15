<#
    Windows shim for the Makefile targets, for machines without GNU make.
    Usage:  .\make.ps1 setup | dev | test | lint | seed | corpus | compose-up
#>
param([Parameter(Position = 0)][string]$Target = "help",
      [Parameter(Position = 1)][string]$Arg)

$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Require-Venv {
    if (-not (Test-Path $Py)) {
        throw "No virtualenv found. Run: .\make.ps1 setup"
    }
}

switch ($Target) {
    "setup" {
        python -m venv .venv
        & $Py -m pip install --upgrade pip
        & $Py -m pip install -r backend/requirements.txt
    }
    { $_ -in "dev", "api" } {
        Require-Venv
        & $Py -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend
    }
    "web" { Push-Location frontend; npm run dev; Pop-Location }
    "test" { Require-Venv; & $Py -m pytest backend/tests -q }
    "lint" {
        Require-Venv
        & $Py -m ruff check backend
        & $Py -m ruff format --check backend
    }
    "seed" { Require-Venv; & $Py scripts/seed_demo.py }
    "samples" { Require-Venv; & $Py scripts/make_sample_documents.py }
    "index" { Require-Venv; & $Py scripts/build_index.py --reset }
    "search" {
        Require-Venv
        if (-not $Arg) { throw 'Usage: .\make.ps1 search "lock-in period"' }
        & $Py scripts/build_index.py --samples-only --search $Arg
    }
    "parse" {
        Require-Venv
        if (-not $Arg) { throw "Usage: .\make.ps1 parse <path-to.pdf>" }
        & $Py scripts/parse_document.py $Arg --summary
    }
    "corpus" { Require-Venv; & $Py scripts/fetch_rbi_corpus.py }
    "compose-up" { docker compose up --build }
    "compose-down" { docker compose down -v }
    default {
        Write-Host "Targets: setup, dev, web, test, lint, seed, samples, parse <file>, index, search <query>, corpus, compose-up, compose-down"
    }
}
