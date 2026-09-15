<#
    Windows shim for the Makefile targets, for machines without GNU make.

    Usage:
        .\make.ps1 setup
        .\make.ps1 test
        .\make.ps1 index
        .\make.ps1 search "is there a lock-in period?"
        .\make.ps1 parse corpus\samples\home_loan_agreement_A.docx
#>
param(
    [Parameter(Position = 0)][string]$Target = "help",
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$Arg = if ($Rest) { $Rest[0] } else { $null }

function Require-Venv {
    if (-not (Test-Path $Py)) {
        throw "No virtualenv found. Run: .\make.ps1 setup"
    }
}

function Invoke-Native {
    <#
        Run an external command and fail only on a non-zero exit code.

        Windows PowerShell treats anything a native program writes to stderr as
        a terminating error under ErrorActionPreference = Stop. pytest and
        PyMuPDF both write harmless notices there -- a SWIG shutdown warning is
        enough to make a passing test run look like a failure -- so the exit
        code is the only thing worth believing.
    #>
    param([Parameter(Mandatory)][scriptblock]$Command)

    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Command
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        throw "Command failed with exit code $code"
    }
}

switch ($Target) {
    "setup" {
        Invoke-Native { python -m venv .venv }
        Invoke-Native { & $Py -m pip install --upgrade pip }
        Invoke-Native { & $Py -m pip install -r backend/requirements.txt }
        Write-Host "`nReady. Next: .\make.ps1 index" -ForegroundColor Green
    }
    { $_ -in "dev", "api" } {
        Require-Venv
        Invoke-Native {
            & $Py -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend
        }
    }
    "web" {
        Push-Location frontend
        try { Invoke-Native { npm run dev } } finally { Pop-Location }
    }
    "test" {
        Require-Venv
        # PyMuPDF's SWIG bindings emit a DeprecationWarning during interpreter
        # shutdown, after pytest has finished and reported. Suppressed by exact
        # message so that warnings from our own code are still visible.
        $env:PYTHONWARNINGS = "ignore:builtin type Swig:DeprecationWarning," +
                              "ignore:builtin type swig:DeprecationWarning"
        Invoke-Native { & $Py -m pytest backend/tests -q }
    }
    "lint" {
        Require-Venv
        Invoke-Native { & $Py -m ruff check backend scripts }
        Invoke-Native { & $Py -m ruff format --check backend scripts }
    }
    "seed" { Require-Venv; Invoke-Native { & $Py scripts/seed_demo.py } }
    "samples" { Require-Venv; Invoke-Native { & $Py scripts/make_sample_documents.py } }
    "index" { Require-Venv; Invoke-Native { & $Py scripts/build_index.py --reset } }
    "search" {
        Require-Venv
        if (-not $Arg) { throw 'Usage: .\make.ps1 search "lock-in period"' }
        Invoke-Native { & $Py scripts/build_index.py --samples-only --search $Arg }
    }
    "parse" {
        Require-Venv
        if (-not $Arg) { throw "Usage: .\make.ps1 parse <path-to-document>" }
        Invoke-Native { & $Py scripts/parse_document.py $Arg --summary }
    }
    "corpus" { Require-Venv; Invoke-Native { & $Py scripts/fetch_rbi_corpus.py } }
    "compose-up" { Invoke-Native { docker compose up --build } }
    "compose-down" { Invoke-Native { docker compose down -v } }
    default {
        Write-Host "Targets:"
        Write-Host "  setup                  Create .venv and install dependencies"
        Write-Host "  test                   Run the backend test suite"
        Write-Host "  lint                   Ruff check + format check"
        Write-Host "  index                  Build the vector index from corpus + samples"
        Write-Host '  search "<question>"    Query the index, printing scored and cited hits'
        Write-Host "  parse <file>           Parse one document to a clause table"
        Write-Host "  samples                Regenerate the synthetic PDF fixtures"
        Write-Host "  dev                    Run the API (health endpoints only until Phase 4)"
        Write-Host "  web                    Run the Next.js frontend (placeholder until Phase 8)"
        Write-Host "  compose-up             Full stack via Docker"
    }
}
