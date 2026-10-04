param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "install",
        "run",
        "test",
        "lint",
        "format",
        "typecheck",
        "lock",
        "sync",
        "docker-build",
        "docker-up",
        "docker-down",
        "docker-logs",
        "clean"
    )]
    [string]$Command
)

$Backend = Join-Path $PSScriptRoot "..\backend"

switch ($Command) {

    "install" {
        Push-Location $Backend
        uv sync
        Pop-Location
    }

    "lock" {
        Push-Location $Backend
        uv lock
        Pop-Location
    }

    "sync" {
        Push-Location $Backend
        uv sync
        Pop-Location
    }

    "run" {
        Push-Location $Backend
        uv run uvicorn app.main:app --reload
        Pop-Location
    }

    "test" {
        Push-Location $Backend
        uv run pytest
        Pop-Location
    }

    "lint" {
        Push-Location $Backend
        uv run ruff check .
        Pop-Location
    }

    "format" {
        Push-Location $Backend
        uv run black .
        Pop-Location
    }

    "typecheck" {
        Push-Location $Backend
        uv run mypy app
        Pop-Location
    }

    "docker-build" {
        docker compose build
    }

    "docker-up" {
        docker compose up
    }

    "docker-down" {
        docker compose down
    }

    "docker-logs" {
        docker compose logs -f
    }

    "clean" {

        Write-Host ""
        Write-Host "Removing Python cache..."
        Write-Host ""

        Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

        Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue

        Write-Host ""
        Write-Host "Removing pytest cache..."
        Write-Host ""

        Remove-Item -Recurse -Force "$Backend\.pytest_cache" -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force "$Backend\.ruff_cache" -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force "$Backend\.mypy_cache" -ErrorAction SilentlyContinue

        Write-Host ""
        Write-Host "Done."
    }
}