# Agentic Insight v2 Benchmark Runner
# This script helps reproduce the official benchmark results.
# Must be run from the repository root directory.
#
# Usage:
#   Single demo query:   .\projects\deep_research\v2\run_benchmark.ps1
#   Full benchmark:      $env:DR_BENCH_ROOT="/path/to/bench"; .\projects\deep_research\v2\run_benchmark.ps1

Write-Host "=========================================" -ForegroundColor Green
Write-Host "Agentic Insight v2 Benchmark Runner" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""

# Locate Python executable early for both modes
$pythonBin = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonBin = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonBin = "python3"
} else {
    Write-Host "Error: Neither 'python' nor 'python3' is available in PATH." -ForegroundColor Red
    exit 1
}

# Verify we are at the repository root
if (-not (Test-Path "ms_agent/cli/cli.py")) {
    Write-Host "Error: This script must be run from the repository root directory." -ForegroundColor Red
    Write-Host "  cd /path/to/ms-agent"
    Write-Host "  .\projects\deep_research\v2\run_benchmark.ps1"
    exit 1
}

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "Error: .env file not found in repository root!" -ForegroundColor Red
    Write-Host "Please create .env file by copying .env.example:"
    Write-Host "  Copy-Item projects\deep_research\.env.example .env"
    Write-Host "  # Then edit .env to add your API keys"
    exit 1
}

# Load environment variables from .env file
Write-Host "Loading environment variables from .env..." -ForegroundColor Green
Get-Content ".env" | ForEach-Object {
    $line = $_.Trim()
    if ($line -and !$line.StartsWith('#')) {
        $parts = $line -split '\s*=\s*', 2
        if ($parts.Length -eq 2) {
            $key = $parts[0]
            $value = $parts[1]
            # Remove quotes if present
            if ($value.StartsWith("'") -and $value.EndsWith("'")) {
                $value = $value.Substring(1, $value.Length - 2)
            } elseif ($value.StartsWith('"') -and $value.EndsWith('"')) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# Validate required environment variables
if ([string]::IsNullOrEmpty($env:OPENAI_API_KEY) -or [string]::IsNullOrEmpty($env:OPENAI_BASE_URL)) {
    Write-Host "Error: OPENAI_API_KEY or OPENAI_BASE_URL not set in .env" -ForegroundColor Red
    exit 1
}

# Check for search engine API key
if ([string]::IsNullOrEmpty($env:EXA_API_KEY) -and [string]::IsNullOrEmpty($env:SERPAPI_API_KEY)) {
    Write-Host "Warning: Neither EXA_API_KEY nor SERPAPI_API_KEY is set." -ForegroundColor Yellow
    Write-Host "The system will use arxiv (academic search only)." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Environment variables loaded successfully!" -ForegroundColor Green
Write-Host "  OPENAI_BASE_URL: $env:OPENAI_BASE_URL"
Write-Host "  EXA_API_KEY: $(if (-not [string]::IsNullOrEmpty($env:EXA_API_KEY)) { "✓ Set" } else { "✗ Not set" })"
Write-Host "  SERPAPI_API_KEY: $(if (-not [string]::IsNullOrEmpty($env:SERPAPI_API_KEY)) { "✓ Set" } else { "✗ Not set" })"
Write-Host ""

# Check if DR_BENCH_ROOT is set
if ([string]::IsNullOrEmpty($env:DR_BENCH_ROOT)) {
    Write-Host "Warning: DR_BENCH_ROOT not set." -ForegroundColor Yellow
    Write-Host "Using default benchmark query..." -ForegroundColor Yellow
    Write-Host ""

    # Run a simple benchmark query
    $query = "Provide a comprehensive survey of recent advances in large language models (LLMs), covering key developments in the last 12 months including architecture innovations, training techniques, and real-world applications."
    $outputDir = "output/deep_research/benchmark_run"

    Write-Host "Running benchmark with query:" -ForegroundColor Green
    Write-Host "  \"$query\""
    Write-Host ""
    Write-Host "Output directory: $outputDir" -ForegroundColor Green
    Write-Host ""

    # Run the benchmark
    $env:PYTHONPATH = "."
    & $pythonBin ms_agent/cli/cli.py run `
        --config projects/deep_research/v2/researcher.yaml `
        --query "$query" `
        --trust_remote_code true `
        --output_dir "$outputDir"

    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "Benchmark completed!" -ForegroundColor Green
    Write-Host "Results saved to: $outputDir" -ForegroundColor Green
    Write-Host "Final report: $outputDir/final_report.md" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green

} else {
    Write-Host "DR_BENCH_ROOT detected: $env:DR_BENCH_ROOT" -ForegroundColor Green
    Write-Host "Running full benchmark suite..." -ForegroundColor Yellow
    Write-Host ""

    # Benchmark subprocess tuning (override via env vars if needed)
    if ([string]::IsNullOrEmpty($env:DR_BENCH_POST_FINISH_GRACE_S)) {
        $env:DR_BENCH_POST_FINISH_GRACE_S = "180"
    }
    if ([string]::IsNullOrEmpty($env:DR_BENCH_POST_REPORT_EXIT_GRACE_S)) {
        $env:DR_BENCH_POST_REPORT_EXIT_GRACE_S = "3600"
    }
    if ([string]::IsNullOrEmpty($env:DR_BENCH_REPORT_STABLE_WINDOW_S)) {
        $env:DR_BENCH_REPORT_STABLE_WINDOW_S = "10"
    }
    if ([string]::IsNullOrEmpty($env:DR_BENCH_SUBPROCESS_POLL_INTERVAL_S)) {
        $env:DR_BENCH_SUBPROCESS_POLL_INTERVAL_S = "0.5"
    }
    if ([string]::IsNullOrEmpty($env:DR_BENCH_SUBPROCESS_TERMINATE_TIMEOUT_S)) {
        $env:DR_BENCH_SUBPROCESS_TERMINATE_TIMEOUT_S = "30"
    }
    if ([string]::IsNullOrEmpty($env:DR_BENCH_SUBPROCESS_KILL_TIMEOUT_S)) {
        $env:DR_BENCH_SUBPROCESS_KILL_TIMEOUT_S = "30"
    }

    # Check if DR_BENCH_ROOT exists
    if (-not (Test-Path "$env:DR_BENCH_ROOT" -PathType Container)) {
        Write-Host "Error: DR_BENCH_ROOT directory not found: $env:DR_BENCH_ROOT" -ForegroundColor Red
        exit 1
    }

    # Check if query file exists
    $queryFile = "$env:DR_BENCH_ROOT/data/prompt_data/query.jsonl"
    if (-not (Test-Path "$queryFile")) {
        Write-Host "Error: Query file not found: $queryFile" -ForegroundColor Red
        exit 1
    }

    # Set default values
    if ([string]::IsNullOrEmpty($env:MODEL_NAME)) {
        $env:MODEL_NAME = "ms_deepresearch_v2_benchmark"
    }
    if ([string]::IsNullOrEmpty($env:OUTPUT_JSONL)) {
        $env:OUTPUT_JSONL = "$env:DR_BENCH_ROOT/data/test_data/raw_data/$env:MODEL_NAME.jsonl"
    }
    if ([string]::IsNullOrEmpty($env:WORK_ROOT)) {
        $env:WORK_ROOT = "temp/benchmark_runs"
    }
    if ([string]::IsNullOrEmpty($env:WORKERS)) {
        $env:WORKERS = "2"
    }
    if ([string]::IsNullOrEmpty($env:LIMIT)) {
        $env:LIMIT = "0"
    }

    # Validate numeric inputs early for clearer errors
    if (-not ($env:WORKERS -match "^[0-9]+$") -or [int]$env:WORKERS -lt 1) {
        Write-Host "Error: WORKERS must be a positive integer. Got: $env:WORKERS" -ForegroundColor Red
        exit 1
    }
    if (-not ($env:LIMIT -match "^[0-9]+$")) {
        Write-Host "Error: LIMIT must be a non-negative integer. Got: $env:LIMIT" -ForegroundColor Red
        exit 1
    }

    Write-Host "Configuration:"
    Write-Host "  Query file: $queryFile"
    Write-Host "  Output JSONL: $env:OUTPUT_JSONL"
    Write-Host "  Model name: $env:MODEL_NAME"
    Write-Host "  Work root: $env:WORK_ROOT"
    Write-Host "  Workers: $env:WORKERS"
    Write-Host "  Limit: $env:LIMIT (0 -eq no limit)"
    Write-Host ""

    # Run the full benchmark
    $env:PYTHONPATH = "."
    & $pythonBin projects/deep_research/v2/eval/dr_bench_runner.py `
        --query_file "$queryFile" `
        --output_jsonl "$env:OUTPUT_JSONL" `
        --model_name "$env:MODEL_NAME" `
        --work_root "$env:WORK_ROOT" `
        --limit "$env:LIMIT" `
        --workers "$env:WORKERS" `
        --trust_remote_code

    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "Full benchmark suite completed!" -ForegroundColor Green
    Write-Host "Results saved to: $env:OUTPUT_JSONL" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
}
