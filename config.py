"""Configuration for the war prediction pipeline."""
import json
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
ARTICLES_PATH = PROJECT_ROOT.parent / "war-prediction-LLMs" / "data" / "processed" / "articles.json"
DATASET_PATH = PROJECT_ROOT / "test_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Load API keys from sibling project config
_config_path = PROJECT_ROOT.parent / "war-prediction-LLMs" / "config.json"
with open(_config_path) as f:
    _secrets = json.load(f)

OPENROUTER_API_KEY = _secrets["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models to evaluate
MODELS = [
    "openai/gpt-5.3-chat",
    "qwen/qwen3.5-35b-a3b",
    "google/gemini-3.1-flash-lite-preview",
    "anthropic/claude-sonnet-4.6",
    "moonshotai/kimi-k2.5",
    "minimax/minimax-m2.5",
]

# Context limits
MAX_ARTICLES = 150
MAX_CONTEXT_CHARS = 30_000
MAX_BODY_CHARS = 500

# API settings
API_TEMPERATURE = 0.3
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2
CALL_DELAY = 1.5  # seconds between API calls
