"""Configuration for the war prediction pipeline."""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

# Paths
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"

# HuggingFace dataset
HF_DATASET_REPO = "war-forecast-arena/war-forecast-bench"

def _ensure_data_file(filename: str) -> Path:
    """Return local path to a data file, downloading from HF if needed."""
    local = PROJECT_ROOT / filename
    if local.exists():
        return local
    print(f"Downloading {filename} from {HF_DATASET_REPO}...")
    return Path(hf_hub_download(repo_id=HF_DATASET_REPO, filename=filename, repo_type="dataset"))

ARTICLES_PATH = _ensure_data_file("articles_clean.json")
DATASET_PATH = _ensure_data_file("test_dataset.json")

# Load API keys from sibling project config
_config_path = PROJECT_ROOT.parent / "war-prediction-LLMs" / "config.json"
with open(_config_path) as f:
    _secrets = json.load(f)

OPENROUTER_API_KEY = _secrets["OPENROUTER_API_KEY"]
OPENAI_API_KEY = _secrets["OPENAI_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://us.api.openai.com/v1"

# Models to evaluate
MODELS = [
    "openai/gpt-5.4",
    "qwen/qwen3.5-35b-a3b",
    "google/gemini-3.1-flash-lite-preview",
    "anthropic/claude-sonnet-4.6",
    "moonshotai/kimi-k2.5",
    "minimax/minimax-m2.5",
]

# Context limits (models have ≥200K token context)
MAX_ARTICLES = 9999  # effectively no cap
MAX_CONTEXT_CHARS = 480_000  # ~120K tokens, leaves room for output within 128K limit
MAX_BODY_CHARS = 2000

# API settings
API_TEMPERATURE = 0.3
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2
CALL_DELAY = 1.5  # seconds between API calls
