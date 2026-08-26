"""Central config. Everything tunable lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRACES = DATA / "traces"      # accumulation: one JSONL per day
RUNS = DATA / "runs"          # batch outputs
for _d in (DATA, TRACES, RUNS):
    _d.mkdir(parents=True, exist_ok=True)

# --- Dispersed ---
LLAMACPP_RECIPE_UUID = "01a0114e-e7f8-4e76-a07a-f09fca41e2be"
GPU_5090_UUID = "0198ac1c-7063-4be8-bb94-57a5bd83548e"
FLEET_SIZE = int(os.environ.get("NIGHTSHIFT_FLEET_SIZE", "8"))
CONTAINER_PORT = 5000
JOB_TITLE_PREFIX = "nightshift-judge"

# generous: image pull + model load on a cold node
MAX_TIMEOUT_START_MS = 30 * 60 * 1000
MAX_TIMEOUT_ASSIGN_MS = 10 * 60 * 1000

# --- Judge ---
JUDGE_MAX_TOKENS = 700
JUDGE_TEMPERATURE = 0.0
JUDGE_CTX = 32768
EPISODE_CHAR_BUDGET = 24000   # keep an episode comfortably inside ctx

# --- Ingest proxy ---
PROXY_PORT = int(os.environ.get("NIGHTSHIFT_PROXY_PORT", "8788"))
UPSTREAM_BASE = os.environ.get("NIGHTSHIFT_UPSTREAM", "https://api.openai.com")
