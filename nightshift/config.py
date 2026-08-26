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
JUDGE_MAX_TOKENS = 900
JUDGE_TEMPERATURE = 0.0
JUDGE_CTX = 32768
EPISODE_CHAR_BUDGET = 24000   # keep an episode comfortably inside ctx

# --- Ingest proxy ---
PROXY_PORT = int(os.environ.get("NIGHTSHIFT_PROXY_PORT", "8788"))
UPSTREAM_BASE = os.environ.get("NIGHTSHIFT_UPSTREAM", "https://api.openai.com")

# --- llama.cpp judge container ---
LLAMA_IMAGE = "dispersednetwork/llama.cpp_accesstoken_preload"
LLAMA_N_PARALLEL = 4      # concurrent slots per GPU — the throughput lever
LLAMA_CTX = 32768         # split across slots: 8k per slot, episodes are ~6k
LLAMA_ENV = {
    "LLAMA_ARG_PORT": "8080",
    "LLAMA_ARG_HOST": "127.0.0.1",
    "LLAMA_ARG_MODELS_DIR": "/opt/dispersedworker/models/",
    "LLAMA_ARG_SPEC_DRAFT_N_MAX": "2",
    "LLAMA_ARG_CACHE_TYPE_K": "q4_0",
    "LLAMA_ARG_CACHE_TYPE_V": "q4_0",
    "LLAMA_ARG_FLASH_ATTN": "1",
}
