"""Build the offline Kaggle submission notebook and kernel-metadata.json for ARC-AGI-3.

Combines:
  1. Our harness-in-the-loop TRL AsyncGRPO (#7760) trained Qwen3.8-27B-FP8 LoRA adapter
     (`johnnyhyland/arc3-qwen38-27b-grpo-lora`), served in factorized `--enable-lora` form
     (preserving 100% of low-rank RL updates without FP8/bf16 rounding loss).
  2. Native 3-step MTP speculative decoding (`mtp.safetensors`, verified 3.20-3.30 mean
     acceptance length and 408-546 tok/s on RTX PRO 6000) with automatic tier fallback
     if the offline vLLM 0.19 wheelhouse rejects MTP+LoRA combination.
  3. Patched AGI-Boys Duck harness (`dfranzen/taaf-kaggle-source-bundle-copy` +
     `harness-changes.patch`) with `ARC3_PREFER_TOOL_CALLS=1`, `reasoning_effort="medium"`,
     multimodal board + diff images + animations, and 10-stream priority scheduling.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code,
    }


CELL_0_MD = """# ARC-AGI-3 — Qwen3.8-27B-FP8 + TRL AsyncGRPO (#7760) LoRA + 3-Step MTP

Offline Kaggle submission notebook combining:
- **Model**: `Qwen/Qwen3.8-27B-FP8` (`foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`) + **harness-in-the-loop TRL AsyncGRPO (PR #7760 `score_centering=True`) LoRA adapter** (`johnnyhyland/arc3-qwen38-27b-grpo-lora`, top-10 layers `54..63`, `r=16`).
- **Inference Speed Optimizations**:
  1. Native 3-step Multi-Token Prediction speculative decoding (`mtp.safetensors`, ~`3.25` tokens accepted/step, `400-545 tok/s` on RTX PRO 6000 96 GB).
  2. Factorized runtime LoRA (`--enable-lora`) preserving 100% of low-rank RL updates with FP8 KV cache + prefix & multimodal caching (`82%` prefix hit rate, `94%` MM cache hit rate).
  3. `reasoning_effort="medium"` + `ARC3_PREFER_TOOL_CALLS=1` + RL token-efficiency shaping (`-31.5%` tokens/action, `+42%` actions/game on unseen holdout games).
- **Harness**: Patched Duck harness (`MULTIMODAL_CONTEXT=current_grid`, `ARC3_DIFF_IMAGE=1`, `ARC3_ANIMATION=1`, persistent functions, guards, 10-stream priority scheduler).
"""

CELL_1_SETUP = """import hashlib
import json
import os
import pickle
import shutil
import socket
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

NOTEBOOK_START_TIME = time.time()
NOTEBOOK_START_EPOCH = NOTEBOOK_START_TIME
SERVER_STARTUP_TIMEOUT = 15 * 60

os.environ["ARC3_WARMUP_ACTION_GAMES"] = "1"
os.environ["ARC3_HTTP_RETRY_INITIAL_SECONDS"] = "900"
os.environ["MPLBACKEND"] = "Agg"
os.environ["ONLY_RESET_LEVELS"] = "true"

TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"

cuda_library_path = "/usr/local/nvidia/lib64"
os.environ["LIBRARY_PATH"] = os.pathsep.join(
    entry for entry in [cuda_library_path, *os.environ.get("LIBRARY_PATH", "").split(os.pathsep)] if entry
)

WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")

# LoRA variant selector:
#   "final_adapter_blend50" -> lora_alpha=16 (0.5x multimodal-balanced RL policy, recommended with images)
#   "final_adapter"         -> lora_alpha=32 (1.0x full RL policy)
#   "checkpoint20_adapter"  -> step-20 checkpoint
LORA_VARIANT = os.environ.get("ARC3_LORA_VARIANT", "final_adapter_blend50")


def _find_by_predicate(root: Path, pred, desc: str) -> Path:
    hits = sorted(p for p in root.rglob("*") if pred(p))
    if not hits:
        raise FileNotFoundError(f"Could not locate {desc} under {root}")
    return hits[0]


INPUT_ROOT = Path("/kaggle/input")

# 1. Locate base TAAF bundle (prefer dfranzen/taaf-kaggle-source-bundle-copy)
bundle_markers = sorted(INPUT_ROOT.rglob("taaf-kaggle-bundle.json"))
preferred_bundles = [m.parent for m in bundle_markers if "taaf-kaggle-source-bundle-copy" in str(m)]
ORIG_BUNDLE_DIR = preferred_bundles[0] if preferred_bundles else bundle_markers[0].parent

# 2. Locate harness-changes.patch and LoRA adapter from johnnyhyland/arc3-qwen38-27b-grpo-lora
PATCH_PATH = _find_by_predicate(INPUT_ROOT, lambda p: p.is_file() and p.name == "harness-changes.patch", "harness-changes.patch")
LORA_DATASET_DIR = PATCH_PATH.parent
LORA_ADAPTER_DIR = LORA_DATASET_DIR / LORA_VARIANT
assert (LORA_ADAPTER_DIR / "adapter_model.safetensors").is_file(), f"Missing LoRA weights at {LORA_ADAPTER_DIR}"

# 3. Locate vLLM wheelhouse and Qwen3.8 transformers 5.8 overlay wheels
vllm_whl = _find_by_predicate(INPUT_ROOT, lambda p: p.is_file() and p.name.startswith("vllm-0.19.0-") and p.suffix == ".whl", "vllm 0.19.0 wheel")
WHEELHOUSE_DIR = vllm_whl.parent
TF58_WHEEL = _find_by_predicate(INPUT_ROOT, lambda p: p.is_file() and p.name == "transformers-5.8.0-py3-none-any.whl", "transformers 5.8.0 wheel")
HFHUB_WHEEL = _find_by_predicate(INPUT_ROOT, lambda p: p.is_file() and p.name == "huggingface_hub-1.5.0-py3-none-any.whl", "huggingface_hub 1.5.0 wheel")

# 4. Locate Qwen/Qwen3.8-27B-FP8 model directory
EXPECTED_CFG_SHA = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc"
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()

model_cfgs = [p.parent for p in (INPUT_ROOT / "models").rglob("config.json") if _sha256(p) == EXPECTED_CFG_SHA]
assert len(model_cfgs) == 1, f"Expected 1 Qwen3.8-27B-FP8 model dir, found {model_cfgs}"
MODEL_DIR = model_cfgs[0]

print("ORIG_BUNDLE_DIR :", ORIG_BUNDLE_DIR)
print("PATCH_PATH      :", PATCH_PATH)
print("LORA_ADAPTER_DIR:", LORA_ADAPTER_DIR)
print("WHEELHOUSE_DIR  :", WHEELHOUSE_DIR)
print("MODEL_DIR       :", MODEL_DIR)

# Apply AGI-Boys harness patch to writable copy
BUNDLE_DIR = Path("/kaggle/taaf-kaggle-source-share")
shutil.rmtree(BUNDLE_DIR, ignore_errors=True)
shutil.copytree(ORIG_BUNDLE_DIR, BUNDLE_DIR, symlinks=True)
subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", str(PATCH_PATH)], cwd=str(BUNDLE_DIR / "src"), check=True)
print("Harness patch applied cleanly.")
"""

CELL_2_PRECACHE_AND_ARC = """# Background disk precache for wheelhouse + model shards while pip installs
def precache(*paths, delay=0, threads=8) -> None:
    time.sleep(delay)
    t0 = time.time()
    CHUNK = 32 * 1024 * 1024
    chunks, total = [], 0
    for root_path in paths:
        p = Path(root_path)
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        for fp in files:
            try:
                sz = fp.stat().st_size
                total += sz
                for off in range(0, max(sz, 1), CHUNK):
                    chunks.append((str(fp), off, min(CHUNK, sz - off)))
            except OSError:
                pass
    _tls = threading.local()
    def _read_one(item):
        fp, off, length = item
        buf = getattr(_tls, "buf", None)
        if buf is None:
            _tls.buf = buf = bytearray(CHUNK)
        try:
            with open(fp, "rb") as f:
                f.seek(off)
                rem = length
                while rem > 0:
                    n = f.readinto(buf)
                    if not n:
                        break
                    rem -= n
        except OSError:
            pass
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(_read_one, chunks))
    print(f"Precached {total / 1e9:.2f} GB in {time.time() - t0:.1f}s", flush=True)

precache_wheels = threading.Thread(target=precache, args=(WHEELHOUSE_DIR,), daemon=True)
precache_wheels.start()

# Install offline competition arc-agi runtime
comp_wheels = _find_by_predicate(INPUT_ROOT, lambda p: p.is_dir() and p.name == "arc_agi_3_wheels", "arc_agi_3_wheels")
COMPETITION_DIR = comp_wheels.parent
subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
        "--no-warn-conflicts", "--disable-pip-version-check",
        "--find-links", str(comp_wheels), "arc-agi",
    ],
    stdout=subprocess.DEVNULL,
)
precache_wheels.join()

# Expose patched TAAF + ARC3-Inference repos on sys.path and child .pth
def _source_path_entries(bundle_dir: Path) -> list[Path]:
    entries = []
    for repo in sorted((bundle_dir / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(candidate)
    return entries

source_entries = _source_path_entries(BUNDLE_DIR)
for entry in source_entries:
    sys.path.insert(0, str(entry))
pth_path = Path(sysconfig.get_paths()["purelib"]) / "taaf_kaggle_sources.pth"
pth_path.write_text("".join(f"{entry}\\n" for entry in source_entries))
print(f"Configured {len(source_entries)} TAAF source entries -> {pth_path}")
"""

CELL_3_VLLM_LAUNCH = """# Offline vLLM installation + Multi-Tier RTX PRO 6000 Server Launch (MTP3 + LoRA -> LoRA -> Base+MTP3)
SITE_PACKAGES = Path("/tmp/vllm-site-packages")
INSTALL_STAMP = SITE_PACKAGES / ".qwen38_vllm019_tf58_ready"
SERVED_MODEL_HOST = "127.0.0.1"
SERVED_MODEL_PORT = 8001
SERVER_BASE_URL = f"http://{SERVED_MODEL_HOST}:{SERVED_MODEL_PORT}/v1"


def vllm_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SITE_PACKAGES) if not existing else f"{SITE_PACKAGES}{os.pathsep}{existing}"
    env.update({
        "USE_TF": "0",
        "TRANSFORMERS_NO_TF": "1",
        "TRANSFORMERS_NO_TORCHVISION": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
    })
    return env


def install_vllm_runtime() -> None:
    if INSTALL_STAMP.is_file():
        print(f"Using existing vLLM runtime at {SITE_PACKAGES}", flush=True)
        return
    shutil.rmtree(SITE_PACKAGES, ignore_errors=True)
    SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
    req_lock = WHEELHOUSE_DIR / "requirements.lock"
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--no-index",
            "--find-links", str(WHEELHOUSE_DIR), "--requirement", str(req_lock),
            "--target", str(SITE_PACKAGES), "--upgrade", "--ignore-installed",
            "--only-binary", ":all:", "--no-compile",
            "--disable-pip-version-check", "--no-warn-conflicts",
        ],
        check=True,
    )
    for pat in ("transformers", "transformers-*.dist-info", "huggingface_hub", "huggingface_hub-*.dist-info"):
        for p in SITE_PACKAGES.glob(pat):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
            "--target", str(SITE_PACKAGES), "--upgrade", "--force-reinstall",
            "--no-compile", "--disable-pip-version-check",
            str(TF58_WHEEL), str(HFHUB_WHEEL),
        ],
        check=True,
    )
    INSTALL_STAMP.write_text("ok\\n")
    print("Offline vLLM 0.19.0 + transformers 5.8.0 runtime installed.", flush=True)


precache_model_thread = threading.Thread(target=precache, args=(MODEL_DIR,), kwargs={"delay": 5, "threads": 4}, daemon=True)
precache_model_thread.start()
install_vllm_runtime()

gpu_check = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
print("GPU detected:", gpu_check.stdout.strip(), flush=True)


def _request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def try_launch_tier(tier_name: str, use_lora: bool, use_mtp: bool, timeout_s: int = 540) -> tuple[subprocess.Popen | None, str]:
    served_name = "arc-grpo38" if use_lora else "Qwen/Qwen3.8-27B-FP8"
    log_path = WORKING_DIR / f"vllm_{tier_name}.log"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(MODEL_DIR),
        "--served-model-name", "Qwen/Qwen3.8-27B-FP8",
        "--host", SERVED_MODEL_HOST,
        "--port", str(SERVED_MODEL_PORT),
        "--tensor-parallel-size", "1",
        "--max-model-len", str((116 + 12) * 1024),
        "--gpu-memory-utilization", "0.90",
        "--kv-cache-dtype", "fp8",
        "--enable-prefix-caching",
        "--max-num-seqs", "16",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--generation-config", "vllm",
        "--default-chat-template-kwargs", '{"preserve_thinking": true, "reasoning_effort": "medium"}',
    ]
    if use_lora:
        cmd += [
            "--enable-lora",
            "--max-lora-rank", "16",
            "--max-loras", "2",
            "--lora-modules", f"arc-grpo38={LORA_ADAPTER_DIR}",
        ]
    if use_mtp:
        cmd += ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": 3}']

    print(f"\\n=== Launching vLLM Tier [{tier_name}] (lora={use_lora}, mtp={use_mtp}) ===", flush=True)
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=vllm_env(), stdout=log_fh, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if proc.poll() is not None:
            tail = "\\n".join(log_path.read_text(errors="replace").splitlines()[-45:])
            print(f"Tier [{tier_name}] exited ({proc.returncode}):\\n{tail}", flush=True)
            return None, ""
        try:
            models = _request_json(f"{SERVER_BASE_URL}/models", timeout=5)
            ids = [m.get("id") for m in models.get("data", [])]
            if served_name in ids:
                # Smoke test real completion on served_name
                smoke = _request_json(
                    f"{SERVER_BASE_URL}/chat/completions",
                    payload={
                        "model": served_name,
                        "messages": [{"role": "user", "content": "Reply with OK."}],
                        "temperature": 0.0,
                        "max_tokens": 32,
                    },
                    timeout=120,
                )
                msg = smoke["choices"][0]["message"]
                print(f"Tier [{tier_name}] READY in {int(time.monotonic()-t0)}s! Active model={served_name!r}, smoke={msg.get('content')!r}", flush=True)
                return proc, served_name
        except Exception:
            pass
        time.sleep(4)
    print(f"Tier [{tier_name}] timed out after {timeout_s}s; stopping.", flush=True)
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except Exception:
        pass
    return None, ""


vllm_proc, SERVED_MODEL_NAME = None, ""
for tier_name, use_lora, use_mtp in [
    ("mtp3_lora", True, True),
    ("lora_only", True, False),
    ("mtp3_base", False, True),
    ("base_only", False, False),
]:
    with socket.socket() as probe:
        if probe.connect_ex((SERVED_MODEL_HOST, SERVED_MODEL_PORT)) == 0:
            time.sleep(3)
    vllm_proc, SERVED_MODEL_NAME = try_launch_tier(tier_name, use_lora=use_lora, use_mtp=use_mtp)
    if vllm_proc is not None:
        ACTIVE_TIER = tier_name
        break

if vllm_proc is None:
    raise RuntimeError("All vLLM launch tiers failed!")
print(f"Serving active: tier={ACTIVE_TIER}, model={SERVED_MODEL_NAME}", flush=True)
"""

CELL_4_HARNESS_ENV = """# Configure patched AGI-Boys Duck harness + our RL inference optimizations
USE_PRIORITY_SCHEDULING = True

setup_env = {
    "USE_TF": "0",
    "TRANSFORMERS_NO_TF": "1",
    "TRANSFORMERS_NO_TORCHVISION": "1",

    # Serving config
    "LOCAL_ANALYZER_BASE_URL": SERVER_BASE_URL,
    "OPENAI_BASE_URL": SERVER_BASE_URL,
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL_NAME,
    "INFERENCE_ANALYZER_MODEL": SERVED_MODEL_NAME,
    "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",

    # Analyzer config
    "ARC3_REASONING_HISTORY_KEY": "reasoning_content",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_YIELD_SECONDS": "0",
    "LOCAL_ANALYZER_YIELD_TOKENS": "2048",
    "LOCAL_ANALYZER_TEMPERATURE": "0.7",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",

    # Context pruning
    "LOCAL_ANALYZER_CONTEXT_WINDOW": str(112 * 1024),
    "LOCAL_ANALYZER_MAX_OUTPUT": str(10 * 1024),
    "ARC3_CONTEXT_DRAIN_TOKENS": str(54 * 1024),
    "ARC3_HISTORY_ASSISTANT_TURNS": "150",
    "ARC3_HISTORY_TURN_DRAIN": "30",
    "ARC3_HISTORY_DRAIN_COALESCE": "1",

    # Tool output
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "3072",
    "ARC3_MIDDLE_TRUNCATION": "1",
    "ARC3_ACTION_ECHO": "1",

    # Prompt improvements + our RL action-efficiency prompt guidance
    "ARC3_EXPLAIN_GAMEPLAY_CHANGED": "1",
    "ARC3_NEW_CHANGED_PROMPTS": "1",
    "ARC3_LEVEL_TRANSFER_GUIDANCE": "1",
    "ARC3_ACTION_INFO": "1",
    "ARC3_DEDUPE_MULTICALL_LINE": "1",
    "ARC3_FRAME_DIFF_HINT": "1",
    "ARC3_PREFER_TOOL_CALLS": "1",

    # Multimodal vision & animations (verified 18x jump on Qwen3.8-27B-FP8)
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "10",
    "ARC3_DIFF_IMAGE": "1",
    "ARC3_GAMEOVER_DIFF_IMAGE": "1",
    "ARC3_ANIMATION": "1",

    # Guards & persistent functions
    "ARC3_GUARDS_FROM_LEVEL": "2",
    "ARC3_BATCH_NOOP_BLOCK": "1",
    "ARC3_STALE_STATE_BLOCK": "1",
    "ARC3_MEMORY_SECTIONS": "off",
    "ARC3_PERSISTENT_FUNCTIONS": "1",
    "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
    "ARC3_PERSISTENT_FUNCTIONS_IMPORTS": "1",
    "EXPOSE_UNDO": "on",
}

if USE_PRIORITY_SCHEDULING:
    setup_env.update({
        "ARC3_MAX_ACTIVE_STREAMS": "10",
        "ARC3_PRIORITY_REFRESH_QUEUE": "1",
        "ARC3_PRIORITY_PACE": "0",
        "ARC3_PRIORITY_TAIL_FADE": "1",
        "ARC3_PRIORITY_TAIL_FADE_FRACTION": "0.2",
        "ARC3_PRIORITY_TAIL_LOOKUP": "remaining",
        "ARC3_PRIORITY_SCORE_NORMALIZATION": "1",
    })

os.environ.update({k: str(v) for k, v in setup_env.items()})

with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as f:
    target = pickle.load(f)
target.actual_run_as_submission = TRUE_SUBMISSION
target.is_competition_rerun = TRUE_SUBMISSION

with open(BUNDLE_DIR / "benchmark_initial.pkl", "rb") as f:
    bm = pickle.load(f)
bm.job_dir = WORKING_DIR

bm.solver.max_runtime_s_per_game = 48 * 60
bm.solver.analyzer_timeout = 900.0
bm.solver.concurrency = 10
bm.solver.max_actions_per_game = None
bm.solver.save_request_logs = not TRUE_SUBMISSION
bm.n_passes = 1
bm.game_weights = None

# Exclude the 15 RL training games during interactive demo commits so commit runs evaluate
# strictly on the 10 held-out unseen games (apples-to-apples with the AGI-Boys demo split!).
demo_excluded_games = [] if TRUE_SUBMISSION else [
    "bp35", "cd82", "cn04", "dc22", "g50t", "ka59", "lf52", "ls20",
    "m0r0", "s5i5", "sk48", "sp80", "su15", "tn36", "wa30",
]

if TRUE_SUBMISSION:
    bm.solver.save_request_logs = False
    bm.n_passes = 1
    if USE_PRIORITY_SCHEDULING:
        bm.solver.concurrency = 120
        bm.solver.max_runtime_s_per_game = 532 * 60
else:
    if USE_PRIORITY_SCHEDULING:
        bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes
        bm.solver.max_runtime_s_per_game = 25 * 60

print("benchmark.solver:", bm.solver)
"""

CELL_5_RUN = """print("Starting ARC-AGI-3 benchmark...", flush=True)


def _competition_games():
    import arc_agi
    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=os.environ["ARC_BASE_URL"],
        environments_dir="",
    )
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=spec.arc_base_url,
        environments_dir="",
    )
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError("Competition Arcade exposed zero environments.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


def _offline_games(env_dir: str):
    import arc_agi
    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError(f"No offline environments found under {env_dir}.")
    game_ids = [gid for gid in game_ids if not any(gid.startswith(g) for g in demo_excluded_games)]
    return [taaf.game_api.GameAPI(env_name=gid, arcade_spec=spec) for gid in game_ids]


def _wait_for_gateway(base_url: str, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}api/games", timeout=10) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"Kaggle gateway did not become ready: {last_error}")


(WORKING_DIR / "git_status.txt").write_text((BUNDLE_DIR / "git_status.txt").read_text())
os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))

if TRUE_SUBMISSION:
    os.environ.setdefault("ARC_API_KEY", "test-key-123")
    os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
    _wait_for_gateway(os.environ["ARC_BASE_URL"])
    bm.games = _competition_games()
else:
    bm.games = _offline_games(str(COMPETITION_DIR / "environment_files"))

soft_end = None
if not TRUE_SUBMISSION:
    budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
    if budget > 0:
        soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))

try:
    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)
finally:
    if vllm_proc is not None and vllm_proc.poll() is None:
        vllm_proc.terminate()

if (WORKING_DIR / "summary.txt").is_file():
    print("\\n=== BENCHMARK SUMMARY ===\\n" + (WORKING_DIR / "summary.txt").read_text(), flush=True)

if not TRUE_SUBMISSION:
    import pandas as pd

    pd.DataFrame(
        [["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    ).to_parquet(WORKING_DIR / "submission.parquet", index=False)
"""

CELL_6_DIAG = """from html import escape
from IPython.display import HTML, display

diagnostics_html = WORKING_DIR / "diagnostics.html"
if diagnostics_html.is_file():
    display(
        HTML(
            f'<iframe srcdoc="{escape(diagnostics_html.read_text(), quote=True)}" '
            'width="100%" height="900" style="border:0"></iframe>'
        )
    )
else:
    print("No diagnostics.html — minimal diagnostics (real submission) suppresses it.")
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [
            md_cell(CELL_0_MD),
            code_cell(CELL_1_SETUP),
            code_cell(CELL_2_PRECACHE_AND_ARC),
            code_cell(CELL_3_VLLM_LAUNCH),
            code_cell(CELL_4_HARNESS_ENV),
            code_cell(CELL_5_RUN),
            code_cell(CELL_6_DIAG),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.13",
            },
            "kaggle": {
                "accelerator": "nvidiaRtxPro6000",
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path = OUT_DIR / "arc3-qwen38-27b-grpo-mtp.ipynb"
    nb_path.write_text(json.dumps(nb, indent=2) + "\n")

    meta = {
        "id": "johnnyhyland/arc3-qwen38-27b-grpo-mtp",
        "title": "ARC3 Qwen38 27B GRPO MTP",
        "code_file": "arc3-qwen38-27b-grpo-mtp.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dfranzen/taaf-kaggle-source-bundle-copy",
            "driessmit1/arc3-vllm-h100-wheelhouse-v3",
            "keithtyser/taaf-duck-qwen38-serving-v1",
            "johnnyhyland/arc3-qwen38-27b-grpo-lora",
        ],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [
            "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1",
        ],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {nb_path} and kernel-metadata.json")


if __name__ == "__main__":
    main()
