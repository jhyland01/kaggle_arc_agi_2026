"""Harness environment config + subprocess launcher shared by eval and RL rollouts.

Starts from the AGI-Boys Kaggle notebook's ``setup_env`` and changes only what a small model on a
shared GPU needs:
  * text-only (no grid/diff images, no animations): TRL AsyncGRPO cannot train on images yet, and
    the baseline must use the same setting as training to be comparable;
  * smaller context window / output budget (9B model, faster turns);
  * no priority scheduler (one game per harness process).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass
class HarnessConfig:
    bundle_dir: str
    env_dir: str
    context_window: int = 32 * 1024
    max_output: int = 6 * 1024
    drain_tokens: int = 12 * 1024
    yield_tokens: int = 2048
    tool_output_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    max_runtime_s: float = 900.0
    max_actions: int = 0
    max_gen_tokens: int = 0
    prefer_tool_calls: bool = False
    multimodal: bool = False
    python_bin: str = sys.executable

    def to_dict(self) -> dict:
        return asdict(self)


def harness_env(cfg: HarnessConfig, base_url: str, model_name: str) -> dict[str, str]:
    env = os.environ.copy()
    for k in ("CUDA_VISIBLE_DEVICES",):
        env.pop(k, None)
    env.update(
        {
            "USE_TF": "0",
            "TRANSFORMERS_NO_TF": "1",
            "TRANSFORMERS_NO_TORCHVISION": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MPLBACKEND": "Agg",
            "ONLY_RESET_LEVELS": "true",
            "TAAF_RUN_AS_SUBMISSION": "0",
            "TAAF_MINIMAL_DIAGNOSTICS": "1",
            "ARC3_WARMUP_ACTION_GAMES": "0",
            "ARC3_HTTP_RETRY_INITIAL_SECONDS": "30",
            # serving
            "LOCAL_ANALYZER_BASE_URL": base_url,
            "OPENAI_BASE_URL": base_url,
            "LOCAL_ANALYZER_PROVIDER": "vllm",
            "OPENAI_PROVIDER": "vllm",
            "LOCAL_ANALYZER_MODEL_ID": model_name,
            "INFERENCE_ANALYZER_MODEL": model_name,
            "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
            "LOCAL_ANALYZER_API_KEY": "EMPTY",
            "OPENAI_API_KEY": "EMPTY",
            # analyzer
            "ARC3_REASONING_HISTORY_KEY": "reasoning_content",
            "LOCAL_ANALYZER_ENABLE_THINKING": "true",
            "LOCAL_ANALYZER_TOOL_STEPS": "0",
            "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
            "LOCAL_ANALYZER_YIELD_SECONDS": "0",
            "LOCAL_ANALYZER_YIELD_TOKENS": str(cfg.yield_tokens),
            "LOCAL_ANALYZER_TEMPERATURE": str(cfg.temperature),
            "LOCAL_ANALYZER_TOP_P": str(cfg.top_p),
            "LOCAL_ANALYZER_TOP_K": str(cfg.top_k),
            "LOCAL_ANALYZER_TIMEOUT": "900",
            # context pruning (scaled down for a 9B model)
            "LOCAL_ANALYZER_CONTEXT_WINDOW": str(cfg.context_window),
            "LOCAL_ANALYZER_MAX_OUTPUT": str(cfg.max_output),
            "ARC3_CONTEXT_DRAIN_TOKENS": str(cfg.drain_tokens),
            "ARC3_HISTORY_ASSISTANT_TURNS": "150",
            "ARC3_HISTORY_TURN_DRAIN": "30",
            "ARC3_HISTORY_DRAIN_COALESCE": "1",
            # tool output
            "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": str(cfg.tool_output_tokens),
            "ARC3_MIDDLE_TRUNCATION": "1",
            "ARC3_ACTION_ECHO": "1",
            # prompt changes (same as the AGI-Boys notebook)
            "ARC3_EXPLAIN_GAMEPLAY_CHANGED": "1",
            "ARC3_NEW_CHANGED_PROMPTS": "1",
            "ARC3_LEVEL_TRANSFER_GUIDANCE": "1",
            "ARC3_ACTION_INFO": "1",
            "ARC3_DEDUPE_MULTICALL_LINE": "1",
            "ARC3_FRAME_DIFF_HINT": "1",
            "ARC3_PREFER_TOOL_CALLS": "1" if cfg.prefer_tool_calls else "0",
            # text-only during RL (multimodal optional for inference/eval)
            "MULTIMODAL_CONTEXT": "current_grid" if cfg.multimodal else "",
            "MULTIMODAL_UPSCALE": "10",
            "ARC3_DIFF_IMAGE": "1" if cfg.multimodal else "0",
            "ARC3_GAMEOVER_DIFF_IMAGE": "1" if cfg.multimodal else "0",
            "ARC3_ANIMATION": "1" if cfg.multimodal else "0",
            # guards
            "ARC3_GUARDS_FROM_LEVEL": "2",
            "ARC3_BATCH_NOOP_BLOCK": "1",
            "ARC3_STALE_STATE_BLOCK": "1",
            "ARC3_MEMORY_SECTIONS": "off",
            # persistent functions
            "ARC3_PERSISTENT_FUNCTIONS": "1",
            "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
            "ARC3_PERSISTENT_FUNCTIONS_IMPORTS": "1",
            "EXPOSE_UNDO": "on",
            "ARC3_MAX_ACTIVE_STREAMS": "1",
        }
    )
    return env


def _die_with_parent() -> None:
    """Linux: SIGKILL this harness subprocess if its parent (the rollout worker / eval runner) dies, so a killed
    training run cannot leave orphaned games still sending requests to the next run."""
    try:
        import ctypes  # noqa: PLC0415
        import signal  # noqa: PLC0415

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG = 1
    except Exception:  # noqa: BLE001 - not Linux (local mac tests)
        pass


async def run_game_subprocess(
    cfg: HarnessConfig, game: str, base_url: str, model_name: str, job_dir: Path, timeout_s: float | None = None
) -> dict:
    """Run one game through the real harness in a subprocess; return its result dict (never raises)."""
    job_dir.mkdir(parents=True, exist_ok=True)
    out = job_dir / "result.json"
    cmd = [
        cfg.python_bin,
        str(HERE / "run_one_game.py"),
        "--game", game,
        "--env-dir", cfg.env_dir,
        "--bundle", cfg.bundle_dir,
        "--job-dir", str(job_dir),
        "--max-runtime-s", str(cfg.max_runtime_s),
        "--max-actions", str(cfg.max_actions),
        "--max-gen-tokens", str(cfg.max_gen_tokens),
        "--out", str(out),
    ]
    log = open(job_dir / "harness.log", "w")  # noqa: SIM115
    proc = await asyncio.create_subprocess_exec(
        *cmd, env=harness_env(cfg, base_url, model_name), stdout=log, stderr=asyncio.subprocess.STDOUT,
        preexec_fn=_die_with_parent,
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout_s or (cfg.max_runtime_s + 600))
    except (TimeoutError, asyncio.TimeoutError):
        proc.kill()
        await proc.wait()
    finally:
        log.close()
    if out.exists():
        try:
            return json.loads(out.read_text())
        except Exception as exc:  # noqa: BLE001
            return {"game": game, "error": f"bad result json: {exc}"}
    return {"game": game, "error": f"no result (exit {proc.returncode})"}
