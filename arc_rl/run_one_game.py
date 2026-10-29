"""Run the patched Duck harness on ONE offline ARC-AGI-3 game and write a JSON result.

The harness reads most of its config from environment variables at import time, so this
script is always launched as a fresh subprocess (by the eval runner or the RL rollout loop)
with the env already set. The LLM endpoint is whatever ``OPENAI_BASE_URL`` /
``LOCAL_ANALYZER_BASE_URL`` point at: a plain vLLM server for evaluation, or the per-rollout
TRL capture proxy during training.

Usage:
    python run_one_game.py --game ft09 --env-dir data/environment_files --bundle work_bundle \
        --job-dir runs/x/ft09 --max-runtime-s 900 --max-actions 120 --out result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import sys
import time
from pathlib import Path


def _add_bundle_to_path(bundle_dir: Path) -> None:
    # Mirror the Kaggle notebook: each bundled repo exposes its importable tree at <repo>/src or <repo>.
    for repo in sorted((bundle_dir / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                sys.path.insert(0, str(candidate))
                break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, help="game id prefix, e.g. ft09")
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--max-runtime-s", type=float, default=900.0)
    ap.add_argument("--max-actions", type=int, default=0)
    ap.add_argument("--max-gen-tokens", type=int, default=0)
    ap.add_argument("--save-request-logs", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("ONLY_RESET_LEVELS", "true")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TAAF_RUN_AS_SUBMISSION", "0")
    os.environ.setdefault("TAAF_MINIMAL_DIAGNOSTICS", "1")
    job_dir = Path(args.job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RECORDINGS_DIR", str(job_dir / "server_recording"))

    bundle = Path(args.bundle).resolve()
    _add_bundle_to_path(bundle)

    import arc_agi  # noqa: PLC0415
    import taaf.game_api  # noqa: PLC0415

    env_dir = str(Path(args.env_dir).resolve())
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    game_ids = [e.game_id for e in arcade.available_environments if e.game_id.startswith(args.game)]
    if not game_ids:
        raise SystemExit(f"game {args.game!r} not found under {env_dir}")

    with open(bundle / "benchmark_initial.pkl", "rb") as f:
        bm = pickle.load(f)
    bm.job_dir = job_dir
    bm.games = [taaf.game_api.GameAPI(env_name=game_ids[0], arcade_spec=spec)]
    bm.n_passes = 1
    bm.game_weights = None
    s = bm.solver
    s.concurrency = 1
    s.max_runtime_s_per_game = float(args.max_runtime_s)
    s.max_actions_per_game = args.max_actions or None
    s.max_generated_tokens_per_game = args.max_gen_tokens or None
    s.save_request_logs = bool(args.save_request_logs)
    s.analyzer_timeout = float(os.environ.get("LOCAL_ANALYZER_TIMEOUT", "900"))

    t0 = time.monotonic()
    error = None
    try:
        asyncio.run(bm.run(soft_end_time=None, runtime_environment=None, minimal_diagnostics=False))
    except BaseException as exc:  # noqa: BLE001 - always write a result
        error = repr(exc)
    wall = time.monotonic() - t0

    result: dict = {"game": args.game, "game_id": game_ids[0], "wall_s": wall, "error": error}
    runs = getattr(bm, "game_runs", None) or []
    if runs:
        run = runs[0]
        hist = list(getattr(run, "history", []) or [])
        result.update(
            state=str(run.state),
            levels_completed=int(run.levels_completed),
            number_of_levels=int(run.number_of_levels),
            final_score=float(run.final_score or 0.0),
            actions=len(hist),
            actions_per_level=list(run.actions_per_level),
            base_actions_per_level=run.base_actions_per_level,
            generated_tokens=int(sum(getattr(r, "generated_tokens", 0) or 0 for r in hist))
            + int(getattr(run, "final_generated_tokens", 0) or 0),
        )
        # Dense, game-grounded exploration stats (used for reward shaping; levels are rare for small models).
        try:
            import hashlib  # noqa: PLC0415

            states = list(getattr(run, "intermediate_states", []) or [])
            hashes, effective, game_overs, prev = set(), 0, 0, None
            for st in states:
                h = hashlib.blake2b(st.frame.data.tobytes(), digest_size=12).hexdigest()
                hashes.add(h)
                if prev is not None and h != prev:
                    effective += 1
                prev = h
                if str(getattr(st.raw.state, "name", st.raw.state)).upper() == "GAME_OVER":
                    game_overs += 1
            result.update(distinct_states=len(hashes), effective_actions=effective, game_overs=game_overs,
                          n_states=len(states))
        except Exception as exc:  # noqa: BLE001
            result["stats_error"] = repr(exc)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
