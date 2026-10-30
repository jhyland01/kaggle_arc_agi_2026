"""Runs INSIDE the Colab kernel as one blocking cell, so the VM never looks idle and is not reclaimed.

Stages (each skipped if already done): setup -> vLLM server -> smoke eval -> baseline eval (held-out games).
Progress goes to /content/driver.log; pull it with `colab download`.

    python colab/driver.py baseline
"""
import os
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("/content/driver.log")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sh(cmd: str, logfile: str, check: bool = True) -> int:
    log(f"$ {cmd}  (log: {logfile})")
    with open(logfile, "a") as f:
        p = subprocess.run(cmd, shell=True, cwd="/content/arc", stdout=f, stderr=subprocess.STDOUT)
    log(f"  exit {p.returncode}")
    if check and p.returncode != 0:
        raise RuntimeError(f"failed: {cmd}")
    return p.returncode


def vllm_up() -> bool:
    return subprocess.run("curl -sf http://127.0.0.1:8000/v1/models > /dev/null", shell=True).returncode == 0


def unload_stale_adapters() -> None:
    """A previous (killed) training run leaves `trl-policy-vN` adapters registered in vLLM; a new run re-publishing
    `trl-policy-v1` would then be rejected. Unload them before training."""
    import json
    import urllib.request

    models = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=30).read())["data"]
    for m in models:
        if m["id"].startswith("trl-policy-"):
            req = urllib.request.Request(
                "http://127.0.0.1:8000/v1/unload_lora_adapter",
                data=json.dumps({"lora_name": m["id"]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=60).read()
            log(f"unloaded stale adapter {m['id']}")


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if stage == "trainsmoke":
        os.environ.setdefault("TRAIN_OUT", "/content/runs/grpo_smoke")
        os.environ.setdefault("TRAIN_ARGS", "--max-steps 2 --num-generations 2 --max-inflight 4 --grad-accum 2 --rollout-runtime-s 150 --save-steps 1 --token-budget 32768")
        stage = "train_only"
    t0 = time.time()
    model = os.environ.setdefault("MODEL", "Qwen/Qwen3.5-9B")  # also read by setup.sh / launch_vllm.sh
    log(f"driver start, stage={stage}, model={model}")
    if not Path("/content/setup.done").exists():
        sh("bash /content/arc/colab/setup.sh", "/content/setup.log")
        Path("/content/setup.done").write_text("ok")
    if not vllm_up():
        sh("bash /content/arc/colab/launch_vllm.sh", "/content/vllm_launch.log")
    log("vLLM is up")
    eval_cmd = (f"PYTHONPATH=/content/arc python -m arc_rl.eval_games --bundle work_bundle "
                f"--env-dir data/environment_files --tokenizer {model}")
    smoke_flag = Path(f"/content/smoke_{model.replace('/', '_')}.done")
    if not smoke_flag.exists():
        sh(f"{eval_cmd} --model {model} --games ft09 --passes 1 --concurrency 1 --max-runtime-s 180 --max-actions 10 "
           f"--out /content/runs/smoke_{model.replace('/', '_')}", "/content/smoke.log")
        smoke_flag.write_text("ok")
    if stage == "confirm":
        # Baseline and adapter evaluated simultaneously (10 concurrent games each), so both see the same GPU load.
        adapter = os.environ.get("ADAPTER", "/content/adapters/arc-grpo1")
        name = os.environ.get("ADAPTER_NAME", "arc-grpo1")
        passes = os.environ.get("PASSES", "4")
        sh(f"curl -sf -X POST http://127.0.0.1:8000/v1/load_lora_adapter -H 'Content-Type: application/json' "
           f"-d '{{\"lora_name\": \"{name}\", \"lora_path\": \"{adapter}\"}}'", "/content/confirm.log", check=False)
        common = f"--games holdout --passes {passes} --concurrency 10 --max-runtime-s 600"
        sh(f"( {eval_cmd} --model {model} {common} --out /content/runs/confirm_base > /content/confirm_base.log 2>&1 & "
           f"{eval_cmd} --model {name} {common} --out /content/runs/confirm_{name} > /content/confirm_adapter.log 2>&1 & wait )",
           "/content/confirm.log")
    if stage in ("baseline", "all"):
        out = os.environ.get("BASELINE_OUT", "/content/runs/baseline_holdout")
        sh(f"{eval_cmd} --model {model} --games holdout --passes 2 --concurrency 20 --max-runtime-s 600 "
           f"--out {out}", "/content/baseline.log")
    if stage in ("train", "train_only", "all"):
        unload_stale_adapters()
        extra = os.environ.get("TRAIN_ARGS", "")
        sh(f"PYTHONPATH=/content/arc python -m arc_rl.train_grpo --model {model} --vllm http://127.0.0.1:8000 "
           f"--bundle work_bundle --env-dir data/environment_files --out {os.environ.get('TRAIN_OUT', '/content/runs/grpo1')} {extra}",
           "/content/train.log")
    if stage in ("posteval", "train", "all"):
        adapter = os.environ.get("ADAPTER", "/content/runs/grpo1/final_adapter")
        name = os.environ.get("ADAPTER_NAME", "arc-grpo1")
        sh(f"curl -sf -X POST http://127.0.0.1:8000/v1/load_lora_adapter -H 'Content-Type: application/json' "
           f"-d '{{\"lora_name\": \"{name}\", \"lora_path\": \"{adapter}\"}}'", "/content/posteval.log", check=False)
        sh(f"{eval_cmd} --model {name} --games holdout --passes 2 --concurrency 20 --max-runtime-s 600 "
           f"--out /content/runs/post_{name}", "/content/posteval.log")
    log(f"driver done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
