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


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if stage == "trainsmoke":
        os.environ.setdefault("TRAIN_OUT", "/content/runs/grpo_smoke")
        os.environ.setdefault("TRAIN_ARGS", "--max-steps 2 --num-generations 2 --max-inflight 4 --grad-accum 2 --rollout-runtime-s 150 --save-steps 1")
        stage = "train_only"
    t0 = time.time()
    log(f"driver start, stage={stage}")
    if not Path("/content/setup.done").exists():
        sh("bash /content/arc/colab/setup.sh", "/content/setup.log")
        Path("/content/setup.done").write_text("ok")
    if not vllm_up():
        sh("bash /content/arc/colab/launch_vllm.sh", "/content/vllm_launch.log")
    log("vLLM is up")
    eval_cmd = "PYTHONPATH=/content/arc python -m arc_rl.eval_games --bundle work_bundle --env-dir data/environment_files"
    if not Path("/content/smoke.done").exists():
        sh(f"{eval_cmd} --games ft09 --passes 1 --concurrency 1 --max-runtime-s 180 --max-actions 10 "
           f"--out /content/runs/smoke", "/content/smoke.log")
        Path("/content/smoke.done").write_text("ok")
    if stage in ("baseline", "all"):
        sh(f"{eval_cmd} --games holdout --passes 2 --concurrency 20 --max-runtime-s 600 "
           f"--out /content/runs/baseline_holdout", "/content/baseline.log")
    if stage in ("train", "train_only", "all"):
        extra = os.environ.get("TRAIN_ARGS", "")
        sh(f"PYTHONPATH=/content/arc python -m arc_rl.train_grpo --model Qwen/Qwen3.5-9B --vllm http://127.0.0.1:8000 "
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
