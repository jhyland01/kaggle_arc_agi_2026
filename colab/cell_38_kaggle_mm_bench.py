"""Full-GPU Kaggle-mode multimodal + speed benchmark on Colab G4 (96 GB RTX PRO 6000).

1. Stops the RL-mode vLLM server (which ran at GPU_UTIL=0.40, text-only, processed_logprobs).
2. Launches vLLM in full Kaggle inference mode:
     - GPU_UTIL=0.88, MAX_LEN=81920, MAX_SEQS=16, KV_DTYPE=fp8, prefix caching
     - Multimodal images ON (MULTIMODAL_CONTEXT=current_grid, ARC3_DIFF_IMAGE=1, ARC3_ANIMATION=1)
     - Native qwen3_coder tool parser + qwen3 reasoning parser + reasoning_effort="medium"
     - Static LoRA module arc-grpo38=/content/runs/grpo38/final_adapter
     - Tests MTP speculative decoding (3 draft tokens) first; falls back cleanly if vLLM rejects MTP+LoRA.
3. Runs a simultaneous 10-stream (5 base + 5 arc-grpo38) 600s multimodal head-to-head on
   5 diagnostic holdout games (ft09,sb26,vc33,lp85,ar25).
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

LOG = Path("/content/mm_bench.log")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def stop_existing_vllm() -> None:
    pid_file = Path("/content/vllm.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            log(f"Sent SIGTERM to existing vLLM pid {pid}")
        except OSError:
            pass
    # Also ensure port 8000 is free
    for _ in range(30):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2)
            time.sleep(1)
        except Exception:
            break
    time.sleep(3)


def try_boot_vllm(use_mtp: bool) -> subprocess.Popen | None:
    vllm_log = Path("/content/vllm_mm_mtp.log" if use_mtp else "/content/vllm_mm.log")
    cmd = [
        "vllm",
        "serve",
        "Qwen/Qwen3.8-27B-FP8",
        "--port",
        "8000",
        "--max-model-len",
        "81920",
        "--gpu-memory-utilization",
        "0.88",
        "--kv-cache-dtype",
        "fp8",
        "--enable-prefix-caching",
        "--max-num-seqs",
        "16",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
        "--generation-config",
        "vllm",
        "--default-chat-template-kwargs",
        '{"preserve_thinking": true, "reasoning_effort": "medium"}',
        "--enable-lora",
        "--max-lora-rank",
        "16",
        "--max-loras",
        "2",
        "--lora-modules",
        "arc-grpo38=/content/runs/grpo38/final_adapter",
    ]
    if use_mtp:
        cmd += ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": 3}']

    env = os.environ.copy()
    env.pop("VLLM_ALLOW_RUNTIME_LORA_UPDATING", None)
    log(f"Booting vLLM (use_mtp={use_mtp}) -> {vllm_log}")
    fh = vllm_log.open("w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    Path("/content/vllm.pid").write_text(str(proc.pid))

    for i in range(90):
        if proc.poll() is not None:
            tail = vllm_log.read_text(errors="replace")[-3000:]
            log(f"vLLM (use_mtp={use_mtp}) exited early ({proc.returncode}):\n{tail}")
            return None
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=3) as r:
                if r.status == 200:
                    body = r.read().decode(errors="replace")
                    log(f"vLLM READY (use_mtp={use_mtp}) after {(i+1)*3}s: {body[:300]}")
                    return proc
        except Exception:
            pass
        time.sleep(3)

    log(f"vLLM (use_mtp={use_mtp}) timed out; terminating")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    return None


def main() -> None:
    stop_existing_vllm()
    proc = try_boot_vllm(use_mtp=True)
    mtp_active = proc is not None
    if proc is None:
        stop_existing_vllm()
        proc = try_boot_vllm(use_mtp=False)
        if proc is None:
            raise RuntimeError("Failed to boot Kaggle-mode vLLM")

    log(f"Starting simultaneous 10-stream Kaggle multimodal eval (mtp_active={mtp_active})")
    common = [
        "python",
        "-m",
        "arc_rl.eval_games",
        "--bundle",
        "/content/arc/work_bundle",
        "--env-dir",
        "/content/arc/data/environment_files",
        "--vllm",
        "http://127.0.0.1:8000",
        "--games",
        "ft09,sb26,vc33,lp85,ar25",
        "--passes",
        "1",
        "--concurrency",
        "5",
        "--context-window",
        "65536",
        "--max-output",
        "8192",
        "--max-runtime-s",
        "600",
        "--prefer-tool-calls",
        "--multimodal",
        "--direct-vllm",
    ]
    run_env = dict(os.environ, PYTHONPATH="/content/arc")
    b_log = Path("/content/mm_base.log").open("w")
    a_log = Path("/content/mm_adapter.log").open("w")
    p_base = subprocess.Popen(
        common + ["--model", "Qwen/Qwen3.8-27B-FP8", "--out", "/content/runs/mm38_base"],
         cwd="/content/arc",
        env=run_env,
        stdout=b_log,
        stderr=subprocess.STDOUT,
    )
    p_adp = subprocess.Popen(
        common + ["--model", "arc-grpo38", "--out", "/content/runs/mm38_arc-grpo38"],
        cwd="/content/arc",
        env=run_env,
        stdout=a_log,
        stderr=subprocess.STDOUT,
    )
    rc1 = p_base.wait()
    rc2 = p_adp.wait()
    b_log.close()
    a_log.close()
    log(f"Multimodal benchmark done: rc_base={rc1}, rc_adapter={rc2}, mtp_active={mtp_active}")


if __name__ == "__main__":
    main()
