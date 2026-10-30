# Colab kernel cell 1: unpack project + setup + vLLM Qwen/Qwen3.8-27B-FP8 + 2-step AsyncGRPO (#7760) smoke test + 1-pass holdout baseline.
import os
import runpy
import subprocess
import sys

if not os.path.exists("/content/arc/colab/driver.py"):
    subprocess.run("mkdir -p /content/arc && tar -xzf /content/arc.tar.gz -C /content/arc", shell=True, check=True)

COMMON_MODEL_ARGS = (
    "--max-model-len 32768 --context-window 26624 --max-output 4096 "
    "--reasoning-effort medium --prefer-tool-calls"
)

os.environ.update({
    "MODEL": "Qwen/Qwen3.8-27B-FP8",
    "KV_DTYPE": "fp8",
    "GPU_UTIL": "0.40",
    "MAX_LEN": "32768",
    "MAX_SEQS": "16",
    "SWAP_VLLM": "1",
    "EVAL_CONCURRENCY": "6",
    "EVAL_PASSES": "1",
    "EVAL_RUNTIME_S": "480",
    "EVAL_EXTRA": COMMON_MODEL_ARGS,
    "SMOKE_TRAIN_OUT": "/content/runs/grpo38_smoke",
    "SMOKE_TRAIN_ARGS": (
        f"--fp8-base --train-layers 10 --max-steps 2 --num-generations 2 --max-inflight 4 "
        f"--grad-accum 1 --rollout-runtime-s 180 --save-steps 1 --token-budget 32768 {COMMON_MODEL_ARGS}"
    ),
    "BASELINE_OUT": "/content/runs/baseline38_holdout",
})
sys.argv = ["driver", "smoke_and_base"]
runpy.run_path("/content/arc/colab/driver.py", run_name="__main__")
