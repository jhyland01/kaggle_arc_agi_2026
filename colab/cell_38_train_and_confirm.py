# Colab kernel cell 2: 30-step AsyncGRPO (#7760) LoRA fine-tuning of Qwen/Qwen3.8-27B-FP8 + simultaneous holdout confirmation eval.
import json
import os
import runpy
import sys

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
    "EVAL_RUNTIME_S": "480",
    "EVAL_EXTRA": COMMON_MODEL_ARGS,
    "TRAIN_OUT": "/content/runs/grpo38",
    "TRAIN_ARGS": (
        f"--fp8-base --train-layers 10 --max-steps 28 --num-generations 4 --max-inflight 6 "
        f"--grad-accum 2 --rollout-runtime-s 360 --save-steps 4 --token-budget 32768 --max-staleness 4 "
        f"--lr 2e-5 {COMMON_MODEL_ARGS} --reward-weights '"
        + json.dumps({"tok_per_action": 0.20, "novelty": 0.50}) + "'"
    ),
    "ADAPTER": "/content/runs/grpo38/final_adapter",
    "ADAPTER_NAME": "arc-grpo38",
    "PASSES": "2",
    "CONFIRM_CONCURRENCY": "3",
    "CONFIRM_BASE_OUT": "/content/runs/confirm38_base",
    "CONFIRM_ADAP_OUT": "/content/runs/confirm38_arc-grpo38",
})
sys.argv = ["driver", "train_and_confirm"]
runpy.run_path("/content/arc/colab/driver.py", run_name="__main__")
