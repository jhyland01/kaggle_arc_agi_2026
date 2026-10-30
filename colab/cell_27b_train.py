# Colab kernel cell: full AsyncGRPO run for Qwen3.5-27B-FP8, then held-out eval of the adapter.
# Stronger tokens-per-action penalty than grpo1: Kaggle inference time is the binding constraint.
import json
import os
import runpy
import sys

os.environ.update({
    "MODEL": "Qwen/Qwen3.5-27B-FP8",
    "KV_DTYPE": "fp8",
    "GPU_UTIL": "0.45",
    "SWAP_VLLM": "1",
    "TRAIN_OUT": "/content/runs/grpo27",
    "TRAIN_ARGS": "--fp8-base --train-layers 16 --max-steps 24 --num-generations 4 --max-inflight 8 "
                  "--grad-accum 8 --rollout-runtime-s 600 --save-steps 4 --token-budget 32768 --max-staleness 4 "
                  "--lr 2e-5 --reward-weights '" + json.dumps({"tok_per_action": 0.15}) + "'",
    "ADAPTER": "/content/runs/grpo27/final_adapter",
    "ADAPTER_NAME": "arc-grpo27",
})
sys.argv = ["driver", "train"]
runpy.run_path("/content/arc/colab/driver.py", run_name="__main__")
