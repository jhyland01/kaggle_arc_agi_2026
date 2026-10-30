# Colab kernel cell: 2-step AsyncGRPO smoke test for Qwen3.5-27B-FP8 (FP8 frozen layers, bf16 LoRA layers).
import os
import runpy
import sys

os.environ.update({
    "MODEL": "Qwen/Qwen3.5-27B-FP8",
    "KV_DTYPE": "fp8",
    "GPU_UTIL": "0.45",
    "SWAP_VLLM": "1",
    "TRAIN_OUT": "/content/runs/grpo27_smoke",
    "TRAIN_ARGS": "--fp8-base --train-layers 16 --max-steps 2 --num-generations 2 --max-inflight 4 "
                  "--grad-accum 2 --rollout-runtime-s 240 --save-steps 1 --token-budget 32768",
})
sys.argv = ["driver", "trainsmoke"]
runpy.run_path("/content/arc/colab/driver.py", run_name="__main__")
