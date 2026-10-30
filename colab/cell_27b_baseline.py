# Colab kernel cell: switch vLLM to Qwen3.5-27B-FP8 (FP8 KV cache) and run the 27B baseline on held-out games.
import os
import runpy
import sys

os.environ.update({
    "MODEL": "Qwen/Qwen3.5-27B-FP8",
    "KV_DTYPE": "fp8",
    "GPU_UTIL": "0.45",
    "SWAP_VLLM": "1",
    "BASELINE_OUT": "/content/runs/baseline27_holdout",
})
sys.argv = ["driver", "baseline"]
runpy.run_path("/content/arc/colab/driver.py", run_name="__main__")
