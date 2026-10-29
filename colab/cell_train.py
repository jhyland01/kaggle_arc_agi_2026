import os, runpy, sys
os.environ["TRAIN_OUT"] = "/content/runs/grpo1"
os.environ["TRAIN_ARGS"] = "--max-steps 50 --num-generations 4 --max-inflight 8 --grad-accum 8 --rollout-runtime-s 480 --save-steps 5 --token-budget 32768 --max-staleness 4 --lr 2e-5"
os.environ["ADAPTER"] = "/content/runs/grpo1/final_adapter"
os.environ["ADAPTER_NAME"] = "arc-grpo1"
sys.argv = ["driver", "train"]
runpy.run_path("/content/arc/colab/driver.py", run_name="__main__")
