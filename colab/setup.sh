#!/usr/bin/env bash
# One-time setup of a Colab GPU VM for harness-in-the-loop AsyncGRPO.
# Expects /content/arc.tar.gz (project: arc_rl/, work_bundle/, data/) uploaded beforehand.
set -euo pipefail
cd /content
mkdir -p /content/arc && tar -xzf /content/arc.tar.gz -C /content/arc
cd /content/arc

python -V
nvidia-smi --query-gpu=name,memory.total --format=csv

# vLLM first (it pins torch), remove Colab's preinstalled incompatible torchao 0.10.0 so PEFT imports cleanly,
# force a transformers new enough for Qwen3.5/3.8 tree packing (>=5.15), then TRL pinned to main @ e29597a,
# which contains huggingface/trl#7760 (AsyncGRPO score centering).
pip install -q uv
uv pip install --system -q "vllm>=0.32,<0.33" 2>&1 | tail -2
uv pip uninstall --system -q torchao || pip uninstall -y -q torchao || true
uv pip install --system -q --no-deps "transformers>=5.15"
uv pip install --system -q --no-deps "git+https://github.com/huggingface/trl.git@e29597afcf973b91b412662988cf7dc88dde4cd1"
uv pip install --system -q "peft>=0.17" accelerate datasets kernels aiohttp rich "huggingface_hub[hf_transfer]"
# Harness runtime (arc-agi engine + framework deps)
uv pip install --system -q --find-links data/arc_agi_3_wheels arc-agi imageio imageio-ffmpeg scipy matplotlib python-dotenv

python - <<'EOF'
import vllm, transformers, trl, peft, torch
from peft import LoraConfig, get_peft_model
print("vllm", vllm.__version__, "transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__, "torch", torch.__version__)
from trl.experimental.async_grpo import AsyncGRPOConfig
c = AsyncGRPOConfig(output_dir="/tmp/x")
assert c.score_centering and c.num_sampler_logprobs == 8, "TRL build lacks #7760 score centering"
print("TRL #7760 score centering present: score_centering =", c.score_centering)
EOF

export HF_HUB_ENABLE_HF_TRANSFER=1
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('${MODEL:-Qwen/Qwen3.5-9B}', allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model']))"
echo SETUP_DONE
