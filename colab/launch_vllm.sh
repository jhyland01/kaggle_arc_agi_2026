#!/usr/bin/env bash
# Start the vLLM sampler for AsyncGRPO (and for baseline/eval runs, so both use the same sampler).
# Flags follow the TRL AsyncGRPO docs: dev mode + weight-transfer config, processed logprobs, sampling-mask
# replay for top-k/top-p, token-in/token-out endpoint, and runtime LoRA loading for adapter sync.
# One API server by default: vLLM refuses VLLM_ALLOW_RUNTIME_LORA_UPDATING with api_server_count > 1, and
# adapter-over-disk sync is what lets trainer + sampler share one GPU. (TRL suggests 8 API servers so the
# top-8 logprob serialization needed by #7760 score centering doesn't throttle generation; we accept that cost.)
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
GPU_UTIL=${GPU_UTIL:-0.45}
MAX_LEN=${MAX_LEN:-40960}
PORT=${PORT:-8000}
LOG=${LOG:-/content/vllm.log}
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}
nohup vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --logprobs-mode processed_logprobs \
  --max-logprobs 20 \
  --return-sampling-mask \
  --enable-scale-out \
  --api-server-count "${API_SERVERS:-1}" \
  --weight-transfer-config '{"backend":"nccl"}' \
  --enable-lora --max-lora-rank 16 --max-loras 6 \
  > "$LOG" 2>&1 &
echo $! > /content/vllm.pid
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then echo "VLLM_READY after $((i*5))s"; exit 0; fi
  if ! kill -0 "$(cat /content/vllm.pid)" 2>/dev/null; then echo "VLLM_DIED"; tail -60 "$LOG"; exit 1; fi
  sleep 5
done
echo "VLLM_TIMEOUT"; tail -60 "$LOG"; exit 1
