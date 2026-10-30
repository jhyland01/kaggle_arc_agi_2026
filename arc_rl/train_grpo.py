"""AsyncGRPO (TRL main @ e29597a, includes huggingface/trl#7760 score centering) LoRA fine-tuning of a
Qwen3.5 / Qwen3.8 model *inside the Duck harness* on offline ARC-AGI-3 games, on ONE GPU.

Single-GPU layout: the vLLM server and this trainer share the GPU. LoRA adapters are synced to vLLM
over the filesystem (`--enable-lora` + VLLM_ALLOW_RUNTIME_LORA_UPDATING), so no NCCL group is needed.
Only the top `--train-layers` decoder layers get LoRA; the rest of the network is frozen and runs
forward-only under `torch.no_grad()` without gradient checkpointing (faster steps, far less activation memory).

    python -m arc_rl.train_grpo --model Qwen/Qwen3.8-27B-FP8 --fp8-base --vllm http://127.0.0.1:8000 --out runs/grpo38
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _get_decoder_layers(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = getattr(base, "model", base)
    lm = getattr(inner, "language_model", inner)
    return getattr(lm, "layers", None)


def _freeze_bottom_layers(model, first_trainable: int) -> None:
    """Keep frozen bottom layers truly forward-only even after HF Trainer.train() calls gradient_checkpointing_enable().

    HF `Trainer.train()` calls `self.model.gradient_checkpointing_enable()`, which sets `gradient_checkpointing=True`
    on every decoder layer and calls `enable_input_require_grads()` on `embed_tokens` -- dragging autograd and saved
    activations through all frozen layers. With `use_reentrant=False`, non-reentrant checkpointing on the top LoRA
    layers produces bit-exact LoRA gradients even when their input `hidden_states` has `requires_grad=False`.
    """
    import torch  # noqa: PLC0415

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for obj in (model, base, getattr(base, "model", None), getattr(getattr(base, "model", None), "language_model", None)):
        if obj is not None and hasattr(obj, "disable_input_require_grads"):
            obj.disable_input_require_grads()

    inner = getattr(base, "model", None)
    if inner is not None and getattr(inner, "visual", None) is not None:
        inner.visual.to("cpu")

    layers = _get_decoder_layers(model)
    if layers is None:
        return
    n_frozen_gc_off = 0
    n_train_gc_on = 0
    for idx, layer in enumerate(layers):
        if idx < first_trainable:
            layer.gradient_checkpointing = False
            n_frozen_gc_off += 1
            if not getattr(layer, "_arc_no_grad_wrapped", False):
                orig_forward = layer.forward

                @functools.wraps(orig_forward)
                def _no_grad_fwd(*a, _fn=orig_forward, **kw):
                    with torch.no_grad():
                        return _fn(*a, **kw)

                layer.forward = _no_grad_fwd
                layer._arc_no_grad_wrapped = True
        else:
            n_train_gc_on += int(bool(getattr(layer, "gradient_checkpointing", False)))
    print(
        f"freeze check: frozen layers 0..{first_trainable - 1} no_grad=True gc_off={n_frozen_gc_off}; "
        f"trainable layers {first_trainable}..{len(layers) - 1} gc_on={n_train_gc_on}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="checkpoint the trainer loads (and vLLM serves)")
    ap.add_argument("--fp8-base", action="store_true",
                    help="checkpoint is fine-grained FP8: keep frozen layers FP8, dequantize LoRA layers to bf16")
    ap.add_argument("--vllm", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", default="work_bundle")
    ap.add_argument("--env-dir", default="data/environment_files")
    ap.add_argument("--games", default="train")
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-inflight", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--train-layers", type=int, default=12, help="LoRA only on the top N decoder layers")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=0, help="0 = vLLM max_model_len")
    ap.add_argument("--max-staleness", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--context-window", type=int, default=32 * 1024)
    ap.add_argument("--max-output", type=int, default=6 * 1024)
    ap.add_argument("--rollout-runtime-s", type=float, default=480.0)
    ap.add_argument("--rollout-max-actions", type=int, default=0)
    ap.add_argument("--reasoning-effort", default="", help="Optional Qwen3.8 reasoning_effort (low/medium/xhigh)")
    ap.add_argument("--prefer-tool-calls", action="store_true", help="Enable ARC3_PREFER_TOOL_CALLS=1 in harness")
    ap.add_argument("--save-steps", type=int, default=5)
    ap.add_argument("--reward-weights", default="{}")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoTokenizer, TrainerCallback

    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
    from trl.experimental.async_grpo import packing as trl_packing

    # TRL's tree-packing attention loads FlashAttention-4 on every compute capability >= 10. FA4 targets datacenter
    # Blackwell (sm100) and fails to import on the RTX PRO 6000 (sm120, the Kaggle/Colab-G4 GPU) with vLLM 0.32's
    # cutlass-dsl. TRL only needs `flash_attn_varlen_func` (causal, bottom-right aligned), which FlashAttention-2
    # provides with the same signature; verified against a reference implementation on sm120 (colab/probe_varlen.py).
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
        from functools import cache

        from kernels import get_kernel

        @cache
        def _fa2_varlen():
            return get_kernel("kernels-community/flash-attn2", version=4)

        trl_packing._flash_attn = _fa2_varlen
        print("tree attention: using FlashAttention-2 (kernels-community/flash-attn2 v4) on sm120", flush=True)

    from arc_rl.eval_games import HOLDOUT_GAMES, TRAIN_GAMES
    from arc_rl.harness_env import HarnessConfig
    from arc_rl.rollout_loop import ArcHarnessRolloutWorker

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    games = TRAIN_GAMES if args.games == "train" else HOLDOUT_GAMES if args.games == "holdout" else args.games.split(",")
    rng = random.Random(args.seed)
    rows = []
    for _ in range(200):  # long shuffled stream of games; one group = num_generations rollouts of one game
        order = games[:]
        rng.shuffle(order)
        rows += [{"prompt": [{"role": "user", "content": g}], "game": g} for g in order]
    dataset = Dataset.from_list(rows)

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    n_layers = getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers")
    layers = list(range(n_layers - args.train_layers, n_layers))
    first_trainable = layers[0]
    model_arg = args.model
    if args.fp8_base:
        from arc_rl.fp8_base import load_fp8_base_with_bf16_top

        model_arg, layers = load_fp8_base_with_bf16_top(args.model, args.train_layers)
        first_trainable = layers[0]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=layers,
        layers_pattern="layers",
        task_type="CAUSAL_LM",
    )

    ct_kwargs: dict = {"enable_thinking": True, "preserve_thinking": True}
    if args.reasoning_effort:
        ct_kwargs["reasoning_effort"] = args.reasoning_effort

    config = AsyncGRPOConfig(
        output_dir=str(out),
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.grad_accum,
        per_device_train_batch_size=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={"dtype": torch.bfloat16},
        num_generations=args.num_generations,
        max_completion_length=args.max_output,
        temperature=0.7,
        top_p=0.95,
        top_k=20,
        chat_template_kwargs=ct_kwargs,
        vllm_server_base_url=args.vllm,
        request_timeout=1800,
        # huggingface/trl#7760: score-centred objective with masked importance sampling against the vLLM sampler.
        score_centering=True,
        num_sampler_logprobs=8,
        vllm_importance_sampling_clip_min=0.5,
        vllm_importance_sampling_clip_max=5.0,
        token_budget=args.token_budget or None,
        max_inflight_tasks=args.max_inflight,
        max_staleness=args.max_staleness,
        weight_sync_steps=1,
        heartbeat_stale_after_s=3600.0,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=1,
        report_to="none",
        seed=args.seed,
    )
    assert config.score_centering and config.num_sampler_logprobs > 0, "TRL #7760 score centering must be on"

    harness_cfg = HarnessConfig(
        bundle_dir=str(Path(args.bundle).resolve()),
        env_dir=str(Path(args.env_dir).resolve()),
        context_window=args.context_window,
        max_output=args.max_output,
        max_runtime_s=args.rollout_runtime_s,
        max_actions=args.rollout_max_actions,
        prefer_tool_calls=args.prefer_tool_calls,
    )
    eos_ids = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None})
    worker = ArcHarnessRolloutWorker(
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],
        processing_class=tok,
        eos_token_ids=eos_ids,
        num_generations=config.num_generations,
        max_inflight_tasks=args.max_inflight,
        queue_maxsize=config.queue_maxsize,
        vllm_server_url=args.vllm,
        max_tokens=args.max_output,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        num_sampler_logprobs=config.num_sampler_logprobs,
        min_p=config.min_p,
        repetition_penalty=config.repetition_penalty,
        request_timeout=config.request_timeout,
        chat_template_kwargs=config.chat_template_kwargs,
        max_tool_calling_iterations=None,
        log_completions=False,
        fork_threshold_tokens=config.fork_threshold_tokens,
        harness_cfg=harness_cfg.to_dict(),
        run_dir=str(out),
        reward_weights=json.loads(args.reward_weights),
        max_model_len=args.max_model_len,
    )

    trainer = AsyncGRPOTrainer(
        model=model_arg,
        args=config,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=peft_config,
        rollout_worker=worker,
    )

    class _FreezeBottomCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            _freeze_bottom_layers(model or trainer.model, first_trainable)

    trainer.add_callback(_FreezeBottomCallback())
    model = trainer.model
    _freeze_bottom_layers(model, first_trainable)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params {n_train / 1e6:.1f}M / {n_all / 1e9:.2f}B; LoRA layers {layers[0]}..{layers[-1]}", flush=True)
    trainer.train()
    trainer.save_model(str(out / "final_adapter"))


if __name__ == "__main__":
    main()
