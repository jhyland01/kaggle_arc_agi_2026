"""Load a fine-grained-FP8 checkpoint (e.g. Qwen/Qwen3.8-27B-FP8) as the AsyncGRPO trainer's base model.

Why: a bf16 27B trainer (~56 GB) plus the FP8 vLLM sampler (~28 GB + KV cache) does not fit on one 96 GB GPU.
LoRA only goes on the top N decoder layers and everything below is frozen and forward-only, so those layers can
stay in FP8. The top N layers are dequantized to bf16 ``nn.Linear`` so PEFT can wrap them and gradients can flow
through them. The unused vision tower is moved to CPU during text-only AsyncGRPO. Trainer and sampler hold the
*same* FP8 weights; the remaining kernel-level mismatch (FP8 matmuls in vLLM vs bf16-dequantized matmuls here) is
what TRL #7760's score centering + masked importance sampling is designed to absorb.
"""

from __future__ import annotations

import torch
from torch import nn


def _dequant_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block: tuple[int, int],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    out_f, in_f = weight.shape
    b0, b1 = block
    s = scale_inv.to(dtype)
    if s.numel() == 1:
        return weight.to(dtype) * s
    if out_f % b0 == 0 and in_f % b1 == 0:
        w = weight.to(dtype).view(out_f // b0, b0, in_f // b1, b1)
        return (w * s[:, None, :, None]).reshape(out_f, in_f)
    s = s.repeat_interleave(b0, dim=0)[:out_f].repeat_interleave(b1, dim=1)[:, :in_f]
    return weight.to(dtype) * s


class DequantFP8Linear(nn.Module):
    """Frozen FP8 (block-scaled) linear layer that dequantizes to bf16 on the fly in forward.

    Replaces transformers' `FP8Linear`, whose triton/hub kernel needs `kernels<0.18` -- incompatible with the
    `kernels` version the FlashAttention-2 tree-attention swap needs. Only one layer's weights are ever materialized
    in bf16 at a time, so memory stays at FP8 levels; weights are frozen (no grads)."""

    def __init__(self, src) -> None:
        super().__init__()
        self.in_features, self.out_features = src.in_features, src.out_features
        self.block = tuple(src.block_size) if src.block_size is not None else (1, 1)
        self.register_buffer("weight", src.weight.data, persistent=False)
        self.register_buffer("weight_scale_inv", src.weight_scale_inv.data, persistent=False)
        self.bias = None if src.bias is None else nn.Parameter(src.bias.data.to(torch.bfloat16), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _dequant_fp8(self.weight, self.weight_scale_inv, self.block, dtype=x.dtype)
        return nn.functional.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))


def load_fp8_base_with_bf16_top(model_id: str, top_layers: int, device: str = "cuda", offload_visual: bool = True):
    from transformers import AutoConfig, AutoModelForImageTextToText  # noqa: PLC0415
    from transformers.integrations.finegrained_fp8 import FP8Linear  # noqa: PLC0415

    cfg = AutoConfig.from_pretrained(model_id)
    # Qwen3.8-27B-FP8 includes MoE router names `*.mlp.gate` in `modules_to_not_convert`. In transformers 5.19
    # `should_convert_module` uses substring matching, so `*.mlp.gate` falsely matches dense `*.mlp.gate_proj`,
    # leaving 64 gate_proj weights unscaled in bf16 (+5.7 GiB VRAM and wrong logits). Keep only real module names.
    qcfg = getattr(cfg, "quantization_config", None)
    mods = qcfg.get("modules_to_not_convert") if isinstance(qcfg, dict) else getattr(qcfg, "modules_to_not_convert", None)
    if mods:
        with torch.device("meta"):
            real_names = {n for n, _ in AutoModelForImageTextToText.from_config(cfg).named_modules()}
        clean_mods = [nc for nc in mods if nc in real_names or "visual." in nc]
        if isinstance(qcfg, dict):
            qcfg["modules_to_not_convert"] = clean_mods
        else:
            qcfg.modules_to_not_convert = clean_mods

    n_layers = cfg.get_text_config().num_hidden_layers
    first = n_layers - top_layers
    model = AutoModelForImageTextToText.from_pretrained(model_id, config=cfg, dtype=torch.bfloat16, device_map=device)

    # AsyncGRPO runs text-only; keep the unquantized bf16 vision tower (~1.3 GiB) on CPU.
    inner = getattr(model, "model", None)
    if offload_visual and inner is not None and getattr(inner, "visual", None) is not None:
        inner.visual.to("cpu")
        for p in inner.visual.parameters():
            p.requires_grad_(False)

    converted = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, FP8Linear):
                continue
            full = f"{name}.{child_name}"
            in_top = False
            if ".layers." in full and "language_model" in full:
                in_top = int(full.split(".layers.")[1].split(".")[0]) >= first
            if not in_top:
                setattr(module, child_name, DequantFP8Linear(child))  # frozen, stays FP8 in memory
                continue
            block = tuple(child.block_size) if child.block_size is not None else (1, 1)
            lin = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None,
                            dtype=torch.bfloat16, device=child.weight.device)
            with torch.no_grad():
                lin.weight.copy_(_dequant_fp8(child.weight.data, child.weight_scale_inv.data, block))
                if child.bias is not None:
                    lin.bias.copy_(child.bias.data.to(torch.bfloat16))
            lin.weight.requires_grad_(False)
            setattr(module, child_name, lin)
            converted += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # transformers' FP8 quantizer reports `is_trainable=False`, which makes Trainer refuse the model. Only LoRA
    # weights on the bf16 layers above are trained, so detach the quantizer flag (the FP8 layers below keep working
    # for the forward pass).
    model.hf_quantizer = None
    model.is_quantized = False
    if hasattr(model.config, "quantization_config"):
        del model.config.quantization_config
    mem_gib = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"fp8 base: dequantized {converted} linear layers in decoder layers {first}..{n_layers - 1} to bf16 "
          f"(visual_cpu={offload_visual}); GPU mem {mem_gib:.1f} GiB", flush=True)
    return model, list(range(first, n_layers))
