"""Merge a PEFT LoRA adapter directly into a block-128 FP8 checkpoint view.

Creates a lightweight directory view where unmodified safetensors shards and
metadata files are symlinked from the read-only base model directory, and ONLY
the shards containing LoRA-targeted layers (e.g. layers 54..63 = ~4 GB for a
top-10-layer 27B adapter) are materialized with merged block-128 FP8 weights.

Benefits for Kaggle / production serving:
  - Only the ~50 MB LoRA adapter needs to be uploaded as a Kaggle dataset
    (base weights come from the attached Kaggle Model mirror).
  - Eliminates runtime LoRA dual-GEMM overhead completely.
  - Preserves 100% compatibility with native MTP speculative decoding
    (`mtp.safetensors`) and CUDA graphs in vLLM and SGLang.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0


def merge_fp8_block_weight(
    weight: torch.Tensor,
    scale_inv: torch.Tensor | None,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    block_size: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return `(merged_weight, merged_scale_inv)` preserving original dtypes."""
    out_dim, in_dim = weight.shape
    delta = (lora_b.float() @ lora_a.float()) * float(scaling)

    if scale_inv is None or weight.dtype != torch.float8_e4m3fn:
        merged = (weight.float() + delta).to(weight.dtype)
        return merged, None

    bo, bi = block_size
    nbo = math.ceil(out_dim / bo)
    nbi = math.ceil(in_dim / bi)
    pad_o = nbo * bo - out_dim
    pad_i = nbi * bi - in_dim

    w_f32 = weight.float()
    if pad_o or pad_i:
        w_f32 = torch.nn.functional.pad(w_f32, (0, pad_i, 0, pad_o))
        delta_pad = torch.nn.functional.pad(delta, (0, pad_i, 0, pad_o))
    else:
        delta_pad = delta

    s_f32 = scale_inv.float()
    blocks = w_f32.view(nbo, bo, nbi, bi) * s_f32[:, None, :, None]
    blocks = blocks + delta_pad.view(nbo, bo, nbi, bi)

    # Zero out any padded tail before computing per-block absmax.
    if pad_o or pad_i:
        flat = blocks.view(nbo * bo, nbi * bi)
        if pad_o:
            flat[out_dim:, :] = 0.0
        if pad_i:
            flat[:, in_dim:] = 0.0
        blocks = flat.view(nbo, bo, nbi, bi)

    amax = blocks.abs().amax(dim=(1, 3))
    new_scale = (amax / FP8_MAX).clamp(min=1e-12).to(scale_inv.dtype)
    denom = new_scale.float().clamp(min=1e-12)[:, None, :, None]
    q_blocks = (blocks / denom).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    q_flat = q_blocks.view(nbo * bo, nbi * bi)[:out_dim, :in_dim].contiguous()
    return q_flat, new_scale.contiguous()


def load_lora_deltas(adapter_dir: Path, device: str = "cpu") -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """Load `{module_prefix: (lora_A, lora_B)}` and `lora_alpha / r` scaling."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    scaling = alpha / r

    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"), device=device)
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        norm = k
        if norm.startswith("base_model.model."):
            norm = norm[len("base_model.model.") :]
        if norm.endswith(".lora_A.weight"):
            lora_a[norm[: -len(".lora_A.weight")]] = v
        elif norm.endswith(".lora_B.weight"):
            lora_b[norm[: -len(".lora_B.weight")]] = v
        elif ".lora_A." in norm and norm.endswith(".weight"):
            lora_a[norm.split(".lora_A.")[0]] = v
        elif ".lora_B." in norm and norm.endswith(".weight"):
            lora_b[norm.split(".lora_B.")[0]] = v

    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for mod, a in lora_a.items():
        if mod not in lora_b:
            raise KeyError(f"Missing lora_B for {mod}")
        pairs[mod] = (a, lora_b[mod])
    if not pairs:
        raise RuntimeError(f"No LoRA A/B weight pairs found in {adapter_dir}")
    return pairs, scaling


def build_merged_checkpoint_view(
    base_dir: Path,
    adapter_dir: Path,
    out_dir: Path,
    device: str | None = None,
) -> dict:
    """Create `out_dir` with symlinks for untouched files and merged shards for LoRA layers."""
    t0 = time.time()
    base_dir = base_dir.resolve()
    adapter_dir = adapter_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = json.loads((base_dir / "config.json").read_text())
    qcfg = cfg.get("quantization_config") or {}
    block_sz = tuple(qcfg.get(" weight_block_size") or qcfg.get("weight_block_size") or (128, 128))

    pairs, scaling = load_lora_deltas(adapter_dir, device=device)

    index_path = base_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map: dict[str, str] = index["weight_map"]
    else:
        weight_map = {f"{m}.weight": "model.safetensors" for m in pairs}

    # Group targeted modules by shard filename.
    shard_to_mods: dict[str, list[str]] = {}
    missing_mods: list[str] = []
    for mod in sorted(pairs):
        w_key = f"{mod}.weight"
        shard = weight_map.get(w_key)
        if shard is None:
            missing_mods.append(mod)
            continue
        shard_to_mods.setdefault(shard, []).append(mod)
    if missing_mods:
        raise KeyError(f"LoRA target modules missing from base weight_map: {missing_mods[:5]}")

    touched_shards = set(shard_to_mods)

    # Symlink every untouched top-level file/dir in base_dir.
    linked_count = 0
    for item in sorted(base_dir.iterdir()):
        if item.name in touched_shards:
            continue
        dest = out_dir / item.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(item)
        linked_count += 1

    # Materialize only the touched shards with merged FP8 weights.
    merged_modules = 0
    written_bytes = 0
    for shard_name, mods in sorted(shard_to_mods.items()):
        src_shard = base_dir / shard_name
        tensors = load_file(str(src_shard), device=device)
        out_tensors: dict[str, torch.Tensor] = dict(tensors)
        for mod in mods:
            w_key = f"{mod}.weight"
            s_key = f"{mod}.weight_scale_inv"
            lora_a, lora_b = pairs[mod]
            new_w, new_s = merge_fp8_block_weight(
                out_tensors[w_key],
                out_tensors.get(s_key),
                lora_a,
                lora_b,
                scaling=scaling,
                block_size=(int(block_sz[0]), int(block_sz[1])),
            )
            out_tensors[w_key] = new_w.cpu()
            if new_s is not None and s_key in out_tensors:
                out_tensors[s_key] = new_s.cpu()
            merged_modules += 1
        out_cpu = {k: (v.cpu() if v.device.type != "cpu" else v) for k, v in out_tensors.items()}
        dst_shard = out_dir / shard_name
        if dst_shard.exists() or dst_shard.is_symlink():
            dst_shard.unlink()
        save_file(out_cpu, str(dst_shard))
        written_bytes += dst_shard.stat().st_size

    elapsed = time.time() - t0
    stats = {
        "base_dir": str(base_dir),
        "adapter_dir": str(adapter_dir),
        "out_dir": str(out_dir),
        "merged_modules": merged_modules,
        "touched_shards": sorted(touched_shards),
        "linked_files": linked_count,
        "written_gb": round(written_bytes / 1e9, 3),
        "elapsed_s": round(elapsed, 2),
    }
    print(f"[merge_fp8_lora] {json.dumps(stats)}", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    build_merged_checkpoint_view(args.base_dir, args.adapter_dir, args.out_dir, device=args.device)


if __name__ == "__main__":
    main()
