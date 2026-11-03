# Kaggle ARC Prize 2026 (`ARC-AGI-3` & `ARC-AGI-2`)

This repository contains our experimental code, reinforcement learning harness, Colab `G4` (`RTX PRO 6000 Blackwell 96 GB`) ablation scripts, and offline Kaggle submission notebooks for the **2026 ARC Prize** competitions:
- **ARC-AGI-3** (Interactive Multi-Turn Game Reasoning)
- **ARC-AGI-2** (Static Grid Abstraction & Program Synthesis)

---

## 1. Foundations & Existing Work Built Upon

Rather than starting from scratch under tight offline hardware constraints (`1× RTX PRO 6000 96 GB` or `4× L4 96 GB`), we built on top of strong open-source community foundations:
- **ARC-AGI-3**:
  - **TAAF (`tufa-arc-agi-framework` / `ARC3-Inference`)**: Multi-turn tool-use agent harness with `python_exec` sandboxing, grid diffing, object segmentation, and async game stream scheduling.
  - **AGI-Boys / Duck Harness Patches**: Multimodal current-grid & frame-diff rendering (`MULTIMODAL_CONTEXT=current_grid`, `ARC3_DIFF_IMAGE=1`, `ARC3_ANIMATION=1`), persistent cross-turn Python functions (`ARC3_PERSISTENT_FUNCTIONS=1`), stale-state/no-op guards, and priority stream scheduling.
  - **Serving Stacks**:
    - `Qwen3.8-27B-FP8` (`vLLM 0.19` + 3-step Multi-Token Prediction `mtp`).
    - `Qwen3.8-Flash-Next` (`80B-A3B` MoE AutoRound `W4A16` + `BF16` Per-Layer Embeddings + 4-token `EAGLE` speculative decoding on `SGLang`, following the `dfranzen` / `albucino` setup).
- **ARC-AGI-2**:
  - **3-Engine Cross-Paradigm Hybrid**:
    1. **Stage 0**: Certified 100%-train-verified symbolic & dihedral (`D4`) ray/occlusion/channel experts (`Arc2SymbolicSolver` + `Astra V23` Leave-One-Out verified experts).
    2. **Phase 1**: `Qwen3.8-27B-FP8` multi-turn REPL program synthesis (`vLLM 0.19` + `MTP`).
    3. **Phase 2**: `Qwen3-4B` (`qwen3_4b_grids15_sft139`) NVARC per-task Test-Time Training (`Unsloth LoRA r=256` across 128 dihedral/color augmentations + length-grouped 8-view batched GPU-logsumexp `turbo_dfs` and `score_kgmon` voting).

---

## 2. What Worked

### A. Scaling Active-Parameter Reasoning Efficiency (`Qwen3.8-Flash-Next 80B-A3B MoE` + `EAGLE-4`)
- Moving from dense `Qwen3.8-27B-FP8` (`27B` active parameters/token) to **`Qwen3.8-Flash-Next` (`80B` total / `3B` active parameters/token, AutoRound `W4A16` + `BF16-PLE` + 4-token `EAGLE` speculative drafter)** on `SGLang` produced the single largest jump in ARC-AGI-3 performance:
  - **Dense `Qwen3.8-27B-FP8`**: `22.29 – 27.85` score on Kaggle (`12–16` levels cleared across 25 public games), bottlenecked by generation latency during deep multi-turn exploration.
  - **`Qwen3.8-Flash-Next` (`v3.2`)**: **`58.48` mean / `60.28` median (`53` levels cleared across 25 public games, `2` full game wins, `0` errors)** on both Colab `G4` and Kaggle `RTX PRO 6000`.

### B. AST Timeout Helper-Retention (`python_tool_sandbox.py` / `repl.py`)
- **Problem**: On complex planning games (`vc33`, `sb26`, `lp85`) and hard ARC-AGI-2 grids, the model often writes reusable helper functions (`def bfs_solve(...)`, `def simulate(...)`) at the top of a `python_exec` snippet, followed by a brute-force search loop that hits the `25s` subprocess timeout. When the subprocess is killed on timeout, the `__retained_funcs__` footer never prints—silently dropping all helper definitions and causing repeated `NameError` failures on subsequent turns.
- **Fix**: Added an AST salvage pass (`_salvage_definitions_on_timeout`) that parses timed-out code via `ast.parse`, extracts only top-level `import`s and `def`s (stripping top-level search loops), and executes them in a fast `2.5s` subprocess so helper functions are preserved.
- **Impact**: Eliminated post-timeout `NameError` loops; `vc33` jumped from **`0%` (`0/7` levels) to `71.4%` (`5/7` levels cleared)** in our 25-game evaluation.

### C. RadixCache-Friendly Hysteresis Context Trimming (`tool_agent.py`)
- **Problem**: Standard per-turn context trimming removes the oldest turn whenever context exceeds `58k` tokens. Because `58k` is immediately exceeded again on the very next turn, the prefix mutates on **every single turn**, destroying `SGLang` RadixAttention prefix cache hits (`0%` hit rate in late-game turns).
- **Fix**: Added `ARC3_CONTEXT_TRIM_HYSTERESIS_TOKENS=6144`. When `58k` is crossed, the harness trims down to `52k` in one shot, allowing the next **6–10 consecutive turns** to append cleanly to a static prefix and hit `100%` of the cached KV prefix (`+27%` late-game token throughput).

### D. Hysteresis Priority Stream Scheduler (`D'` + `1.10x` Stickiness)
- **Problem**: With `10` concurrent game slots competing for `25` games, small fluctuations in token-efficiency priority caused frequent pause/resume switching between games with nearly tied scores, evicting their KV caches from `SGLang`'s `43,808`-token RadixCache pool.
- **Fix**: Combined the token-aware `D'` priority formula with a `1.10x` stickiness multiplier for currently active streams so a paused game must beat an active game's priority by `>10%` to preempt it.

### E. FUSE Page-Cache Pre-Warming & Gateway Keepalive (`v3.2`)
- **Problem**: Reading `44 GB` of `W4A16` + `BF16-PLE` weights over Kaggle's network FUSE mount took `17–22 minutes` on cold competition workers, causing `bm.run()` to time out or the competition gateway to drop idle sessions.
- **Fix**: Added parallel 64 MiB background page-cache pre-warming while `uv` installs the wheelhouse, released `bm.run()` at `12 minutes` so `ARC3_WARMUP_ACTION_GAMES=1` immediately sends a `RESET` keepalive to the gateway, and extended the HTTP client retry window to `40 minutes` (`2400s`) while `SGLang` finishes loading.

### F. Cross-Paradigm Union Arbitration for ARC-AGI-2 (`kaggle_arc2/`)
- **Insight**: Symbolic program synthesis (`Qwen3.8-27B-FP8` REPL) and neural test-time training (`Qwen3-4B` NVARC `LoRA r=256` + 8-view logsumexp DFS) fail on orthogonal subsets of ARC-AGI-2:
  - REPL synthesis has **high precision (`~85%+`) whenever a Python `def transform(grid)` passes 100% of training demonstrations**, plus we auto-verify any `transform()` defined inside exploratory `python` tool calls.
  - NVARC TTT excels at holistic/noisy pattern completion where explicit Python rules time out.
- **Arbitrator**: Prioritizes tasks unverified by Phase 1 REPL in the Phase 2 NVARC queue, and assigns `attempt_1 = Verified REPL Program` and `attempt_2 = NVARC Top-1` whenever both exist and differ—capturing the union `V_repl ∪ N_nvarc` across the two submission slots.

---

## 3. What Didn't Work (Negative Results & Post-Mortem)

### A. Online Harness-in-the-Loop GRPO on Dense `Qwen3.8-27B-FP8` (`arc_rl/`)
- **What we built**: A full online RL pipeline (`arc_rl/train_grpo.py`, `arc_rl/harness_proxy.py`, `arc_rl/fp8_base.py`) using Hugging Face TRL's `AsyncGRPOTrainer` with **score centering (`score_centering=True`, TRL PR `#7760`)**, an aiohttp token-in/token-out proxy to avoid chat re-tokenization drift, and custom on-the-fly FP8 dequantization freezing the bottom 54 layers so a `27B` model could train a `rank-16` LoRA alongside `vLLM` on a single `96 GB` GPU.
- **What happened**:
  - Training converged cleanly and shaped token efficiency as intended: on 5 unseen holdout games, the GRPO LoRA reduced **tokens per action by `-31.5%`** (`529.7 -> 362.8`) and increased **actions executed per game by `+42%`** (`36.0 -> 51.2`).
  - However, on the full Kaggle evaluation (`22.29` score), the dense `27B` model was fundamentally capacity- and speed-bound compared to the `80B-A3B` `Flash-Next` MoE (`58.48`), and enabling Punica LoRA (`--enable-lora`) in `vLLM` disabled fused Cutlass FP8 kernels on LoRA layers.

### B. Aggressive Context Shrinking & Early `reasoning_effort="low"` Escalation (`v4` Ablation)
- **What we tested**: Shrinking the context window from `58k -> 48k`, truncating `python_exec` stdout from `3,072 -> 2,048` tokens, and escalating `reasoning_effort` down to `"low"` after just 2 context trims.
- **Result**: Regressed mean score by **`-5.00` points (`56.90 -> 51.90`)** and levels cleared from `19 -> 17` on our 8-game Colab `G4` benchmark (`ar25`, `cd87`, `ls20`, `sb26`, `sc25`, `tn36`, `tr87`, `vc33`). Truncating tool outputs cut off ASCII board state dumps, and dropping to `reasoning_effort="low"` caused shallow repetitive action loops on puzzle games (`tn36`, `vc33`).

### C. `FP8` KV Cache on `SGLang` with Hybrid Gated-DeltaNet + `EAGLE` (`flash_next_kv_mtp_bench.py`)
- **What we tested**: Passing `--kv-cache-dtype fp8_e4m3` to `SGLang` on `Qwen3.8-Flash-Next` to double the RadixCache token pool (`43,808 -> ~87,000` tokens) and increasing `--speculative-num-draft-tokens` from `4 -> 5`.
- **Result**: `SGLang`'s Triton KV-cache gather kernel crashed during EAGLE tree verification on the hybrid Gated-DeltaNet/Attention layers (`ValueError: Pointer argument (at 0) cannot be accessed from Triton`). Standard `bf16` KV cache with `--chunked-prefill-size 8192` and `4` draft tokens proved rock-solid.

### D. `FlashInfer` FP8 + Speculative `MTP` on Ada `SM89` (`4× L4`) in `vLLM 0.19`
- **What we found**: On `4× L4` (`SM89`), `vLLM 0.19` boots `/v1/models` with `FlashInfer` + `qwen3_5_mtp`, but crashes `EngineCore` on the very first token generated. Enforcing `--attention-backend TRITON_ATTN` (`VLLM_FORCE_ATTN_BACKEND=TRITON_ATTN`) and gating server readiness on a live `/v1/chat/completions` generation smoke test resolved the crash completely.

### E. Kaggle Container Python `3.13` vs. `cp312` Offline Wheelhouses
- **What we found**: Newly pushed CLI notebooks on Kaggle boot into a **Python `3.13` (`cp313`)** container, breaking offline `cp312` binary wheelhouses (`vLLM 0.19`, `SGLang`, `Unsloth`, `FlashAttention-2.8.2`). Extracting a standalone relocatable `CPython 3.12.16` build (`cpython-3.12-linux-x86_64.blob`) to `/tmp/py312` made all submissions immune to container Python upgrades.

---

## 4. Repository Structure

| Path | Description |
| :--- | :--- |
| [`kaggle_flash_next_v3_2/`](kaggle_flash_next_v3_2/) | **Final ARC-AGI-3 Submission (`v3.2`)** — `Qwen3.8-Flash-Next` (`80B-A3B` W4A16 + BF16-PLE + EAGLE-4 on `SGLang`) with AST timeout helper-retention, hysteresis context trimming, `D'` stickiness scheduler, and 40-min FUSE startup protection (`58.48` mean / `53` levels cleared). |
| [`kaggle_flash_next/`](kaggle_flash_next/), [`v3/`](kaggle_flash_next_v3/), [`v3_1/`](kaggle_flash_next_v3_1/) | Earlier iterative versions (`v2`, `v3`, `v3.1`) of the `Qwen3.8-Flash-Next` ARC-AGI-3 notebook. |
| [`kaggle_sub/`](kaggle_sub/) | ARC-AGI-3 `Qwen3.8-27B-FP8` + `vLLM 0.19` 3-step `MTP` + optional `TRL AsyncGRPO` LoRA submission notebook. |
| [`kaggle_arc2/`](kaggle_arc2/) | **Final ARC-AGI-2 Submission (`arc2-2026-hybrid-v1`)** — 3-Engine Hybrid (Stage 0 Symbolic/Astra V23 + Phase 1 `Qwen3.8-27B-FP8` `v3.2` REPL Agent + Phase 2 `Qwen3-4B` NVARC TTT + Phase 3 Cross-Paradigm Union Arbitrator) with universal `G4 (1×96 GB)` & `4×L4` auto-adaptation. |
| [`arc_rl/`](arc_rl/) | Online harness-in-the-loop `TRL AsyncGRPO` (`score_centering=True`) training pipeline, token-in/token-out `HarnessProxy`, and custom FP8 dequantization (`fp8_base.py`). |
| [`colab/`](colab/) | Colab `G4` (`RTX PRO 6000 96 GB`) benchmark drivers and controlled A/B ablation scripts (`flash_next_g4_bench.py`, `flash_next_v4_bench.py`, `flash_next_kv_mtp_bench.py`, `flash_next_v3_2_25game_bench.py`). |
| [`work_bundle/`](work_bundle/) | Patched `TAAF` + `ARC3-Inference` source tree with all `v3.2` agent, sandbox, vision, and scheduler upgrades. |
| [`reference/`](reference/) | Upstream reference `TAAF` bundle and AGI-Boys baseline patch for diff comparison. |
