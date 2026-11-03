"""Colab G4 (RTX PRO 6000 Blackwell 96 GB) Multi-Drain + Priority-Contention v3 vs v4 A/B Benchmark.

Designed specifically to reproduce the 9-hour 110-game competition dynamics that short 16-min 10-game
sprints miss:
  1. 15 games competing for 10 active GPU streams (MAX_ACTIVE_STREAMS=10 < 15 games), so _PriorityGate
     actively queues, preempts on 118k -> 58k context drains, and re-ranks games by progress pace & point density.
  2. 2100s (35 min) warm play budget per arm (~1.6M+ generated tokens per arm), pushing active games across
     multiple 118k -> 58k context drains so SOLVED_LEVEL_MEMORY_PATCH is genuinely exercised.
  3. Reports both the 15-game overall mean AND the 10-game unseen holdout mean, plus count of SOLVED_LEVEL_MEMORY
     pins and D' scheduler replacements.

Arms:
  - v3 (Control, identical to Kaggle arc3-flash-next-v3):
      dfranzen Milestone 2 setup_env (YIELD_TOKENS=2048) + Cache-Safe Drain-Synchronized SOLVED_LEVEL_MEMORY.
  - v4 (Candidate for Nov 2 final slot):
      v3 + ARC3_NOOP_REPEAT_GUARD=1 (+16 levels replicated on 25 public games, overridable on insistence)
         + D' Priority Scheduler (shiiin9 #26 38.98 LB: FormPace mean tokens/level + N-level triangular point
           density pricing on unstarted games + TAIL_FADE_FRACTION=0.4).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import sysconfig
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

STATUS_LOG = Path("/content/flash_v4_status.log")
WORK_ROOT = Path("/content/flash_work")

SERVED_MODEL_NAME = "flashnext"
SERVED_MODEL_PORT = 8001
PREFIX = "/tmp/sgl-intel"
SERVE_LOG = Path("/content/serve.log")

BF16_PLE_HANDLE = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
FP8_PLE_HANDLE = "dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple/transformers/default/1"
DRAFTER_HANDLE = "dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1"
PENNYROYAL_DATASET = "dfranzen/pennyroyal-v253"
BUNDLE_DATASET = "dfranzen/taaf-kaggle-source-bundle-copy"
COMPETITION_NAME = "arc-prize-2026-arc-agi-3"

TOKEN_MAP_SHA = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
EXPERT_TARGET = r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"
DENSE_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"

HOLDOUT_10 = ["ar25", "ft09", "lp85", "r11l", "re86", "sb26", "sc25", "tr87", "tu93", "vc33"]
CONTENTION_15 = HOLDOUT_10 + ["bp35", "sp80", "wa30", "ls20", "dc22"]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with STATUS_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def disk_free_gb(path: str = "/") -> float:
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / (1024 ** 3)


def run_cmd(cmd: list[str], *, check: bool = True, env: dict[str, str] | None = None, cwd: str | None = None):
    cmd_str = [str(c) for c in cmd]
    log(f"$ {shlex.join(cmd_str)}")
    res = subprocess.run(cmd_str, text=True, capture_output=True, env=env, cwd=cwd)
    if res.stdout.strip():
        print(res.stdout.strip()[-4000:], flush=True)
    if res.stderr.strip() and res.returncode != 0:
        print(res.stderr.strip()[-4000:], flush=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {shlex.join(cmd_str)}\n{res.stderr[-3000:]}")
    return res


def cleanup_kagglehub_archives() -> None:
    cache_root = Path.home() / ".cache/kagglehub"
    if not cache_root.exists():
        return
    removed = 0
    for ext in ("*.archive", "*.tar.gz", "*.tgz", "*.zip"):
        for p in cache_root.rglob(ext):
            if p.is_file():
                removed += p.stat().st_size
                p.unlink(missing_ok=True)
    if removed:
        log(f"Cleaned {removed / (1024**3):.2f} GiB of temporary kagglehub archives (free={disk_free_gb():.1f} GiB)")


def download_model_file_by_file(handle: str, target_dir: Path, workers: int = 3) -> Path:
    import kagglehub

    target_dir.mkdir(parents=True, exist_ok=True)
    meta_files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "hybrid_sources.json",
        "generation_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
    ]
    for mf in meta_files:
        dest = target_dir / mf
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        try:
            src = Path(kagglehub.model_download(handle, path=mf))
            if src.is_file() and not dest.exists():
                shutil.move(str(src), str(dest))
        except Exception:
            pass

    idx_path = target_dir / "model.safetensors.index.json"
    if not idx_path.is_file():
        raise FileNotFoundError(f"model.safetensors.index.json missing in {target_dir}")
    idx = json.loads(idx_path.read_text())
    shards = sorted(set(idx["weight_map"].values()))
    todo = [s for s in shards if not ((target_dir / s).is_file() and (target_dir / s).stat().st_size > 1024)]
    log(f"Model {handle}: {len(shards)} total shards, {len(todo)} remaining (free={disk_free_gb():.1f} GiB)")

    t0 = time.time()

    def _fetch_one(shard_name: str) -> tuple[str, float]:
        dest = target_dir / shard_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = Path(kagglehub.model_download(handle, path=shard_name))
        sz_gb = src.stat().st_size / (1024 ** 3)
        if src.resolve() != dest.resolve():
            shutil.move(str(src), str(dest))
        return shard_name, sz_gb

    done_count = 0
    done_gb = 0.0
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_fetch_one, s): s for s in todo}
            for fut in as_completed(futs):
                _, sz_gb = fut.result()
                done_count += 1
                done_gb += sz_gb
                if done_count % 5 == 0 or done_count == len(todo):
                    elapsed = max(0.1, time.time() - t0)
                    log(
                        f"  downloaded {done_count}/{len(todo)} shards ({done_gb:.1f} GiB in {elapsed:.0f}s, "
                        f"{done_gb * 1024 / elapsed:.0f} MiB/s, free={disk_free_gb():.1f} GiB)"
                    )
    cleanup_kagglehub_archives()
    return target_dir


def install_pennyroyal_venv_and_free_wheels(wheelhouse_dir: Path) -> Path:
    """Install Pennyroyal venv before downloading 169 GiB BF16-PLE, then reclaim ~4.4 GiB of .whl archives."""
    Path(PREFIX).mkdir(parents=True, exist_ok=True)
    venv = Path(PREFIX) / "venv"
    py_bin = venv / "bin/python"
    sgl_bin = venv / "bin/sglang"
    marker = Path(PREFIX) / "installed-bundle.txt"
    hot_dest = Path(PREFIX) / "hot_tokens_64k.pt"

    if not hot_dest.is_file():
        tok = find_unique(wheelhouse_dir, "hot_tokens_64k.pt", required=False) or find_unique(wheelhouse_dir, "flash-next-64k.pt")
        shutil.copy2(str(tok), str(hot_dest))

    if not sgl_bin.exists() or not marker.is_file():
        wheels = find_unique(wheelhouse_dir, "wheels", directory=True)
        lock = find_unique(wheelhouse_dir, "requirements.lock")
        sgl_wheel = sorted(wheels.glob("sglang-*.whl"))[0]
        for pkg in ("openai", "httpx", "pydantic", "pillow", "numpy", "matplotlib"):
            run_cmd(
                [sys.executable, "-m", "pip", "install", "--quiet", "--no-index", f"--find-links={wheels}", pkg],
                check=False,
            )
        log("Installing Pennyroyal wheelhouse into /tmp/sgl-intel/venv...")
        install_env = dict(os.environ, UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1")
        install_env.pop("PYTHONPATH", None)
        uv_wheel = sorted(wheels.glob("uv-*.whl"))[0]
        uv_bin = Path(PREFIX) / "bin/uv"
        uv_bin.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(uv_wheel) as zf:
            member = [n for n in zf.namelist() if Path(n).name == "uv" and not n.endswith("/")][0]
            uv_bin.write_bytes(zf.read(member))
        uv_bin.chmod(0o755)
        if not py_bin.exists():
            py312 = shutil.which("python3.12") or sys.executable
            run_cmd([str(uv_bin), "venv", "--python", py312, str(venv)], env=install_env)
        inst = [str(uv_bin), "pip", "install", "--python", str(py_bin), "--no-index", "--find-links", str(wheels)]
        run_cmd(inst + ["-r", str(lock)], env=install_env)
        run_cmd(inst + ["--reinstall", "--no-deps", str(sgl_wheel)], env=install_env)
        marker.write_text("installed")

        # Reclaim ~4.4 GiB of wheel files now that venv + host deps + hot_tokens_64k.pt are installed
        shutil.rmtree(wheels, ignore_errors=True)
        log(f"Reclaimed Pennyroyal .whl directory after venv install (free={disk_free_gb():.1f} GiB)")

    return hot_dest


def ensure_assets() -> dict[str, Path]:
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "kaggle", "kagglehub", "pandas", "pyarrow"])
    import kagglehub

    log(f"Initial free disk: {disk_free_gb():.1f} GiB")

    wheelhouse_dir = Path(kagglehub.dataset_download(PENNYROYAL_DATASET))
    cleanup_kagglehub_archives()
    log(f"Pennyroyal wheelhouse -> {wheelhouse_dir} (free={disk_free_gb():.1f} GiB)")
    hot_tokens_path = install_pennyroyal_venv_and_free_wheels(wheelhouse_dir)

    orig_bundle_dir = Path(kagglehub.dataset_download(BUNDLE_DATASET))
    cleanup_kagglehub_archives()
    log(f"TAAF bundle -> {orig_bundle_dir}")

    cached_comp = Path("/root/.cache/kagglehub/competitions") / COMPETITION_NAME
    comp_dir = cached_comp if (cached_comp / "environment_files").is_dir() else (WORK_ROOT / "comp_arc3")
    if not (comp_dir / "environment_files").is_dir():
        comp_dl = Path(kagglehub.competition_download(COMPETITION_NAME))
        cleanup_kagglehub_archives()
        if comp_dl.is_dir():
            comp_dir = comp_dl
        elif comp_dl.is_file() and zipfile.is_zipfile(comp_dl):
            comp_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(comp_dl) as zf:
                zf.extractall(comp_dir)
            comp_dl.unlink(missing_ok=True)
    log(f"Competition assets -> {comp_dir} (free={disk_free_gb():.1f} GiB)")

    drafter_dir = Path(kagglehub.model_download(DRAFTER_HANDLE))
    cleanup_kagglehub_archives()
    log(f"Drafter -> {drafter_dir} (free={disk_free_gb():.1f} GiB)")

    # After venv + drafter + bundle are already installed, BF16-PLE needs 168.8 GiB.
    # Require >=176.5 GiB free (leaves ~8 GiB post-download for JIT cache, transcripts & logs).
    bf16_dir = WORK_ROOT / "flash_next_bf16ple"
    fp8_dir = WORK_ROOT / "flash_next_fp8ple"
    prefer_fp8 = os.environ.get("FORCE_FP8_PLE", "0").strip() == "1"
    if (bf16_dir / "model.safetensors.index.json").is_file():
        model_dir = download_model_file_by_file(BF16_PLE_HANDLE, bf16_dir, workers=2)
    elif not prefer_fp8 and disk_free_gb() >= 176.5:
        log(f"Free disk {disk_free_gb():.1f} GiB >= 176.5 GiB -> downloading Intel BF16-PLE ({BF16_PLE_HANDLE})")
        model_dir = download_model_file_by_file(BF16_PLE_HANDLE, bf16_dir, workers=2)
    else:
        log(f"Free disk {disk_free_gb():.1f} GiB < 176.5 GiB (or FORCE_FP8_PLE=1) -> using Albucino FP8-PLE ({FP8_PLE_HANDLE})")
        model_dir = download_model_file_by_file(FP8_PLE_HANDLE, fp8_dir, workers=4)

    return {
        "wheelhouse": wheelhouse_dir,
        "hot_tokens": hot_tokens_path,
        "orig_bundle": orig_bundle_dir,
        "comp": comp_dir,
        "drafter": drafter_dir,
        "model": model_dir,
    }


def find_unique(root: Path, name: str, *, directory: bool = False, required: bool = True) -> Path | None:
    root = Path(root)
    direct = root if root.name == name else root / name
    valid = (lambda p: p.is_dir()) if directory else (lambda p: p.is_file())
    if valid(direct):
        return direct
    hits = sorted(p for p in root.rglob(name) if valid(p))
    if len(hits) > 1:
        raise RuntimeError(f"Multiple {name} matches under {root}: {hits}")
    if hits:
        return hits[0]
    if required:
        raise FileNotFoundError(f"{name} not found under {root}")
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def indexed_shards(root: Path):
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    for name in names:
        p = root / name
        if not p.is_file() or p.stat().st_size < 8:
            raise RuntimeError(f"Missing/empty shard: {p}")
    return index, names


def prepare_draft_view(target: Path, source_root: Path, work_root: Path) -> Path:
    configs = [source_root / "config.json"] if (source_root / "config.json").is_file() else []
    configs += sorted(p for p in source_root.rglob("config.json") if p not in configs)
    candidates = []
    for path in configs:
        cfg = json.loads(path.read_text())
        q = cfg.get("quantization_config", {})
        if q.get("quant_method") == "compressed-tensors" and "mtp_routed_experts" in q.get("config_groups", {}):
            candidates.append((path.parent, cfg))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected 1 MTP checkpoint under {source_root}, found {len(candidates)}")
    source, cfg = candidates[0]
    _, shards = indexed_shards(source)
    cfg = copy.deepcopy(cfg)
    q = cfg["quantization_config"]
    q["config_groups"]["mtp_routed_experts"]["targets"] = [EXPERT_TARGET]
    q["ignore"] = [DENSE_IGNORE]
    identity = json.dumps({"source": str(source.resolve()), "target": str(target.resolve()), "config": cfg}, sort_keys=True)
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    view = work_root / f"draft-view-{suffix}"
    view.mkdir(parents=True, exist_ok=True)
    (view / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    links = {name: source / name for name in shards}
    links["model.safetensors.index.json"] = source / "model.safetensors.index.json"
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"):
        if (target / name).is_file():
            links[name] = target / name
    for name, src in links.items():
        dest = view / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest.exists() or dest.is_symlink()):
            dest.symlink_to(src.resolve())
    return view


def setup_patched_bundle(orig_bundle: Path, comp_dir: Path, wheelhouse: Path) -> tuple[Path, str]:
    bundle_dir = WORK_ROOT / "taaf-source-share"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    shutil.copytree(orig_bundle, bundle_dir)
    run_cmd(["chmod", "-R", "u+w", str(bundle_dir)])

    base_patch = Path("/content/harness-changes.patch")
    raw = base_patch.read_bytes()
    if not raw.endswith(b"\n"):
        base_patch.write_bytes(raw + b"\n")
    run_cmd(["git", "apply", "--include=ARC3-Inference/*", "-v", str(base_patch)], cwd=str(bundle_dir / "src"))

    wheels_dir = find_unique(comp_dir, "arc_agi_3_wheels", directory=True)
    site_packages = Path(sysconfig.get_path("purelib"))
    run_cmd(
        [
            sys.executable, "-m", "pip", "install", "--quiet",
            "--no-index", "--no-deps",
            f"--find-links={wheels_dir}",
            f"--target={site_packages}",
            "arc-agi", "arcengine",
        ]
    )
    wh_wheels = find_unique(wheelhouse, "wheels", directory=True, required=False)
    if wh_wheels is not None:
        for pkg in ("openai", "httpx", "pydantic", "pillow", "numpy", "matplotlib"):
            run_cmd(
                [sys.executable, "-m", "pip", "install", "--quiet", "--no-index", f"--find-links={wh_wheels}", pkg],
                check=False,
            )
    return bundle_dir, sys.executable


def ensure_sglang_server(assets: dict[str, Path]) -> float:
    url_health = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health"
    try:
        with urllib.request.urlopen(url_health, timeout=3) as r:
            if r.status == 200:
                log("SGLang server already healthy on :8001!")
                return 0.0
    except Exception:
        pass

    model_dir = assets["model"]
    drafter_dir = assets["drafter"]
    tok = assets["hot_tokens"]

    Path(PREFIX).mkdir(parents=True, exist_ok=True)
    venv = Path(PREFIX) / "venv"
    sgl_bin = venv / "bin/sglang"
    assert sgl_bin.exists(), f"Expected installed Pennyroyal SGLang at {sgl_bin}"

    cu13 = sorted(venv.glob("lib/python*/site-packages/nvidia/cu13"))[0]
    cuda_home = str(cu13)
    libdir = cu13 / "lib"
    lib64 = cu13 / "lib64"
    if not lib64.exists() and not lib64.is_symlink():
        lib64.symlink_to("lib")
    for so in sorted(libdir.glob("*.so.*")):
        link = libdir / re.sub(r"\.so\..*$", ".so", so.name)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(so.name)
    if not (libdir / "libcuda.so").exists():
        cands = []
        for d in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
            if Path(d).is_dir():
                cands += sorted(p for p in Path(d).glob("libcuda.so*") if p.is_file())
        if not cands:
            raise RuntimeError("libcuda.so not found")
        (libdir / "libcuda.so").symlink_to(cands[0].resolve())

    cxx = next((c for c in ("g++-15", "g++-14", "g++-13", "g++-12", "g++-11", "g++") if shutil.which(c)), None)
    cc = cxx.replace("g++", "gcc")
    cache = f"{PREFIX}/cache"
    for name in ("huggingface", "torch", "torchinductor", "triton", "flashinfer", "cuda", "sglang/jit"):
        Path(cache, name).mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({
        "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_HOME": cuda_home, "CUDACXX": f"{cuda_home}/bin/nvcc",
        "PATH": f"{cuda_home}/bin:{venv}/bin:" + env.get("PATH", ""),
        "LD_LIBRARY_PATH": f"{cuda_home}/lib64:{cuda_home}/lib:/usr/lib64-nvidia:" + env.get("LD_LIBRARY_PATH", ""),
        "CC": cc, "CXX": cxx, "CUDAHOSTCXX": cxx,
        "TORCH_CUDA_ARCH_LIST": "12.0",
        "HF_HOME": f"{cache}/huggingface", "XDG_CACHE_HOME": cache,
        "TORCH_HOME": f"{cache}/torch", "CUDA_CACHE_PATH": f"{cache}/cuda",
        "TORCHINDUCTOR_CACHE_DIR": f"{cache}/torchinductor", "TRITON_CACHE_DIR": f"{cache}/triton",
        "FLASHINFER_WORKSPACE_BASE": f"{cache}/flashinfer",
        "SGLANG_CACHE_DIR": f"{cache}/sglang", "SGLANG_JIT_CACHE_DIR": f"{cache}/sglang/jit",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",
        "MAX_JOBS": "8", "CMAKE_BUILD_PARALLEL_LEVEL": "8",
        "FLASHINFER_NINJA_JOBS": "8", "FLASHINFER_NVCC_THREADS": "2", "TORCHINDUCTOR_COMPILE_THREADS": "8",
        "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM": "1", "SGLANG_SM120_ONLINE_MXFP8": "0",
        "SGLANG_SM120_LOWM_FP8_WEIGHT": "0", "SGLANG_SM120_LM_HEAD_FP8": "0",
        "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_NUMA_BIND_V2": "false",
        "SGLANG_MM_PREPROCESS_DEVICE": "cpu", "NUMPY_MADVISE_HUGEPAGE": "0",
    })

    draft_view = prepare_draft_view(model_dir, drafter_dir, Path(PREFIX))
    assert sha256_file(tok) == TOKEN_MAP_SHA
    assert sha256_file(model_dir / "tokenizer.json") == TOKENIZER_SHA

    ctx = (116 + 12 + 8) * 1024
    graph_bs = [1, 2, 4, 7, 8, 9, 10]
    args = [
        str(sgl_bin), "serve", "--model-path", str(model_dir), "--load-format", "safetensors",
        "--model-loader-extra-config", '{"enable_multithread_load":false}',
        "--served-model-name", SERVED_MODEL_NAME, "--host", "127.0.0.1",
        "--port", str(SERVED_MODEL_PORT), "--tensor-parallel-size", "1",
        "--dtype", "bfloat16", "--quantization", "auto-round",
        "--kv-cache-dtype", "fp8_e4m3", "--mem-fraction-static", "0.96",
        "--context-length", str(ctx), "--page-size", "64",
        "--max-running-requests", "10", "--chunked-prefill-size", "8192",
        "--max-prefill-tokens", "16384",
        "--cuda-graph-max-bs-decode", "10",
        "--cuda-graph-bs-decode", *map(str, graph_bs),
        "--mamba-ssm-dtype", "bfloat16", "--max-mamba-cache-size", "60",
        "--mamba-radix-cache-strategy", "extra_buffer", "--mamba-track-interval", "64",
        "--mamba-backend", "flashinfer",
        "--linear-attn-decode-backend", "flashinfer",
        "--linear-attn-prefill-backend", "flashinfer",
        "--moe-runner-backend", "auto", "--ple-offload-embedding", "--trust-remote-code",
        "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
        "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
        "--default-chat-template-kwargs", '{"preserve_thinking":true}',
        "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
        "--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging",
        "--weight-loader-prefetch-checkpoints", "--weight-loader-drop-cache-after-load",
        "--gdn-mtp-cache-mode", "none",
        "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4",
        "--speculative-draft-model-path", str(draft_view),
        "--speculative-draft-model-quantization", "compressed-tensors",
        "--speculative-moe-runner-backend", "auto",
        "--speculative-draft-kv-cache-dtype", "fp8_e4m3",
        "--speculative-accept-threshold-single", "1.0",
        "--speculative-accept-threshold-acc", "1.0",
        "--speculative-token-map", str(tok),
    ]
    tmpl = model_dir / "chat_template.jinja"
    if tmpl.is_file():
        args += ["--chat-template", str(tmpl)]

    log("Launching Pennyroyal SGLang server...")
    t0 = time.time()
    with SERVE_LOG.open("ab", buffering=0) as logf:
        proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    log(f"SGLang pid={proc.pid}, waiting for /health...")
    last_rep = -30
    while time.time() - t0 < 2400:
        if proc.poll() is not None:
            tail = SERVE_LOG.read_text(errors="replace")[-12000:]
            raise RuntimeError(f"SGLang exited early ({proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(url_health, timeout=5) as r:
                if r.status == 200:
                    boot_s = time.time() - t0
                    log(f"SGLang READY after {boot_s:.1f}s!")
                    return boot_s
        except Exception:
            pass
        el = int(time.time() - t0)
        if el - last_rep >= 30:
            log(f"  SGLang loading... {el}s")
            last_rep = el
        time.sleep(3)
    raise RuntimeError("SGLang startup timed out after 2400s")


def flush_sglang_cache() -> None:
    req = urllib.request.Request(f"http://127.0.0.1:{SERVED_MODEL_PORT}/flush_cache", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            log(f"POST /flush_cache -> status {r.status}")
    except Exception as exc:
        log(f"Warning: /flush_cache returned {exc!r}")
    time.sleep(2)


ARM_RUNNER_CODE = r'''import asyncio
import math
import os
import pickle
import re as _re_slm
import sys
import threading
import time
import types
from datetime import datetime, timedelta
from pathlib import Path

bundle_dir = Path(sys.argv[1])
env_dir = Path(sys.argv[2])
work_dir = Path(sys.argv[3])
runtime_s = float(sys.argv[4])
games_filter = [g.strip() for g in sys.argv[5].split(",") if g.strip()] if len(sys.argv) > 5 else []
arm_mode = sys.argv[6] if len(sys.argv) > 6 else "v3"

src_root = bundle_dir / "src"
for repo in sorted(src_root.iterdir()):
    if repo.is_dir():
        cand = repo / "src" if (repo / "src").is_dir() else repo
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))

work_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLBACKEND"] = "Agg"
os.environ["TAAF_RUN_AS_SUBMISSION"] = "0"
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "0"
os.environ["ONLY_RESET_LEVELS"] = "true"
os.environ["RECORDINGS_DIR"] = str(work_dir / "server_recording")

with open(bundle_dir / "deploy_target.pkl", "rb") as f:
    target = pickle.load(f)
target.actual_run_as_submission = False
target.is_competition_rerun = False

with open(bundle_dir / "benchmark_initial.pkl", "rb") as f:
    bm = pickle.load(f)
bm.job_dir = work_dir

# --- 1. Cache-Safe Drain-Synchronized SOLVED-LEVEL MEMORY (enabled in both v3 and v4) ---
import inference.agent.tool_agent as _ta_slm
from inference.agent.runtime_state import load_runtime_state as _slm_load_state

_SLM_ACTIONS = 30
_SLM_REASONING_CHARS = 900
_SLM_RULE_CHARS = 600
_SLM_FULL_LEVELS = 3
_SLM_ASK_TURNS = 2
_SLM_HEADER = "=== SOLVED LEVELS (facts from this game + your own conclusions; kept for the whole game) ==="
_SLM_FOOTER = "=== END SOLVED LEVELS ==="


def _slm_text(msg):
    parts = []
    for key in ("content", "reasoning_content", "reasoning"):
        value = msg.get(key) if isinstance(msg, dict) else None
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts += [p.get("text", "") for p in value if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return "\n".join(p for p in parts if p)


def _slm_level_actions(entries, level):
    out, prev = [], None
    for entry in entries or []:
        action = str(getattr(entry, "action", "") or "").strip()
        if action and prev == level:
            if action.upper().startswith("RESET"):
                out = []
            else:
                out.append(action)
        lv = getattr(getattr(entry, "frame", None), "level", None)
        if isinstance(lv, int):
            prev = lv
    return out


def _slm_find_rule(text, level):
    m = _re_slm.search(r"Rule for level\s*%d\s*:[ \t]*" % int(level), text or "", _re_slm.I)
    if not m:
        return None
    lines = text[m.end():].split("\n")
    first = lines[0].strip()
    if len(first) >= 40:
        rule = first
    else:
        out = [first] if first else []
        for line in lines[1:6]:
            if not line.strip():
                if out:
                    break
                continue
            out.append(line.strip())
        rule = " ".join(out)
    rule = " ".join(rule.split())[:_SLM_RULE_CHARS]
    return rule if len(rule) >= 20 else None


def _slm_block(levels):
    if not levels:
        return ""
    lines = [_SLM_HEADER]
    full_from = max(0, len(levels) - _SLM_FULL_LEVELS)
    for i, rec in enumerate(levels):
        lines.append(f"Level {rec['level']} - cleared after {rec['n_actions']} actions.")
        lines.append(f"  Your rule: {rec['rule'] if rec.get('rule') else 'not stated'}")
        if i >= full_from:
            tail = rec["actions"]
            skipped = rec["n_actions"] - len(tail)
            lines.append(
                "  Last actions before it cleared"
                + (f" ({skipped} earlier omitted)" if skipped > 0 else "")
                + ": "
                + (", ".join(tail) if tail else "none recorded")
            )
            if rec.get("reasoning"):
                lines.append("  Your reasoning in the turn that cleared it: " + rec["reasoning"])
    lines.append("Reuse these where the new level has the same controls; verify with one probe before relying on them.")
    lines.append(_SLM_FOOTER)
    return "\n".join(lines)


def _slm_refresh_system_prompt(self):
    block = _slm_block(getattr(self, "_slm_levels", []))
    if block == getattr(self, "_slm_block_text", ""):
        return False
    prompt = self._system_prompt
    old = getattr(self, "_slm_block_text", "")
    if old and old in prompt:
        prompt = prompt.replace(old, block)
    else:
        prompt = prompt.rstrip("\n") + "\n\n" + block
    self._system_prompt = prompt
    self._slm_block_text = block
    print(f"SOLVED_LEVEL_MEMORY pinned: {len(self._slm_levels)} level(s), {len(block)} chars", flush=True)
    return True


def _slm_before_turn(self, state_path):
    if state_path is None:
        return
    levels = self.__dict__.setdefault("_slm_levels", [])
    for rec in levels:
        if not rec.get("rule") and rec.get("asked"):
            for msg in reversed(self._history_messages[-8:]):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    rule = _slm_find_rule(_slm_text(msg), rec["level"])
                    if rule:
                        rec["rule"] = rule
                        break
    current = int((self._last_step_summary or {}).get("level") or 1)
    recorded = {rec["level"] for rec in levels}
    missing = [lv for lv in range(1, current) if lv not in recorded]
    if not missing:
        return
    _, entries = _slm_load_state(state_path)
    reasoning = ""
    for msg in reversed(self._history_messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            text = str(msg.get("reasoning_content") or msg.get("reasoning") or msg.get("content") or "")
            reasoning = " ".join(text.split())[-_SLM_REASONING_CHARS:]
            break
    for lv in missing:
        actions = _slm_level_actions(entries, lv)
        levels.append({
            "level": lv,
            "n_actions": len(actions),
            "actions": actions[-_SLM_ACTIONS:],
            "reasoning": reasoning if lv == missing[-1] else "",
            "rule": None,
            "asks_left": _SLM_ASK_TURNS,
        })
    levels.sort(key=lambda rec: rec["level"])


_slm_original_analyze = _ta_slm.ToolAgent.analyze
_slm_original_user_prompt = _ta_slm.ToolAgent._build_user_prompt
_slm_original_evicted = _ta_slm.ToolAgent._note_history_evicted
_slm_original_trim = _ta_slm.ToolAgent._trim_messages_for_context


def _slm_analyze(self, state_path, *args, **kwargs):
    self._slm_state_path = state_path
    try:
        _slm_before_turn(self, state_path)
    except Exception as exc:
        print(f"SOLVED_LEVEL_MEMORY error (turn unchanged): {exc}", flush=True)
    return _slm_original_analyze(self, state_path, *args, **kwargs)


def _slm_user_prompt(self, *args, **kwargs):
    prompt = _slm_original_user_prompt(self, *args, **kwargs)
    try:
        for rec in getattr(self, "_slm_levels", []):
            if not rec.get("rule") and rec.get("asks_left", 0) > 0:
                rec["asks_left"] -= 1
                rec["asked"] = True
                prompt += (
                    f"\n\nBefore your tool call, write a line starting `Rule for level {rec['level']}:` "
                    "that states the goal, what each control did and the plan that won (at most 5 lines). "
                    "It is kept for the rest of the game."
                )
                break
    except Exception:
        pass
    return prompt


def _slm_note_evicted(self):
    _slm_original_evicted(self)
    try:
        _slm_before_turn(self, getattr(self, "_slm_state_path", None))
        _slm_refresh_system_prompt(self)
    except Exception as exc:
        print(f"SOLVED_LEVEL_MEMORY error (prompt unchanged): {exc}", flush=True)


def _slm_trim(self, messages, *args, **kwargs):
    result = _slm_original_trim(self, messages, *args, **kwargs)
    if (
        len(result) < len(messages)
        and result
        and result[0].get("role") == "system"
        and result[0].get("content") != self._system_prompt
    ):
        result = [{**result[0], "content": self._system_prompt}, *result[1:]]
    return result


_ta_slm.ToolAgent.analyze = _slm_analyze
_ta_slm.ToolAgent._build_user_prompt = _slm_user_prompt
_ta_slm.ToolAgent._note_history_evicted = _slm_note_evicted
_ta_slm.ToolAgent._trim_messages_for_context = _slm_trim
print("SOLVED_LEVEL_MEMORY_PATCH on", flush=True)

# --- 2. Optional D' Priority Scheduler (enabled in v4; attribution: shiiin9/affectify-arc-dfranzen-form-d1, CC-BY 4.0) ---
if arm_mode == "v4":
    import inference.framework.solver as _fp_solver

    D_PRIME = {
        "level_pow": 0.5, "eff_const": 300.0, "eff_pow": 2.5, "pace_pow": 0.4,
        "pace_ref": 30_000.0, "c_action": 0.1, "action_scale": 115.0, "c_token": 0.9,
        "token_scale": 225_000.0, "lin_floor": 0.1, "rel_pow": 0.5,
        "b1": 10.0, "b2": 14.0, "b3": 16.0, "fade": 0.4,
    }
    SCALE = 1000.0
    _TL = threading.local()

    class FormPace:
        def __init__(self):
            self.completed = 0
            self.tokens = 0.0

        def record_completion(self, tokens_used: float, reference_tokens: float = 0.0) -> None:
            self.completed += 1
            self.tokens += max(0.0, float(tokens_used))

        def cost_multiplier(self) -> float:
            return self.tokens / self.completed if self.completed else 0.0

    def _levels(total):
        try:
            n = int(total)
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

    def d_parts(level: int, actions: float, tokens: float, pace: float, total_levels, params=D_PRIME):
        level = max(1, int(level))
        n = _levels(total_levels) or 10
        n_clamped = min(10, max(6, n))
        norm = 55 / (n_clamped * (n_clamped + 1) / 2)
        a = max(0.0, float(actions))
        t = max(0.0, float(tokens))
        A = (1.0 + params["level_pow"] * (level - 1)) * norm
        A *= (params["eff_const"] / (params["eff_const"] + a)) ** params["eff_pow"]
        if pace > 0:
            A *= min(4.0, max(0.25, (params["pace_ref"] / max(1.0, pace)) ** params["pace_pow"]))
        scale = params["token_scale"]
        rel = pace if pace > 0 else params["pace_ref"]
        scale *= min(2.0, max(0.5, rel / params["pace_ref"])) ** params["rel_pow"]
        floor = params["lin_floor"]
        C = (
            params["c_action"] * max(floor, 1.0 - a / params["action_scale"])
            + params["c_token"] * max(floor, 1.0 - t / scale)
        )
        remaining = max(0, n - level)
        B = (0.0, params["b1"], params["b2"], params["b3"])[min(remaining, 3)]
        return A * C, B

    def d_priority(state, *, tail_fraction=None, **_ignored) -> int:
        phi = 1.0 if tail_fraction is None else min(1.0, max(0.0, float(tail_fraction)))
        u, v = d_parts(
            state.level,
            state.actions,
            state.tokens,
            float(getattr(state, "cost_multiplier", 0.0) or 0.0),
            getattr(state, "total_levels", None),
        )
        return max(1, int((u + v * phi) * SCALE))

    _ta_slm.priority_value = d_priority
    _ta_slm.ProgressPace = FormPace
    os.environ["ARC3_PRIORITY_PACE"] = "1"
    os.environ["ARC3_PRIORITY_REFRESH_QUEUE"] = "1"
    os.environ["ARC3_PRIORITY_TAIL_FADE"] = "1"
    os.environ["ARC3_PRIORITY_TAIL_FADE_FRACTION"] = str(D_PRIME["fade"])

    session_cls = _fp_solver._HarnessGameSession
    orig_play = session_cls.play

    def _d_play(self, *args, **kwargs):
        run = getattr(getattr(self, "game", None), "game_run", None)
        _TL.n = getattr(run, "number_of_levels", None)
        try:
            return orig_play(self, *args, **kwargs)
        finally:
            _TL.n = None

    session_cls.play = _d_play

    gate_cls = _ta_slm._PriorityGate
    orig_acquire = gate_cls.acquire
    fresh_floor = _ta_slm._PRIORITY_UNTRIMMED_BASE - _ta_slm._PRIORITY_BAND
    snapshot_cls = _ta_slm.PrioritySnapshot

    def _d_acquire(self, priority):
        if priority < fresh_floor:
            return orig_acquire(self, priority)
        snap = snapshot_cls(1, 0, 0.0, 0.0, getattr(_TL, "n", None))
        value = self._snapshot_priority(snap, time.monotonic())
        with self._cond:
            self._enqueue_and_wait(value, snap)

    gate_cls.acquire = _d_acquire
    print("D_PRIME_PRIORITY_PATCH on (tail_fade=0.4, pace=FormPace, fresh-first repriced)", flush=True)

import arc_agi
import taaf.game_api

spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(env_dir))
arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(env_dir))
game_ids = [e.game_id for e in arcade.available_environments]
if games_filter:
    game_ids = [gid for gid in game_ids if any(gid.startswith(gf) for gf in games_filter)]
bm.games = [taaf.game_api.GameAPI(env_name=gid, arcade_spec=spec) for gid in game_ids]

bm.solver.analyzer_timeout = 2400.0
bm.solver.concurrency = len(bm.games)
bm.solver.max_runtime_s_per_game = runtime_s
bm.solver.max_actions_per_game = None
bm.solver.save_request_logs = False
bm.n_passes = 1
bm.game_weights = None

print(f"Running arm [{arm_mode}] in {work_dir} on {len(bm.games)} games ({[g.env_name for g in bm.games]}), runtime_s={runtime_s}", flush=True)
soft_end = datetime.now() + timedelta(seconds=runtime_s + 180.0)
asyncio.run(bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=False))
'''


def base_v3_env() -> dict[str, str]:
    server_url = f"http://127.0.0.1:{SERVED_MODEL_PORT}/v1"
    return {
        "USE_TF": "0",
        "TRANSFORMERS_NO_TF": "1",
        "TRANSFORMERS_NO_TORCHVISION": "1",
        "LOCAL_ANALYZER_BASE_URL": server_url,
        "OPENAI_BASE_URL": server_url,
        "LOCAL_ANALYZER_PROVIDER": "vllm",
        "OPENAI_PROVIDER": "vllm",
        "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL_NAME,
        "INFERENCE_ANALYZER_MODEL": SERVED_MODEL_NAME,
        "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
        "ARC3_REASONING_HISTORY_KEY": "reasoning_content",
        "LOCAL_ANALYZER_ENABLE_THINKING": "true",
        "LOCAL_ANALYZER_TOOL_STEPS": "0",
        "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
        "LOCAL_ANALYZER_YIELD_SECONDS": "0",
        "LOCAL_ANALYZER_YIELD_TOKENS": "2048",
        "LOCAL_ANALYZER_TEMPERATURE": "0.7",
        "LOCAL_ANALYZER_TOP_P": "0.95",
        "LOCAL_ANALYZER_TOP_K": "20",
        "LOCAL_ANALYZER_CONTEXT_WINDOW": str((116 + 12) * 1024),
        "LOCAL_ANALYZER_MAX_OUTPUT": str(12 * 1024),
        "ARC3_CONTEXT_DRAIN_TOKENS": str(58 * 1024),
        "ARC3_HISTORY_ASSISTANT_TURNS": "150",
        "ARC3_HISTORY_TURN_DRAIN": "30",
        "ARC3_HISTORY_DRAIN_COALESCE": "1",
        "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "3072",
        "ARC3_MIDDLE_TRUNCATION": "1",
        "ARC3_ACTION_ECHO": "1",
        "ARC3_EXPLAIN_GAMEPLAY_CHANGED": "1",
        "ARC3_NEW_CHANGED_PROMPTS": "1",
        "ARC3_LEVEL_TRANSFER_GUIDANCE": "1",
        "ARC3_ACTION_INFO": "1",
        "ARC3_DEDUPE_MULTICALL_LINE": "1",
        "ARC3_FRAME_DIFF_HINT": "1",
        "MULTIMODAL_CONTEXT": "current_grid",
        "MULTIMODAL_UPSCALE": "10",
        "ARC3_DIFF_IMAGE": "1",
        "ARC3_GAMEOVER_DIFF_IMAGE": "1",
        "ARC3_ANIMATION": "1",
        "ARC3_GUARDS_FROM_LEVEL": "2",
        "ARC3_BATCH_NOOP_BLOCK": "1",
        "ARC3_STALE_STATE_BLOCK": "1",
        "ARC3_MEMORY_SECTIONS": "off",
        "ARC3_PERSISTENT_FUNCTIONS": "1",
        "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
        "ARC3_PERSISTENT_FUNCTIONS_IMPORTS": "1",
        "EXPOSE_UNDO": "on",
        "ARC3_MAX_ACTIVE_STREAMS": "10",
        "ARC3_PRIORITY_REFRESH_QUEUE": "1",
        "ARC3_PRIORITY_PACE": "0",
        "ARC3_PRIORITY_TAIL_FADE": "1",
        "ARC3_PRIORITY_TAIL_FADE_FRACTION": "0.2",
        "ARC3_PRIORITY_TAIL_LOOKUP": "remaining",
        "ARC3_PRIORITY_SCORE_NORMALIZATION": "1",
    }


def candidate_v4_env() -> dict[str, str]:
    env = base_v3_env()
    env.update({
        "ARC3_NOOP_REPEAT_GUARD": "1",
    })
    return env


def run_arm(
    name: str,
    arm_env: dict[str, str],
    runner_py: str,
    bundle_dir: Path,
    env_dir: Path,
    runtime_s: float,
    games_csv: str,
) -> dict:
    flush_sglang_cache()
    out_dir = WORK_ROOT / f"out_{name}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner_script = WORK_ROOT / "_run_v4_arm.py"
    runner_script.write_text(ARM_RUNNER_CODE)

    full_env = dict(os.environ)
    full_env.update(arm_env)
    arm_log = WORK_ROOT / f"arm_{name}.log"
    log(f"=== Starting Arm [{name}] (runtime_s={runtime_s}, games={games_csv}) ===")
    t0 = time.time()
    with arm_log.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(
            [runner_py, str(runner_script), str(bundle_dir), str(env_dir), str(out_dir), str(runtime_s), games_csv, name],
            env=full_env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    wall_s = time.time() - t0
    bm_json = out_dir / "benchmark.json"
    log_text = arm_log.read_text(errors="replace") if arm_log.is_file() else ""
    slm_pins = log_text.count("SOLVED_LEVEL_MEMORY pinned:")
    summary: dict = {"arm": name, "returncode": proc.returncode, "wall_s": round(wall_s, 1), "slm_pins": slm_pins}
    if bm_json.is_file():
        data = json.loads(bm_json.read_text())
        game_runs = data.get("game_runs") or data.get("games") or data.get("results") or []
        per_game = {}
        scores_all = []
        scores_holdout10 = []
        levels_all = 0
        levels_holdout10 = 0
        for g in game_runs:
            if isinstance(g, dict):
                gid = g.get("game_id") or g.get("env_name") or "?"
                sc = float(g.get("final_score") if g.get("final_score") is not None else (g.get("score") or 0.0))
                lv = int(g.get("levels_completed") or 0)
                scores_all.append(sc)
                levels_all += lv
                if any(gid.startswith(h) for h in HOLDOUT_10):
                    scores_holdout10.append(sc)
                    levels_holdout10 += lv
                per_game[gid] = {
                    "final_score": round(sc, 3),
                    "levels_completed": lv,
                    "number_of_levels": g.get("number_of_levels"),
                    "actions_per_level": g.get("actions_per_level"),
                    "solver_note": g.get("solver_note"),
                    "state": g.get("state"),
                }
        summary["mean_score_15"] = round(sum(scores_all) / len(scores_all), 3) if scores_all else None
        summary["levels_15"] = levels_all
        summary["mean_score_holdout10"] = round(sum(scores_holdout10) / len(scores_holdout10), 3) if scores_holdout10 else None
        summary["levels_holdout10"] = levels_holdout10
        summary["per_game"] = per_game
    else:
        summary["error_tail"] = log_text[-3000:]
    log(
        f"=== Finished Arm [{name}] in {wall_s:.1f}s -> "
        f"mean_15={summary.get('mean_score_15')} ({summary.get('levels_15')} lvls), "
        f"holdout10={summary.get('mean_score_holdout10')} ({summary.get('levels_holdout10')} lvls), "
        f"slm_pins={slm_pins} ==="
    )
    return summary


def main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    runtime_s = float(os.environ.get("BENCH_RUNTIME_S", "2100"))
    arms_requested = [a.strip() for a in os.environ.get("BENCH_ARMS", "v4,v3").split(",") if a.strip()]
    games_csv = os.environ.get("BENCH_GAMES", ",".join(CONTENTION_15)).strip()

    assets = ensure_assets()
    bundle_dir, runner_py = setup_patched_bundle(assets["orig_bundle"], assets["comp"], assets["wheelhouse"])
    env_dir = next(p for p in [assets["comp"] / "environment_files", *assets["comp"].rglob("environment_files")] if p.is_dir())
    boot_s = ensure_sglang_server(assets)

    results = {
        "model_dir": str(assets["model"]),
        "sglang_cold_boot_s": round(boot_s, 1),
        "runtime_s_per_arm": runtime_s,
        "games_csv": games_csv,
        "arms": {},
    }
    for arm_name in arms_requested:
        arm_cfg = candidate_v4_env() if arm_name == "v4" else base_v3_env()
        results["arms"][arm_name] = run_arm(arm_name, arm_cfg, runner_py, bundle_dir, env_dir, runtime_s, games_csv)
        Path("/content/v4_ab_summary.json").write_text(json.dumps(results, indent=2))

    serve_tail = WORK_ROOT / "serve_tail.log"
    if SERVE_LOG.is_file():
        serve_tail.write_text(SERVE_LOG.read_text(errors="replace")[-60000:])
    with tarfile.open("/content/v4_ab_results.tar.gz", "w:gz") as tf:
        tf.add("/content/v4_ab_summary.json", arcname="v4_ab_summary.json")
        if serve_tail.is_file():
            tf.add(str(serve_tail), arcname="serve_tail.log")
        for arm_name in arms_requested:
            out_dir = WORK_ROOT / f"out_{arm_name}"
            if (out_dir / "benchmark.json").is_file():
                tf.add(str(out_dir / "benchmark.json"), arcname=f"{arm_name}_benchmark.json")
            alog = WORK_ROOT / f"arm_{arm_name}.log"
            if alog.is_file():
                tf.add(str(alog), arcname=f"arm_{arm_name}.log")

    log(f"ALL DONE in {(time.time() - t_start)/60.0:.1f} min! Summary:\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
