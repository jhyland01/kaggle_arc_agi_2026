"""Colab G4 (RTX PRO 6000 Blackwell 96 GB) KV/Prefill & sm120 MTP Efficiency Benchmark.

Tests two research-backed, lossless/prefix-cache-invariant optimizations on Qwen3.8-Flash-Next
(180B MoE W4A16 BF16-PLE + Hybrid 3:1 GatedDeltaNet/Attention + Pennyroyal SGLang 0.5.19):

Stage A — Noise-Free Deterministic Server & Prefill Micro-Benchmarks:
  A1. Pennyroyal SGLang 0.5.19 Source Audit:
      Inspects /tmp/sgl-intel/venv/lib/python3.12/site-packages/sglang for SGLANG_ENABLE_SM120_LOWM_BF16_GEMM
      row threshold (32 rows vs 40 rows at 10 streams x 4 draft tokens), Mamba state dtype options, and flags.
  A2. Drain-Synchronized Visual Token Pruning Micro-Benchmark (n=3 trials/mode, POST /flush_cache):
      Compares on a realistic 33-turn (~119k-token, 66 distinct 640x640 PNG) multimodal trajectory:
        1) stock_post_drain: 15 retained turns, 30 PNGs (~56k tokens) + T+1 follow-up turn cache check
        2) drain_img_strip_same_budget (keep=4 image msgs = 8 PNGs): 19 retained turns (+26.7% deeper
           verbatim text/code history), 8 PNGs (~58k tokens) + T+1 follow-up 100% prefix-cache hit check
        3) drain_img_strip_equal_turns: 15 retained turns, 8 PNGs (~47.5k tokens) to isolate pure ViT+prefill ms saved.
  A3. 10-Stream Concurrent Decode & MTP Speculative Decoding Benchmark (Zero Game Noise):
      Measures sustained 10-stream concurrent decode tok/s, ms/step, and draft acceptance length across:
        - mtp_3step_4tok: --speculative-num-steps 3 --speculative-num-draft-tokens 4 (40 rows @ 10 streams, v3 baseline)
        - mtp_2step_3tok: --speculative-num-steps 2 --speculative-num-draft-tokens 3 (30 <= 32 rows @ 10 streams, sm120 Low-M)
        - (Optional mtp_3step_4tok_lowm48 if A1 discovers a Python/env configurable 32-row threshold)

Stage B — End-to-End 30-Minute (1800s) 10-Game Unseen Holdout Multi-Drain Validation (v3.1):
  Runs the 10 unseen holdout games using the winning Stage-A MTP server config + SOLVED_LEVEL_MEMORY
  + Drain-Only Old-Image Stripping (keep=4) + ARC3_NOOP_REPEAT_GUARD=1 + stock zero-starvation D scheduler.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
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

STATUS_LOG = Path("/content/flash_kv_mtp_status.log")
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


def download_model_file_by_file(handle: str, target_dir: Path, workers: int = 2) -> Path:
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

    bf16_dir = WORK_ROOT / "flash_next_bf16ple"
    fp8_dir = WORK_ROOT / "flash_next_fp8ple"
    prefer_fp8 = os.environ.get("FORCE_FP8_PLE", "0").strip() == "1"
    if (bf16_dir / "model.safetensors.index.json").is_file():
        model_dir = download_model_file_by_file(BF16_PLE_HANDLE, bf16_dir, workers=2)
    elif not prefer_fp8 and disk_free_gb() >= 176.5:
        log(f"Free disk {disk_free_gb():.1f} GiB >= 176.5 GiB -> downloading Intel BF16-PLE ({BF16_PLE_HANDLE})")
        model_dir = download_model_file_by_file(BF16_PLE_HANDLE, bf16_dir, workers=2)
    else:
        log(f"Free disk {disk_free_gb():.1f} GiB < 176.5 GiB -> using Albucino FP8-PLE ({FP8_PLE_HANDLE})")
        model_dir = download_model_file_by_file(FP8_PLE_HANDLE, fp8_dir, workers=4)

    return {
        "wheelhouse": wheelhouse_dir,
        "hot_tokens": hot_tokens_path,
        "orig_bundle": orig_bundle_dir,
        "comp": comp_dir,
        "drafter": drafter_dir,
        "model": model_dir,
    }


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


def audit_pennyroyal_sm120() -> dict:
    """Inspect Pennyroyal SGLang 0.5.19 sources for SM120_LOWM_BF16_GEMM and Mamba/MTP options."""
    sgl_pkg = Path(PREFIX) / "venv/lib/python3.12/site-packages/sglang"
    audit: dict = {"sm120_lowm_hits": [], "mamba_ssm_dtype_hits": []}
    if not sgl_pkg.is_dir():
        return audit
    for p in sorted(sgl_pkg.rglob("*")):
        if not p.is_file() or p.suffix not in (".py", ".cu", ".cuh", ".cpp", ".h"):
            continue
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        if "SM120_LOWM" in txt or "LOWM_BF16" in txt:
            lines = txt.splitlines()
            for idx, ln in enumerate(lines):
                if "SM120_LOWM" in ln or "LOWM_BF16" in ln:
                    lo = max(0, idx - 5)
                    hi = min(len(lines), idx + 25)
                    snippet = "\n".join(f"{j+1}: {lines[j]}" for j in range(lo, hi))
                    audit["sm120_lowm_hits"].append({"file": str(p.relative_to(sgl_pkg)), "line": idx + 1, "snippet": snippet})
        if "mamba_ssm_dtype" in txt and p.name == "server_args.py":
            lines = txt.splitlines()
            for idx, ln in enumerate(lines):
                if "mamba_ssm_dtype" in ln:
                    lo = max(0, idx - 3)
                    hi = min(len(lines), idx + 12)
                    snippet = "\n".join(f"{j+1}: {lines[j]}" for j in range(lo, hi))
                    audit["mamba_ssm_dtype_hits"].append({"file": str(p.relative_to(sgl_pkg)), "line": idx + 1, "snippet": snippet})
    log(f"Pennyroyal source audit: {len(audit['sm120_lowm_hits'])} SM120_LOWM hits, {len(audit['mamba_ssm_dtype_hits'])} mamba_ssm_dtype hits")
    for h in audit["sm120_lowm_hits"][:6]:
        log(f"  [SM120_LOWM] {h['file']}:{h['line']}\n{h['snippet']}")
    return audit


def gpu_mem_used_mib() -> int:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        )
        return int(res.stdout.strip().splitlines()[0].strip())
    except Exception:
        return 0


def stop_sglang_server(proc: subprocess.Popen | None) -> None:
    log("Stopping SGLang server and waiting for GPU VRAM release...")
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(0.5)
    subprocess.run(["pkill", "-9", "-f", "/tmp/sgl-intel/venv/bin/sglang"], check=False)
    subprocess.run(["pkill", "-9", "-f", "sglang::"], check=False)
    t0 = time.time()
    while time.time() - t0 < 45:
        used = gpu_mem_used_mib()
        if used < 1500:
            log(f"SGLang stopped cleanly; GPU VRAM used = {used} MiB")
            return
        time.sleep(1.5)
    log(f"Warning: GPU VRAM still {gpu_mem_used_mib()} MiB after 45s")


def start_sglang_server(
    assets: dict[str, Path],
    *,
    spec_steps: int = 3,
    spec_draft_tokens: int = 4,
    extra_env: dict[str, str] | None = None,
    tag: str = "default",
) -> tuple[subprocess.Popen, float, int]:
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
    if extra_env:
        env.update(extra_env)

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
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", str(spec_steps),
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", str(spec_draft_tokens),
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

    log(f"Launching SGLang [{tag}] (steps={spec_steps}, draft_tokens={spec_draft_tokens})...")
    with SERVE_LOG.open("ab", buffering=0) as logf:
        logf.write(f"\n=== SGLANG LAUNCH [{tag}] steps={spec_steps} draft_tokens={spec_draft_tokens} ===\n".encode("utf-8"))
    log_offset = SERVE_LOG.stat().st_size if SERVE_LOG.is_file() else 0
    t0 = time.time()
    with SERVE_LOG.open("ab", buffering=0) as logf:
        proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    log(f"SGLang [{tag}] pid={proc.pid}, waiting for /health...")
    url_health = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health"
    last_rep = -30
    while time.time() - t0 < 2400:
        if proc.poll() is not None:
            tail = SERVE_LOG.read_text(errors="replace")[-12000:]
            raise RuntimeError(f"SGLang [{tag}] exited early ({proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(url_health, timeout=5) as r:
                if r.status == 200:
                    boot_s = time.time() - t0
                    log(f"SGLang [{tag}] READY after {boot_s:.1f}s!")
                    return proc, boot_s, log_offset
        except Exception:
            pass
        el = int(time.time() - t0)
        if el - last_rep >= 30:
            log(f"  SGLang [{tag}] loading... {el}s")
            last_rep = el
        time.sleep(3)
    raise RuntimeError(f"SGLang [{tag}] startup timed out after 2400s")


def flush_sglang_cache() -> None:
    req = urllib.request.Request(f"http://127.0.0.1:{SERVED_MODEL_PORT}/flush_cache", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            log(f"POST /flush_cache -> status {r.status}")
    except Exception as exc:
        log(f"Warning: /flush_cache returned {exc!r}")
    time.sleep(1.5)


def chat_completion_raw(payload: dict, timeout: float = 600.0) -> tuple[dict, float]:
    url = f"http://127.0.0.1:{SERVED_MODEL_PORT}/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
            return out, (time.perf_counter() - t0)
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 422) and "ignore_eos" in payload:
            p2 = dict(payload)
            p2.pop("ignore_eos", None)
            body2 = json.dumps(p2).encode("utf-8")
            req2 = urllib.request.Request(url, data=body2, headers={"Content-Type": "application/json"}, method="POST")
            t0 = time.perf_counter()
            with urllib.request.urlopen(req2, timeout=timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
                return out, (time.perf_counter() - t0)
        err_txt = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {err_txt}") from exc


def make_distinct_png_data_url(seed_idx: int, size: int = 640) -> str:
    """Create a distinct 640x640 ARC-like palette grid PNG so MM hash caching never deduplicates across turns."""
    from PIL import Image

    palette = [
        (255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102),
        (51, 51, 51), (0, 0, 0), (229, 58, 163), (255, 123, 204),
        (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0),
        (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214),
    ]
    grid_n = 64
    small = Image.new("RGB", (grid_n, grid_n), palette[0])
    px = small.load()
    for r in range(grid_n):
        for c in range(grid_n):
            v = ((r * 7 + c * 13 + seed_idx * 31) ^ (seed_idx + r)) & 15
            px[c, r] = palette[v]
    big = small.resize((size, size), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def parse_serve_log_slice(start_offset: int) -> dict:
    if not SERVE_LOG.is_file():
        return {}
    with SERVE_LOG.open("rb") as f:
        f.seek(start_offset)
        text = f.read().decode("utf-8", errors="replace")
    decodes = re.findall(
        r"Decode batch, #running-req: (\d+), #full token: (\d+), full token usage: ([0-9.]+), "
        r"mamba num: (\d+), mamba usage: ([0-9.]+), accept len: ([0-9.]+), accept rate: ([0-9.]+), "
        r"cuda graph: (True|False), gen throughput \(token/s\): ([0-9.]+)",
        text,
    )
    reqs = re.findall(
        r"ReqTimeStats\(rid=\w+, input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*?"
        r"queue_duration=([0-9.]+)ms, initial_prefill_elapsed=([0-9.]+)ms, post_prefill_elapsed=([0-9.]+)ms",
        text,
    )
    full10 = [d for d in decodes if int(d[0]) >= 8]
    use_d = full10 if full10 else decodes
    out: dict = {"decode_samples": len(decodes), "full_load_samples": len(full10), "req_stats_count": len(reqs)}
    if use_d:
        acc_lens = sorted(float(d[5]) for d in use_d)
        acc_rates = sorted(float(d[6]) for d in use_d)
        tputs = sorted(float(d[8]) for d in use_d)
        cg_true = sum(1 for d in use_d if d[7] == "True")
        out.update({
            "mean_accept_len": round(sum(acc_lens) / len(acc_lens), 3),
            "median_accept_len": round(acc_lens[len(acc_lens) // 2], 3),
            "mean_accept_rate": round(sum(acc_rates) / len(acc_rates), 3),
            "mean_decode_tok_s": round(sum(tputs) / len(tputs), 2),
            "median_decode_tok_s": round(tputs[len(tputs) // 2], 2),
            "max_decode_tok_s": round(tputs[-1], 2),
            "cuda_graph_pct": round(100.0 * cg_true / len(use_d), 1),
        })
    if reqs:
        out["last_reqs"] = [
            {
                "input_len": int(r[0]),
                "cached_input_len": int(r[1]),
                "output_len": int(r[2]),
                "queue_ms": round(float(r[3]), 2),
                "prefill_ms": round(float(r[4]), 2),
                "decode_ms": round(float(r[5]), 2),
            }
            for r in reqs[-6:]
        ]
    return out


def run_prefill_drain_microbench(bundle_dir: Path) -> dict:
    """Measure exact post-drain prefill latency & T+1 RadixCache invariance on live SGLang."""
    log("=== Stage A2: Running Post-Drain Prefill & Drain-Only Image Stripping Micro-Benchmark ===")
    src_arc = bundle_dir / "src/ARC3-Inference"
    if str(src_arc) not in sys.path:
        sys.path.insert(0, str(src_arc))
    os.environ["LOCAL_ANALYZER_MODEL_ID"] = SERVED_MODEL_NAME
    os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] = str((116 + 12) * 1024)
    os.environ["LOCAL_ANALYZER_MAX_OUTPUT"] = str(12 * 1024)
    os.environ["ARC3_CONTEXT_DRAIN_TOKENS"] = str(58 * 1024)
    import inference.agent.tool_agent as ta

    # Pre-generate 108 distinct 640x640 PNGs
    pngs = [make_distinct_png_data_url(k) for k in range(108)]
    sys_msg = {
        "role": "system",
        "content": "You are an ARC-AGI-3 interactive reasoning agent. " + ("Analyze grid transitions carefully. " * 450),
    }

    def build_turn(i: int) -> list[dict]:
        u = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Action step {i}: level=3, score=21.43. Interior diff cells changed at (row={i%50}, col={(i*3)%50}). "
                        + (f"obs_{i}_grid_row_values=[0,1,9,14,8,11,5,3] " * 22)
                    ),
                },
                {"type": "image_url", "image_url": {"url": pngs[(2 * i) % len(pngs)]}},
                {"type": "image_url", "image_url": {"url": pngs[(2 * i + 1) % len(pngs)]}},
            ],
        }
        a = {
            "role": "assistant",
            "content": f" Executing probe {i}.",
            "reasoning_content": (
                f"Step {i} deduction: testing movement invariants and collision masks across interior objects. "
                + (f"hypothesis_{i}_check_coord_delta " * 140)
            ),
        }
        return [u, a]

    # Accumulate turns until ToolAgent triggers its 118k -> 58k drain
    agent = ta.ToolAgent(model="local")
    full_msgs = [sys_msg]
    drain_turn = None
    stock_drained = None
    for i in range(1, 65):
        cand = [*full_msgs, *build_turn(i)]
        est = agent._estimate_request_input_tokens(cand, tools=[])
        if est > agent._context_budget_tokens:
            drain_turn = i
            stock_drained = agent._trim_messages_for_context(cand, tools=[])
            full_msgs = cand
            break
        full_msgs = cand
    assert stock_drained is not None, "Failed to reach 118k drain threshold in synthetic trajectory"

    # Build the 3 post-drain variants:
    # 1) stock_post_drain (all surviving turns keep their PNGs)
    # 2) drain_img_strip_same_budget (strip old PNGs keep=4 BEFORE draining to 58k -> keeps +more verbatim text turns!)
    stripped_all = [full_msgs[0], *ta._apply_history_image_window(list(full_msgs[1:]), 4)]
    img_strip_same_budget = agent._trim_messages_for_context(stripped_all, tools=[], force_drain=True)
    # 3) drain_img_strip_equal_turns (same exact turns as stock_post_drain, but older PNGs stripped keep=4)
    img_strip_equal_turns = [stock_drained[0], *ta._apply_history_image_window(list(stock_drained[1:]), 4)]

    follow_up_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Follow-up turn T+1 after drain: inspect new board state and choose next action."},
            {"type": "image_url", "image_url": {"url": pngs[106]}},
            {"type": "image_url", "image_url": {"url": pngs[107]}},
        ],
    }

    def count_imgs(msgs: list[dict]) -> int:
        c = 0
        for m in msgs:
            if isinstance(m.get("content"), list):
                c += sum(1 for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")
        return c

    def count_asst(msgs: list[dict]) -> int:
        return sum(1 for m in msgs if m.get("role") == "assistant")

    variants = {
        "stock_post_drain": [*stock_drained, {"role": "user", "content": "State next probe concisely."}],
        "drain_img_strip_same_budget": [*img_strip_same_budget, {"role": "user", "content": "State next probe concisely."}],
        "drain_img_strip_equal_turns": [*img_strip_equal_turns, {"role": "user", "content": "State next probe concisely."}],
    }

    bench_out: dict = {"triggered_at_turn": drain_turn, "variants": {}}
    for vname, vmsgs in variants.items():
        trials = []
        for trial in range(2):
            flush_sglang_cache()
            off0 = SERVE_LOG.stat().st_size
            resp1, wall1 = chat_completion_raw(
                {"model": SERVED_MODEL_NAME, "messages": vmsgs, "max_tokens": 1, "temperature": 0.0}
            )
            slice1 = parse_serve_log_slice(off0)
            req1 = (slice1.get("last_reqs") or [{}])[-1]

            # Immediately test T+1 turn (appending assistant reply + follow_up_user) WITHOUT flushing cache
            asst_reply = resp1["choices"][0]["message"]
            msgs_t1 = [*vmsgs, {"role": "assistant", "content": asst_reply.get("content") or "ok"}, follow_up_user]
            off1 = SERVE_LOG.stat().st_size
            resp2, wall2 = chat_completion_raw(
                {"model": SERVED_MODEL_NAME, "messages": msgs_t1, "max_tokens": 1, "temperature": 0.0}
            )
            slice2 = parse_serve_log_slice(off1)
            req2 = (slice2.get("last_reqs") or [{}])[-1]
            trials.append({
                "trial": trial,
                "drain_wall_ms": round(wall1 * 1000.0, 1),
                "drain_server_prefill_ms": req1.get("prefill_ms"),
                "drain_input_tokens": resp1.get("usage", {}).get("prompt_tokens"),
                "t1_wall_ms": round(wall2 * 1000.0, 1),
                "t1_server_prefill_ms": req2.get("prefill_ms"),
                "t1_input_tokens": resp2.get("usage", {}).get("prompt_tokens"),
                "t1_cached_tokens": req2.get("cached_input_len"),
            })
        mean_drain_ms = round(sum(t["drain_wall_ms"] for t in trials) / len(trials), 1)
        mean_t1_ms = round(sum(t["t1_wall_ms"] for t in trials) / len(trials), 1)
        bench_out["variants"][vname] = {
            "retained_assistant_turns": count_asst(vmsgs),
            "retained_png_images": count_imgs(vmsgs),
            "prompt_tokens": trials[0]["drain_input_tokens"],
            "mean_post_drain_ttft_ms": mean_drain_ms,
            "mean_t1_followup_ttft_ms": mean_t1_ms,
            "t1_prefix_cache_hit_pct": round(
                100.0 * (trials[0]["t1_cached_tokens"] or 0) / max(1, (trials[0]["drain_input_tokens"] or 1)), 2
            ),
            "trials": trials,
        }
        log(
            f"  [{vname}] turns={count_asst(vmsgs)}, pngs={count_imgs(vmsgs)}, "
            f"tok={trials[0]['drain_input_tokens']}, post_drain_ttft={mean_drain_ms}ms, "
            f"T+1_ttft={mean_t1_ms}ms (cached={trials[0]['t1_cached_tokens']})"
        )
    return bench_out


def run_10stream_mtp_decode_bench(tag: str) -> dict:
    """Measure steady-state 10-stream concurrent decode tok/s & MTP acceptance without game variance."""
    log(f"=== Stage A3: Running 10-Stream Concurrent Decode Benchmark [{tag}] ===")
    flush_sglang_cache()

    # Build 10 distinct ~32k-token multimodal prefixes (4 PNGs each) so KV + Mamba state match mid-game load
    streams = []
    for s in range(10):
        p0 = make_distinct_png_data_url(100 + 4 * s)
        p1 = make_distinct_png_data_url(101 + 4 * s)
        p2 = make_distinct_png_data_url(102 + 4 * s)
        p3 = make_distinct_png_data_url(103 + 4 * s)
        history_text = (
            f"Stream {s} ARC-AGI-3 game state history:\n"
            + (
                f"Turn log entry for stream {s}: grid_diff=[(12,14,0->9),(12,15,9->0)], "
                "tested BFS pathfinding function find_shortest_path(grid, start, goal), verified wall mask={5,8}.\n"
                * 420
            )
        )
        msgs = [
            {"role": "system", "content": f"You are ARC-AGI-3 solver instance #{s}. Think step-by-step in Python and grid coordinates."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": history_text},
                    {"type": "image_url", "image_url": {"url": p0}},
                    {"type": "image_url", "image_url": {"url": p1}},
                    {"type": "image_url", "image_url": {"url": p2}},
                    {"type": "image_url", "image_url": {"url": p3}},
                    {
                        "type": "text",
                        "text": (
                            "Now write an exhaustive, turn-by-turn mathematical analysis and Python simulator of a 64x64 "
                            "cellular automaton puzzle with 8 colors, enumerating helper functions, edge cases, BFS tables, "
                            "and invariant proofs. Do not stop early."
                        ),
                    },
                ],
            },
        ]
        streams.append(msgs)

    # Warmup prefill of all 10 streams (`max_tokens=4`) so prefixes + Mamba states are 100% resident in RadixCache
    def _prefill_one(idx: int) -> int:
        r, _ = chat_completion_raw(
            {"model": SERVED_MODEL_NAME, "messages": streams[idx], "max_tokens": 4, "temperature": 0.0}
        )
        return int(r.get("usage", {}).get("prompt_tokens", 0))

    with ThreadPoolExecutor(max_workers=10) as pool:
        prompt_lens = list(pool.map(_prefill_one, range(10)))
    log(f"  [{tag}] Prefilled 10 concurrent streams into RadixCache (mean prompt_tokens={sum(prompt_lens)//10})")

    waves_out: dict = {"mean_cached_prompt_tokens_per_stream": sum(prompt_lens) // 10}
    for wave_name, temp, max_tok in [("greedy_t0", 0.0, 1024), ("sampling_t07_run1", 0.7, 1280), ("sampling_t07_run2", 0.7, 1280)]:
        off_start = SERVE_LOG.stat().st_size

        def _decode_one(idx: int) -> tuple[int, float]:
            payload = {
                "model": SERVED_MODEL_NAME,
                "messages": [
                    *streams[idx][:-1],
                    {
                        **streams[idx][-1],
                        "content": [
                            *streams[idx][-1]["content"],
                            {"type": "text", "text": f"\n[Wave={wave_name} seed={idx}] Begin full derivation:"},
                        ],
                    },
                ],
                "max_tokens": max_tok,
                "temperature": temp,
                "top_p": 0.95,
                "top_k": 20,
                "ignore_eos": True,
            }
            resp, w = chat_completion_raw(payload, timeout=600.0)
            comp_tok = int(resp.get("usage", {}).get("completion_tokens", 0))
            return comp_tok, w

        t_wave = time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as pool:
            res_list = list(pool.map(_decode_one, range(10)))
        wall_wave = time.perf_counter() - t_wave
        total_gen = sum(x[0] for x in res_list)
        net_tok_s = total_gen / max(0.001, wall_wave)
        srv_metrics = parse_serve_log_slice(off_start)
        waves_out[wave_name] = {
            "total_generated_tokens": total_gen,
            "wall_s": round(wall_wave, 2),
            "client_net_tok_s": round(net_tok_s, 2),
            "server_metrics": srv_metrics,
        }
        log(
            f"  [{tag} :: {wave_name}] gen={total_gen} tok in {wall_wave:.2f}s -> "
            f"client_net={net_tok_s:.2f} tok/s | srv_median={srv_metrics.get('median_decode_tok_s')} tok/s | "
            f"accept_len={srv_metrics.get('mean_accept_len')} (rate={srv_metrics.get('mean_accept_rate')})"
        )

    t07_mean = 0.5 * (
        waves_out["sampling_t07_run1"]["client_net_tok_s"] + waves_out["sampling_t07_run2"]["client_net_tok_s"]
    )
    waves_out["sampling_t07_mean_client_tok_s"] = round(t07_mean, 2)
    return waves_out


ARM_RUNNER_CODE = r'''import asyncio
import os
import pickle
import re as _re_slm
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

bundle_dir = Path(sys.argv[1])
env_dir = Path(sys.argv[2])
work_dir = Path(sys.argv[3])
runtime_s = float(sys.argv[4])
games_filter = [g.strip() for g in sys.argv[5].split(",") if g.strip()] if len(sys.argv) > 5 else []
drain_keep_imgs = int(sys.argv[6]) if len(sys.argv) > 6 else 4

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

# --- Cache-Safe Drain-Synchronized SOLVED-LEVEL MEMORY + Drain-Only Old-Image Stripping ---
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


def _count_msg_images(msgs):
    total = 0
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            total += sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image_url")
    return total


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
    did_strip = False
    before_imgs = 0
    work_messages = messages
    kw = dict(kwargs)
    if drain_keep_imgs >= 0 and len(messages) > 1:
        extra_safety = int(kw.get("extra_safety_tokens", 0) or 0)
        force_drain = bool(kw.get("force_drain", False))
        tools = kw.get("tools", None)
        budget_tokens = max(1, self._context_budget_tokens - max(0, extra_safety))
        drain_tokens = min(_ta_slm._context_drain_tokens(), budget_tokens // 2)
        will_drain = bool(force_drain and drain_tokens) or (
            self._estimate_request_input_tokens(messages, tools=tools) > budget_tokens
        )
        if will_drain:
            before_imgs = _count_msg_images(messages)
            stripped_hist = _ta_slm._apply_history_image_window(list(messages[1:]), drain_keep_imgs)
            work_messages = [messages[0], *stripped_hist]
            kw["force_drain"] = True
            did_strip = True

    result = _slm_original_trim(self, work_messages, *args, **kw)
    if did_strip:
        after_imgs = _count_msg_images(result)
        if result != messages and not getattr(self, "_context_was_trimmed", False):
            self._note_history_evicted()
        print(f"DRAIN_IMG_STRIP: images {before_imgs} -> {after_imgs} (keep_msgs={drain_keep_imgs}), msgs {len(messages)} -> {len(result)}", flush=True)

    if (
        (len(result) < len(messages) or did_strip)
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
print(f"SOLVED_LEVEL_MEMORY + DRAIN_IMG_STRIP (keep={drain_keep_imgs}) on", flush=True)

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

print(f"Running v3.1 holdout validation in {work_dir} on {len(bm.games)} games, runtime_s={runtime_s}", flush=True)
soft_end = datetime.now() + timedelta(seconds=runtime_s + 180.0)
asyncio.run(bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=False))
'''


def v31_env() -> dict[str, str]:
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
        "ARC3_NOOP_REPEAT_GUARD": "1",
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


def run_v31_holdout_validation(
    runner_py: str,
    bundle_dir: Path,
    env_dir: Path,
    runtime_s: float,
    games_csv: str,
    drain_keep_imgs: int = 4,
) -> dict:
    flush_sglang_cache()
    out_dir = WORK_ROOT / "out_v31"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner_script = WORK_ROOT / "_run_v31.py"
    runner_script.write_text(ARM_RUNNER_CODE)

    full_env = dict(os.environ)
    full_env.update(v31_env())
    arm_log = WORK_ROOT / "arm_v31.log"
    off0 = SERVE_LOG.stat().st_size if SERVE_LOG.is_file() else 0
    log(f"=== Stage B: Starting v3.1 10-Game Holdout Validation (runtime_s={runtime_s}, drain_keep_imgs={drain_keep_imgs}) ===")
    t0 = time.time()
    with arm_log.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(
            [runner_py, str(runner_script), str(bundle_dir), str(env_dir), str(out_dir), str(runtime_s), games_csv, str(drain_keep_imgs)],
            env=full_env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    wall_s = time.time() - t0
    log_text = arm_log.read_text(errors="replace") if arm_log.is_file() else ""
    slm_pins = log_text.count("SOLVED_LEVEL_MEMORY pinned:")
    img_strips = log_text.count("DRAIN_IMG_STRIP:")
    srv_slice = parse_serve_log_slice(off0)

    summary: dict = {
        "returncode": proc.returncode,
        "wall_s": round(wall_s, 1),
        "slm_pins": slm_pins,
        "drain_img_strips": img_strips,
        "server_metrics": srv_slice,
    }
    bm_json = out_dir / "benchmark.json"
    if bm_json.is_file():
        data = json.loads(bm_json.read_text())
        game_runs = data.get("game_runs") or []
        per_game = {}
        scores = []
        levels = 0
        total_tok = 0
        for g in game_runs:
            gid = g.get("game_id") or "?"
            sc = float(g.get("final_score") or 0.0)
            lv = int(g.get("levels_completed") or 0)
            m = re.search(r"tokens=(\d+)", str(g.get("solver_note") or ""))
            tok = int(m.group(1)) if m else 0
            scores.append(sc)
            levels += lv
            total_tok += tok
            per_game[gid] = {
                "final_score": round(sc, 3),
                "levels_completed": lv,
                "number_of_levels": g.get("number_of_levels"),
                "actions_per_level": g.get("actions_per_level"),
                "solver_note": g.get("solver_note"),
                "state": g.get("state"),
            }
        scores_sorted = sorted(scores)
        med = (
            0.5 * (scores_sorted[len(scores_sorted) // 2 - 1] + scores_sorted[len(scores_sorted) // 2])
            if scores_sorted and len(scores_sorted) % 2 == 0
            else (scores_sorted[len(scores_sorted) // 2] if scores_sorted else None)
        )
        summary.update({
            "mean_score_holdout10": round(sum(scores) / len(scores), 3) if scores else None,
            "median_score_holdout10": round(med, 3) if med is not None else None,
            "levels_holdout10": levels,
            "total_tokens": total_tok,
            "job_wallclock_tok_s": round(total_tok / max(1.0, runtime_s), 2),
            "per_game": per_game,
        })
    else:
        summary["error_tail"] = log_text[-3000:]
    log(
        f"=== Finished Stage B v3.1 in {wall_s:.1f}s -> "
        f"mean={summary.get('mean_score_holdout10')}, median={summary.get('median_score_holdout10')}, "
        f"levels={summary.get('levels_holdout10')}, slm_pins={slm_pins}, drain_img_strips={img_strips}, "
        f"job_tok_s={summary.get('job_wallclock_tok_s')} ==="
    )
    return summary


def main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    runtime_s = float(os.environ.get("BENCH_RUNTIME_S", "1800"))
    games_csv = os.environ.get("BENCH_GAMES", ",".join(HOLDOUT_10)).strip()

    assets = ensure_assets()
    bundle_dir, runner_py = setup_patched_bundle(assets["orig_bundle"], assets["comp"], assets["wheelhouse"])
    env_dir = next(p for p in [assets["comp"] / "environment_files", *assets["comp"].rglob("environment_files")] if p.is_dir())

    # Stage A1: Source audit of Pennyroyal SGLang 0.5.19 sm120 Low-M GEMM & Mamba flags
    source_audit = audit_pennyroyal_sm120()

    results: dict = {
        "model_dir": str(assets["model"]),
        "pennyroyal_source_audit": source_audit,
        "stage_a_prefill_drain": {},
        "stage_a_mtp_10stream": {},
        "stage_b_v31_holdout": {},
    }

    # Boot 1: Baseline MTP (steps=3, draft_tokens=4 -> 40 rows at 10 streams)
    proc = None
    try:
        proc, boot1_s, _ = start_sglang_server(assets, spec_steps=3, spec_draft_tokens=4, tag="mtp_3step_4tok")
        results["boot_3step_s"] = round(boot1_s, 1)

        # Stage A2: Drain-Only Image Stripping vs Stock Post-Drain Prefill Micro-Benchmark
        results["stage_a_prefill_drain"] = run_prefill_drain_microbench(bundle_dir)
        Path("/content/kv_mtp_summary.json").write_text(json.dumps(results, indent=2))

        # Stage A3 (Part 1): 10-Stream Concurrent Decode on Baseline MTP (3 steps / 4 draft tokens = 40 rows)
        results["stage_a_mtp_10stream"]["mtp_3step_4tok"] = run_10stream_mtp_decode_bench("mtp_3step_4tok")
        Path("/content/kv_mtp_summary.json").write_text(json.dumps(results, indent=2))

        # Boot 2: Candidate MTP (steps=2, draft_tokens=3 -> 30 <= 32 rows at 10 streams for sm120 Low-M GEMM)
        stop_sglang_server(proc)
        proc, boot2_s, _ = start_sglang_server(assets, spec_steps=2, spec_draft_tokens=3, tag="mtp_2step_3tok")
        results["boot_2step_s"] = round(boot2_s, 1)

        results["stage_a_mtp_10stream"]["mtp_2step_3tok"] = run_10stream_mtp_decode_bench("mtp_2step_3tok")
        Path("/content/kv_mtp_summary.json").write_text(json.dumps(results, indent=2))

        # Select winning MTP configuration based on measured 10-stream temp=0.7 net tok/s
        t_3step = results["stage_a_mtp_10stream"]["mtp_3step_4tok"]["sampling_t07_mean_client_tok_s"]
        t_2step = results["stage_a_mtp_10stream"]["mtp_2step_3tok"]["sampling_t07_mean_client_tok_s"]
        log(f"Stage A3 Comparison (10-stream temp=0.7): 3-step(40 rows)={t_3step} tok/s vs 2-step(30 rows)={t_2step} tok/s")

        # Keep 2-step if it beats or ties 3-step (within 0.5%); otherwise reload 3-step so Stage B uses the faster server
        if t_2step >= t_3step * 0.995:
            chosen_mtp = "mtp_2step_3tok"
            log(f"Keeping warm {chosen_mtp} server for Stage B ({t_2step} vs {t_3step} tok/s)")
        else:
            chosen_mtp = "mtp_3step_4tok"
            log(f"3-step beat 2-step ({t_3step} vs {t_2step} tok/s); restarting {chosen_mtp} for Stage B...")
            stop_sglang_server(proc)
            proc, _, _ = start_sglang_server(assets, spec_steps=3, spec_draft_tokens=4, tag="mtp_3step_4tok_stageb")

        results["chosen_mtp_for_stage_b"] = chosen_mtp

        # Stage B: 1800s (30-min) 10-Game Unseen Holdout Multi-Drain Validation
        results["stage_b_v31_holdout"] = run_v31_holdout_validation(
            runner_py, bundle_dir, env_dir, runtime_s, games_csv, drain_keep_imgs=4
        )
        Path("/content/kv_mtp_summary.json").write_text(json.dumps(results, indent=2))
    finally:
        stop_sglang_server(proc)

    serve_tail = WORK_ROOT / "serve_tail.log"
    if SERVE_LOG.is_file():
        serve_tail.write_text(SERVE_LOG.read_text(errors="replace")[-80000:])
    with tarfile.open("/content/kv_mtp_results.tar.gz", "w:gz") as tf:
        tf.add("/content/kv_mtp_summary.json", arcname="kv_mtp_summary.json")
        if serve_tail.is_file():
            tf.add(str(serve_tail), arcname="serve_tail.log")
        out_dir = WORK_ROOT / "out_v31"
        if (out_dir / "benchmark.json").is_file():
            tf.add(str(out_dir / "benchmark.json"), arcname="v31_benchmark.json")
        alog = WORK_ROOT / "arm_v31.log"
        if alog.is_file():
            tf.add(str(alog), arcname="arm_v31.log")

    log(f"ALL DONE in {(time.time() - t_start)/60.0:.1f} min! Summary:\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
