"""Colab G4 (RTX PRO 6000 Blackwell 96 GB) Warm SGLang Flash-Next Head-to-Head A/B Benchmark.

1. Downloads inside Colab datacenter (zero local Wi-Fi, zero archive duplication on disk):
   - dfranzen/pennyroyal-v253
   - dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple/transformers/default/1 (~125 GB, shard-by-shard)
   - dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1
   - dfranzen/taaf-kaggle-source-bundle-copy
   - arc-prize-2026-arc-agi-3 (arc_agi_3_wheels + environment_files)
2. Applies agi_boys harness-changes.patch + harness-v2-overlay.patch (v2 features gated by ARC3_HARNESS_V2).
3. Boots Pennyroyal SGLang server (127.0.0.1:8001, MAXREQ=10, 3-step NEXTN MTP + FR-Spec, --ple-offload-embedding)
   and measures exact cold-load startup wallclock with albucino FP8-PLE + parallel precache.
4. Runs clean, cache-flushed, isolated-process 10-game holdout A/B benchmark:
   - Arm v1: Stock agi_boys configuration (ARC3_HARNESS_V2=0)
   - Arm v2: Duck Harness v2 configuration (ARC3_HARNESS_V2=1 + prefix-safe short yields + guards + ledger + BFS)
5. Writes side-by-side summary to /content/ab_summary.json and tarballs transcripts/benchmarks to /content/ab_results.tar.gz.
"""
from __future__ import annotations

import copy
import glob
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
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

STATUS_LOG = Path("/content/flash_g4_status.log")
WORK_ROOT = Path("/content/flash_work")
WORK_ROOT.mkdir(parents=True, exist_ok=True)

SERVED_MODEL_NAME = "flashnext"
SERVED_MODEL_PORT = 8001
PREFIX = "/tmp/sgl-intel"
SERVE_LOG = Path("/content/serve.log")

MODEL_HANDLE = "dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple/transformers/default/1"
DRAFTER_HANDLE = "dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1"
PENNYROYAL_DATASET = "dfranzen/pennyroyal-v253"
BUNDLE_DATASET = "dfranzen/taaf-kaggle-source-bundle-copy"
COMPETITION_NAME = "arc-prize-2026-arc-agi-3"

TOKEN_MAP_SHA = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
EXPERT_TARGET = r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"
DENSE_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"


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


def download_model_file_by_file(handle: str, target_dir: Path, workers: int = 6) -> Path:
    """Download a large Kaggle model shard-by-shard without creating a 125 GB archive copy."""
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
            if src.is_file():
                if not dest.exists():
                    shutil.move(str(src), str(dest))
        except Exception:
            pass

    idx_path = target_dir / "model.safetensors.index.json"
    if not idx_path.is_file():
        raise FileNotFoundError(f"model.safetensors.index.json missing in {target_dir}")
    idx = json.loads(idx_path.read_text())
    shards = sorted(set(idx["weight_map"].values()))
    todo = [s for s in shards if not ((target_dir / s).is_file() and (target_dir / s).stat().st_size > 1024)]
    log(f"Model {handle}: {len(shards)} total shards, {len(todo)} remaining to download (free={disk_free_gb():.1f} GiB)")

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
                sname, sz_gb = fut.result()
                done_count += 1
                done_gb += sz_gb
                if done_count % 4 == 0 or done_count == len(todo):
                    elapsed = max(0.1, time.time() - t0)
                    log(
                        f"  downloaded {done_count}/{len(todo)} shards ({done_gb:.1f} GiB in {elapsed:.0f}s, "
                        f"{done_gb * 1024 / elapsed:.0f} MiB/s, free={disk_free_gb():.1f} GiB)"
                    )
    cleanup_kagglehub_archives()
    return target_dir


def ensure_assets() -> dict[str, Path]:
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "kaggle", "kagglehub", "pandas", "pyarrow"])
    import kagglehub

    log(f"Initial free disk: {disk_free_gb():.1f} GiB")

    # 1. Pennyroyal wheelhouse (~4.5 GB)
    wheelhouse_dir = Path(kagglehub.dataset_download(PENNYROYAL_DATASET))
    cleanup_kagglehub_archives()
    log(f"Pennyroyal wheelhouse -> {wheelhouse_dir} (free={disk_free_gb():.1f} GiB)")

    # 2. TAAF source bundle
    orig_bundle_dir = Path(kagglehub.dataset_download(BUNDLE_DATASET))
    cleanup_kagglehub_archives()
    log(f"TAAF bundle -> {orig_bundle_dir}")

    # 3. Competition wheels & environment_files
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

    # 4. Drafter (~2.8 GB, preserves subdirectory tree runtime/mtp-int4-g32/...)
    drafter_dir = Path(kagglehub.model_download(DRAFTER_HANDLE))
    cleanup_kagglehub_archives()
    log(f"Drafter -> {drafter_dir} (free={disk_free_gb():.1f} GiB)")

    # 5. Target Flash-Next FP8-PLE (~125 GB, file-by-file)
    model_dir = WORK_ROOT / "flash_next_fp8ple"
    download_model_file_by_file(MODEL_HANDLE, model_dir, workers=6)

    return {
        "wheelhouse": wheelhouse_dir,
        "orig_bundle": orig_bundle_dir,
        "comp": comp_dir,
        "drafter": drafter_dir,
        "model": model_dir,
    }


def target_python312() -> str:
    return shutil.which("python3.12") or sys.executable


def ensure_uv_binary(wheelhouse_dir: Path) -> Path:
    uv_bin = Path(PREFIX) / "bin/uv"
    if uv_bin.is_file():
        return uv_bin
    wheels = find_unique(wheelhouse_dir, "wheels", directory=True)
    assert wheels is not None
    uv_whl = sorted(wheels.glob("uv-*.whl"))[0]
    uv_bin.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(uv_whl) as zf:
        member = [n for n in zf.namelist() if Path(n).name == "uv" and not n.endswith("/")][0]
        uv_bin.write_bytes(zf.read(member))
    uv_bin.chmod(0o755)
    return uv_bin


def setup_patched_bundle(orig_bundle: Path, comp_dir: Path, wheelhouse_dir: Path) -> tuple[Path, str]:
    # Build dedicated Python 3.12 runner venv matching Kaggle's cp312 runtime
    uv_bin = ensure_uv_binary(wheelhouse_dir)
    runner_venv = Path("/tmp/arc3-py312")
    runner_py = str(runner_venv / "bin/python")
    if not Path(runner_py).is_file():
        run_cmd([str(uv_bin), "venv", "--python", target_python312(), str(runner_venv)])

    wheels_dir = next(p for p in [comp_dir / "arc_agi_3_wheels", *comp_dir.rglob("arc_agi_3_wheels")] if p.is_dir())
    run_cmd(
        [
            str(uv_bin),
            "pip",
            "install",
            "--python",
            runner_py,
            "--no-index",
            "--find-links",
            str(wheels_dir),
            "arc-agi",
        ]
    )
    run_cmd(
        [
            str(uv_bin),
            "pip",
            "install",
            "--python",
            runner_py,
            "matplotlib",
            "scipy",
            "imageio",
            "python-dotenv",
            "requests",
            "pandas",
            "pyarrow",
            "psutil",
        ]
    )

    bundle_dir = WORK_ROOT / "taaf-patched"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    shutil.copytree(orig_bundle, bundle_dir)

    for patch_file in (Path("/content/harness-changes.patch"), Path("/content/harness-v2-overlay.patch")):
        raw = patch_file.read_bytes()
        if not raw.endswith(b"\n"):
            patch_file.write_bytes(raw + b"\n")

    run_cmd(
        ["git", "apply", "--include=ARC3-Inference/*", "-v", "/content/harness-changes.patch"],
        cwd=str(bundle_dir / "src"),
    )
    run_cmd(
        ["git", "apply", "--include=ARC3-Inference/*", "-v", "/content/harness-v2-overlay.patch"],
        cwd=str(bundle_dir / "src"),
    )

    entries = []
    for repo in sorted((bundle_dir / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(candidate)
    purelib = subprocess.check_output(
        [runner_py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True,
    ).strip()
    pth_path = Path(purelib) / "taaf_kaggle_sources.pth"
    pth_path.write_text("".join(f"{e}\n" for e in entries))
    log(f"Patched TAAF bundle ready at {bundle_dir} -> {pth_path} ({len(entries)} source roots)")
    return bundle_dir, runner_py


def precache_paths(*paths: Path, threads: int = 8) -> None:
    t0 = time.time()
    chunk_size = 32 * 1024 * 1024
    chunks = []
    total_bytes = 0
    for idx, p in enumerate(paths):
        for root, _, files in os.walk(p):
            for fn in files:
                fp = os.path.join(root, fn)
                if os.path.isfile(fp):
                    sz = os.path.getsize(fp)
                    total_bytes += sz
                    for off in range(0, max(sz, 1), chunk_size):
                        chunks.append((idx, fp, off))
    chunks.sort()
    tls = threading.local()

    def _read(item):
        buf = getattr(tls, "buf", None)
        if buf is None:
            buf = tls.buf = bytearray(chunk_size)
        _, fp, off = item
        try:
            with open(fp, "rb") as f:
                f.seek(off)
                rem = chunk_size
                while rem > 0:
                    n = f.readinto(buf)
                    if not n:
                        break
                    rem -= n
        except OSError:
            pass

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(_read, chunks))
    log(f"Precached {total_bytes / 1e9:.2f} GB into page cache in {time.time() - t0:.1f}s")


def find_unique(root: Path, name: str, *, directory: bool = False, required: bool = True) -> Path | None:
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def indexed_shards(root: Path):
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    for nm in names:
        p = root / nm
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
    view = work_root / f"draft-view-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
    view.mkdir(parents=True, exist_ok=True)
    (view / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    links = {nm: source / nm for nm in shards}
    links["model.safetensors.index.json"] = source / "model.safetensors.index.json"
    for nm in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"):
        if (target / nm).is_file():
            links[nm] = target / nm
    for nm, src in links.items():
        dest = view / nm
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() and not dest.is_symlink():
            dest.symlink_to(src.resolve())
    return view


def sglang_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_sglang_server(assets: dict[str, Path]) -> float:
    if sglang_healthy():
        log("SGLang server already healthy on port 8001; reusing warm server!")
        return 0.0

    model_dir = assets["model"].resolve()
    drafter_dir = assets["drafter"].resolve()
    wheelhouse_dir = assets["wheelhouse"].resolve()

    venv = f"{PREFIX}/venv"
    py_bin = f"{venv}/bin/python"
    sgl_bin = f"{venv}/bin/sglang"
    Path(PREFIX).mkdir(parents=True, exist_ok=True)
    wheels = find_unique(wheelhouse_dir, "wheels", directory=True)
    lock = find_unique(wheelhouse_dir, "requirements.lock")
    assert wheels is not None and lock is not None
    sgl_wheel = sorted(wheels.glob("sglang-*.whl"))[0]
    install_marker = Path(PREFIX) / "installed-bundle.txt"

    # Start precaching model shards in background WHILE Pennyroyal wheels install!
    precache_thread = threading.Thread(target=precache_paths, args=(model_dir, drafter_dir), kwargs={"threads": 8}, daemon=True)
    precache_thread.start()

    install_env = dict(os.environ, UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1")
    install_env.pop("PYTHONPATH", None)
    if not Path(sgl_bin).exists() or not install_marker.is_file():
        log("Installing Pennyroyal SGLang offline wheelhouse (Python 3.12)...")
        uv_bin = ensure_uv_binary(wheelhouse_dir)
        if not Path(py_bin).exists():
            run_cmd([str(uv_bin), "venv", "--python", target_python312(), venv], env=install_env)
        inst = [str(uv_bin), "pip", "install", "--python", py_bin, "--no-index", "--find-links", str(wheels)]
        run_cmd(inst + ["-r", str(lock)], env=install_env)
        run_cmd(inst + ["--reinstall", "--no-deps", str(sgl_wheel)], env=install_env)
        install_marker.write_text("installed")

    cu13 = sorted(Path(venv).glob("lib/python*/site-packages/nvidia/cu13"))[0]
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
        for d in ("/usr/lib64-nvidia", "/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
            if Path(d).is_dir():
                cands += sorted(p for p in Path(d).glob("libcuda.so*") if p.is_file())
        if not cands:
            code = 'import torch; torch.cuda.init(); print(open("/proc/self/maps").read())'
            maps = run_cmd([py_bin, "-c", code], env=install_env).stdout
            cands = [Path(p) for p in re.findall(r"(/[^\s]*libcuda\.so[^\s]*)", maps) if Path(p).exists()]
        if not cands:
            raise RuntimeError("GPU driver libcuda.so not found")
        link = libdir / "libcuda.so"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(cands[0].resolve())

    cxx = next(c for c in ("g++-15", "g++-14", "g++-13", "g++-12", "g++-11", "g++") if shutil.which(c))
    cc = cxx.replace("g++", "gcc")
    cache = f"{PREFIX}/cache"
    for nm in ("huggingface", "torch", "torchinductor", "triton", "flashinfer", "cuda", "sglang/jit"):
        Path(cache, nm).mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "UV_OFFLINE": "1",
            "UV_PYTHON_DOWNLOADS": "never",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_HOME": cuda_home,
            "CUDACXX": f"{cuda_home}/bin/nvcc",
            "PATH": f"{cuda_home}/bin:{venv}/bin:" + env.get("PATH", ""),
            "LD_LIBRARY_PATH": f"{cuda_home}/lib64:{cuda_home}/lib:" + env.get("LD_LIBRARY_PATH", ""),
            "CC": cc,
            "CXX": cxx,
            "CUDAHOSTCXX": cxx,
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "HF_HOME": f"{cache}/huggingface",
            "XDG_CACHE_HOME": cache,
            "TORCH_HOME": f"{cache}/torch",
            "CUDA_CACHE_PATH": f"{cache}/cuda",
            "TORCHINDUCTOR_CACHE_DIR": f"{cache}/torchinductor",
            "TRITON_CACHE_DIR": f"{cache}/triton",
            "FLASHINFER_WORKSPACE_BASE": f"{cache}/flashinfer",
            "SGLANG_CACHE_DIR": f"{cache}/sglang",
            "SGLANG_JIT_CACHE_DIR": f"{cache}/sglang/jit",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "false",
            "MAX_JOBS": "8",
            "CMAKE_BUILD_PARALLEL_LEVEL": "8",
            "FLASHINFER_NINJA_JOBS": "8",
            "FLASHINFER_NVCC_THREADS": "2",
            "TORCHINDUCTOR_COMPILE_THREADS": "8",
            "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM": "1",
            "SGLANG_SM120_ONLINE_MXFP8": "0",
            "SGLANG_SM120_LOWM_FP8_WEIGHT": "0",
            "SGLANG_SM120_LM_HEAD_FP8": "0",
            "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
            "SGLANG_NUMA_BIND_V2": "false",
            "SGLANG_MM_PREPROCESS_DEVICE": "cpu",
            "NUMPY_MADVISE_HUGEPAGE": "0",
        }
    )

    draft_view = prepare_draft_view(model_dir, drafter_dir, Path(PREFIX))
    tok = find_unique(wheelhouse_dir, "hot_tokens_64k.pt", required=False) or find_unique(wheelhouse_dir, "flash-next-64k.pt")
    assert tok is not None and sha256(tok) == TOKEN_MAP_SHA
    assert sha256(model_dir / "tokenizer.json") == TOKENIZER_SHA

    ctx = (116 + 12 + 8) * 1024
    graph_bs = [1, 2, 4, 7, 8, 9, 10]
    args = [
        sgl_bin,
        "serve",
        "--model-path",
        str(model_dir),
        "--load-format",
        "safetensors",
        "--model-loader-extra-config",
        '{"enable_multithread_load":false}',
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVED_MODEL_PORT),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--quantization",
        "auto-round",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.96",
        "--context-length",
        str(ctx),
        "--page-size",
        "64",
        "--max-running-requests",
        "10",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--cuda-graph-max-bs-decode",
        "10",
        "--cuda-graph-bs-decode",
        *map(str, graph_bs),
        "--mamba-ssm-dtype",
        "bfloat16",
        "--max-mamba-cache-size",
        "60",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "64",
        "--mamba-backend",
        "flashinfer",
        "--linear-attn-decode-backend",
        "flashinfer",
        "--linear-attn-prefill-backend",
        "flashinfer",
        "--moe-runner-backend",
        "auto",
        "--ple-offload-embedding",
        "--trust-remote-code",
        "--mm-feature-transport",
        "cpu",
        "--image-processor-backend",
        "pil",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--default-chat-template-kwargs",
        '{"preserve_thinking":true}',
        "--watchdog-timeout",
        "1800",
        "--schedule-policy",
        "lpm",
        "--warmups",
        "structured_output",
        "--enable-cache-report",
        "--enable-metrics",
        "--enable-request-time-stats-logging",
        "--weight-loader-prefetch-checkpoints",
        "--weight-loader-drop-cache-after-load",
        "--chat-template",
        str(model_dir / "chat_template.jinja"),
        "--gdn-mtp-cache-mode",
        "none",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-draft-model-path",
        str(draft_view),
        "--speculative-draft-model-quantization",
        "compressed-tensors",
        "--speculative-moe-runner-backend",
        "auto",
        "--speculative-draft-kv-cache-dtype",
        "fp8_e4m3",
        "--speculative-accept-threshold-single",
        "1.0",
        "--speculative-accept-threshold-acc",
        "1.0",
        "--speculative-token-map",
        str(tok),
    ]

    log("Launching Pennyroyal SGLang Flash-Next (albucino FP8-PLE + 3-step MTP + FR-Spec)...")
    t_boot = time.time()
    with SERVE_LOG.open("ab", buffering=0) as lf:
        proc = subprocess.Popen(args, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    Path("/content/sglang.pid").write_text(str(proc.pid))

    last_rep = -30
    while time.time() - t_boot < 1500:
        if proc.poll() is not None:
            tail = SERVE_LOG.read_text(errors="replace")[-8000:]
            raise RuntimeError(f"SGLang exited early ({proc.returncode}):\n{tail}")
        if sglang_healthy():
            boot_s = time.time() - t_boot
            log(f"SGLang READY after {boot_s:.1f}s! (vs 889.8s in agi_boys BF16-PLE)")
            return boot_s
        el = int(time.time() - t_boot)
        if el - last_rep >= 30:
            log(f"  waiting for SGLang /health ... {el}s")
            last_rep = el
        time.sleep(3)
    raise RuntimeError("SGLang startup timed out after 1500s")


def flush_sglang_cache() -> None:
    for endpoint in ("flush_cache",):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{SERVED_MODEL_PORT}/{endpoint}", method="POST")
            with urllib.request.urlopen(req, timeout=15) as r:
                log(f"POST /{endpoint} -> HTTP {r.status}")
        except Exception as exc:
            log(f"POST /{endpoint} warning: {exc}")
    time.sleep(2)


ARM_RUNNER_CODE = r"""
import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

bundle_dir = Path(sys.argv[1])
env_dir = Path(sys.argv[2])
work_dir = Path(sys.argv[3])
runtime_s = float(sys.argv[4])
games_filter = [g.strip() for g in sys.argv[5].split(",") if g.strip()] if len(sys.argv) > 5 else []

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

demo_excluded_games = [
    "bp35", "cd82", "cn04", "dc22", "g50t", "ka59", "lf52", "ls20",
    "m0r0", "s5i5", "sk48", "sp80", "su15", "tn36", "wa30",
]

import arc_agi
import taaf.game_api

spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(env_dir))
arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(env_dir))
game_ids = [
    e.game_id for e in arcade.available_environments
    if not any(e.game_id.startswith(g) for g in demo_excluded_games)
]
if games_filter:
    game_ids = [gid for gid in game_ids if any(gid.startswith(gf) for gf in games_filter)]
bm.games = [taaf.game_api.GameAPI(env_name=gid, arcade_spec=spec) for gid in game_ids]

bm.solver.analyzer_timeout = 900.0
bm.solver.concurrency = len(bm.games)
bm.solver.max_runtime_s_per_game = runtime_s
bm.solver.max_actions_per_game = None
bm.solver.save_request_logs = False
bm.n_passes = 1
bm.game_weights = None

print(f"Running arm in {work_dir} on {len(bm.games)} games ({[g.env_name for g in bm.games]}), runtime_s={runtime_s}", flush=True)
soft_end = datetime.now() + timedelta(seconds=runtime_s + 180.0)
asyncio.run(bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=False))
"""


def base_agi_boys_env() -> dict[str, str]:
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


def harness_v2_env() -> dict[str, str]:
    env = base_agi_boys_env()
    env.update(
        {
            "ARC3_HARNESS_V2": "1",
            "ARC3_STALL_ALERT_THRESHOLD": "4",
            "LOCAL_ANALYZER_YIELD_TOKENS": "4096",
            "ARC3_YIELD_RESUME_PROMPT": "short",
            "ARC3_RESUME_PROMPT_IMAGES": "1",
            "ARC3_PRUNE_CONTROL_CONTEXT": "0",
            "ARC3_PREFER_TOOL_CALLS": "1",
            "ARC3_DEATH_LEDGER": "1",
            "ARC3_DEATH_REPEAT_GUARD": "1",
            "ARC3_NOOP_REPEAT_GUARD": "1",
            "ARC3_REPEAT_STATE_GUARD": "0",
            "ARC3_GAMEOVER_DIFF": "1",
            "ARC3_LEVEL_INVENTORY": "1",
            "ARC3_BOARD_DUMP_GUARD": "1",
            "ARC3_OPENER_COMMIT_HINT": "1",
        }
    )
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
    runner_script = WORK_ROOT / "_run_arm.py"
    runner_script.write_text(ARM_RUNNER_CODE)

    full_env = dict(os.environ)
    full_env.update(arm_env)
    arm_log = WORK_ROOT / f"arm_{name}.log"
    log(f"=== Starting Arm [{name}] (runtime_s={runtime_s}, games={games_csv or '10-game-holdout'}) ===")
    t0 = time.time()
    with arm_log.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(
            [runner_py, str(runner_script), str(bundle_dir), str(env_dir), str(out_dir), str(runtime_s), games_csv],
            env=full_env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    wall_s = time.time() - t0
    bm_json = out_dir / "benchmark.json"
    summary: dict = {"arm": name, "returncode": proc.returncode, "wall_s": round(wall_s, 1)}
    if bm_json.is_file():
        data = json.loads(bm_json.read_text())
        game_runs = data.get("game_runs") or data.get("games") or data.get("results") or []
        per_game = {}
        scores = []
        for g in game_runs:
            if isinstance(g, dict):
                gid = g.get("game_id") or g.get("env_name") or "?"
                sc = float(g.get("final_score") if g.get("final_score") is not None else (g.get("score") or 0.0))
                scores.append(sc)
                per_game[gid] = {
                    "final_score": round(sc, 3),
                    "levels_completed": g.get("levels_completed"),
                    "number_of_levels": g.get("number_of_levels"),
                    "actions_per_level": g.get("actions_per_level"),
                    "solver_note": g.get("solver_note"),
                    "state": g.get("state"),
                }
        summary["mean_score"] = round(sum(scores) / len(scores), 3) if scores else None
        summary["per_game"] = per_game
    else:
        summary["error_tail"] = arm_log.read_text(errors="replace")[-3000:]
    log(f"=== Finished Arm [{name}] in {wall_s:.1f}s -> mean_score={summary.get('mean_score')} ===")
    return summary


def main() -> None:
    t_start = time.time()
    runtime_s = float(os.environ.get("BENCH_RUNTIME_S", "900"))
    arms_requested = [a.strip() for a in os.environ.get("BENCH_ARMS", "v2,v1").split(",") if a.strip()]
    games_csv = os.environ.get("BENCH_GAMES", "").strip()

    assets = ensure_assets()
    bundle_dir, runner_py = setup_patched_bundle(assets["orig_bundle"], assets["comp"], assets["wheelhouse"])
    env_dir = next(p for p in [assets["comp"] / "environment_files", *assets["comp"].rglob("environment_files")] if p.is_dir())
    boot_s = ensure_sglang_server(assets)

    results = {"sglang_cold_boot_s": round(boot_s, 1), "runtime_s_per_arm": runtime_s, "arms": {}}
    for arm_name in arms_requested:
        arm_cfg = harness_v2_env() if arm_name.startswith("v2") else base_agi_boys_env()
        results["arms"][arm_name] = run_arm(arm_name, arm_cfg, runner_py, bundle_dir, env_dir, runtime_s, games_csv)
        Path("/content/ab_summary.json").write_text(json.dumps(results, indent=2))

    # Package benchmark.json + transcripts + serve.log tail into compact /content/ab_results.tar.gz
    SERVE_TAIL = WORK_ROOT / "serve_tail.log"
    if SERVE_LOG.is_file():
        SERVE_TAIL.write_text(SERVE_LOG.read_text(errors="replace")[-60000:])
    with tarfile.open("/content/ab_results.tar.gz", "w:gz") as tf:
        tf.add("/content/ab_summary.json", arcname="ab_summary.json")
        if SERVE_TAIL.is_file():
            tf.add(str(SERVE_TAIL), arcname="serve_tail.log")
        for arm_name in arms_requested:
            out_dir = WORK_ROOT / f"out_{arm_name}"
            if (out_dir / "benchmark.json").is_file():
                tf.add(str(out_dir / "benchmark.json"), arcname=f"{arm_name}_benchmark.json")
            t_dir = out_dir / "transcripts"
            if t_dir.is_dir():
                tf.add(str(t_dir), arcname=f"{arm_name}_transcripts")
            alog = WORK_ROOT / f"arm_{arm_name}.log"
            if alog.is_file():
                tf.add(str(alog), arcname=f"arm_{arm_name}.log")

    log(f"ALL DONE in {(time.time() - t_start)/60.0:.1f} min! Summary:\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
