# Paste this entire file into the launcher cell.
# Required notebook variables: MODEL_DIR, DRAFT_MODEL_DIR, WHEELHOUSE_DIR,
# WORKING_DIR, SERVED_MODEL_NAME, SERVED_MODEL_PORT.
# MODEL_DIR: Intel AutoRound INT4 target (BF16 PLE or albucino FP8 PLE).
# DRAFT_MODEL_DIR: downloaded albucino runtime/mtp-int4-g32, or its dataset root.
import os, sys, glob, time, shutil, subprocess, urllib.request, urllib.error
import copy, hashlib, json, re, shlex, socket, zipfile
from pathlib import Path

PREFIX = "/tmp/sgl-intel"       # Separate from the previous fork's /tmp/sgl venv.
LOG = str(Path(WORKING_DIR) / "serve.log")
CFG = dict(
    CTX=(116+12+8)*1024,
    MEMFRAC=0.96,
    MAXREQ=10,
    CUDAGRAPH_MAXBS=10,
    MAMBA_CACHE=60,
    CHUNK=8192,
    MAX_PREFILL=16384,
    SPEC=True,
    FRSPEC=True,
    AUTOTUNE=True,
    # Preserve the original Kaggle checkpoint prefetch/cache behavior.
    PREFETCH_CHECKPOINTS=True,
    DROP_CACHE_AFTER_LOAD=True,
    LINEAR_BACKEND="flashinfer",
    KVDTYPE="fp8_e4m3",
    SSM_DTYPE="bfloat16",
    MAMBA_RADIX="extra_buffer",
    GDN_MTP_CACHE_MODE="none",
    SERVED_NAME=SERVED_MODEL_NAME,
    SPEC_ACCEPT_SINGLE=1.0,
    SPEC_ACCEPT_ACC=1.0,
    SPEC_STEPS=3,
)
VENV = f"{PREFIX}/venv"
PYTHON = f"{VENV}/bin/python"
SGLANG = f"{VENV}/bin/sglang"
TOKEN_MAP_SHA = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
EXPERT_TARGET = r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"
DENSE_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"


def run(cmd, *, check=True, env=None):
    cmd = list(map(str, cmd))
    print("$", shlex.join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if result.stdout.strip(): print(result.stdout.strip()[-6000:])
    if result.stderr.strip() and result.returncode: print(result.stderr.strip()[-6000:])
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(cmd)}")
    return result


def find_unique(root, name, *, directory=False, required=True):
    root = Path(root)
    direct = root if root.name == name else root / name
    valid = lambda p: p.is_dir() if directory else p.is_file()
    if valid(direct): return direct
    hits = sorted(p for p in root.rglob(name) if valid(p))
    if len(hits) > 1:
        raise RuntimeError(f"Multiple {name} matches under {root}; use a more specific directory: {hits}")
    if hits: return hits[0]
    if required: raise FileNotFoundError(f"{name} not found under {root}")
    return None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()


def indexed_shards(root):
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if not names: raise RuntimeError(f"Empty weight index: {root}")
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Invalid shard path: {name}")
        p = root / rel
        if not p.is_file() or p.stat().st_size < 8:
            raise RuntimeError(f"Missing/empty shard: {p}")
    return index, names


def prepare_draft_view(target, source_root, work_root):
    """Adapt metadata only; link original dataset weights without copying them."""
    target, source_root, work_root = map(Path, (target, source_root, work_root))
    configs = [source_root / "config.json"] if (source_root / "config.json").is_file() else []
    configs += sorted(p for p in source_root.rglob("config.json") if p not in configs)
    candidates = []
    for path in configs:
        cfg = json.loads(path.read_text())
        q = cfg.get("quantization_config", {})
        if q.get("quant_method") == "compressed-tensors" and "mtp_routed_experts" in q.get("config_groups", {}):
            candidates.append((path.parent, cfg))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one INT4 g32 MTP checkpoint under DRAFT_MODEL_DIR={source_root}; found {len(candidates)}")
    source, cfg = candidates[0]
    index, shards = indexed_shards(source)
    if not any(k.startswith("mtp.layers.0.mlp.experts.") for k in index["weight_map"]):
        raise RuntimeError("Draft index does not contain the expected MTP experts")
    cfg = copy.deepcopy(cfg)
    q = cfg["quantization_config"]
    group = q["config_groups"]["mtp_routed_experts"]
    w = group["weights"]
    if (w["num_bits"], w["group_size"], w["symmetric"]) != (4, 32, True):
        raise RuntimeError("Expected symmetric INT4 group32 draft experts")
    if group["targets"] not in (["RoutedExperts"], [EXPERT_TARGET]):
        raise RuntimeError(f"Unexpected draft target rules: {group['targets']}")
    if q.get("ignore", []) not in ([], [DENSE_IGNORE]):
        raise RuntimeError("Unexpected draft ignore rules")
    group["targets"] = [EXPERT_TARGET]
    q["ignore"] = [DENSE_IGNORE]
    # Content/path-specific directory makes notebook reruns idempotent without
    # overwriting an unrelated view. We never modify the read-only datasets.
    identity = json.dumps({"source": str(source.resolve()), "target": str(target.resolve()), "config": cfg}, sort_keys=True)
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    view = work_root / f"draft-view-{suffix}"
    view.mkdir(parents=True, exist_ok=True)
    config_path = view / "config.json"
    config_text = json.dumps(cfg, indent=2) + "\n"
    if config_path.exists() or config_path.is_symlink():
        if config_path.is_symlink() or config_path.read_text() != config_text:
            raise RuntimeError(f"Unexpected existing draft-view config: {config_path}")
    else:
        config_path.write_text(config_text)
    links = {name: source / name for name in shards}
    links["model.safetensors.index.json"] = source / "model.safetensors.index.json"
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"):
        if (target / name).is_file(): links[name] = target / name
    for name, src in links.items():
        dest = view / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if not dest.is_symlink() or dest.resolve() != src.resolve():
                raise RuntimeError(f"Unexpected existing draft-view file: {dest}")
        else:
            dest.symlink_to(src.resolve())
    print(f"Draft source: {source}\nSGLang draft view: {view} ({len(shards)} linked weight shards)")
    return view


# ---- validate notebook inputs / identify the offline bundle ----
MODEL_DIR = str(Path(MODEL_DIR).expanduser().resolve())
assert Path(MODEL_DIR).is_dir(), f"model dir not found: {MODEL_DIR}"
assert CFG["CUDAGRAPH_MAXBS"] >= CFG["MAXREQ"], "CUDAGRAPH_MAXBS must cover MAXREQ"
assert CFG["CHUNK"] > 0 and CFG["CHUNK"] % 64 == 0
# Refuse to start a second server on an occupied port; do not kill other jobs.
with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", int(SERVED_MODEL_PORT))) == 0:
        raise RuntimeError(f"Port {SERVED_MODEL_PORT} is occupied; stop the existing server before rerunning")
Path(PREFIX).mkdir(parents=True, exist_ok=True)
Path(LOG).parent.mkdir(parents=True, exist_ok=True)
WHEELS = find_unique(WHEELHOUSE_DIR, "wheels", directory=True)
LOCK = find_unique(WHEELHOUSE_DIR, "requirements.lock")
sglang_wheels = sorted(WHEELS.glob("sglang-*.whl"))
assert len(sglang_wheels) == 1, f"Expected one SGLang wheel, found {sglang_wheels}"
wheel = sglang_wheels[0]
install_marker = Path(PREFIX) / "installed-bundle.txt"

# ---- one-time offline install, isolated from the old mratsim environment ----
install_env = dict(os.environ, UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1")
install_env.pop("PYTHONPATH", None)
if not Path(SGLANG).exists() or not install_marker.is_file():
    print("=== installing Pennyroyal (offline) ===")
    # Bootstrap uv directly from its wheel; no global pip changes/ensurepip.
    uv_wheels = sorted(WHEELS.glob("uv-*.whl"))
    assert len(uv_wheels) == 1, "Expected one bundled uv wheel"
    uv_binary = Path(PREFIX) / "bin/uv"
    uv_binary.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(uv_wheels[0]) as archive:
        members = [n for n in archive.namelist() if Path(n).name == "uv" and not n.endswith("/")]
        assert len(members) == 1, "Cannot identify the Linux uv executable in its wheel"
        binary = archive.read(members[0])
        assert binary.startswith(b"\x7fELF"), "Bundled uv executable is not a Linux ELF binary"
        uv_binary.write_bytes(binary)
    uv_binary.chmod(0o755)
    uv = [str(uv_binary)]
    if not Path(PYTHON).exists():
        run(uv + ["venv", "--python", sys.executable, VENV], env=install_env)
    install = uv + ["pip", "install", "--python", PYTHON, "--no-index", "--find-links", str(WHEELS)]
    run(install + ["-r", str(LOCK)], env=install_env)
    run(install + ["--reinstall", "--no-deps", str(wheel)], env=install_env)
    install_marker.write_text('install')
else:
    print(f"Matching offline bundle already installed at {VENV}")

# ---- bundled CUDA toolkit and linker symlinks ----
cu13 = sorted(Path(VENV).glob("lib/python*/site-packages/nvidia/cu13"))
if not cu13 or not (cu13[0] / "bin/nvcc").is_file():
    raise RuntimeError("Bundled CUDA 13 nvcc missing; check that the complete Pennyroyal wheelhouse was uploaded")
CUDA_HOME = str(cu13[0])
libdir = Path(CUDA_HOME) / "lib"
lib64 = Path(CUDA_HOME) / "lib64"
if not lib64.exists() and not lib64.is_symlink(): lib64.symlink_to("lib")
for so in sorted(libdir.glob("*.so.*")):
    link = libdir / re.sub(r"\.so\..*$", ".so", so.name)
    if not link.exists() and not link.is_symlink(): link.symlink_to(so.name)
assert (libdir / "libcudart.so").exists(), "libcudart.so missing: JIT linker cannot find CUDA runtime"
if not (libdir / "libcuda.so").exists():
    candidates = []
    for directory in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
        candidates += sorted(p for p in Path(directory).glob("libcuda.so*") if p.is_file())
    if not candidates:
        code = 'import torch; torch.cuda.init(); print(open("/proc/self/maps").read())'
        maps = run([PYTHON, "-c", code], env=install_env).stdout
        candidates = [Path(p) for p in re.findall(r"(/[^\s]*libcuda\.so[^\s]*)", maps)]
    if not candidates: raise RuntimeError("GPU driver libcuda.so not found")
    link = libdir / "libcuda.so"
    if link.is_symlink(): link.unlink()
    link.symlink_to(candidates[0].resolve())
CXX = next((c for c in ("g++-15", "g++-14", "g++-13", "g++-12", "g++-11", "g++") if shutil.which(c)), None)
assert CXX, "Host C++ compiler not found"
CC = CXX.replace("g++", "gcc")
assert shutil.which(CC), f"Matching C compiler missing: {CC}"

# ---- runtime environment ----
env = dict(os.environ)
env.pop("PYTHONPATH", None)
cache = f"{PREFIX}/cache"
for name in ("huggingface", "torch", "torchinductor", "triton", "flashinfer", "cuda", "sglang/jit"):
    Path(cache, name).mkdir(parents=True, exist_ok=True)
env.update({
    "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_HOME": CUDA_HOME, "CUDACXX": f"{CUDA_HOME}/bin/nvcc",
    "PATH": f"{CUDA_HOME}/bin:{VENV}/bin:" + env.get("PATH", ""),
    "LD_LIBRARY_PATH": f"{CUDA_HOME}/lib64:{CUDA_HOME}/lib:" + env.get("LD_LIBRARY_PATH", ""),
    "CC": CC, "CXX": CXX, "CUDAHOSTCXX": CXX,
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
run([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], env=env)
print("Host compiler:", CXX)
# Report the optional later caching patch. This cell does not modify wheel code.
run([PYTHON, "-c", '''import pathlib, sglang, torch
print("sglang", sglang.__version__, "| torch", torch.__version__)
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
'''], env=env)

# ---- validate target and create the read-only draft's SGLang view ----
target_cfg = json.loads(Path(MODEL_DIR, "config.json").read_text())
q = target_cfg.get("quantization_config", {})
assert q.get("quant_method") == "auto-round" and q.get("bits") == 4, "MODEL_DIR must be the Intel/albucino AutoRound INT4 checkpoint"
_, target_shards = indexed_shards(MODEL_DIR)
print("Target:", MODEL_DIR, "| shards:", len(target_shards), "| PLE dtype:", target_cfg.get("text_config", {}).get("ple_embedding_dtype", "checkpoint default (typically BF16)"))
DRAFT_VIEW = None
if CFG["SPEC"]:
    assert Path(DRAFT_MODEL_DIR).is_dir(), f"draft dir not found: {DRAFT_MODEL_DIR}"
    DRAFT_VIEW = prepare_draft_view(MODEL_DIR, DRAFT_MODEL_DIR, PREFIX)

# ---- assemble Pennyroyal / AutoRound launch arguments ----
graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})
graph_bs = [n for n in graph_bs if n <= CFG["CUDAGRAPH_MAXBS"]]
args = [SGLANG, "serve", "--model-path", MODEL_DIR, "--load-format", "safetensors",
    "--model-loader-extra-config", '{"enable_multithread_load":false}',
    "--served-model-name", str(CFG["SERVED_NAME"]), "--host", "127.0.0.1",
    "--port", str(SERVED_MODEL_PORT), "--tensor-parallel-size", "1",
    "--dtype", "bfloat16", "--quantization", "auto-round",
    "--kv-cache-dtype", CFG["KVDTYPE"], "--mem-fraction-static", str(CFG["MEMFRAC"]),
    "--context-length", str(CFG["CTX"]), "--page-size", "64",
    "--max-running-requests", str(CFG["MAXREQ"]), "--chunked-prefill-size", str(CFG["CHUNK"]),
    "--max-prefill-tokens", str(CFG["MAX_PREFILL"]),
    "--cuda-graph-max-bs-decode", str(CFG["CUDAGRAPH_MAXBS"]),
    "--cuda-graph-bs-decode", *map(str, graph_bs),
    "--mamba-ssm-dtype", CFG["SSM_DTYPE"], "--max-mamba-cache-size", str(CFG["MAMBA_CACHE"]),
    "--mamba-radix-cache-strategy", CFG["MAMBA_RADIX"], "--mamba-track-interval", "64",
    "--mamba-backend", CFG["LINEAR_BACKEND"],
    "--linear-attn-decode-backend", CFG["LINEAR_BACKEND"],
    "--linear-attn-prefill-backend", CFG["LINEAR_BACKEND"],
    "--moe-runner-backend", "auto", "--ple-offload-embedding", "--trust-remote-code",
    "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
    "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
    "--default-chat-template-kwargs", '{"preserve_thinking":true}',
    "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
    "--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging"]
if CFG["PREFETCH_CHECKPOINTS"]: args += ["--weight-loader-prefetch-checkpoints"]
if CFG["DROP_CACHE_AFTER_LOAD"]: args += ["--weight-loader-drop-cache-after-load"]
tmpl = Path(MODEL_DIR, "chat_template.jinja")
if tmpl.is_file(): args += ["--chat-template", str(tmpl)]
if not CFG["AUTOTUNE"]: args += ["--disable-flashinfer-autotune"]
if CFG["GDN_MTP_CACHE_MODE"]: args += ["--gdn-mtp-cache-mode", CFG["GDN_MTP_CACHE_MODE"]]
if CFG["SPEC"]:
    args += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(CFG["SPEC_STEPS"]),
        "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(CFG["SPEC_STEPS"]+1),
        "--speculative-draft-model-path", str(DRAFT_VIEW),
        "--speculative-draft-model-quantization", "compressed-tensors", "--speculative-moe-runner-backend", "auto",
        "--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],
        "--speculative-accept-threshold-single", str(CFG["SPEC_ACCEPT_SINGLE"]),
        "--speculative-accept-threshold-acc", str(CFG["SPEC_ACCEPT_ACC"])]
    if CFG["FRSPEC"]:
        tok = find_unique(WHEELHOUSE_DIR, "hot_tokens_64k.pt", required=False)
        if tok is None: tok = find_unique(WHEELHOUSE_DIR, "flash-next-64k.pt")
        assert sha256(tok) == TOKEN_MAP_SHA, "Wrong FR-Spec map; use the map from the pinned Pennyroyal bundle"
        assert sha256(Path(MODEL_DIR, "tokenizer.json")) == TOKENIZER_SHA, "Tokenizer differs from the FR-Spec map; set CFG['FRSPEC']=False"
        args += ["--speculative-token-map", str(tok)]

# ---- launch detached and wait for health ----
precache_model_thread.start()
print("\n" + shlex.join(args) + "\n")
log_offset = Path(LOG).stat().st_size if Path(LOG).exists() else 0
with open(LOG, "ab", buffering=0) as logf:
    proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
print(f"pid {proc.pid} -> {LOG}")

def show_log_tail():
    with open(LOG, "rb") as f:
        f.seek(0, 2)
        f.seek(max(log_offset, f.tell()-20000))
        print(f.read().decode(errors="replace"))

url, t0 = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", time.time()
last_report = -30
while time.time() - NOTEBOOK_START_TIME < SERVER_STARTUP_TIMEOUT:
    if proc.poll() is not None:
        show_log_tail()
        break
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status == 200:
                print(f"\nREADY after {int(time.time()-t0)}s -> {url.replace('/health','/v1')}")
                break
    except Exception:
        pass
    elapsed = int(time.time()-t0)
    if elapsed-last_report >= 30:
        print(f"  {elapsed}s ...", flush=True); last_report=elapsed
    time.sleep(3)
else:
    print(f"\nDEADLINE at {int(time.time()-NOTEBOOK_START_TIME)}s from notebook start; "
          f"releasing the benchmark with the server still loading")