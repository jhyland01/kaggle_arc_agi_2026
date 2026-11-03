import ast
import json
import os
from pathlib import Path

OUT_DIR = Path("/Users/johnny/Documents/ARC-AGI-3-abstraction-reasoning/kaggle_arc2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

nb_q38 = json.load(open("/tmp/arc2_top/arc-agent-sub-q38-direct-b98k-c16-792ef9/arc-agent-sub-q38-direct-b98k-c16-792ef9.ipynb"))
nb_sym = json.load(open("/tmp/arc2_top/arc-agi2-sota-33-89-plus-cognitive-solver/arc-agi2-sota-33-89-plus-cognitive-solver.ipynb"))
nb_evg = json.load(open("/tmp/arc2_top/arc-2-2026-6-6-local-boyko-nvarc/arc-2-2026-6-6-local-boyko-nvarc.ipynb"))
nb_wog = json.load(open("/tmp/arc2_top/arc2-nvarc-full-cost-sota-triple-pass/arc2-nvarc-full-cost-sota-triple-pass.ipynb"))

def strip_magic(src: str) -> str:
    lines = src.splitlines()
    if lines and lines[0].strip().startswith("%%writefile"):
        return "\n".join(lines[1:]) + "\n"
    return src

# ==============================================================================
# STAGE 0: Symbolic + Astra V23 Experts
# ==============================================================================
arc2_symbolic_py = strip_magic("".join(nb_sym["cells"][5]["source"]))
exact_helper = """
    def solve_exact_only(self, task: dict):
        tests = [to_arr(x["input"]) for x in task["test"]]
        by_test = [[] for _ in tests]
        for cand in self._exact_programs(task):
            if len(cand.outputs) == len(tests) and all(valid_grid(o) for o in cand.outputs):
                for i, out in enumerate(cand.outputs):
                    self._add(by_test[i], out, cand.score, cand.name)
        out_per_test = []
        for i in range(len(tests)):
            ordered = []
            seen = set()
            for score, name, arr in sorted(by_test[i], key=lambda x: (-x[0], x[1])):
                key = to_grid(arr)
                if key not in seen:
                    seen.add(key)
                    ordered.append({"grid": arr.astype(int).tolist(), "score": float(score), "name": name})
            out_per_test.append(ordered)
        return out_per_test
"""
target_anchor = "    def solve_task(self, task: dict) -> List[List[np.ndarray]]:"
assert target_anchor in arc2_symbolic_py
arc2_symbolic_py = arc2_symbolic_py.replace(target_anchor, exact_helper.lstrip("\n") + "\n" + target_anchor, 1)

arc2_d4_tools_py = strip_magic("".join(nb_evg["cells"][12]["source"]))
candidate_solver_py = strip_magic("".join(nb_evg["cells"][13]["source"]))
new_experts_py = strip_magic("".join(nb_evg["cells"][14]["source"]))
astra_bridge_py = strip_magic("".join(nb_evg["cells"][15]["source"]))

run_stage0_py = '''"""Stage 0: Immediate fallback submission + Certified Symbolic & Astra V23 Experts."""
import glob
import json
import os
import time
from pathlib import Path

from arc2_symbolic import Arc2SymbolicSolver
from astra_bridge import expert_guess
from arc2_d4_tools import validate_grid


def find_challenge_file(name="arc-agi_test_challenges.json"):
    for pat in (f"/kaggle/input/**/{name}", f"/**/{name}"):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


def ok_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30
            and all(isinstance(r, list) and 1 <= len(r) <= 30 and len(r) == len(g[0]) for r in g)
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))


def distinct_fallback(inp):
    import numpy as np
    arr = np.asarray(inp, dtype=int)
    for cand in (np.rot90(arr, 1).tolist(), np.fliplr(arr).tolist(), np.flipud(arr).tolist()):
        if ok_grid(cand) and cand != inp:
            return cand
    c = [r[:] for r in inp]
    c[0][0] = (c[0][0] + 1) % 10
    return c


def write_atomic(obj, path="/kaggle/working/submission.json"):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def run_stage0(challenges_path=None, sub_path="/kaggle/working/submission.json",
               verified_out="/kaggle/working/stage0_verified.json"):
    t0 = time.time()
    if challenges_path is None:
        challenges_path = find_challenge_file("arc-agi_test_challenges.json")
    challenges = json.load(open(challenges_path))

    # 1. Immediate valid fallback submission
    sub = {}
    for tid, task in challenges.items():
        rows = []
        for x in task["test"]:
            inp = [[int(v) for v in r] for r in x["input"]]
            rows.append({"attempt_1": inp, "attempt_2": distinct_fallback(inp)})
        sub[tid] = rows
    write_atomic(sub, sub_path)

    # 2. Certified Symbolic + Astra V23 exact experts
    sym_solver = Arc2SymbolicSolver()
    stage0_verified = {}
    n_astra = 0
    n_sym = 0

    for tid, task in challenges.items():
        n_tests = len(task["test"])
        per_test_cands = [[] for _ in range(n_tests)]

        # Priority 1: Astra V23 (D4 context occlusion + Diagonal rays + Directional channel fill, LOO verified)
        try:
            train = task["train"]
            for ti, q in enumerate(task["test"]):
                g_astra, info = expert_guess(train, q["input"])
                if g_astra is not None:
                    validate_grid(g_astra)
                    per_test_cands[ti].append({
                        "grid": [[int(v) for v in r] for r in g_astra],
                        "source": f"astra:{info.get('expert', 'v23')}",
                        "score": 1.0,
                    })
                    n_astra += 1
        except Exception:
            pass

        # Priority 2: Arc2SymbolicSolver 100%-train-verified exact programs
        try:
            exact_lists = sym_solver.solve_exact_only(task)
            for ti, cands in enumerate(exact_lists):
                seen = { json.dumps(x["grid"]) for x in per_test_cands[ti] }
                inp_ti = [[int(v) for v in r] for r in task["test"][ti]["input"]]
                for c in cands:
                    k = json.dumps(c["grid"])
                    if k not in seen and ok_grid(c["grid"]) and c["grid"] != inp_ti:
                        seen.add(k)
                        per_test_cands[ti].append({
                            "grid": c["grid"],
                            "source": f"sym:{c['name']}",
                            "score": c["score"],
                        })
                        n_sym += 1
        except Exception:
            pass

        if any(per_test_cands):
            stage0_verified[tid] = per_test_cands
            for ti, cands in enumerate(per_test_cands):
                if cands:
                    inp = [[int(v) for v in r] for r in task["test"][ti]["input"]]
                    a1 = cands[0]["grid"]
                    a2 = cands[1]["grid"] if len(cands) > 1 else (inp if inp != a1 else distinct_fallback(inp))
                    sub[tid][ti] = {"attempt_1": a1, "attempt_2": a2}

    write_atomic(sub, sub_path)
    write_atomic(stage0_verified, verified_out)
    print(f"[Stage 0] Completed in {time.time() - t0:.2f}s | tasks={len(challenges)} | "
          f"verified_tasks={len(stage0_verified)} (astra_hits={n_astra}, sym_hits={n_sym})", flush=True)
    return stage0_verified


if __name__ == "__main__":
    run_stage0()
'''

# ==============================================================================
# PHASE 1: Qwen3.8-27B-FP8 Agent (ARC-AGI-3 v3.2 REPL + Direct Rescue)
# ==============================================================================
grids_py = strip_magic("".join(nb_q38["cells"][2]["source"]))
sandbox_py = strip_magic("".join(nb_q38["cells"][3]["source"]))
backend_py = strip_magic("".join(nb_q38["cells"][4]["source"]))
agent_py = strip_magic("".join(nb_q38["cells"][5]["source"]))
repl_py_orig = strip_magic("".join(nb_q38["cells"][6]["source"]))
client_py = strip_magic("".join(nb_q38["cells"][7]["source"]))
repl_agent_py_orig = strip_magic("".join(nb_q38["cells"][8]["source"]))
run_direct_py = strip_magic("".join(nb_q38["cells"][9]["source"]))

salvage_code = '''
def _salvage_kept_on_timeout(code, store, mem_bytes=2 << 30):
    """ARC-AGI-3 v3.2 fix: when a python snippet times out inside an exploratory search loop,
    extract top-level imports and FunctionDefs via AST and run only those definitions in a fast
    2.5s subprocess so helper functions and candidate transform() definitions are retained."""
    if store is None or not code:
        return ""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    kept_nodes = []
    has_func = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not getattr(node, "level", 0):
            seg = ast.get_source_segment(code, node)
            if seg:
                kept_nodes.append(seg)
        elif isinstance(node, ast.FunctionDef):
            seg = ast.get_source_segment(code, node)
            if seg:
                kept_nodes.append(seg)
                has_func = True
    if not has_func:
        return ""
    defs_only = "\\n\\n".join(kept_nodes)
    job = {"code": defs_only, "helpers": HELPERS,
           "train_inputs": [], "train_outputs": [], "test_inputs": [],
           "retained": store}
    try:
        p = subprocess.run([sys.executable, "-c", RUNNER], input=json.dumps(job),
                           capture_output=True, text=True, timeout=2.5,
                           env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
        for line in reversed(p.stdout.splitlines()):
            if line.startswith("@@KEPT@@"):
                kept = json.loads(line[len("@@KEPT@@"):])
                return _absorb(store, kept)
    except Exception:
        pass
    return ""
'''

old_timeout_block = '''    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout:.0f}s]"'''
new_timeout_block = '''    except subprocess.TimeoutExpired:
        salvaged = _salvage_kept_on_timeout(code, store, mem_bytes=mem_bytes)
        return f"[timed out after {timeout:.0f}s]" + salvaged'''
assert old_timeout_block in repl_py_orig
repl_py = repl_py_orig.replace(old_timeout_block, new_timeout_block, 1) + "\n" + salvage_code

old_py_tool_block = '''            if c["name"] == "python":
                att.python_calls += 1
                out = run_snippet(code, task, timeout=snippet_timeout, store=store)
                since = att.python_calls - att.python_at_submit
                if nudge_every and since and since % nudge_every == 0:
                    out += NUDGE.format(n=since)
                    att.nudges += 1'''

new_py_tool_block = '''            if c["name"] == "python":
                prev_transform = store["defs"].get("transform") if store else None
                att.python_calls += 1
                out = run_snippet(code, task, timeout=snippet_timeout, store=store)
                cur_transform = store["defs"].get("transform") if store else None
                if store and cur_transform and cur_transform != prev_transform:
                    full_auto, added_auto = with_kept("pass", store)
                    n_ok_auto, test_out_auto, out_auto = verify(full_auto, task)
                    ntr_auto = len(task["train"])
                    if n_ok_auto > 0 and all(o is not None for o in test_out_auto):
                        att.programs.append({"turn": att.turns, "code": full_auto, "n_ok": n_ok_auto,
                                             "n_train": ntr_auto, "test": test_out_auto, "auto_from_python": True})
                        if n_ok_auto > best_ok:
                            best_ok = n_ok_auto
                        if n_ok_auto == ntr_auto:
                            out += f"\\n[auto-verified kept transform(): {out_auto}]"
                            done = True
                since = att.python_calls - att.python_at_submit
                if not done and nudge_every and since and since % nudge_every == 0:
                    out += NUDGE.format(n=since)
                    att.nudges += 1'''
assert old_py_tool_block in repl_agent_py_orig
repl_agent_py = repl_agent_py_orig.replace(old_py_tool_block, new_py_tool_block, 1)

run_submit_py = '''"""Phase 1 Competition Runner: ARC-AGI-3 v3.2 Supercharged REPL Agent + Direct xhigh Rescue."""
import glob
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import ChatClient                      # noqa: E402
from repl_agent import solve_repl, AGENT_OPTS      # noqa: E402
from run_direct import answer as direct_answer     # noqa: E402

SUB = "/kaggle/working/submission.json"
REPL_RESULTS = "/kaggle/working/repl_results.json"
REPL_VERIFIED = "/kaggle/working/repl_verified.json"
DIRECT_RESULTS = "/kaggle/working/direct_results.json"


def find(name, root="/kaggle/input"):
    hits = glob.glob(f"{root}/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def write_atomic(obj, path=SUB):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def ranked_answers_detailed(records, task):
    """Per test input: returns (verified_grids, partial_grids, all_ranked_grids)."""
    ntr = len(task["train"])
    ver_out, part_out, all_out = [], [], []
    for j in range(len(task["test"])):
        score = {}
        grids = {}
        for r in records:
            for p in r["programs"]:
                g = p["test"][j] if j < len(p["test"]) else None
                if g is None:
                    continue
                k = json.dumps(g)
                grids[k] = g
                verified = int(p["n_ok"] == ntr)
                s = score.get(k, (0, 0, 0))
                score[k] = (s[0] + verified, max(s[1], p["n_ok"]), s[2] + 1)
        ordered_keys = sorted(score, key=lambda k: score[k], reverse=True)
        ver_out.append([grids[k] for k in ordered_keys if score[k][0] > 0])
        part_out.append([grids[k] for k in ordered_keys if score[k][0] == 0 and score[k][1] >= max(1, ntr - 1)])
        all_out.append([grids[k] for k in ordered_keys])
    return ver_out, part_out, all_out


def run_direct_rescue(cfg, tasks, client, deadline, sub, sub_path, t0):
    """Run direct xhigh reasoning on any remaining Phase 1 time for unverified tasks."""
    lock = threading.Lock()
    got = {}
    if os.path.exists(DIRECT_RESULTS):
        try:
            got = json.load(open(DIRECT_RESULTS))
        except Exception:
            got = {}
    samples = cfg.get("direct_samples", 1)
    jobs = [(t, k) for k in range(samples) for t in tasks]
    count = {}

    def work(job):
        t, k = job
        if deadline - time.time() < cfg.get("min_start_s", 360):
            return
        ts = time.time()
        rec = direct_answer(client, tasks[t], 1000 * k + 7, deadline,
                            max_tokens=cfg.get("direct_max_tokens", 49152),
                            rescue_tokens=cfg.get("rescue_tokens", 8192))
        rec["wall"] = round(time.time() - ts, 1)
        with lock:
            if rec.get("grids"):
                got.setdefault(t, []).append(rec["grids"])
                write_atomic(got, DIRECT_RESULTS)
                for j, x in enumerate(tasks[t]["test"]):
                    cands = []
                    for grids in got[t]:
                        if j < len(grids) and grids[j] not in cands:
                            cands.append(grids[j])
                    if cands:
                        cur_a1 = sub[t][j]["attempt_1"]
                        if cur_a1 == x["input"]:
                            sub[t][j]["attempt_1"] = cands[0]
                        elif sub[t][j]["attempt_2"] == x["input"] and cands[0] != cur_a1:
                            sub[t][j]["attempt_2"] = cands[0]
                write_atomic(sub, sub_path)
            st = rec["status"].split(":")[0]
            count[st] = count.get(st, 0) + 1
            print(f"[Direct {time.time() - t0:6.0f}s] {t}#{k} {rec['status'][:30]:12s} "
                  f"tok {rec.get('gen_tokens', 0):6d} | {count}", flush=True)

    with ThreadPoolExecutor(max_workers=cfg.get("direct_concurrency", 12)) as ex:
        list(ex.map(work, jobs))


def main(cfg):
    t0 = time.time()
    rerun = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
    sub_path = cfg.get("sub_path", SUB)
    ch = json.load(open(find(cfg.get("challenges", "arc-agi_test_challenges.json"),
                             cfg.get("data_dir", "/kaggle/input"))))
    if not rerun and not cfg.get("full_interactive"):
        ch = {t: ch[t] for t in sorted(ch)[:cfg.get("interactive_tasks", 2)]}

    stage0_ver = {}
    if os.path.exists("/kaggle/working/stage0_verified.json"):
        try:
            stage0_ver = json.load(open("/kaggle/working/stage0_verified.json"))
        except Exception:
            stage0_ver = {}

    n = len(ch)
    deadline = cfg["deadline_epoch"]
    avail = max(0.0, deadline - time.time())
    supply = cfg.get("tok_per_s", 260) * avail
    cap = int(min(cfg.get("max_cap", 28000), max(cfg.get("min_cap", 12000), supply / max(n * 1.4, 1))))
    print(f"[Phase 1 REPL] rerun={rerun} tasks={n} avail={avail / 3600:.2f}h est_supply={supply / 1e6:.2f}M "
          f"-> per-attempt cap={cap}", flush=True)

    # Order: cheapest prompt first across all tasks
    order = sorted(ch, key=lambda t: (len(json.dumps(ch[t])), t))
    tasks = {t: ch[t] for t in order}

    client = ChatClient(cfg["base_url"], cfg["model"], max_tokens=cfg.get("max_tokens", 16384),
                        temperature=cfg.get("temperature", 0.6), top_p=cfg.get("top_p", 0.95),
                        top_k=cfg.get("top_k", 20), thinking=cfg.get("thinking", True),
                        reasoning_effort=cfg.get("reasoning_effort", "high"),
                        thinking_token_budget=cfg.get("thinking_token_budget"),
                        template_kwargs=cfg.get("template_kwargs"))

    sub = json.load(open(sub_path))
    lock = threading.Lock()
    done = {}
    repl_summary = {}
    repl_verified = {}

    def on_done(rec):
        with lock:
            tid = rec["task_id"]
            done.setdefault(tid, []).append(rec)
            task = tasks[tid]
            ver_out, part_out, all_out = ranked_answers_detailed(done[tid], task)
            repl_summary[tid] = {
                "verified_grids": ver_out,
                "partial_grids": part_out,
                "all_grids": all_out,
                "best_ok": max((r.get("best_ok", -1) for r in done[tid]), default=-1),
                "n_train": len(task["train"]),
            }
            if all(len(v) > 0 for v in ver_out):
                repl_verified[tid] = ver_out
                write_atomic(repl_verified, REPL_VERIFIED)
            write_atomic(repl_summary, REPL_RESULTS)

            for j, cands in enumerate(all_out):
                if cands:
                    inp = task["test"][j]["input"]
                    cur = sub[tid][j]
                    a1 = cands[0]
                    a2 = cands[1] if len(cands) > 1 else (cur["attempt_1"] if cur["attempt_1"] != a1 else inp)
                    sub[tid][j] = {"attempt_1": a1, "attempt_2": a2}
            write_atomic(sub, sub_path)

    _, report = solve_repl(
        tasks, client, attempts=cfg.get("attempts", 2), concurrency=cfg.get("concurrency", 28),
        deadline=deadline, stop_after=1, max_turns=cfg.get("max_turns", 18), max_gen_tokens=cap,
        context_budget=cfg.get("context_budget", 28000), nudge_every=cfg.get("nudge_every", 5),
        turn_max_tokens=cfg.get("turn_max_tokens", 6144), in_python=cfg.get("in_python", True),
        log=lambda m: print(m, flush=True), on_done=on_done,
        **{k: cfg[k] for k in AGENT_OPTS if k in cfg and k not in ("turn_max_tokens", "in_python")})

    ver_tasks = len(repl_verified)
    print(f"[Phase 1 REPL] Done: {ver_tasks}/{n} tasks 100%-demo-verified, "
          f"{sum(a['gen_tokens'] for a in report['attempts'])} tokens, {time.time() - t0:.0f}s", flush=True)

    # If time remains in Phase 1 window, run Direct xhigh rescue on unverified tasks
    rem = deadline - time.time()
    unverified_tasks = {t: tasks[t] for t in order if t not in repl_verified and t not in stage0_ver}
    if rem > 480 and unverified_tasks:
        print(f"[Phase 1 Direct Rescue] {rem:.0f}s remaining in Phase 1 -> running Direct xhigh on "
              f"{len(unverified_tasks)} unverified tasks", flush=True)
        direct_client = ChatClient(
            cfg["base_url"], cfg["model"], max_tokens=cfg.get("direct_max_tokens", 49152),
            temperature=1.0, top_p=0.95, top_k=20, thinking=True,
            thinking_token_budget=cfg.get("direct_thinking_budget", 40960),
            template_kwargs={"reasoning_effort": "xhigh"}, timeout=3600)
        run_direct_rescue(cfg, unverified_tasks, direct_client, deadline, sub, sub_path, t0)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
'''

submit_main_py = '''"""Phase 1 Server Lifecycle Manager: Page-Cache Weight Preload + TRITON_ATTN MTP vLLM + Real Smoke Test."""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = Path("/kaggle/working")
SITE = Path("/tmp/vllm-site-packages")
LOG = WORK / "vllm-server.log"


def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        raise FileNotFoundError(pattern)
    return Path(hits[0])


def preload_weights(model_dir: Path, workers: int = 8):
    """Preload safetensors into OS page cache in 64 MiB chunks to cut vLLM cold-start time by >50%."""
    t0 = time.time()
    files = sorted(p for p in model_dir.rglob("*") if p.is_file() and p.suffix in {".safetensors", ".bin", ".pt"})
    if not files:
        return

    def _read_file(p: Path):
        n = 0
        with open(p, "rb") as f:
            while chunk := f.read(64 * 1024 * 1024):
                n += len(chunk)
        return n

    with ThreadPoolExecutor(max_workers=workers) as ex:
        total = sum(ex.map(_read_file, files))
    print(f"[Preload] Cached {total / (1024**3):.1f} GiB across {len(files)} files in {time.time() - t0:.1f}s", flush=True)


VLLM_PYTHON = os.environ.get("VLLM_PYTHON", sys.executable)


def install(cfg):
    wh = one("/kaggle/input/**/arc3-vllm-h100-wheelhouse-v3/**/requirements.lock").parent
    t = time.time()
    SITE.mkdir(parents=True, exist_ok=True)
    subprocess.run([VLLM_PYTHON, "-m", "pip", "install", "--no-index", "--find-links", str(wh),
                    "--requirement", str(wh / "requirements.lock"), "--target", str(SITE),
                    "--upgrade", "--ignore-installed", "--only-binary", ":all:", "--no-compile",
                    "--disable-pip-version-check", "--no-warn-conflicts", "-q"], check=True)
    print(f"wheelhouse installed to {SITE} using {VLLM_PYTHON} in {time.time() - t:.0f}s", flush=True)
    if cfg.get("overlay"):
        wheels = sorted(glob.glob(cfg["overlay"], recursive=True))
        if not wheels:
            raise FileNotFoundError(cfg["overlay"])
        for pat in ("transformers", "transformers-*.dist-info", "huggingface_hub",
                    "huggingface_hub-*.dist-info"):
            for p in SITE.glob(pat):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        subprocess.run([VLLM_PYTHON, "-m", "pip", "install", "--no-index", "--no-deps",
                        "--target", str(SITE), "--upgrade", "--force-reinstall", "--no-compile",
                        "--disable-pip-version-check", "-q", *wheels], check=True)
        print("overlay installed:", [Path(w).name for w in wheels], flush=True)


def cuda_link_dir():
    d = Path("/tmp/cuda-link")
    d.mkdir(parents=True, exist_ok=True)
    hits = []
    for pat in ("/usr/local/nvidia/lib64/libcuda.so*", "/usr/lib/x86_64-linux-gnu/libcuda.so*",
                "/usr/lib64/libcuda.so*", "/usr/local/cuda/compat/libcuda.so*"):
        hits += sorted(glob.glob(pat))
    link = d / "libcuda.so"
    if hits and not os.path.lexists(link):
        os.symlink(next((h for h in hits if h.endswith(".so.1")), hits[0]), link)
    return str(d)


def smoke_test_generation():
    """Verify actual token generation (catches SM89 FlashInfer FP8+MTP crashes that /v1/models misses)."""
    payload = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": "Reply with the single digit 4."}],
        "max_tokens": 16,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:1234/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        return bool(choices)


def detect_gpu_topology():
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        lines = [l.strip() for l in smi.stdout.splitlines() if l.strip()] if smi.returncode == 0 else []
    except Exception:
        lines = []
    n_gpus = max(1, len(lines))
    gpu_name = lines[0].split(",")[0].strip() if lines else "Unknown"
    try:
        vram_mib = int(float(lines[0].split(",")[1].strip())) if lines else 0
    except Exception:
        vram_mib = 0
    return n_gpus, gpu_name, vram_mib


def start_server(cfg, extra, tp_size=4):
    model = one(cfg["model_glob"]).parent
    py_prefix = Path(VLLM_PYTHON).resolve().parent.parent
    ld_parts = [str(py_prefix / "lib")] if (py_prefix / "lib").is_dir() else []
    if os.environ.get("LD_LIBRARY_PATH"):
        ld_parts.append(os.environ["LD_LIBRARY_PATH"])
    path_parts = [str(py_prefix / "bin")] if (py_prefix / "bin").is_dir() else []
    if os.environ.get("PATH"):
        path_parts.append(os.environ["PATH"])
    env = dict(os.environ,
               PYTHONPATH=str(SITE),
               PYTHONNOUSERSITE="1",
               LD_LIBRARY_PATH=":".join(ld_parts),
               PATH=":".join(path_parts),
               USE_TF="0",
               TRANSFORMERS_NO_TF="1",
               TRANSFORMERS_NO_TORCHVISION="1",
               VLLM_NO_USAGE_STATS="1",
               TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
               LIBRARY_PATH=":".join([cuda_link_dir(), "/usr/local/nvidia/lib64",
                                      os.environ.get("LIBRARY_PATH", "")]))
    if "--attention-backend" in extra and "TRITON_ATTN" in extra:
        env["VLLM_FORCE_ATTN_BACKEND"] = "TRITON_ATTN"

    cmd = [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
           "--served-model-name", "qwen", "--host", "127.0.0.1", "--port", "1234",
           "--tensor-parallel-size", str(tp_size), "--enable-auto-tool-choice", *cfg["parsers"],
           "--generation-config", "vllm", "--enable-prefix-caching",
           "--max-model-len", str(cfg.get("max_model_len", 65536)), "--gpu-memory-utilization", "0.90",
           "--max-num-seqs", str(cfg.get("max_num_seqs", 28)), *extra]
    print(" ".join(cmd), flush=True)
    t = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    while time.time() - t < 900:
        text = LOG.read_text(errors="replace") if LOG.exists() else ""
        if proc.poll() is not None or "EngineCore failed" in text or "RuntimeError:" in text:
            break
        try:
            urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5)
            if smoke_test_generation():
                print(f"server ready & generation smoke-verified after {time.time() - t:.0f}s", flush=True)
                return proc
            else:
                print("server /v1/models up but smoke generation returned empty", flush=True)
                break
        except urllib.error.HTTPError as he:
            if he.code >= 500:
                print(f"smoke generation HTTP {he.code} -> engine crashed", flush=True)
                break
            time.sleep(8)
        except Exception:
            time.sleep(8)
    print("server FAILED to start; tail:\\n" + "\\n".join(
        LOG.read_text(errors="replace").splitlines()[-30:]), flush=True)
    kill(proc)
    return None


def kill(proc):
    """Kill the server's whole process group and wait for all 4 L4 GPUs to drain completely."""
    import signal
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=45)
        except Exception:
            pass
    subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"], check=False)
    for _ in range(36):
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()
        if used and all(int(u) < 1000 for u in used if u.strip().isdigit()):
            print(f"[GPU Drain] All GPUs drained cleanly: {used} MiB", flush=True)
            return
        time.sleep(4)


def main(cfg):
    proc = None
    try:
        model_dir = one(cfg["model_glob"]).parent
        preload_weights(model_dir)
        install(cfg)
        n_gpus, gpu_name, vram_mib = detect_gpu_topology()
        tp_size = 1 if (n_gpus == 1 or vram_mib >= 45000) else min(4, n_gpus)
        print(f"[Phase 1 Hardware] Detected {n_gpus}x {gpu_name} ({vram_mib} MiB/GPU) -> tensor_parallel_size={tp_size}", flush=True)
        ladder = list(cfg.get("server_ladder", [[]]))
        if tp_size == 1 and ("PRO 6000" in gpu_name.upper() or "BLACKWELL" in gpu_name.upper() or vram_mib >= 80000):
            # On G4 (RTX PRO 6000 Blackwell 96 GB, SM120), prioritize native FlashInfer + Cutlass FP8 + 3-step MTP (400-545 tok/s)
            g4_fast_tier = [
                "--reasoning-config", json.dumps({"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}),
                "--kv-cache-dtype", "fp8",
                "--language-model-only",
                "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3}),
            ]
            ladder = [g4_fast_tier] + ladder
        for i, extra in enumerate(ladder):
            proc = start_server(cfg, extra, tp_size=tp_size)
            if proc is not None:
                print(f"server up with option set {i}: {extra}", flush=True)
                break
            print(f"option set {i} failed -> next", flush=True)
            time.sleep(10)
        if proc is None:
            return
        sys.path.insert(0, HERE)
        import run_submit
        run_submit.main(dict(cfg, base_url="http://127.0.0.1:1234/v1", model="qwen"))
    except BaseException:
        print("submit_main caught:\\n" + traceback.format_exc(), flush=True)
    finally:
        kill(proc)
        shutil.rmtree(SITE, ignore_errors=True)


if __name__ == "__main__":
    try:
        main(json.loads(sys.argv[1]))
    except BaseException:
        print(traceback.format_exc(), flush=True)
    sys.exit(0)
'''

# ==============================================================================
# PHASE 2: Qwen3-4B NVARC TTT + GPU-Logsumexp 8-View Turbo-DFS + Priority Queue
# ==============================================================================
arc_loader_py = strip_magic("".join(nb_wog["cells"][3]["source"]))
arc_decoder_py = strip_magic("".join(nb_wog["cells"][4]["source"]))
arc_sched_py = strip_magic("".join(nb_wog["cells"][5]["source"]))
arc_solver_py_orig = strip_magic("".join(nb_wog["cells"][6]["source"]))
starter_py_orig = strip_magic("".join(nb_wog["cells"][7]["source"]))

# Patch resolve_model_dir() in arc_solver.py to find either sorokin/qwen3_4b_grids15_sft139 or konstantinboyko/qwen3-4b-bfloat16
old_resolve = '''def resolve_model_dir():
    if CFG["model_path"] and os.path.isfile(os.path.join(CFG["model_path"], "config.json")):
        return CFG["model_path"]
    candidates = [
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    import glob
    for c in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(c)
        if "grids15" in d and os.path.isfile(os.path.join(d, "tokenizer.json")):
            return d
    raise FileNotFoundError("qwen3_4b_grids15_sft139 model directory not found under /kaggle/input")'''

new_resolve = '''def resolve_model_dir():
    if CFG["model_path"] and os.path.isfile(os.path.join(CFG["model_path"], "config.json")):
        return CFG["model_path"]
    candidates = [
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
        "/kaggle/input/models/konstantinboyko/qwen3-4b-bfloat16-v02-01-01/transformers/default/1",
        "/kaggle/input/models/konstantinboyko/qwen3-4b-bfloat16-v02-01-01/Transformers/default/1",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    import glob
    for c in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(c)
        if ("grids15" in d or "qwen3-4b" in d or "qwen3_4b" in d) and os.path.isfile(os.path.join(d, "tokenizer.json")):
            return d
    raise FileNotFoundError("Qwen3-4B NVARC model directory not found under /kaggle/input")'''
assert old_resolve in arc_solver_py_orig
arc_solver_py = arc_solver_py_orig.replace(old_resolve, new_resolve, 1)

# Patch starter.py to prioritize tasks NOT yet demo-verified by Stage 0 or Phase 1 REPL
old_order = '''    est = {k: est_task_seconds(t) for k, t in data.items()}
    # cheapest (and most often solved) tasks first: whatever the time pressure on the
    # hidden set turns out to be, it can only squeeze the large low-yield tail
    order = sorted(data.keys(), key=lambda k: (est[k], k))'''

new_order = '''    est = {k: est_task_seconds(t) for k, t in data.items()}
    verified_tasks = set()
    for vpath in ("/kaggle/working/repl_verified.json",):
        if os.path.exists(vpath):
            try:
                verified_tasks |= set(json.load(open(vpath)).keys())
            except Exception:
                pass
    # Priority 0: tasks NOT yet 100%-demo-verified by Stage 0 or Phase 1 REPL (cheapest first)
    # Priority 1: tasks already demo-verified (run after unverified tasks to provide orthogonal attempt_2)
    order = sorted(data.keys(), key=lambda k: (1 if k in verified_tasks else 0, est[k], k))
    print(f"[starter] Priority queue: {sum(1 for k in order if k not in verified_tasks)} unverified tasks first, "
          f"then {sum(1 for k in order if k in verified_tasks)} already-verified tasks", flush=True)'''
assert old_order in starter_py_orig
starter_py = starter_py_orig.replace(old_order, new_order, 1)
starter_py = starter_py.replace(
    'def local_worker(rank, end_time, shared, test_path):\n\n    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)\n\n    torch.set_default_device("cpu")',
    'def detect_n_gpus():\n    import subprocess\n    try:\n        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)\n        lines = [l for l in smi.stdout.splitlines() if l.strip().startswith("GPU ")] if smi.returncode == 0 else []\n    except Exception:\n        lines = []\n    return max(1, len(lines))\n\n\ndef local_worker(rank, end_time, shared, test_path, n_gpus=4):\n\n    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % max(1, n_gpus))\n\n    torch.set_default_device("cpu")\n    if n_gpus == 1 and N_WORKERS > 1:\n        try:\n            torch.cuda.set_per_process_memory_fraction(min(0.92 / N_WORKERS, 0.23), 0)\n        except Exception:\n            pass',
    1
)
starter_py = starter_py.replace(
    '    mp.spawn(worker_fn, args=(args.end_time, shared, test_path), nprocs=N_WORKERS, join=True)',
    '    n_gpus = detect_n_gpus()\n    print(f"[starter] Launching {N_WORKERS} workers across {n_gpus} visible GPU(s) (modulo mapping rank % {n_gpus})", flush=True)\n    mp.spawn(worker_fn, args=(args.end_time, shared, test_path, n_gpus), nprocs=N_WORKERS, join=True)',
    1
)

# ==============================================================================
# PHASE 3: Monotonic Cross-Paradigm Union Arbitrator (final_arbitrator.py)
# ==============================================================================
final_arbitrator_py = '''"""Phase 3: Cross-Paradigm Union Arbitrator & Strict Submission Verifier.

Combines:
  1. V_repl   : 100% Demo-Verified Python programs from Phase 1 (Qwen3.8-27B-FP8 REPL)
  2. N_nvarc  : Per-task LoRA TTT + 16-view Turbo-DFS score_kgmon grids from Phase 2 (Qwen3-4B NVARC)
  3. V_sym    : 100% Train-Verified Symbolic & Astra V23 experts from Stage 0
  4. D_direct : Direct xhigh reasoning grids from Phase 1 rescue
  5. P_repl   : High-agreement partial Python programs from Phase 1 (n_ok >= n_train - 1)
"""
import glob
import json
import os
import numpy as np

from arc_loader import ArcDataset
from arc_decoder import ArcDecoder


def find_data_dir():
    for d in ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2", "/kaggle/input/arc-prize-2026-arc-agi-2"):
        if os.path.isdir(d):
            return d
    hits = glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("arc-prize-2026-arc-agi-2 input directory not found")


def ok_grid(g):
    if not isinstance(g, list) or not (1 <= len(g) <= 30):
        return False
    w = len(g[0]) if isinstance(g[0], list) else 0
    if not (1 <= w <= 30):
        return False
    for r in g:
        if not isinstance(r, list) or len(r) != w:
            return False
        for v in r:
            if not isinstance(v, int) or isinstance(v, bool) or not (0 <= v <= 9):
                return False
    return True


def clean_grid(g, inp=None):
    if isinstance(g, np.ndarray):
        g = g.astype(int).tolist()
    if not ok_grid(g):
        return None
    if g == [[0]]:
        return None
    res = [[int(v) for v in r] for r in g]
    if inp is not None and res == inp:
        return None
    return res


def distinct_fallbacks(inp):
    arr = np.asarray(inp, dtype=int)
    out = []
    for cand in (inp, np.rot90(arr, 1).tolist(), np.fliplr(arr).tolist(), np.flipud(arr).tolist()):
        cg = clean_grid(cand) or [[int(v) for v in r] for r in inp]
        if cg not in out:
            out.append(cg)
    c = [r[:] for r in out[0]]
    c[0][0] = (c[0][0] + 1) % 10
    if c not in out:
        out.append(c)
    return out


def load_json_safe(path):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception as e:
            print(f"[Arbitrator] Warning loading {path}: {e!r}")
    return {}


def main():
    rerun = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
    data_dir = find_data_dir()
    test_file = os.path.join(data_dir, "arc-agi_test_challenges.json")
    challenges = json.load(open(test_file))

    # Load NVARC decoded results if present
    nvarc_by_task = {}
    try:
        ds = ArcDataset.from_file(test_file)
        decoder = ArcDecoder(ds.split_multi_replies(), n_guesses=2)
        for cdir in ("/kaggle/inference_outputs", "/tmp/inference_outputs", "/kaggle/working/inference_outputs"):
            if os.path.isdir(cdir) and os.listdir(cdir):
                decoder.load_decoded_results(cdir)
                print(f"[Arbitrator] Loaded NVARC decoded beams from {cdir}")
                break
        nvarc_sub = ds.get_submission(decoder.run_selection_algo())
        for tid, rows in nvarc_sub.items():
            nvarc_by_task[tid] = []
            for r in rows:
                cands = []
                for k in ("attempt_1", "attempt_2"):
                    cg = clean_grid(r.get(k))
                    if cg is not None and cg not in cands:
                        cands.append(cg)
                nvarc_by_task[tid].append(cands)
    except Exception as e:
        print(f"[Arbitrator] NVARC decode load warning: {e!r}")

    stage0_ver = load_json_safe("/kaggle/working/stage0_verified.json")
    repl_res = load_json_safe("/kaggle/working/repl_results.json")
    direct_res = load_json_safe("/kaggle/working/direct_results.json")

    submission = {}
    stats = {
        "union_repl_plus_nvarc": 0,
        "agreed_repl_and_nvarc": 0,
        "repl_verified_only": 0,
        "nvarc_plus_sym": 0,
        "nvarc_only": 0,
        "sym_only": 0,
        "direct_or_partial": 0,
        "fallback_slots": 0,
    }

    for tid, task in challenges.items():
        n_tests = len(task["test"])
        submission[tid] = []

        t_repl = repl_res.get(tid, {})
        t_sym = stage0_ver.get(tid, [])
        t_nvarc = nvarc_by_task.get(tid, [[] for _ in range(n_tests)])
        t_direct = direct_res.get(tid, [])

        for j in range(n_tests):
            inp = [[int(v) for v in r] for r in task["test"][j]["input"]]

            v_repl = []
            for g in (t_repl.get("verified_grids", [])[j] if j < len(t_repl.get("verified_grids", [])) else []):
                cg = clean_grid(g, inp)
                if cg is not None and cg not in v_repl:
                    v_repl.append(cg)

            p_repl = []
            for g in (t_repl.get("partial_grids", [])[j] if j < len(t_repl.get("partial_grids", [])) else []):
                cg = clean_grid(g, inp)
                if cg is not None and cg not in p_repl:
                    p_repl.append(cg)

            v_sym = []
            for item in (t_sym[j] if j < len(t_sym) else []):
                cg = clean_grid(item.get("grid") if isinstance(item, dict) else item, inp)
                if cg is not None and cg not in v_sym:
                    v_sym.append(cg)

            n_nvarc = []
            for g in (t_nvarc[j] if j < len(t_nvarc) else []):
                cg = clean_grid(g, inp)
                if cg is not None and cg not in n_nvarc:
                    n_nvarc.append(cg)

            d_direct = []
            for sample_grids in t_direct:
                if isinstance(sample_grids, list) and j < len(sample_grids):
                    cg = clean_grid(sample_grids[j], inp)
                    if cg is not None and cg not in d_direct:
                        d_direct.append(cg)

            # Cross-Paradigm Union Selection for (attempt_1, attempt_2)
            chosen = []
            if v_repl and n_nvarc:
                if v_repl[0] != n_nvarc[0]:
                    chosen = [v_repl[0], n_nvarc[0]]
                    stats["union_repl_plus_nvarc"] += 1
                else:
                    chosen = [v_repl[0]]
                    stats["agreed_repl_and_nvarc"] += 1
                    for g in v_repl[1:] + v_sym + n_nvarc[1:] + d_direct + p_repl:
                        if g not in chosen:
                            chosen.append(g)
                            break
            elif v_repl:
                stats["repl_verified_only"] += 1
                for g in v_repl + v_sym + d_direct + p_repl:
                    if g not in chosen:
                        chosen.append(g)
                        if len(chosen) == 2:
                            break
            elif n_nvarc:
                chosen = [n_nvarc[0]]
                if v_sym and v_sym[0] != n_nvarc[0]:
                    chosen.append(v_sym[0])
                    stats["nvarc_plus_sym"] += 1
                else:
                    stats["nvarc_only"] += 1
                    for g in n_nvarc[1:] + v_sym + d_direct + p_repl:
                        if g not in chosen:
                            chosen.append(g)
                            break
            elif v_sym:
                stats["sym_only"] += 1
                for g in v_sym + d_direct + p_repl:
                    if g not in chosen:
                        chosen.append(g)
                        if len(chosen) == 2:
                            break
            elif d_direct or p_repl:
                stats["direct_or_partial"] += 1
                for g in d_direct + p_repl:
                    if g not in chosen:
                        chosen.append(g)
                        if len(chosen) == 2:
                            break

            for fb in distinct_fallbacks(inp):
                if len(chosen) >= 2:
                    break
                if fb not in chosen:
                    chosen.append(fb)
                    stats["fallback_slots"] += 1

            submission[tid].append({"attempt_1": chosen[0], "attempt_2": chosen[1]})

    tmp = "/kaggle/working/submission.json.tmp"
    out_path = "/kaggle/working/submission.json"
    with open(tmp, "w") as f:
        json.dump(submission, f, separators=(",", ":"))
    os.replace(tmp, out_path)
    print(f"[Arbitrator] Wrote {out_path} ({os.path.getsize(out_path)} bytes) | stats={stats}", flush=True)
    audit_submission(out_path, challenges)


def audit_submission(path, raw_tasks):
    assert os.path.exists(path), f"Submission file {path} missing!"
    with open(path, "r") as f:
        sub = json.load(f)
    assert isinstance(sub, dict), "Submission is not a dict!"
    assert len(sub) == len(raw_tasks), f"Task count mismatch: {len(sub)} vs {len(raw_tasks)}"
    distinct_pairs = 0
    total_pairs = 0
    for tid, tdata in raw_tasks.items():
        assert tid in sub, f"Missing task {tid} in submission!"
        tests = tdata.get("test", [])
        preds = sub[tid]
        assert len(preds) == len(tests), f"Test count mismatch for {tid}: {len(preds)} vs {len(tests)}"
        for p in preds:
            total_pairs += 1
            assert "attempt_1" in p and "attempt_2" in p, f"Missing attempts in {tid}"
            for att in ("attempt_1", "attempt_2"):
                g = p[att]
                assert ok_grid(g), f"Invalid grid in {tid} {att}"
            if p["attempt_1"] != p["attempt_2"]:
                distinct_pairs += 1
    print(f"[VERIFIED] {path} passed ALL integrity checks: {len(sub)} tasks, "
          f"{distinct_pairs}/{total_pairs} distinct (attempt_1 != attempt_2) test slots!", flush=True)


if __name__ == "__main__":
    main()
'''

# ==============================================================================
# Build the Jupyter Notebook Cells
# ==============================================================================
cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()]
    })

def add_code(code):
    # Verify syntax if not starting with %%writefile
    lines = code.splitlines()
    body = "\n".join(lines[1:]) if (lines and lines[0].startswith("%%writefile")) else code
    ast.parse(body)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": [line + "\n" for line in lines]
    })

add_md("""# ARC Prize 2026 — ARC-AGI-2: 3-Engine Cross-Paradigm Hybrid (`v1`)

Combines three orthogonal solver paradigms inside the 12-hour `4×L4` budget with strict monotonic `pass@2` (`attempt_1` / `attempt_2`) union arbitration:
1. **Stage 0 (Instant CPU)**: Certified 100%-train-verified Symbolic & Geometric Invariant Experts (`Astra V23` D4 occlusion / diagonal ray / directional channel experts with LOO validation + `Arc2SymbolicSolver`).
2. **Phase 1 (`4×L4` GPU — `Qwen3.8-27B-FP8` with ARC-AGI-3 v3.2 Learnings)**:
   - Served via `vLLM` in an isolated `/tmp/vllm-site-packages` target with **parallel OS page-cache weight preloading**, **`TRITON_ATTN` + `--kv-cache-dtype fp8` + `qwen3_5_mtp` speculative decoding**, and a **live generation smoke test** on startup.
   - Runs the **ARC-AGI-3 v3.2 REPL Program-Synthesis Agent (`python` + `submit` tools)**:
     - **AST Timeout Helper-Retention (`_salvage_kept_on_timeout`)**: preserves top-level helper functions and `transform()` definitions even when an exploratory search snippet times out.
     - **Auto-Verification of `python`-defined `transform()`**: automatically verifies any `def transform(grid)` written during exploratory `python` calls against all training examples.
     - **Multi-turn reasoning continuity (`keep_reasoning=True`, `preserve_thinking=True`)** + **hysteresis context trimming (`trim_drain=6000`)** for `>90%` Radix prefix-cache hit rate + **adaptive reasoning-effort ladder (`["medium", "low"]`)**.
3. **Phase 2 (`4×L4` GPU — Priority-Paced `Qwen3-4B` NVARC Per-Task Test-Time Training + 8-View Batched GPU-Logsumexp Turbo-DFS)**:
   - Runs **unverified tasks first** (cheapest-first via online-calibrated `Pacer`), then already-verified tasks (to provide an orthogonal neural `attempt_2`), followed by a second TTT pass on uncertain outputs.
4. **Phase 3 (Cross-Paradigm Union Arbitrator)**:
   - Pairs 100% Demo-Verified REPL Python programs (`attempt_1`) with NVARC `score_kgmon` Top-1 (`attempt_2`) whenever both exist and differ, capturing the union of both paradigms' solves.""")

add_code("""# Cell 1: Global Clock, Directories & Immediate Fallback Safety Submission
import glob, json, os, time
from pathlib import Path

START = time.time()
Path("/kaggle/working/start_epoch.txt").write_text(str(START))
os.makedirs("/kaggle/working/agent", exist_ok=True)

try:
    hits = glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True)
    if hits:
        ch = json.load(open(hits[0]))
        with open("/kaggle/working/submission.json", "w") as f:
            json.dump({t: [{"attempt_1": x["input"], "attempt_2": x["input"]} for x in task["test"]]
                       for t, task in ch.items()}, f)
        print(f"[Init] Fallback submission.json written for {len(ch)} tasks | rerun={os.environ.get('KAGGLE_IS_COMPETITION_RERUN')}")
except Exception as e:
    print("[Init] Fallback warning:", repr(e))""")

add_code("%%writefile /kaggle/working/arc2_symbolic.py\n" + arc2_symbolic_py)
add_code("%%writefile /kaggle/working/arc2_d4_tools.py\n" + arc2_d4_tools_py)
add_code("%%writefile /kaggle/working/candidate_solver.py\n" + candidate_solver_py)
add_code("%%writefile /kaggle/working/new_experts.py\n" + new_experts_py)
add_code("%%writefile /kaggle/working/astra_bridge.py\n" + astra_bridge_py)
add_code("%%writefile /kaggle/working/run_stage0.py\n" + run_stage0_py)

add_code("%%writefile /kaggle/working/agent/grids.py\n" + grids_py)
add_code("%%writefile /kaggle/working/agent/sandbox.py\n" + sandbox_py)
add_code("%%writefile /kaggle/working/agent/backend.py\n" + backend_py)
add_code("%%writefile /kaggle/working/agent/agent.py\n" + agent_py)
add_code("%%writefile /kaggle/working/agent/repl.py\n" + repl_py)
add_code("%%writefile /kaggle/working/agent/client.py\n" + client_py)
add_code("%%writefile /kaggle/working/agent/repl_agent.py\n" + repl_agent_py)
add_code("%%writefile /kaggle/working/agent/run_direct.py\n" + run_direct_py)
add_code("%%writefile /kaggle/working/agent/run_submit.py\n" + run_submit_py)
add_code("%%writefile /kaggle/working/agent/submit_main.py\n" + submit_main_py)

add_code("%%writefile /kaggle/working/arc_loader.py\n" + arc_loader_py)
add_code("%%writefile /kaggle/working/arc_decoder.py\n" + arc_decoder_py)
add_code("%%writefile /kaggle/working/arc_sched.py\n" + arc_sched_py)
add_code("%%writefile /kaggle/working/arc_solver.py\n" + arc_solver_py)
add_code("%%writefile /kaggle/working/starter.py\n" + starter_py)
add_code("%%writefile /kaggle/working/final_arbitrator.py\n" + final_arbitrator_py)

orchestrator_code = r'''# Master Execution Orchestrator: Stage 0 -> Phase 1 (Qwen3.8-27B-FP8 REPL+Direct) -> Phase 2 (Qwen3-4B NVARC TTT) -> Phase 3 (Union Arbitrator)
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

START = float(Path("/kaggle/working/start_epoch.txt").read_text())
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
print(f"=== ARC-AGI-2 3-Engine Hybrid Orchestrator | RERUN={RERUN} ===")

# -------------------------------------------------------------------------
# Stage 0: Run Certified Symbolic + Astra V23 Experts (<20s CPU)
# -------------------------------------------------------------------------
rc0 = subprocess.call([sys.executable, "/kaggle/working/run_stage0.py"], cwd="/kaggle/working")
print(f"[Orchestrator] Stage 0 exit code: {rc0}")

# Phase 1 Configuration (Qwen3.8-27B-FP8 with ARC-AGI-3 v3.2 REPL upgrades + TRITON_ATTN MTP)
q38_cfg = {
    "model_glob": "/kaggle/input/**/qwen3-8-27b-fp8-repacked-v1/**/config.json",
    "overlay": "/kaggle/input/**/taaf-duck-qwen38-serving-v1/serving_wheels/*.whl",
    "parsers": ["--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3"],
    "server_ladder": [
        [
            "--attention-backend", "TRITON_ATTN",
            "--reasoning-config", "{\"reasoning_start_str\": \"<think>\", \"reasoning_end_str\": \"</think>\"}",
            "--kv-cache-dtype", "fp8",
            "--language-model-only",
            "--speculative-config", "{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": 2}"
        ],
        [
            "--attention-backend", "TRITON_ATTN",
            "--reasoning-config", "{\"reasoning_start_str\": \"<think>\", \"reasoning_end_str\": \"</think>\"}",
            "--kv-cache-dtype", "fp8",
            "--speculative-config", "{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": 2}"
        ],
        [
            "--reasoning-config", "{\"reasoning_start_str\": \"<think>\", \"reasoning_end_str\": \"</think>\"}",
            "--kv-cache-dtype", "fp8",
            "--language-model-only",
            "--speculative-config", "{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": 2}"
        ],
        [
            "--attention-backend", "TRITON_ATTN",
            "--reasoning-config", "{\"reasoning_start_str\": \"<think>\", \"reasoning_end_str\": \"</think>\"}",
            "--kv-cache-dtype", "fp8"
        ],
        [
            "--reasoning-config", "{\"reasoning_start_str\": \"<think>\", \"reasoning_end_str\": \"</think>\"}",
            "--kv-cache-dtype", "fp8"
        ],
        [
            "--reasoning-config", "{\"reasoning_start_str\": \"<think>\", \"reasoning_end_str\": \"</think>\"}"
        ],
        []
    ],
    "max_model_len": 65536,
    "max_num_seqs": 28,
    "mode": "repl",
    "attempts": 2,
    "concurrency": 28,
    "max_turns": 18,
    "turn_max_tokens": 6144,
    "max_tokens": 6144,
    "max_cap": 28000,
    "min_cap": 12000,
    "context_budget": 28000,
    "nudge_every": 5,
    "force_submit_after": 7,
    "in_python": True,
    "retain_functions": True,
    "keep_reasoning": True,
    "reasoning_key": "reasoning_content",
    "keep_recent": 6,
    "trim_drain": 6000,
    "effort_ladder": ["medium", "low"],
    "degenerate_guard": True,
    "timeout_as_cut": True,
    "snippet_timeout": 25.0,
    "reasoning_effort": "high",
    "template_kwargs": {"reasoning_effort": "high", "preserve_thinking": True},
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "tok_per_s": 280
}


def ensure_py312(target_dir="/tmp/py312") -> str:
    """Extract standalone CPython 3.12 when Kaggle container defaults to Python 3.13."""
    if sys.version_info[:2] == (3, 12):
        return sys.executable
    py312_bin = Path(target_dir) / "python/bin/python3.12"
    if not py312_bin.is_file():
        blobs = sorted(Path("/kaggle/input").rglob("cpython-3.12-linux-x86_64.blob"))
        if blobs:
            print(f"[Orchestrator] Extracting standalone Python 3.12 from {blobs[0]} -> {target_dir}...", flush=True)
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            import tarfile
            with tarfile.open(blobs[0], "r:gz") as tf:
                tf.extractall(target_dir)
            py312_bin.chmod(0o755)
    if py312_bin.is_file():
        return str(py312_bin)
    return sys.executable


PY312 = ensure_py312("/tmp/py312")
py312_ver = subprocess.check_output([PY312, "--version"], text=True).strip()
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    gpu_summary = " | ".join(smi.stdout.strip().splitlines()) if smi.returncode == 0 and smi.stdout.strip() else "NO GPU"
except Exception:
    gpu_summary = "NO GPU"
print(f"[Orchestrator] Container Python={sys.version.split()[0]} ({sys.executable}) | PY312={PY312} ({py312_ver}) | GPUs={gpu_summary}", flush=True)


def py312_env(py_bin: str) -> dict:
    env = dict(os.environ)
    py_prefix = Path(py_bin).resolve().parent.parent
    if (py_prefix / "lib").is_dir():
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{py_prefix / 'lib'}:{ld}" if ld else str(py_prefix / "lib")
    if (py_prefix / "bin").is_dir():
        env["PATH"] = f"{py_prefix / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONNOUSERSITE"] = "1"
    env["VLLM_PYTHON"] = py_bin
    return env


NVARC_PKGS = [
    "unsloth==2025.9.7", "unsloth_zoo==2025.9.9",
    "torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0", "torchao==0.14.1",
    "transformers==4.55.4", "accelerate==1.11.0", "triton==3.4.0",
    "numpy==2.2.6", "matplotlib==3.10.6", "scikit-learn==1.7.2",
    "xformers==0.0.32.post2", "peft==0.17.1", "trl==0.22.2", "bitsandbytes==0.48.2",
    "pillow==12.0.0", "datasets==3.6.0", "wandb==0.20.1",
]


def install_nvarc_py312(py_bin: str = PY312):
    """Install patched Unsloth + FlashAttention-2.8.2 (cp312) into py_bin for Phase 2 Qwen3-4B NVARC TTT."""
    t0 = time.time()
    unsloth_hits = glob.glob("/kaggle/input/**/unsloth-2025.9.7-py3-none-any.whl", recursive=True)
    flash_hits = glob.glob("/kaggle/input/**/flash_attn*cp312*linux_x86_64.whl", recursive=True)
    if not unsloth_hits or not flash_hits:
        raise FileNotFoundError(f"Missing py312 wheels: unsloth={unsloth_hits}, flash={flash_hits}")
    unsloth_dir = os.path.dirname(unsloth_hits[0])
    flash_whl = flash_hits[0]
    env_p = py312_env(py_bin)
    if Path(py_bin).resolve() == Path(sys.executable).resolve():
        subprocess.run("uv pip uninstall tensorflow tensorflow-decision-forests keras keras-tuner keras-hub",
                       shell=True, check=False)
    subprocess.run(
        [py_bin, "-m", "pip", "install", "--no-index", "--find-links", unsloth_dir,
         "--upgrade", "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts", "-q", *NVARC_PKGS],
        env=env_p,
        check=True,
    )
    subprocess.run(
        [py_bin, "-m", "pip", "install", "--no-index", "--no-deps", "--no-build-isolation",
         "--no-compile", "--disable-pip-version-check", "-q", flash_whl],
        env=env_p,
        check=True,
    )
    # Probe whether FlashAttention-2.8.2 supports the active GPU architecture (SM89 L4 = yes; SM120 G4 Blackwell = fallback to cuDNN SDPA)
    probe = subprocess.run(
        [py_bin, "-c", (
            "import torch; assert torch.cuda.is_available(); "
            "from flash_attn import flash_attn_func; "
            "q = torch.randn(1, 4, 2, 64, device='cuda', dtype=torch.bfloat16); "
            "out = flash_attn_func(q, q, q); torch.cuda.synchronize(); "
            "print('FLASH_ATTN_PROBE_OK', torch.cuda.get_device_capability(0))"
        )],
        env=env_p, capture_output=True, text=True
    )
    if probe.returncode != 0:
        print(f"[Phase 2 Setup] FlashAttn-2 kernel unsupported on this GPU ({probe.stderr.strip().splitlines()[-1] if probe.stderr else 'SM120'}); switching Unsloth to native PyTorch 2.8 cuDNN SDPA...", flush=True)
        subprocess.run([py_bin, "-m", "pip", "uninstall", "-y", "-q", "flash-attn"], env=env_p, check=False)
        for q3_path in Path(py_bin).resolve().parent.parent.rglob("unsloth/models/qwen3.py"):
            s = q3_path.read_text()
            old_fa = "    Qnn = Qn.transpose(1, 2)\n    Knn = Kn.transpose(1, 2)\n    Vnn = Vn.transpose(1, 2)\n\n    A = flash_attn_func(Qnn, Knn, Vnn)\n"
            new_sdpa = "    is_c = (attention_mask is None and Qn.shape[-2] == Kn.shape[-2] and Qn.shape[-2] > 1)\n    A = scaled_dot_product_attention(Qn, Kn, Vn, attn_mask=attention_mask, is_causal=is_c, enable_gqa=True).transpose(1, 2)\n"
            if old_fa in s:
                q3_path.write_text(s.replace(old_fa, new_sdpa, 1))
                print(f"[Phase 2 Setup] Patched {q3_path} to native PyTorch 2.8 cuDNN GQA SDPA!", flush=True)
    else:
        print(f"[Phase 2 Setup] {probe.stdout.strip()}", flush=True)
    print(f"[Phase 2 Setup] Installed Unsloth + Attention backend (cp312) into {py_bin} in {time.time() - t0:.1f}s", flush=True)


if not RERUN:
    # -------------------------------------------------------------------------
    # Fast Commit Pre-Flight Verification & Self-Test Suite (~60s total)
    # -------------------------------------------------------------------------
    print("[Commit Self-Test] Verifying all attached Kaggle model & wheelhouse mounts...")
    checks = {
        "arc_test": "/kaggle/input/**/arc-agi_test_challenges.json",
        "arc_eval": "/kaggle/input/**/arc-agi_evaluation_challenges.json",
        "qwen38_27b_fp8": "/kaggle/input/**/qwen3-8-27b-fp8-repacked-v1/**/config.json",
        "vllm_wheelhouse": "/kaggle/input/**/arc3-vllm-h100-wheelhouse-v3/**/requirements.lock",
        "qwen38_overlay": "/kaggle/input/**/taaf-duck-qwen38-serving-v1/serving_wheels/*.whl",
        "unsloth_py312": "/kaggle/input/**/unsloth-2025.9.7-py3-none-any.whl",
        "flash_attn_py312": "/kaggle/input/**/flash_attn*cp312*linux_x86_64.whl",
        "cpython_312_blob": "/kaggle/input/**/cpython-3.12-linux-x86_64.blob",
        "qwen3_4b_nvarc": "/kaggle/input/**/config.json",
    }
    for name, pat in checks.items():
        hits = glob.glob(pat, recursive=True)
        if name == "qwen3_4b_nvarc":
            hits = [h for h in hits if ("grids15" in h or "qwen3-4b" in h or "qwen3_4b" in h)]
        print(f"  [Check] {name:18s}: {len(hits)} match(es) -> {hits[0] if hits else 'MISSING!'}")
        assert hits, f"Missing required input mount for {name}: {pat}"
    assert "3.12." in py312_ver, f"Expected Python 3.12.x in PY312, got {py312_ver}"

    # Verify cp312 wheel dependency resolution for BOTH Phase 1 (vLLM 0.19) and Phase 2 (Unsloth + FlashAttn 2.8.2)
    wh_req = Path(glob.glob(checks["vllm_wheelhouse"], recursive=True)[0])
    subprocess.run(
        [PY312, "-m", "pip", "install", "--dry-run", "--no-index", "--find-links", str(wh_req.parent),
         "--requirement", str(wh_req), "--only-binary", ":all:", "--disable-pip-version-check", "-q"],
        env=py312_env(PY312),
        check=True,
    )
    print("  [Self-Test] Phase 1 vLLM 0.19 cp312 wheelhouse dependency closure verified!")

    unsloth_dir = os.path.dirname(glob.glob(checks["unsloth_py312"], recursive=True)[0])
    flash_whl = glob.glob(checks["flash_attn_py312"], recursive=True)[0]
    subprocess.run(
        [PY312, "-m", "pip", "install", "--dry-run", "--no-index", "--find-links", unsloth_dir,
         "--only-binary", ":all:", "--disable-pip-version-check", "-q", *NVARC_PKGS],
        env=py312_env(PY312),
        check=True,
    )
    subprocess.run(
        [PY312, "-m", "pip", "install", "--dry-run", "--no-index", "--no-deps",
         "--disable-pip-version-check", "-q", flash_whl],
        env=py312_env(PY312),
        check=True,
    )
    print("  [Self-Test] Phase 2 Unsloth + FlashAttn-2.8.2 cp312 wheelhouse dependency closure verified!")

    # Verify REPL sandbox + ARC-AGI-3 v3.2 AST Timeout Helper-Retention + Auto-Verified transform()
    sys.path.insert(0, "/kaggle/working/agent")
    from repl import new_store, run_snippet, with_kept
    synth_task = {
        "train": [
            {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
            {"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]},
        ],
        "test": [{"input": [[0, 1], [0, 1]]}],
    }
    st = new_store()
    timeout_code = (
        "def helper_flip(g):\n"
        "    return [[1 - int(c) for c in row] for row in g]\n\n"
        "def transform(grid):\n"
        "    return helper_flip(grid)\n\n"
        "import time\n"
        "time.sleep(5.0)\n"
    )
    out_to = run_snippet(timeout_code, synth_task, timeout=1.0, store=st)
    print("  [Self-Test] Timeout Helper-Retention output:", out_to.replace("\n", " "))
    assert "helper_flip" in st["defs"] and "transform" in st["defs"], f"Salvage failed: {st}"
    full_src, added = with_kept("pass", st)
    from repl_agent import verify
    n_ok, test_out, msg = verify(full_src, synth_task)
    print(f"  [Self-Test] Auto-Verify salvaged transform(): n_ok={n_ok}/2, test_out={test_out}, msg={msg}")
    assert n_ok == 2 and test_out == [[[1, 0], [1, 0]]]

    # Verify Universal G4 (1x96GB) + 4xL4 (4x24GB) hardware topology detectors
    import submit_main, starter
    print("  [Self-Test] Phase 1 detect_gpu_topology():", submit_main.detect_gpu_topology())
    print("  [Self-Test] Phase 2 starter.detect_n_gpus():", starter.detect_n_gpus())

    # Run Phase 3 Arbitrator & Strict Audit on submission.json
    rc_arb = subprocess.call([sys.executable, "/kaggle/working/final_arbitrator.py"], cwd="/kaggle/working")
    assert rc_arb == 0, f"final_arbitrator.py failed with {rc_arb}"
    print("[Commit Self-Test] ALL checks passed! Ready to Submit to Competition.")
else:
    # -------------------------------------------------------------------------
    # Full 12-Hour Competition Rerun Pipeline
    #   Phase 1 (0.0h -> 3.85h): Qwen3.8-27B-FP8 REPL Agent + Direct Rescue
    #   Phase 2 (3.85h -> 11.70h): Qwen3-4B NVARC TTT + 8-View Batched GPU-Logsumexp DFS
    #   Phase 3 (11.70h -> 11.80h): Cross-Paradigm Union Arbitrator & Final Audit
    # -------------------------------------------------------------------------
    phase1_deadline = START + int(3.85 * 3600)
    phase2_deadline = START + int(11.70 * 3600)

    # Phase 1: Qwen3.8-27B-FP8 (vLLM server runs under PY312; REPL sandbox runs under container Python)
    q38_cfg["deadline_epoch"] = phase1_deadline
    try:
        t_p1 = time.time()
        rc1 = subprocess.run(
            [sys.executable, "/kaggle/working/agent/submit_main.py", json.dumps(q38_cfg)],
            env=py312_env(PY312),
            timeout=max(120, phase1_deadline + 12 * 60 - time.time())
        ).returncode
        print(f"[Orchestrator] Phase 1 finished with rc={rc1} in {(time.time() - t_p1)/60:.1f} min", flush=True)
    except Exception as e:
        print(f"[Orchestrator] Phase 1 exception caught safely: {e!r}", flush=True)
        subprocess.run(["pkill", "-9", "-f", "vllm"], check=False)

    # Intermediate Arbitration Checkpoint after Phase 1
    subprocess.call([sys.executable, "/kaggle/working/final_arbitrator.py"], cwd="/kaggle/working")

    # Phase 2: Priority-Paced Qwen3-4B NVARC TTT + 8-View GPU-Logsumexp Turbo-DFS (runs under PY312)
    if time.time() < phase2_deadline - 900:
        try:
            install_nvarc_py312(PY312)
            env2 = py312_env(PY312)
            env2.update(
                UNSLOTH_DISABLE_STATISTICS="1",
                TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
                OMP_NUM_THREADS="12",
                ARC_BENCH="0",
            )
            t_p2 = time.time()
            rc2 = subprocess.call(
                [PY312, "/kaggle/working/starter.py", "--end-time", f"{phase2_deadline}"],
                env=env2,
                cwd="/kaggle/working",
            )
            print(f"[Orchestrator] Phase 2 finished with rc={rc2} in {(time.time() - t_p2)/60:.1f} min", flush=True)
        except Exception as e:
            print(f"[Orchestrator] Phase 2 exception caught safely: {e!r}", flush=True)

    # Phase 3: Final Cross-Paradigm Union Arbitration (runs in fresh subprocess)
    rc3 = subprocess.call([sys.executable, "/kaggle/working/final_arbitrator.py"], cwd="/kaggle/working")
    print(f"[Orchestrator] Phase 3 final_arbitrator exit code: {rc3}")
'''

add_code(orchestrator_code)

add_code("""# Last Line of Defense: Ensure /kaggle/working/submission.json is 100% compliant
import glob, json
try:
    ch = json.load(open(glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True)[0]))
    sub = json.load(open("/kaggle/working/submission.json"))
    def ok_grid(g):
        return (isinstance(g, list) and 1 <= len(g) <= 30
                and all(isinstance(r, list) and 1 <= len(r) <= 30 and len(r) == len(g[0]) for r in g)
                and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))
    bad = [t for t, task in ch.items()
           if t not in sub or len(sub[t]) != len(task["test"])
           or not all(ok_grid(r.get("attempt_1")) and ok_grid(r.get("attempt_2")) for r in sub[t])]
    if bad:
        print("REPAIRING", len(bad), "tasks in final defense cell")
        for t in bad:
            sub[t] = [{"attempt_1": x["input"], "attempt_2": x["input"]} for x in ch[t]["test"]]
        json.dump(sub, open("/kaggle/working/submission.json", "w"))
    changed_a1 = sum(r["attempt_1"] != x["input"] for t in ch for r, x in zip(sub[t], ch[t]["test"]))
    distinct = sum(r["attempt_1"] != r["attempt_2"] for t in ch for r in sub[t])
    print(f"[Final Check] submission.json valid: {len(sub)} tasks, {changed_a1} non-identity attempt_1, {distinct} distinct attempt pairs")
except Exception as e:
    print("final validation warning:", repr(e))""")

notebook = {
    "cells": cells,
    "metadata": {
        "kaggle": {
            "accelerator": "nvidiaL4x4",
            "isGpuEnabled": True,
            "isInternetEnabled": False,
            "language": "python",
            "sourceType": "notebook"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_path = OUT_DIR / "arc2-2026-hybrid-v1.ipynb"
with open(nb_path, "w") as f:
    json.dump(notebook, f, indent=2)

kernel_meta = {
    "id": "johnnyhyland/arc2-2026-hybrid-v1",
    "title": "ARC2 2026 Hybrid v1",
    "code_file": "arc2-2026-hybrid-v1.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3",
        "keithtyser/taaf-duck-qwen38-serving-v1",
        "konstantinboyko/flash-attn-2-8-2",
        "johnnyhyland/arc3-qwen38-27b-grpo-lora"
    ],
    "kernel_sources": [
        "konstantinboyko/unsloth-2025-9-7-torch-2-8-0-py3-12-patched"
    ],
    "competition_sources": [
        "arc-prize-2026-arc-agi-2"
    ],
    "model_sources": [
        "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1",
        "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
        "konstantinboyko/qwen3-4b-bfloat16-v02-01-01/Transformers/default/1"
    ]
}

meta_path = OUT_DIR / "kernel-metadata.json"
with open(meta_path, "w") as f:
    json.dump(kernel_meta, f, indent=2)

print(f"Successfully generated {nb_path} ({len(cells)} cells) and {meta_path}!")
