"""Compare a post-training eval against the baseline eval (same games, passes, budgets)."""
import json
import statistics as st
import sys


def load(path):
    return [json.loads(line) for line in open(path)]


def summarize(rows):
    ok = [r for r in rows if "final_score" in r]
    tok = sum(r["proxy"]["completion_tokens"] for r in ok)
    act = sum(r["actions"] for r in ok)
    calls = sum(r["proxy"]["n_tool_calls"] for r in ok)
    errs = sum(r["proxy"]["n_tool_errors"] for r in ok)
    return {
        "runs": len(ok),
        "mean_score": round(st.mean(r["final_score"] for r in ok), 3),
        "levels_total": sum(r["levels_completed"] for r in ok),
        "runs_with_level": sum(r["levels_completed"] > 0 for r in ok),
        "mean_actions": round(st.mean(r["actions"] for r in ok), 1),
        "mean_distinct_states": round(st.mean(r.get("distinct_states", 0) or 0 for r in ok), 1),
        "gen_tok_per_run": round(tok / len(ok)),
        "gen_tok_per_action": round(tok / max(1, act)),
        "tool_err_rate": round(errs / max(1, calls), 3),
    }


def per_game(rows):
    out = {}
    for r in rows:
        if "final_score" in r:
            out.setdefault(r["game"], []).append(r)
    return out


base = load(sys.argv[1])
post = load(sys.argv[2])
sb, sp = summarize(base), summarize(post)
print(f"{'metric':24s} {'baseline':>10s} {'trained':>10s}")
for k in sb:
    print(f"{k:24s} {sb[k]!s:>10s} {sp[k]!s:>10s}")
gb, gp = per_game(base), per_game(post)
print("\nper game: levels / mean actions / mean distinct states (baseline -> trained)")
for g in sorted(set(gb) | set(gp)):
    b, p = gb.get(g, []), gp.get(g, [])
    def f(rs):
        if not rs:
            return "-"
        return (f"{sum(r['levels_completed'] for r in rs)}L "
                f"{st.mean(r['actions'] for r in rs):.0f}a "
                f"{st.mean(r.get('distinct_states', 0) or 0 for r in rs):.0f}s")
    print(f"  {g}: {f(b):>16s} -> {f(p):>16s}")
