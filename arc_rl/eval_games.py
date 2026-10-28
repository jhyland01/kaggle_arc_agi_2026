"""Evaluate a model inside the real Duck harness on offline ARC-AGI-3 games.

Each game runs in its own harness subprocess; all LLM calls go through ``HarnessProxy`` which
renders prompts with the same (TRL training) chat template used during RL and samples
token-in/token-out from vLLM. Works for the base model or any LoRA adapter loaded in vLLM
(pass the adapter name as --model).

    python -m arc_rl.eval_games --games ft09,sb26 --passes 2 --model Qwen/Qwen3.5-9B \
        --vllm http://127.0.0.1:8000 --tokenizer Qwen/Qwen3.5-9B --out runs/baseline
    python -m arc_rl.eval_games --mock ...   # no GPU: random-action fake LLM (plumbing test)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp

from arc_rl.harness_env import HarnessConfig, run_game_subprocess
from arc_rl.harness_proxy import HarnessProxy, TurnOutput

TRAIN_GAMES = ["bp35", "cd82", "cn04", "dc22", "g50t", "ka59", "lf52", "ls20", "m0r0", "s5i5", "sk48", "sp80", "su15", "tn36", "wa30"]
HOLDOUT_GAMES = ["ar25", "ft09", "lp85", "r11l", "re86", "sb26", "sc25", "tr87", "tu93", "vc33"]

MOCK_CODE = (
    "import random\n"
    "acts = [a for a in valid_actions if str(a).upper() not in ('MOUSE','UNDO','ACTION7')]\n"
    "if acts:\n"
    "    action([random.choice(acts)])\n"
    "else:\n"
    "    action([{'action':'MOUSE','row':random.randrange(64),'col':random.randrange(64)}])\n"
    "print(current_frame.step, current_frame.level)\n"
)


def load_tokenizer_and_template(name: str):
    from transformers import AutoTokenizer  # noqa: PLC0415

    from trl.chat_template_utils import (  # noqa: PLC0415
        add_response_schema,
        get_training_chat_template,
        is_chat_template_prefix_preserving,
    )

    tok = AutoTokenizer.from_pretrained(name)
    if getattr(tok, "response_template", None) is None and getattr(tok, "response_schema", None) is None:
        tok = add_response_schema(tok)
    template = None if is_chat_template_prefix_preserving(tok) else get_training_chat_template(tok)
    return tok, template


def make_vllm_generate(session: aiohttp.ClientSession, vllm_url: str, cfg: HarnessConfig):
    async def gen(prompt_ids: list[int], model: str, max_tokens: int) -> TurnOutput:
        payload = {
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "logprobs": 0,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "seed": random.getrandbits(63),
        }
        for attempt in range(20):
            try:
                async with session.post(f"{vllm_url}/v1/completions", json=payload, timeout=aiohttp.ClientTimeout(total=1800)) as r:
                    r.raise_for_status()
                    out = await r.json()
                ch = out["choices"][0]
                return TurnOutput(ids=ch["token_ids"], logprobs=(ch.get("logprobs") or {}).get("token_logprobs") or [])
            except Exception:  # noqa: BLE001
                if attempt == 19:
                    raise
                await asyncio.sleep(min(2**attempt, 15))
        raise RuntimeError("unreachable")

    return gen


def make_mock_generate(tok):
    async def gen(prompt_ids: list[int], model: str, max_tokens: int) -> TurnOutput:
        text = (
            "I will try a random valid action to explore.\n</think>\n\n"
            "<tool_call>\n<function=python>\n<parameter=code>\n" + MOCK_CODE + "</parameter>\n</function>\n</tool_call>"
        )
        ids = tok(text, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
        await asyncio.sleep(0.05)
        return TurnOutput(ids=ids, logprobs=[0.0] * len(ids))

    return gen


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "final_score" in r]
    by_game: dict[str, list[dict]] = {}
    for r in ok:
        by_game.setdefault(r["game"], []).append(r)
    per_game = {
        g: {
            "score": statistics.mean(x["final_score"] for x in rs),
            "levels": statistics.mean(x["levels_completed"] for x in rs),
            "actions": statistics.mean(x["actions"] for x in rs),
            "gen_tokens": statistics.mean(x.get("proxy", {}).get("completion_tokens", 0) for x in rs),
            "n": len(rs),
        }
        for g, rs in sorted(by_game.items())
    }
    tot_tok = sum(r.get("proxy", {}).get("completion_tokens", 0) for r in ok)
    tot_act = sum(r["actions"] for r in ok)
    return {
        "n_runs": len(results),
        "n_ok": len(ok),
        "mean_score": statistics.mean(r["final_score"] for r in ok) if ok else 0.0,
        "mean_levels": statistics.mean(r["levels_completed"] for r in ok) if ok else 0.0,
        "frac_any_level": (sum(r["levels_completed"] > 0 for r in ok) / len(ok)) if ok else 0.0,
        "gen_tokens_per_action": tot_tok / max(1, tot_act),
        "tool_error_rate": sum(r.get("proxy", {}).get("n_tool_errors", 0) for r in ok)
        / max(1, sum(r.get("proxy", {}).get("n_tool_calls", 0) for r in ok)),
        "per_game": per_game,
    }


async def amain(args) -> None:
    cfg = HarnessConfig(
        bundle_dir=args.bundle,
        env_dir=args.env_dir,
        context_window=args.context_window,
        max_output=args.max_output,
        max_runtime_s=args.max_runtime_s,
        max_actions=args.max_actions,
        max_gen_tokens=args.max_gen_tokens,
    )
    tok, template = load_tokenizer_and_template(args.tokenizer)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    games = HOLDOUT_GAMES if args.games == "holdout" else TRAIN_GAMES if args.games == "train" else args.games.split(",")

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=256)) as session:
        gen = make_mock_generate(tok) if args.mock else make_vllm_generate(session, args.vllm, cfg)
        proxy = HarnessProxy(
            tokenizer=tok,
            chat_template=template,
            chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
            generate_fn=gen,
            max_tokens_cap=args.max_output,
            max_model_len=args.max_model_len,
        )
        await proxy.start()
        sem = asyncio.Semaphore(args.concurrency)
        results: list[dict] = []

        async def one(game: str, p: int) -> None:
            async with sem:
                ctx = proxy.open(args.model, record=True)
                t0 = time.monotonic()
                res = await run_game_subprocess(cfg, game, proxy.url_for(ctx), args.model, out_dir / f"{game}_p{p}")
                proxy.close(ctx.rollout_id)
                res["pass"] = p
                res["proxy"] = {
                    "requests": ctx.n_requests,
                    "prompt_tokens": ctx.prompt_tokens,
                    "completion_tokens": ctx.completion_tokens,
                    "n_tool_calls": ctx.n_tool_calls,
                    "n_tool_errors": ctx.n_tool_errors,
                    "n_parse_fail": ctx.n_parse_fail,
                    "n_truncated": ctx.n_truncated,
                    "wall_s": time.monotonic() - t0,
                }
                results.append(res)
                print(
                    f"[{len(results)}] {game} p{p}: score={res.get('final_score')} levels={res.get('levels_completed')}"
                    f" actions={res.get('actions')} gen_tok={ctx.completion_tokens} req={ctx.n_requests}"
                    f" err={res.get('error')}",
                    flush=True,
                )
                (out_dir / "results.jsonl").open("a").write(json.dumps(res) + "\n")

        await asyncio.gather(*(one(g, p) for p in range(args.passes) for g in games))
        await proxy.stop()
    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_game"}, indent=2))
    for g, s in summary["per_game"].items():
        print(f"  {g}: {s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="holdout")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--vllm", default="http://127.0.0.1:8000")
    ap.add_argument("--bundle", default="work_bundle")
    ap.add_argument("--env-dir", default="data/environment_files")
    ap.add_argument("--context-window", type=int, default=32 * 1024)
    ap.add_argument("--max-output", type=int, default=6 * 1024)
    ap.add_argument("--max-model-len", type=int, default=40 * 1024)
    ap.add_argument("--max-runtime-s", type=float, default=900.0)
    ap.add_argument("--max-actions", type=int, default=0)
    ap.add_argument("--max-gen-tokens", type=int, default=0)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", required=True)
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
