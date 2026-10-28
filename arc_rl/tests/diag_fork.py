"""Show where turn N+1's prompt diverges from turn N's prompt+output (fork diagnosis)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trl.experimental.async_grpo.async_rollout_worker import TurnRecord  # noqa: E402

from arc_rl.eval_games import load_tokenizer_and_template, make_mock_generate  # noqa: E402
from arc_rl.harness_env import HarnessConfig, run_game_subprocess  # noqa: E402
from arc_rl.harness_proxy import HarnessProxy  # noqa: E402


async def main():
    tok, template = load_tokenizer_and_template("Qwen/Qwen3.5-9B")
    cfg = HarnessConfig(bundle_dir="work_bundle", env_dir="data/environment_files", max_runtime_s=60, max_actions=3)
    proxy = HarnessProxy(tokenizer=tok, chat_template=template, chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
                         generate_fn=make_mock_generate(tok), max_tokens_cap=6144, max_model_len=40960,
                         turn_record_factory=TurnRecord)
    await proxy.start()
    ctx = proxy.open("mock")
    await run_game_subprocess(cfg, "sb26", proxy.url_for(ctx), "mock", Path("runs/mock_chain/diag"))
    t0, t1 = ctx.turns[0], ctx.turns[1]
    full = t0.prompt_ids + t0.output_ids
    n = 0
    while n < min(len(full), len(t1.prompt_ids)) and full[n] == t1.prompt_ids[n]:
        n += 1
    print("prompt0", len(t0.prompt_ids), "out0", len(t0.output_ids), "common prefix", n)
    print("=== sampled (from divergence) ===")
    print(repr(tok.decode(full[n - 20:n + 200])))
    print("=== re-rendered (from divergence) ===")
    print(repr(tok.decode(t1.prompt_ids[n - 20:n + 300])))
    await proxy.stop()


asyncio.run(main())
