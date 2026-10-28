"""Local check: run one mock game, record TurnRecords, and inspect TRL's sequence reconciliation."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trl.experimental.async_grpo.async_rollout_worker import TurnRecord, _chain_to_sequences  # noqa: E402

from arc_rl.eval_games import load_tokenizer_and_template, make_mock_generate  # noqa: E402
from arc_rl.harness_env import HarnessConfig, run_game_subprocess  # noqa: E402
from arc_rl.harness_proxy import HarnessProxy  # noqa: E402


async def main():
    tok, template = load_tokenizer_and_template("Qwen/Qwen3.5-9B")
    print("training template swapped in:", template is not None)
    cfg = HarnessConfig(bundle_dir="work_bundle", env_dir="data/environment_files", max_runtime_s=90, max_actions=25)
    proxy = HarnessProxy(tokenizer=tok, chat_template=template, chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
                         generate_fn=make_mock_generate(tok), max_tokens_cap=6144, max_model_len=40960,
                         turn_record_factory=TurnRecord)
    await proxy.start()
    ctx = proxy.open("mock")
    res = await run_game_subprocess(cfg, "sb26", proxy.url_for(ctx), "mock", Path("runs/mock_chain/sb26"))
    print(res)
    turns = ctx.turns
    print("turns", len(turns), "prompt lens", [len(t.prompt_ids) for t in turns][:30])
    seqs, tally = _chain_to_sequences(turns, ctx.rollout_id, 1024)
    print("sequences", len(seqs), "tally", tally)
    for s in seqs:
        print(" seq len", len(s.input_ids), "trained tokens", sum(s.completion_mask))
    print("---- first prompt (tail) ----")
    print(tok.decode(turns[0].prompt_ids)[-1500:])
    print("---- second prompt (tail) ----")
    print(tok.decode(turns[1].prompt_ids)[-1800:])
    await proxy.stop()


asyncio.run(main())
