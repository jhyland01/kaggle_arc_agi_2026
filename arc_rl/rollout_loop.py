"""TRL AsyncGRPO rollout loop that plays ARC-AGI-3 games through the real Duck harness.

Built on TRL main @ e29597a, which contains huggingface/trl#7760 (AsyncGRPO score centering +
masked importance sampling). Each rollout:

1. opens a route on an in-process ``HarnessProxy`` pinned to the current policy adapter,
2. launches the patched Duck harness on one game in a subprocess pointed at that route,
3. every harness LLM call is sampled token-in/token-out via TRL's own ``_generate_one_turn``
   (vLLM ``/inference/v1/generate``), which returns the sampled ids, behaviour logprobs, the
   top-k/top-p candidate set and the sampler's top-``num_sampler_logprobs`` logprobs that
   #7760's score-centred objective needs; each is stored as a TRL ``TurnRecord``,
4. when the harness finishes, the game result becomes the rollout reward, and TRL's
   ``_chain_to_sequences`` reconciles the turns (history trims fork new rows) into training rows.

A group is ``num_generations`` rollouts of the same game; the advantage is group-relative.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
import uuid
from pathlib import Path
from typing import Any

from trl.chat_template_utils import get_training_chat_template, is_chat_template_prefix_preserving
from trl.experimental.async_grpo.async_rollout_worker import (
    AsyncRolloutWorker,
    TurnRecord,
    _AsyncRolloutLoop,
    _chain_to_sequences,
)

from arc_rl.harness_env import HarnessConfig, run_game_subprocess
from arc_rl.harness_proxy import HarnessProxy, TurnOutput


def _tok_id(entry: dict) -> int:
    """Token id of one top-logprob entry. vLLM 0.32's `/inference/v1/generate` returns `{"token_id": 8, ...}`;
    TRL @ e29597a (and older vLLM) expects `{"token": "token_id:8", ...}`. Accept both."""
    if entry.get("token_id") is not None:
        return int(entry["token_id"])
    return int(str(entry["token"]).removeprefix("token_id:"))


def game_reward(result: dict, ctx_stats: dict, weights: dict) -> float | None:
    """Scalar reward for one harness rollout.

    Primary signal is the competition's own metric (levels completed, action efficiency). For a
    9B model levels are rare within a short rollout, so small dense terms help GRPO get a
    non-zero group variance: tool-call hygiene and token efficiency (inference time is the
    binding constraint on Kaggle).
    """
    if "final_score" not in result:
        return None  # harness crashed: unscorable, excluded from the group baseline
    levels = float(result.get("levels_completed", 0))
    score = float(result.get("final_score", 0.0))
    actions = max(1, int(result.get("actions", 0)))
    n_calls = max(1, ctx_stats.get("n_tool_calls", 0))
    tool_err = ctx_stats.get("n_tool_errors", 0) / n_calls
    gen_tok = ctx_stats.get("completion_tokens", 0)
    tok_per_action = gen_tok / actions
    # Exploration: distinct boards reached, and the share of actions that changed the board (vs no-ops).
    novelty = min(float(result.get("distinct_states", 0)) / 40.0, 1.0)
    effective = float(result.get("effective_actions", 0)) / actions
    game_overs = float(result.get("game_overs", 0))
    r = (
        weights.get("level", 1.0) * levels
        + weights.get("score", 0.01) * score
        + weights.get("novelty", 0.4) * novelty
        + weights.get("effective", 0.2) * effective
        - weights.get("game_over", 0.02) * min(game_overs, 10.0)
        - weights.get("tool_err", 0.3) * tool_err
        - weights.get("tok_per_action", 0.05) * min(tok_per_action / 1000.0, 5.0)
        - weights.get("truncated", 0.05) * ctx_stats.get("n_truncated", 0)
    )
    return float(r)


class ArcHarnessLoop(_AsyncRolloutLoop):
    _provides_rollout_reward = True

    def __init__(self, *, harness_cfg: dict, run_dir: str, reward_weights: dict | None = None,
                 max_model_len: int = 40960, **loop_kwargs: Any):
        super().__init__(**loop_kwargs)
        self.reward_func_names.append("game_reward")
        self.harness_cfg = HarnessConfig(**harness_cfg)
        self.run_dir = Path(run_dir)
        self.reward_weights = reward_weights or {}
        self.max_model_len = max_model_len
        # The harness owns the tools, so TRL never sees them; force the prefix-preserving training template.
        if self.chat_template is None and not is_chat_template_prefix_preserving(self.tokenizer):
            self.chat_template = get_training_chat_template(self.tokenizer)
        self._proxy: HarnessProxy | None = None
        self._results_path = self.run_dir / "rollouts.jsonl"

    async def _gen_for_proxy(self, prompt_ids: list[int], model: str, max_tokens: int) -> TurnOutput:
        ids, lps, mask, slp = await self._generate_one_turn_for(prompt_ids, model, max_tokens)
        return TurnOutput(ids=ids, logprobs=lps, sampling_mask=mask, sampler_logprobs=slp)

    async def _generate_one_turn_for(self, prompt_ids, model, max_tokens):
        """`_generate_one_turn` with the model (adapter version) and token budget chosen per call.

        The rollout pins the adapter version it started under, so a long game is sampled by one policy;
        falls back to the current version if that adapter was already unloaded.
        """
        sampling_params = {
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "n": 1,
            "logprobs": self.num_sampler_logprobs,
            "seed": random.getrandbits(63),
        }
        if self.min_p is not None:
            sampling_params["min_p"] = self.min_p
        for use_model in (model, self._request_model):
            try:
                if not self.sampling_replay:
                    payload = {"model": use_model, "prompt": prompt_ids, "return_token_ids": True,
                               "return_tokens_as_token_ids": True, **sampling_params}
                    output = await self._retry(lambda: self._post("/v1/completions", payload, self.request_timeout),
                                               max_attempts=30, label="vllm /v1/completions")
                    choice = output["choices"][0]
                    slp = None
                    if self.num_sampler_logprobs:
                        slp = [{int(k.removeprefix("token_id:")) if isinstance(k, str) else int(k): v for k, v in t.items()}
                               for t in choice["logprobs"]["top_logprobs"]]
                    return choice["token_ids"], choice["logprobs"]["token_logprobs"], None, slp
                payload = {"model": use_model, "token_ids": prompt_ids, "sampling_params": sampling_params}
                output = await self._retry(lambda: self._post("/inference/v1/generate", payload, self.request_timeout),
                                           max_attempts=30, label="vllm /inference/v1/generate")
                choice = output["choices"][0]
                content = choice["logprobs"]["content"]
                lps = [t["logprob"] for t in content]
                slp = None
                if self.num_sampler_logprobs:
                    slp = [{_tok_id(tt): tt["logprob"] for tt in t["top_logprobs"]} for t in content]
                return choice["token_ids"], lps, choice["sampling_mask"], slp
            except Exception:
                if use_model == self._request_model:
                    raise
        raise RuntimeError("unreachable")

    async def _run_loops(self, stop_event):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._proxy = HarnessProxy(
            tokenizer=self.tokenizer,
            chat_template=self.chat_template,
            chat_template_kwargs=self.chat_template_kwargs,
            generate_fn=self._gen_for_proxy,
            max_tokens_cap=self.max_tokens,
            max_model_len=self.max_model_len,
            turn_record_factory=TurnRecord,
        )
        await self._proxy.start()
        try:
            await super()._run_loops(stop_event)
        finally:
            await self._proxy.stop()

    async def _generate_one(self, prompt, tool_dict, tools, group_id=0):
        t_dispatch = time.monotonic()
        game = prompt[-1]["content"].strip()
        ctx = self._proxy.open(self._request_model, record=True)
        job_dir = self.run_dir / "games" / f"g{group_id:05d}_{game}_{ctx.rollout_id[:8]}"
        result = await run_game_subprocess(self.harness_cfg, game, self._proxy.url_for(ctx), ctx.model, job_dir)
        self._proxy.close(ctx.rollout_id)
        stats = {
            "n_requests": ctx.n_requests, "prompt_tokens": ctx.prompt_tokens, "completion_tokens": ctx.completion_tokens,
            "n_tool_calls": ctx.n_tool_calls, "n_tool_errors": ctx.n_tool_errors, "n_parse_fail": ctx.n_parse_fail,
            "n_truncated": ctx.n_truncated,
        }
        reward = game_reward(result, stats, self.reward_weights)
        turns: list[TurnRecord] = ctx.turns
        rec = {"t": time.time(), "group": group_id, "game": game, "model": ctx.model, "reward": reward,
               "result": {k: result.get(k) for k in ("final_score", "levels_completed", "actions", "distinct_states", "effective_actions", "game_overs", "error", "wall_s")},
               **stats}
        with self._results_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if not turns:
            return [], [], [], 0, 0, None
        sequences, tally = _chain_to_sequences(turns, ctx.rollout_id, self._fork_threshold_tokens)
        completion_ids = [t for turn in turns for t in turn.output_ids]
        self._push_rollout_metrics(turns=len(turns), sequences=len(sequences), completion_ids=completion_ids,
                                   tally=tally, loop_exhausted=False, duration_s=time.monotonic() - t_dispatch)
        if reward is not None:
            self._rates["game/levels"][0] += float(result.get("levels_completed", 0))
            self._rates["game/levels"][1] += 1
            self._rates["game/score"][0] += float(result.get("final_score", 0.0))
            self._rates["game/score"][1] += 1
            self._rates["game/actions"][0] += float(result.get("actions", 0))
            self._rates["game/actions"][1] += 1
        completion = [{"role": "assistant", "content": f"[{game}] levels={result.get('levels_completed')} "
                                                       f"score={result.get('final_score')}"}]
        return completion, completion_ids, sequences, ctx.n_tool_calls, ctx.n_tool_errors, reward


class ArcHarnessRolloutWorker(AsyncRolloutWorker):
    _loop_cls = ArcHarnessLoop
