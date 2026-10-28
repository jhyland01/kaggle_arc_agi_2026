"""OpenAI-compatible chat proxy that sits between the Duck harness and vLLM.

Why a proxy instead of pointing the harness straight at vLLM's ``/v1/chat/completions``:

* TRL AsyncGRPO (incl. #7760 score centering) needs, for every sampled token, the exact engine
  token ids, the sampler logprob, the top-k/top-p candidate set and the sampler's top-8 logprobs.
  A chat endpoint re-tokenizes and drops most of that. The proxy renders the harness's messages
  with the *training* chat template itself and samples token-in/token-out, so what the trainer
  later scores is byte-identical to what was sampled.
* The same proxy is used for baseline/eval runs, so baseline and fine-tuned numbers are measured
  through identical prompt rendering and sampling.

One proxy serves many concurrent harness subprocesses; each rollout gets its own route
``/r/<rollout_id>/v1/chat/completions`` so its turns can be attributed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

logger = logging.getLogger("arc_rl.proxy")


@dataclass
class TurnOutput:
    ids: list[int]
    logprobs: list[float]
    sampling_mask: Any = None  # list[list[int]] | None (top-k/top-p candidate set per token)
    sampler_logprobs: Any = None  # list[dict[int, float]] | None (top-N sampler logprobs per token)


# generate_fn(prompt_ids, model_name, max_tokens) -> TurnOutput
GenerateFn = Callable[[list[int], str, int], Awaitable[TurnOutput]]


@dataclass
class RolloutCtx:
    rollout_id: str
    model: str
    record: bool = True
    turns: list = field(default_factory=list)  # list[TurnRecord] (training) or list[dict] (eval stats)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_requests: int = 0
    n_parse_fail: int = 0
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    n_truncated: int = 0
    _seen_tool_ids: set = field(default_factory=set)


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
            # image parts are dropped: the harness runs text-only (AsyncGRPO cannot train on images yet)
        return "\n".join(parts)
    return str(content)


def normalize_messages(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages (as the harness sends them) -> what HF chat templates expect."""
    out = []
    for m in messages:
        role = m.get("role")
        nm: dict[str, Any] = {"role": role, "content": _text_of(m.get("content"))}
        if role == "assistant":
            rc = m.get("reasoning_content") or m.get("reasoning")
            if rc:
                nm["reasoning_content"] = rc
            tcs = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"code": args}
                tcs.append({"type": "function", "id": tc.get("id"), "function": {"name": fn.get("name"), "arguments": args}})
            if tcs:
                nm["tool_calls"] = tcs
        elif role == "tool":
            if m.get("tool_call_id"):
                nm["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                nm["name"] = m["name"]
        out.append(nm)
    return out


def _tool_result_failed(text: str) -> bool:
    try:
        d = json.loads(text)
        if isinstance(d, dict) and "returncode" in d:
            return int(d.get("returncode") or 0) != 0
    except Exception:
        pass
    t = text.lower()
    return "traceback (most recent call last)" in t


class HarnessProxy:
    def __init__(
        self,
        *,
        tokenizer,
        chat_template: str | None,
        chat_template_kwargs: dict | None,
        generate_fn: GenerateFn,
        max_tokens_cap: int,
        max_model_len: int,
        turn_record_factory: Callable[..., Any] | None = None,
        parse_fn: Callable[..., dict] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.generate_fn = generate_fn
        self.max_tokens_cap = max_tokens_cap
        self.max_model_len = max_model_len
        self.turn_record_factory = turn_record_factory
        if parse_fn is None:
            from trl.chat_template_utils import parse_response as parse_fn  # noqa: PLC0415
        self.parse_fn = parse_fn
        self.routes: dict[str, RolloutCtx] = {}
        self._runner: web.AppRunner | None = None
        self.base_url: str | None = None

    # ---------------------------------------------------------------- lifecycle
    async def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_post("/r/{rid}/v1/chat/completions", self._handle)
        app.router.add_get("/r/{rid}/v1/models", self._models)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()
        real_port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
        self.base_url = f"http://{host}:{real_port}"
        logger.info("harness proxy listening on %s", self.base_url)
        return self.base_url

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    def open(self, model: str, record: bool = True) -> RolloutCtx:
        rid = uuid.uuid4().hex
        ctx = RolloutCtx(rollout_id=rid, model=model, record=record)
        self.routes[rid] = ctx
        return ctx

    def close(self, rid: str) -> RolloutCtx | None:
        return self.routes.pop(rid, None)

    def url_for(self, ctx: RolloutCtx) -> str:
        return f"{self.base_url}/r/{ctx.rollout_id}/v1"

    # ---------------------------------------------------------------- handlers
    async def _models(self, request: web.Request) -> web.Response:
        rid = request.match_info["rid"]
        ctx = self.routes.get(rid)
        name = ctx.model if ctx else "unknown"
        return web.json_response({"object": "list", "data": [{"id": name, "object": "model"}]})

    def _render(self, messages: list[dict], tools: list | None, kwargs: dict) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            chat_template=self.chat_template,
            **kwargs,
        )

    async def _handle(self, request: web.Request) -> web.Response:
        rid = request.match_info["rid"]
        ctx = self.routes.get(rid)
        if ctx is None:
            return web.json_response({"error": {"message": f"unknown rollout {rid}"}}, status=404)
        body = await request.json()
        dump_dir = os.environ.get("ARC_RL_DUMP_REQUESTS")
        if dump_dir:
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
            (Path(dump_dir) / f"{rid}_{ctx.n_requests:04d}.json").write_text(json.dumps(body))
        raw_messages = body.get("messages") or []
        tools = body.get("tools") or None
        messages = normalize_messages(raw_messages)

        # Count tool-call failures the model caused (new tool results only).
        for m in raw_messages:
            if m.get("role") == "tool":
                tid = m.get("tool_call_id") or str(hash(_text_of(m.get("content"))[:200]))
                if tid not in ctx._seen_tool_ids:  # noqa: SLF001
                    ctx._seen_tool_ids.add(tid)  # noqa: SLF001
                    ctx.n_tool_calls += 1
                    if _tool_result_failed(_text_of(m.get("content"))):
                        ctx.n_tool_errors += 1

        kwargs = dict(self.chat_template_kwargs)
        for k, v in (body.get("chat_template_kwargs") or {}).items():
            kwargs.setdefault(k, v)
        loop = asyncio.get_running_loop()
        try:
            prompt_ids = await loop.run_in_executor(None, self._render, messages, tools, kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat template render failed")
            return web.json_response({"error": {"message": f"template error: {exc}"}}, status=400)

        req_max = int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_tokens_cap)
        budget = min(req_max, self.max_tokens_cap, self.max_model_len - len(prompt_ids) - 8)
        if budget < 64:
            return web.json_response(
                {
                    "error": {
                        "message": (
                            f"This model's maximum context length is {self.max_model_len} tokens. However, you "
                            f"requested {len(prompt_ids) + req_max} tokens ({len(prompt_ids)} in the messages)."
                        ),
                        "type": "BadRequestError",
                        "code": "context_length_exceeded",
                    }
                },
                status=400,
            )

        t0 = time.monotonic()
        out = await self.generate_fn(prompt_ids, ctx.model, budget)
        dt = time.monotonic() - t0
        ctx.n_requests += 1
        ctx.prompt_tokens += len(prompt_ids)
        ctx.completion_tokens += len(out.ids)
        truncated = len(out.ids) >= budget
        ctx.n_truncated += int(truncated)

        if ctx.record and self.turn_record_factory is not None:
            ctx.turns.append(
                self.turn_record_factory(
                    prompt_ids,
                    out.ids,
                    out.logprobs,
                    output_sampling_mask=out.sampling_mask,
                    output_sampler_logprobs=out.sampler_logprobs,
                )
            )
        elif ctx.record:
            ctx.turns.append({"prompt_len": len(prompt_ids), "completion_len": len(out.ids), "latency_s": dt})

        try:
            parsed = await loop.run_in_executor(
                None, lambda: self.parse_fn(self.tokenizer, out.ids, prefix=prompt_ids, tools=tools)
            )
        except Exception:  # noqa: BLE001
            parsed = {"role": "assistant", "content": self.tokenizer.decode(out.ids, skip_special_tokens=True)}
            ctx.n_parse_fail += 1

        tool_calls = []
        for tc in parsed.get("tool_calls") or []:
            fn = tc.get("function", tc)
            args = fn.get("arguments", {})
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": fn.get("name"),
                        "arguments": args if isinstance(args, str) else json.dumps(args),
                    },
                }
            )
        msg: dict[str, Any] = {"role": "assistant", "content": parsed.get("content") or ""}
        rc = parsed.get("reasoning_content")
        if rc:
            msg["reasoning_content"] = rc
            msg["reasoning"] = rc
        if tool_calls:
            msg["tool_calls"] = tool_calls
        finish = "length" if truncated else ("tool_calls" if tool_calls else "stop")
        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": ctx.model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(out.ids),
                "total_tokens": len(prompt_ids) + len(out.ids),
            },
        }
        return web.json_response(resp)
