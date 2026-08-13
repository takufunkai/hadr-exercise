"""The agentic tool loop: the claw's Loop, model- and task-agnostic.

HOLE (stage 1): `agent_loop` is the one loop every agent in this kit shares -
the dispatcher (dispatch.py) is a configuration of it, not a different animal.
The contract is checked by the LOOP-* scenarios in features/agent_loop.feature
against a scripted model - deterministic, no tokens.

Once built, run it standalone with toy tools and no engine:
`uv run hadr-agent "roll two dice and tell me the time"`.

GOTCHA: the dispatch model puts its reasoning INLINE in message.content as
<think>...</think>. Feed the assistant turn back verbatim so the provider's
implicit prompt cache can hit; strip only for display (see strip_think).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from openai import OpenAI
from openai.types.chat import ChatCompletionToolUnionParam

from .util import TokenUse

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


@dataclass
class LoopResult:
    usage: TokenUse = field(default_factory=TokenUse)
    # Debug hooks: rounds shows when the loop hits max_rounds; final is the
    # model's closing free-text answer (<think> stripped).
    rounds: int = 0
    final: str = ""


async def agent_loop(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    tool_defs: list[ChatCompletionToolUnionParam],
    run_tool: Callable[[str, dict], Awaitable[dict]],
    max_rounds: int,
    max_tokens: int,
) -> LoopResult:
    """HOLE (stage 1): the agentic tool loop. Each round is one call to:

    client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=...,
        tools=tool_defs, temperature=0)

    The contract:

    - messages start as:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    - Append the assistant turn to messages VERBATIM - content including <think>,
      tool_calls unchanged - or the provider's implicit prompt cache never hits.
    - Execute each tool call via `run_tool(name, args)` (json.loads the
      arguments; a parse failure is an empty dict) and append exactly one
      role:"tool" message per call, content json.dumps of the result, matched
      by tool_call_id.
    - Accumulate result.usage.add(resp.usage) and result.rounds every round.
    - No tool_calls means the model yields: set result.final =
      strip_think(msg.content) and stop.
    - At most max_rounds rounds.
    - We set temperature=0 here for reproducibility. In practice, you would check
      the model card of the model you are using for the initial value, then tune it
      on your task.
    """
    result = LoopResult()
    messages: list = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        temperature=0,
    )
    result.usage.add(resp.usage)
    result.rounds += 1

    msg = resp.choices[0].message
    result.final = msg.content or ""

    return result


# ---- hadr-agent toy tools ---------------------------------------------------
# Try: "roll two dice", "what is 2**31 - 1?", "what's the secret message?",
# "what files are in this project and what is it for?"

_CALC_OPS: dict = {}  # filled lazily in main() to keep import cost off the hot path


def _caesar(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if ch.isalpha():
            base = ord("a") if ch.islower() else ord("A")
            out.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            out.append(ch)
    return "".join(out)


def _calc(node) -> float:
    import ast

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_calc(node.left), _calc(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_calc(node.operand))
    raise ValueError("numbers and + - * / // % ** only")


def main() -> None:
    """`hadr-agent`: the loop standalone, with toy tools. Proves the loop is
    general before it ever dispatches a truck."""
    import argparse
    import ast
    import asyncio
    import operator
    import random
    from datetime import datetime
    from pathlib import Path

    from dotenv import load_dotenv

    from .util import NUM, STR, tool
    from .util import client as llm_client

    _CALC_OPS.update(
        {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
    )

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("task", help="what to ask the agent")
    parser.add_argument("--model", default="minimax-m3")
    parser.add_argument("--max-rounds", type=int, default=6)
    args = parser.parse_args()
    load_dotenv()

    root = Path.cwd().resolve()

    def _safe(rel: str) -> Path:
        p = (root / rel).resolve()
        if not p.is_relative_to(root):
            raise ValueError("path escapes the project")
        return p

    async def run_tool(name: str, a: dict) -> dict:
        try:
            if name == "roll_dice":
                return {"value": random.randint(1, int(a.get("sides", 6)))}
            if name == "now":
                return {"now": datetime.now().isoformat(timespec="seconds")}
            if name == "calc":
                expr = ast.parse(a["expression"], mode="eval").body
                return {"value": _calc(expr)}
            if name == "secret_message":
                return {
                    "message": _caesar("the fire is out at the school", 7),
                    "shift": 7,
                }
            if name == "decode":
                return {"text": _caesar(a["text"], -int(a["shift"]))}
            if name == "ls":
                p = _safe(a.get("path", "."))
                return {
                    "entries": sorted(c.name + ("/" if c.is_dir() else "") for c in p.iterdir())[
                        :50
                    ]
                }
            if name == "read_file":
                return {"text": _safe(a["path"]).read_text()[:4000]}
            return {"error": f"unknown tool {name}"}
        except Exception as e:  # same convention as dispatch: errors go back to the model
            return {"error": str(e)[:200]}

    result = asyncio.run(
        agent_loop(
            client=llm_client(),
            model=args.model,
            system="You are a helpful agent. Use tools when they help; "
            "when done, yield with a short answer.",
            user=args.task,
            tool_defs=[
                tool(
                    "roll_dice",
                    "Roll a die with the given number of sides.",
                    {"sides": NUM},
                ),
                tool("now", "Current local date and time.", {}),
                tool(
                    "calc",
                    "Evaluate an arithmetic expression exactly.",
                    {"expression": STR},
                ),
                tool("secret_message", "Fetch the secret message (Caesar-shifted).", {}),
                tool(
                    "decode",
                    "Undo a Caesar shift on text.",
                    {"text": STR, "shift": NUM},
                ),
                tool("ls", "List a directory in the project.", {"path": STR}, required=[]),
                tool(
                    "read_file",
                    "Read a text file in the project (first 4000 chars).",
                    {"path": STR},
                ),
            ],
            run_tool=run_tool,
            max_rounds=args.max_rounds,
            max_tokens=900,
        )
    )
    print(f"rounds: {result.rounds}  tokens: {result.usage.total}")
    print(result.final)
