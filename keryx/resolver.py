"""Principal resolver: answers the voice model's mid-call ask_principal questions.

Hermes is blocked inside place_call while the call runs, so it cannot answer. Instead:
1. a quick model call (Hermes's own model via ctx.llm) over the brief, optionally using
   allow-listed read-only Hermes tools (e.g. a calendar lookup);
2. if that is not confident, ask the user directly through Hermes's clarify prompt (any surface);
3. otherwise return "no answer" so the voice model says it will confirm later.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable, Optional

from .brief import Brief

logger = logging.getLogger(__name__)

LlmFn = Callable[[list[dict[str, str]]], str]
ToolFn = Callable[[str, dict[str, Any]], str]
AskUserFn = Callable[[str, float], Optional[str]]

_SYSTEM = """You help a phone assistant that is on a live call for {principal}. The person on the call asked a
question the assistant could not answer from its brief. Decide quickly and reply with ONE JSON object only:
  {{"action": "answer", "answer": "<short spoken answer>", "confident": true|false}}
  {{"action": "tool", "tool": "<name>", "args": {{...}}}}     (only these read-only tools: {tools})
  {{"action": "ask_user"}}                                     (only {principal} can decide)
Rules: answer only from the brief or tool results. Never invent facts. Never approve spending, commitments or
sharing information beyond the brief's limits and may_share list; for those choose ask_user.
Brief: {brief}"""


def _parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        val = json.loads(m.group(0))
        return val if isinstance(val, dict) else {}
    except ValueError:
        return {}


class PrincipalResolver:
    def __init__(self, brief: Brief, *, llm: Optional[LlmFn] = None, dispatch_tool: Optional[ToolFn] = None,
                 readonly_tools: Optional[list[str]] = None, ask_user: Optional[AskUserFn] = None,
                 timeout_s: float = 30.0, max_steps: int = 3) -> None:
        self.brief = brief
        self.llm, self.dispatch_tool, self.ask_user = llm, dispatch_tool, ask_user
        self.readonly_tools = list(readonly_tools or []) if dispatch_tool else []
        self.timeout_s = timeout_s
        self.max_steps = max_steps
        self.log: list[dict[str, Any]] = []

    async def resolve(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()[:1000]
        if not question:
            return {"answer": None, "source": "none", "note": "Empty question."}
        try:
            result = await asyncio.wait_for(self._resolve(question), self.timeout_s)
        except asyncio.TimeoutError:
            result = {"answer": None, "source": "timeout"}
        except Exception as e:  # never break the call over a resolver failure
            logger.warning("keryx resolver failed: %s", e)
            result = {"answer": None, "source": "error"}
        if not result.get("answer"):
            result["note"] = ("No answer available. Tell them you will confirm and get back to them; "
                              "do not guess or commit.")
        self.log.append({"question": question, **result})
        return result

    async def _resolve(self, question: str) -> dict[str, Any]:
        if self.llm is not None:
            ans = await self._ask_model(question)
            if ans is not None:
                return ans
        if self.ask_user is not None:
            prompt = (f"📞 On the call with {self.brief.callee}, they ask: \"{question}\". "
                      f"How should I answer? (They are waiting on the line.)")
            reply = await asyncio.to_thread(self.ask_user, prompt, max(5.0, self.timeout_s - 2))
            if reply and reply.strip():
                return {"answer": reply.strip(), "source": "principal"}
        return {"answer": None, "source": "unknown"}

    async def _ask_model(self, question: str) -> Optional[dict[str, Any]]:
        system = _SYSTEM.format(principal=self.brief.principal, tools=", ".join(self.readonly_tools) or "none",
                                brief=json.dumps(self.brief.to_dict(), ensure_ascii=False))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Question: {question}"}]
        for _ in range(self.max_steps):
            text = await asyncio.to_thread(self.llm, messages)
            decision = _parse_json(text)
            action = decision.get("action")
            if action == "answer" and decision.get("answer"):
                if decision.get("confident") is False:
                    return None  # fall through to the user
                return {"answer": str(decision["answer"])[:1000], "source": "brief"}
            if action == "tool" and decision.get("tool") in self.readonly_tools and self.dispatch_tool:
                name, args = decision["tool"], decision.get("args") or {}
                try:
                    out = await asyncio.to_thread(self.dispatch_tool, name, args if isinstance(args, dict) else {})
                except Exception as e:
                    out = json.dumps({"error": str(e)})
                messages += [{"role": "assistant", "content": json.dumps(decision)},
                             {"role": "user", "content": f"Result of {name}: {str(out)[:4000]}"}]
                continue
            return None  # ask_user, unparseable, or a tool outside the allow-list
        return None
