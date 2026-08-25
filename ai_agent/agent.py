"""A conversational agent over the pricing audit trail.

Answers "why was this price what it was" by retrieving the decision, the
competitor rates around it and the forecast behind it, then explaining the
arithmetic that is already recorded. It does not compute prices and it cannot
change them; see :mod:`ai_agent.tools` for how that is enforced rather than
requested.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai_agent.tools import AgentUnavailable, _load_anthropic, build_tools
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

#: Opus 4.8. The task is retrieval and careful attribution over structured
#: facts, and the cost of a fluent wrong answer here is a revenue manager acting
#: on it.
MODEL = os.getenv("AGENT_MODEL", "claude-opus-4-8")

#: Retrieval and narration over a handful of tool calls -- not deep reasoning.
#: `high` buys little here and costs latency on every turn. Worth sweeping
#: against your own question set rather than taking either setting on faith.
EFFORT = os.getenv("AGENT_EFFORT", "medium")

#: Enough for a multi-step answer with tool results in context, bounded so a
#: runaway loop cannot quietly spend a lot of money.
MAX_TOKENS = 8_000

#: Hard ceiling on tool calls per question. The SDK's runner loops until the
#: model stops asking; without a cap, one badly-formed question could walk the
#: whole competitor table.
MAX_ITERATIONS = 12


SYSTEM_PROMPT = """\
You are a revenue-management analyst for a dynamic hotel pricing system covering
eight properties across six Indian cities. Your job is to explain pricing
decisions the system has already made, using the tools to retrieve the facts.

HOW THE SYSTEM PRICES
A base rate is multiplied by additive adjustments -- occupancy, lead time, day of
week, seasonality, local events, competitor position. The adjustments are summed
into one multiplier (never chained), each is individually clamped, and the whole
multiplier is then scaled by the demand model's confidence with a floor of 0.5.
The result passes through guardrails in a fixed order: low-occupancy block, then
the daily-change cap, then competitor bands, then the per-room floor and ceiling,
then the global MIN_PRICE and MAX_PRICE. Relative rules run before absolute ones,
so a floor is never undercut by a percentage rule.

Two consequences worth remembering, because they explain most surprises:
- Low model confidence shrinks the multiplier towards the base rate. A price that
  looks unresponsive to obvious demand is often a wide prediction interval.
- The daily-change cap frequently clips a correct, large move. The demand signal
  is still in the number; it was outranked.

RULES
- Never state a price, rate, occupancy, date or percentage that you did not read
  from a tool result. If a tool did not return something, say you do not have it.
  Do not estimate, and do not fill a gap from general knowledge about hotels.
- When a guardrail changed a price, name the specific identifier from
  guardrails_applied. Do not paraphrase it into a cause of your own -- saying
  "demand was soft" when the daily_change_cap fired is the single most damaging
  mistake you can make here, because it is plausible, confident and wrong.
- Check whether competitor observations are stale before explaining anything by
  competitor behaviour. If competitor data looks thin, call the ingestion status
  tool: a dead feed explains far more than a story about the market.
- You cannot change prices. You read and you simulate. If asked to set a rate,
  run the simulation, show what it produces, and say the change is theirs to make.
- Lead with the answer in one or two sentences, then the supporting numbers. The
  reader is a revenue manager, not an engineer: give figures, not schema names.
"""


@dataclass
class AgentTurn:
    """One question, its answer, and what the model did to get there."""

    question: str
    answer: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stopped_early: bool = False

    @property
    def estimated_cost_usd(self) -> float:
        """Opus 4.8 list pricing: $5/MTok in, $25/MTok out."""
        return self.input_tokens / 1e6 * 5.0 + self.output_tokens / 1e6 * 25.0


class PricingAgent:
    """A multi-turn agent session.

    Holds conversation history so follow-ups ("and the night before?") work.
    Construct one per user session; it is not thread-safe.
    """

    def __init__(
        self,
        *,
        model: str = MODEL,
        effort: str = EFFORT,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        anthropic = _load_anthropic()
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise AgentUnavailable(
                "No ANTHROPIC_API_KEY in the environment. The agent needs one; "
                "the rest of the pricing system does not."
            )

        self._client = anthropic.Anthropic()
        self._tools = build_tools()
        self.model = model
        self.effort = effort
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Forget the conversation, keep the configuration."""
        self.messages = []

    def ask(self, question: str) -> AgentTurn:
        """Answer one question, updating the conversation history.

        Raises:
            AgentUnavailable: If the API rejects the credentials.
        """
        import anthropic as _sdk

        self.messages.append({"role": "user", "content": question})

        runner = self._client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                    # The prompt and the tool list are byte-identical on every
                    # turn, so this prefix is worth caching: repeat input drops
                    # to roughly a tenth of list price. The question comes after
                    # it and varies, which is exactly the right way round.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=self._tools,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=self.messages,
        )

        tool_calls: List[Dict[str, Any]] = []
        input_tokens = output_tokens = 0
        final = None
        stopped_early = False

        try:
            for iteration, message in enumerate(runner, start=1):
                final = message
                input_tokens += message.usage.input_tokens or 0
                output_tokens += message.usage.output_tokens or 0

                for block in message.content:
                    if block.type == "tool_use":
                        tool_calls.append({"name": block.name, "input": block.input})

                if iteration >= MAX_ITERATIONS:
                    stopped_early = True
                    logger.warning(
                        "Agent hit the %d-iteration cap on: %.80s",
                        MAX_ITERATIONS,
                        question,
                    )
                    break
        except _sdk.AuthenticationError as exc:
            raise AgentUnavailable(
                "The Anthropic API rejected the credentials. Check ANTHROPIC_API_KEY."
            ) from exc
        except _sdk.RateLimitError as exc:
            raise AgentUnavailable(
                "Rate limited by the Anthropic API. Wait a moment and retry."
            ) from exc

        if final is None:  # pragma: no cover - the runner always yields once
            raise AgentUnavailable("The model returned no response.")

        answer = "\n".join(b.text for b in final.content if b.type == "text").strip()
        if stopped_early and answer:
            answer += (
                f"\n\n_(Stopped after {MAX_ITERATIONS} tool calls. Ask something "
                f"narrower for a complete answer.)_"
            )
        elif stopped_early:
            answer = (
                f"I made {MAX_ITERATIONS} tool calls without reaching an answer. "
                f"Try narrowing the question to one hotel and a small date range."
            )

        # The full content, not just the text: tool_use blocks have to be echoed
        # back or the next turn's history is inconsistent and the API rejects it.
        self.messages.append({"role": "assistant", "content": final.content})

        return AgentTurn(
            question=question,
            answer=answer,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stopped_early=stopped_early,
        )


def ask(question: str, agent: Optional[PricingAgent] = None) -> AgentTurn:
    """One-shot convenience wrapper. Builds a throwaway session.

    Fine for a script; use :class:`PricingAgent` directly when follow-up
    questions should see the earlier ones.
    """
    return (agent or PricingAgent()).ask(question)


#: Questions the dashboard offers as starting points. Chosen to show the tools
#: chaining rather than to flatter the model: each needs at least two calls.
EXAMPLE_QUESTIONS = (
    "Why did we drop the price on our Goa suites last week?",
    "Which guardrail fires most often across the portfolio, and what is it costing us?",
    "Is our competitor data still arriving? When was the last observation?",
    "What would happen to next weekend's Jaipur deluxe rate if occupancy were 20 points higher?",
    "Compare our pricing against the competitor set for Mumbai over the next 7 nights.",
)


__all__ = [
    "EXAMPLE_QUESTIONS",
    "MODEL",
    "SYSTEM_PROMPT",
    "AgentTurn",
    "PricingAgent",
    "ask",
]
