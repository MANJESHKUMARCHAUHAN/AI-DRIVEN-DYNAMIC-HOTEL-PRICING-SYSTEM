"""AI Agent: ask why a price was what it was.

Every pricing decision is already stored with its full arithmetic -- the base
rate, each adjustment, the demand score, the guardrails that fired. Reading it
today means writing SQL and decoding JSON. This page puts language on top of data
that is already complete.

The agent reads and simulates. It does not price anything: that is
``pricing/``, which is deterministic, reproducible and auditable, and none of
those survive being replaced by a language model. See ``docs/ai_agent_design.md``.

Optional. When the SDK or the API key is missing this page explains what to
install; every other page is unaffected, which is the intended dependency
direction for anything non-deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from ai_agent.tools import AgentUnavailable, agent_status  # noqa: E402
from dashboard.components.ui import bootstrap, sidebar_status  # noqa: E402

bootstrap("AI Agent")
sidebar_status()

status = agent_status()

# --------------------------------------------------------------------------- #
# Unavailable: explain, do not traceback
# --------------------------------------------------------------------------- #

if not status["available"]:
    st.info(
        "The AI agent is an optional add-on. The pricing system, the API, the "
        "models and every other page work without it."
    )
    for problem in status["problems"]:
        st.warning(problem)

    st.markdown("##### Enable it")
    st.code(
        'pip install -e ".[agent]"\n'
        "export ANTHROPIC_API_KEY=sk-ant-...      # or add it to .env\n"
        "# then restart the dashboard",
        language="bash",
    )

    with st.expander("What it would do, and what it deliberately will not"):
        st.markdown(
            """
**It reads and it simulates.**

- Explains a past pricing decision from the stored breakdown, naming the exact
  guardrail that changed the number.
- Cross-references competitor rates and the demand forecast to check whether an
  explanation actually holds.
- Runs what-if pricing with `persist=false`, so simulations never enter the
  audit trail.

**It does not price anything.** Pricing needs reproducibility, auditability and
a 28 ms budget; a language model gives up all three. The tempting middle
position -- let the model choose the multipliers and keep the guardrails -- is
worse than either extreme, because guardrails clamp the *output* and cannot
detect that the reasoning was wrong. A hallucinated festival produces a price
that is inside every band, confidently justified, and false.

**It cannot write.** The tool layer holds a fixed allowlist of read-only calls
and refuses anything else, and `persist=False` is hardcoded rather than exposed
as a parameter the model could set. A constraint that lives only in a system
prompt is advice to a probabilistic system, not a constraint.
            """
        )
    st.stop()

# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #

from ai_agent.agent import EXAMPLE_QUESTIONS, MODEL, PricingAgent  # noqa: E402

if "agent" not in st.session_state:
    try:
        st.session_state["agent"] = PricingAgent()
        st.session_state["agent_turns"] = []
    except AgentUnavailable as exc:
        st.error(str(exc))
        st.stop()

agent: PricingAgent = st.session_state["agent"]
turns = st.session_state["agent_turns"]

with st.sidebar:
    st.divider()
    st.caption(f"**Model** {MODEL}")
    st.caption(f"**Tools** {status['tool_count']}, all read-only")
    if turns:
        spent = sum(t.estimated_cost_usd for t in turns)
        st.caption(f"**This session** {len(turns)} question(s), ~${spent:.3f}")
    if st.button("Clear conversation", use_container_width=True):
        agent.reset()
        st.session_state["agent_turns"] = []
        st.rerun()

st.caption(
    "Reads the pricing audit trail, competitor rates and demand forecasts through "
    "the same API the rest of this dashboard uses. It can simulate prices but "
    "cannot change one."
)

# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #

for turn in turns:
    with st.chat_message("user"):
        st.write(turn.question)
    with st.chat_message("assistant"):
        st.markdown(turn.answer)
        if turn.tool_calls:
            # The tool trace is shown, not hidden. An answer whose citations
            # cannot be checked is an answer nobody should act on.
            with st.expander(f"{len(turn.tool_calls)} tool call(s)"):
                for call in turn.tool_calls:
                    st.code(
                        f"{call['name']}({', '.join(f'{k}={v!r}' for k, v in call['input'].items())})",
                        language="python",
                    )
        st.caption(
            f"{turn.input_tokens:,} in / {turn.output_tokens:,} out "
            f"(~${turn.estimated_cost_usd:.3f})"
        )

if not turns:
    st.markdown("##### Try one of these")
    columns = st.columns(2)
    for index, example in enumerate(EXAMPLE_QUESTIONS):
        if columns[index % 2].button(example, key=f"ex_{index}", use_container_width=True):
            st.session_state["pending_question"] = example
            st.rerun()

question = st.chat_input("Ask about a pricing decision...")
if not question:
    question = st.session_state.pop("pending_question", None)

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Reading the audit trail..."):
            try:
                turn = agent.ask(question)
            except AgentUnavailable as exc:
                st.error(str(exc))
                st.stop()

        st.markdown(turn.answer)
        if turn.tool_calls:
            with st.expander(f"{len(turn.tool_calls)} tool call(s)"):
                for call in turn.tool_calls:
                    st.code(
                        f"{call['name']}({', '.join(f'{k}={v!r}' for k, v in call['input'].items())})",
                        language="python",
                    )
        st.caption(
            f"{turn.input_tokens:,} in / {turn.output_tokens:,} out "
            f"(~${turn.estimated_cost_usd:.3f})"
        )

    turns.append(turn)
    st.session_state["agent_turns"] = turns
