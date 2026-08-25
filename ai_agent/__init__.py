"""An LLM agent that reads the pricing system and explains it.

WHAT THIS PACKAGE IS ALLOWED TO DO
----------------------------------
Read, and simulate. Nothing here decides a price. Prices are decided by
:mod:`pricing`, which is deterministic, unit-tested, reproducible and returns the
arithmetic as its own explanation -- and none of those properties survive being
replaced by a language model. See ``docs/ai_agent_design.md`` for the full
argument, including why the tempting middle position ("let the model pick the
multipliers, keep the guardrails") is worse than either extreme: guardrails clamp
outputs, they cannot detect that the reasoning behind an in-band number was
wrong.

What the system genuinely lacks is language. Every pricing decision is stored
with a full arithmetic breakdown, and reading it means writing SQL and decoding
JSON. That is the gap this package fills.

OPTIONAL BY CONSTRUCTION
------------------------
``anthropic`` is an optional extra in ``pyproject.toml`` and is imported
lazily. The API, the pricing engine, the models and the dashboard all start and
run normally when it is absent; the agent page renders an explanation of what
to install instead of a traceback. That is the correct dependency direction for
anything non-deterministic: deleting this package must leave a working pricing
system behind, and it does.

Install it with::

    pip install -e ".[agent]"
    export ANTHROPIC_API_KEY=sk-ant-...
"""

from ai_agent.tools import (
    AGENT_TOOLS,
    AgentUnavailable,
    ToolCallDenied,
    agent_status,
)

__all__ = [
    "AGENT_TOOLS",
    "AgentUnavailable",
    "ToolCallDenied",
    "agent_status",
]
