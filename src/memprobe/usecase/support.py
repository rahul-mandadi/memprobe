"""Customer-support domain surface: system prompt + probe phrasing.

Kept intentionally thin. The generator (scenarios/) owns the fact vocabulary and ground truth;
this module only supplies the domain's *voice* — how the agent is framed and how a probe is
posed as a support question. Swap this file to change domains without touching the harness.

Implemented (static templates; no model).
"""

from __future__ import annotations

from memprobe.scenarios.schema import Probe

SYSTEM_PROMPT = (
    "You are a customer-support assistant for a SaaS product. Use what you remember about the "
    "user to help them. If you don't know something, say so — never guess a stored detail."
)


def probe_question(probe: Probe) -> str:
    """Render a probe as a natural support question the agent must answer from memory."""
    key = probe.fact_key.replace("_", " ")
    return f"Quick check before I proceed — what's my {key} on file?"
