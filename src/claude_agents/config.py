"""Central configuration, sourced from environment variables.

All secrets and endpoints come from the environment (see ``.env.example``).
Nothing here should ever contain a real key or an inline default secret.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# Model IDs are pinned to the current Claude generation. Opus 4.8 is the most
# capable model for long-horizon agentic loops; Haiku 4.5 is a cheap default for
# sub-agents and cheap sub-steps. Adaptive thinking is on by default.
DEFAULT_MODEL = "claude-opus-4-8"
FAST_MODEL = "claude-haiku-4-5"


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment."""

    anthropic_api_key: str | None = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", DEFAULT_MODEL))
    fast_model: str = field(
        default_factory=lambda: os.getenv("AGENT_FAST_MODEL", FAST_MODEL)
    )
    # Effort trades thoroughness for token spend: low | medium | high | xhigh | max.
    effort: str = field(default_factory=lambda: os.getenv("AGENT_EFFORT", "high"))
    # Hard ceiling on loop iterations so a runaway agent can't spin forever.
    max_steps: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_STEPS", "8")))
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_TOKENS", "16000"))
    )

    def require_api_key(self) -> str:
        """Return the API key or raise a clear error if it is missing."""
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "populate it, or export the variable in your shell."
            )
        return self.anthropic_api_key


settings = Settings()
