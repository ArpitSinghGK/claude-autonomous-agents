"""Thin, idiomatic wrapper around the Anthropic Python SDK.

Every agent loop in this repo drives Claude through this one seam, so model
selection, adaptive thinking, streaming, and the tool round-trip live in a
single place instead of being copy-pasted into each loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import anthropic

from ..config import settings


@dataclass
class LLMResponse:
    """Normalized view of a single Claude turn."""

    text: str
    tool_calls: list["ToolCall"]
    stop_reason: str
    raw: Any  # the underlying anthropic Message, for callers that need it


@dataclass
class ToolCall:
    """A tool_use block Claude emitted, ready for the harness to execute."""

    id: str
    name: str
    input: dict[str, Any]


class ClaudeClient:
    """Wraps ``anthropic.Anthropic`` with the defaults these agents rely on.

    Construction is lazy about credentials: the SDK resolves ``ANTHROPIC_API_KEY``
    (or an ``ant auth login`` profile) from the environment, so we only assert the
    key exists when a real call is made.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.model
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            settings.require_api_key()
            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        stream: bool = True,
    ) -> LLMResponse:
        """Run one Claude turn and return a normalized :class:`LLMResponse`.

        We default to streaming so large ``max_tokens`` requests don't trip the
        SDK's HTTP timeout, then collect the final message via the SDK helper.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": settings.max_tokens,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": settings.effort},
            "messages": messages,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = list(tools)

        if stream:
            with self.client.messages.stream(**params) as s:
                message = s.get_final_message()
        else:
            message = self.client.messages.create(**params)

        return self._normalize(message)

    @staticmethod
    def _normalize(message: Any) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )
        return LLMResponse(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=message.stop_reason,
            raw=message,
        )
