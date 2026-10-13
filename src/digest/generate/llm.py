"""OpenAI-compatible client for the internal LLM endpoint (vLLM / Ollama)."""

from __future__ import annotations

import logging

import openai
from openai import OpenAI
from pydantic import BaseModel

from digest.config import Settings
from digest.generate.prompts import LLMInitiative, ResponseMode

log = logging.getLogger(__name__)


class LLM:
    """Chat client that asks for structured JSON.

    Starts in the configured response mode; if the server rejects `json_schema`, it switches to
    `json_object` (schema then goes in the prompt) for the rest of the run.
    """

    def __init__(self, cfg: Settings, client: OpenAI | None = None):
        if client is None and not cfg.llm.base_url:
            raise ValueError("LLM_BASE_URL is not set; use --dry-run for extractive summaries")
        self.cfg = cfg
        self.model = cfg.llm.model
        self.mode: ResponseMode = cfg.generate.response_format  # type: ignore[assignment]
        self.client = client or OpenAI(
            base_url=cfg.llm.base_url, api_key=cfg.llm.api_key, timeout=cfg.llm.timeout_s, max_retries=1
        )

    def _response_format(self, schema: type[BaseModel]) -> dict:
        if self.mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__.lower(), "schema": schema.model_json_schema()},
            }
        return {"type": "json_object"}

    def complete(self, messages: list[dict[str, str]], schema: type[BaseModel] = LLMInitiative) -> str:
        extra = {"reasoning_effort": self.cfg.llm.reasoning_effort} if self.cfg.llm.reasoning_effort else {}
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.cfg.generate.temperature,
                max_tokens=self.cfg.llm.max_tokens,
                response_format=self._response_format(schema),
                **extra,
            )
        except openai.BadRequestError as exc:
            if self.mode != "json_schema":
                raise
            log.warning("server rejected json_schema (%s); falling back to json_object", exc)
            self.mode = "json_object"
            return self.complete(messages, schema)
        return resp.choices[0].message.content or ""
