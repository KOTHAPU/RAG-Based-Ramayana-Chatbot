"""
generation.py — LLM generation clients for Groq and Ollama.

Provides a unified GenerationClient interface with:
- Groq API (llama-3.3-70b-versatile and friends)
- Ollama (local inference)
- Streaming support
- Rate-limit retry with exponential back-off
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Generator, Iterator

logger = logging.getLogger(__name__)


@dataclass
class GenerationResponse:
    """Result from a single LLM generation call."""

    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"

    @property
    def is_empty(self) -> bool:
        return not self.content.strip()


class GenerationClient:
    """
    Unified LLM generation interface supporting Groq and Ollama.
    """

    def __init__(
        self,
        provider: str = "groq",
        model: str = "llama-3.3-70b-versatile",
        api_key: str = "",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._groq_client = None

    def _get_groq_client(self):
        """Lazy-init the Groq client."""
        if self._groq_client is not None:
            return self._groq_client
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=self.api_key)
            return self._groq_client
        except ImportError:
            raise RuntimeError("groq package not installed. Run: pip install groq")

    # ── Non-streaming generation ──────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        retries: int = 3,
    ) -> GenerationResponse:
        """Generate a complete response (non-streaming)."""
        _max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        _temperature = temperature if temperature is not None else self.temperature

        if self.provider == "groq":
            return self._generate_groq(
                system_prompt, user_prompt, _max_tokens, _temperature, retries
            )
        elif self.provider == "ollama":
            return self._generate_ollama(
                system_prompt, user_prompt, _max_tokens, _temperature
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    def _generate_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        retries: int = 3,
    ) -> GenerationResponse:
        """Call the Groq API with exponential back-off retry."""
        client = self._get_groq_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content or ""
                usage = response.usage
                return GenerationResponse(
                    content=content,
                    model=self.model,
                    provider="groq",
                    input_tokens=getattr(usage, "prompt_tokens", 0),
                    output_tokens=getattr(usage, "completion_tokens", 0),
                    finish_reason=response.choices[0].finish_reason or "stop",
                )
            except Exception as exc:
                error_str = str(exc).lower()
                if "rate_limit" in error_str or "429" in error_str:
                    wait = 2 ** attempt
                    logger.warning("Groq rate limit hit. Retrying in %ds…", wait)
                    time.sleep(wait)
                elif attempt < retries - 1:
                    logger.warning("Groq API error (attempt %d): %s", attempt + 1, exc)
                    time.sleep(1)
                else:
                    raise RuntimeError(f"Groq generation failed after {retries} attempts: {exc}") from exc

        raise RuntimeError("Groq generation failed after max retries.")

    def _generate_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResponse:
        """Call the Ollama local API."""
        try:
            import requests

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            resp = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            return GenerationResponse(
                content=content,
                model=self.model,
                provider="ollama",
                finish_reason=data.get("done_reason", "stop"),
            )
        except Exception as exc:
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc

    # ── Streaming generation ──────────────────────────────────────────────────

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """
        Stream tokens from the LLM.
        Yields string deltas as they arrive.
        Falls back to non-streaming on error.
        """
        _max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        _temperature = temperature if temperature is not None else self.temperature

        if self.provider == "groq":
            yield from self._stream_groq(system_prompt, user_prompt, _max_tokens, _temperature)
        elif self.provider == "ollama":
            yield from self._stream_ollama(system_prompt, user_prompt, _max_tokens, _temperature)
        else:
            # Fallback: return full response as a single chunk
            response = self.generate(system_prompt, user_prompt, _max_tokens, _temperature)
            yield response.content

    def _stream_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        """Stream tokens from Groq using create() with stream=True."""
        client = self._get_groq_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                text = getattr(delta, 'content', None)
                if text:
                    yield text
        except Exception as exc:
            logger.warning("Streaming failed, falling back to non-streaming: %s", exc)
            response = self._generate_groq(system_prompt, user_prompt, max_tokens, temperature)
            yield response.content

    def _stream_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        """Stream tokens from Ollama."""
        try:
            import requests, json

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            with requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=180,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        delta = data.get("message", {}).get("content", "")
                        if delta:
                            yield delta
                        if data.get("done"):
                            break
        except Exception as exc:
            logger.warning("Ollama streaming failed: %s", exc)
            response = self._generate_ollama(system_prompt, user_prompt, max_tokens, temperature)
            yield response.content

    def is_available(self) -> bool:
        """Check if the configured LLM backend is reachable."""
        try:
            if self.provider == "groq":
                if not self.api_key:
                    return False
                client = self._get_groq_client()
                # Try listing models as a health check
                client.models.list()
                return True
            elif self.provider == "ollama":
                import requests
                resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                return resp.status_code == 200
        except Exception:
            return False
        return False
