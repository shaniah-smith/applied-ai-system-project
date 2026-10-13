"""Thin wrapper around the Gemini API so the rest of the app never talks to the SDK directly."""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger("vibefinder.llm")

DEFAULT_MODEL = "gemini-3.8-flash"
TRANSIENT = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "timed out")
RETRY_WAITS = (2, 5)  # seconds to wait before the 2nd and 3rd tries


class LLMError(Exception):
    """Raised when the language model can't be reached or returns nothing usable."""


class GeminiClient:
    """Calls Gemini and returns plain text (or JSON text when asked)."""

    def __init__(self, api_key: str, model: Optional[str] = None):
        """Create the SDK client. Imported here so tests and offline mode don't need the package."""
        from google import genai
        from google.genai import types

        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, json_mode: bool = False, temperature: float = 0.3) -> str:
        """Send one prompt and return the text reply, raising LLMError on any failure."""
        config = self._types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        for attempt in range(len(RETRY_WAITS) + 1):
            try:
                response = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
                text = (response.text or "").strip()
                break
            except Exception as err:  # network, auth, quota, bad model name...
                busy = any(code in str(err) for code in TRANSIENT)
                if busy and attempt < len(RETRY_WAITS):
                    wait = RETRY_WAITS[attempt]
                    logger.warning("Gemini busy (%s); retrying in %ss", str(err)[:60], wait)
                    time.sleep(wait)
                    continue
                logger.error("Gemini call failed: %s", err)
                raise LLMError(str(err)) from err
        if not text:
            logger.error("Gemini returned an empty response")
            raise LLMError("empty response")
        return text


def get_llm() -> Optional[GeminiClient]:
    """Return a Gemini client if GEMINI_API_KEY is set, otherwise None (offline mode)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        logger.warning("GEMINI_API_KEY not set; running in offline mode")
        return None
    try:
        return GeminiClient(key)
    except Exception as err:
        logger.error("Could not create Gemini client: %s", err)
        return None
