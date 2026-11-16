"""Talking to the language model: one small interface, several interchangeable backends."""

from cairn.llm.base import (
    BadRequestError,
    GenRequest,
    GenResponse,
    LLMBackend,
    LLMError,
    ServerUnavailableError,
    Usage,
)
from cairn.llm.cache import CachedBackend
from cairn.llm.mock import DemoReaderBackend, FunctionBackend, RouterBackend, ScriptedBackend
from cairn.llm.openai_compat import OpenAICompatBackend, RetryPolicy

__all__ = [
    "BadRequestError",
    "CachedBackend",
    "DemoReaderBackend",
    "FunctionBackend",
    "GenRequest",
    "GenResponse",
    "LLMBackend",
    "LLMError",
    "OpenAICompatBackend",
    "RetryPolicy",
    "RouterBackend",
    "ScriptedBackend",
    "ServerUnavailableError",
    "Usage",
]
