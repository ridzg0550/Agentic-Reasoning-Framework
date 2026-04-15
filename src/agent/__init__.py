"""Agent module for agentic reasoning system."""

# src/agent/__init__.py
# Keep compatibility: many modules import OllamaClient from src.agent
from .llm_client import OllamaClient

# Optionally expose Executor if other modules import it directly
from .executor import Executor  # safe fallback; if executor is elsewhere, this keeps common access
from .params import *

__all__ = ['OllamaClient']
