"""Orchestrates the three-stage visualization generation flow."""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from typing import Any

from pydantic import ValidationError

from deeptutor.core.context import Attachment
from deeptutor.services.llm.exceptions import (
    LLMAPIError,
    LLMError,
    LLMParseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

from .agents import AnalysisAgent, CodeGeneratorAgent, ReviewAgent
from .models import ReviewResult, VisualizationAnalysis

logger = logging.getLogger(__name__)


def _is_retryable_llm_error(error: LLMError) -> bool:
    """Return whether a provider failure can succeed on one immediate retry."""
    if isinstance(
        error,
        (LLMProviderError, LLMParseError, LLMRateLimitError, LLMTimeoutError),
    ):
        return True
    return isinstance(error, LLMAPIError) and (
        error.status_code is None or error.status_code >= 500
    )


def _should_retry_visualization_error(error: Exception) -> bool:
    """Limit retries to transient provider and structured-output failures."""
    if isinstance(error, (json.JSONDecodeError, ValidationError)):
        return True
    return isinstance(error, LLMError) and _is_retryable_llm_error(error)


class VisualizePipeline:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        api_version: str | None,
        language: str = "zh",
        trace_callback: Callable[[dict[str, Any]], Any] | None = None,
        retry_attempts: int = 1,
    ) -> None:
        self.retry_attempts = max(1, retry_attempts)
        self.analysis_agent = AnalysisAgent(
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            language=language,
        )
        self.code_agent = CodeGeneratorAgent(
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            language=language,
        )
        self.review_agent = ReviewAgent(
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            language=language,
        )
        self.set_trace_callback(trace_callback)

    def set_trace_callback(self, callback: Callable[[dict[str, Any]], Any] | None) -> None:
        for agent in (self.analysis_agent, self.code_agent, self.review_agent):
            agent.set_trace_callback(callback)

    async def run_analysis(
        self,
        *,
        user_input: str,
        history_context: str,
        render_mode: str = "auto",
        attachments: list[Attachment] | None = None,
    ) -> VisualizationAnalysis:
        for attempt in range(self.retry_attempts):
            try:
                return await self.analysis_agent.process(
                    user_input=user_input,
                    history_context=history_context,
                    render_mode=render_mode,
                    attachments=attachments,
                )
            except (json.JSONDecodeError, ValidationError, LLMError) as exc:
                if not _should_retry_visualization_error(exc) or (
                    attempt + 1 >= self.retry_attempts
                ):
                    raise
                logger.warning(
                    "Visualization analysis failed; retrying once",
                    exc_info=True,
                )

        raise RuntimeError("Visualization analysis retry loop ended unexpectedly")

    async def run_code_generation(
        self,
        *,
        user_input: str,
        history_context: str,
        analysis: VisualizationAnalysis,
        validator: Callable[[str], bool] | None = None,
    ) -> str:
        last_code = ""
        for attempt in range(self.retry_attempts):
            try:
                code = await self.code_agent.process(
                    user_input=user_input,
                    history_context=history_context,
                    analysis=analysis,
                )
                last_code = code or ""
                is_valid = bool(last_code.strip()) and (validator is None or validator(last_code))
                if is_valid or attempt + 1 >= self.retry_attempts:
                    return last_code
            except LLMError as exc:
                if not _should_retry_visualization_error(exc) or (
                    attempt + 1 >= self.retry_attempts
                ):
                    raise
                logger.warning(
                    "Visualization code generation failed; retrying once",
                    exc_info=True,
                )
                continue

            logger.warning("Visualization code failed local validation; retrying once")

        return last_code

    async def run_repair(
        self,
        *,
        user_input: str,
        analysis: VisualizationAnalysis,
        code: str,
        error: str,
    ) -> ReviewResult:
        return await self.review_agent.process(
            user_input=user_input,
            analysis=analysis,
            code=code,
            error=error,
        )


__all__ = ["VisualizePipeline"]
