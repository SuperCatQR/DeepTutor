from __future__ import annotations

import json

import pytest

from deeptutor.agents.visualize.models import ReviewResult, VisualizationAnalysis
from deeptutor.agents.visualize.pipeline import VisualizePipeline
from deeptutor.agents.visualize.utils import validate_self_contained_html


@pytest.fixture(autouse=True)
def _stub_agent_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deeptutor.agents.base_agent.get_agent_params", lambda _: {})


def _analysis() -> VisualizationAnalysis:
    return VisualizationAnalysis(
        render_type="html",
        description="A small interactive demo",
    )


def _pipeline(*, retry_attempts: int = 1) -> VisualizePipeline:
    return VisualizePipeline(
        api_key="test",
        base_url="https://example.invalid/v1",
        api_version=None,
        retry_attempts=retry_attempts,
    )


@pytest.mark.asyncio
async def test_run_analysis_retries_transient_structured_output_failure() -> None:
    pipeline = _pipeline(retry_attempts=2)
    calls = 0

    async def fake_process(**_: object) -> VisualizationAnalysis:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise json.JSONDecodeError("empty JSON response", "", 0)
        return _analysis()

    pipeline.analysis_agent.process = fake_process  # type: ignore[method-assign]

    result = await pipeline.run_analysis(
        user_input="Make a demo",
        history_context="",
        render_mode="html",
    )

    assert calls == 2
    assert result.render_type == "html"


@pytest.mark.asyncio
async def test_run_analysis_uses_one_attempt_by_default() -> None:
    pipeline = _pipeline()
    calls = 0

    async def fake_process(**_: object) -> VisualizationAnalysis:
        nonlocal calls
        calls += 1
        raise json.JSONDecodeError("empty JSON response", "", 0)

    pipeline.analysis_agent.process = fake_process  # type: ignore[method-assign]

    with pytest.raises(json.JSONDecodeError):
        await pipeline.run_analysis(
            user_input="Make a demo",
            history_context="",
            render_mode="html",
        )

    assert calls == 1


@pytest.mark.asyncio
async def test_run_code_generation_retries_invalid_output() -> None:
    pipeline = _pipeline(retry_attempts=2)
    outputs = iter(["", "<!doctype html><html><body>ok</body></html>"])

    async def fake_process(**_: object) -> str:
        return next(outputs)

    pipeline.code_agent.process = fake_process  # type: ignore[method-assign]

    result = await pipeline.run_code_generation(
        user_input="Make a demo",
        history_context="",
        analysis=_analysis(),
        validator=lambda value: "<html" in value.lower(),
    )

    assert result.startswith("<!doctype html>")


@pytest.mark.asyncio
async def test_run_repair_retries_transient_structured_output_failure() -> None:
    pipeline = _pipeline(retry_attempts=2)
    calls = 0

    async def fake_process(**_: object) -> ReviewResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise json.JSONDecodeError("empty JSON response", "", 0)
        return ReviewResult(optimized_code="<svg viewBox='0 0 1 1'></svg>")

    pipeline.review_agent.process = fake_process  # type: ignore[method-assign]

    result = await pipeline.run_repair(
        user_input="Make a demo",
        analysis=_analysis(),
        code="<svg></svg>",
        error="SVG root is missing a viewBox attribute.",
    )

    assert calls == 2
    assert result.optimized_code.startswith("<svg")


@pytest.mark.parametrize(
    "html",
    [
        "<div>placeholder</div>",
        "<html><body>broken</html></body>",
        '<script>const fake = "<html><body></body></html>";</script>',
    ],
)
def test_validate_self_contained_html_rejects_incomplete_documents(html: str) -> None:
    assert not validate_self_contained_html(html)[0]


def test_validate_self_contained_html_accepts_complete_fenced_document() -> None:
    html = "```html\n<!doctype html><html><body>ok</body></html>\n```"

    assert validate_self_contained_html(html)[0]
