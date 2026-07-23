"""Utility helpers for the visualize pipeline."""

from __future__ import annotations

from html.parser import HTMLParser
import json
import re

import defusedxml.ElementTree as ET

from deeptutor.agents._shared.json_output import extract_json_object

_MERMAID_KEYWORDS = (
    "graph",
    "flowchart",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram-v2",
    "stateDiagram",
    "erDiagram",
    "gantt",
    "mindmap",
    "pie",
    "journey",
    "gitGraph",
    "timeline",
    "quadrantChart",
    "requirementDiagram",
    "sankey-beta",
    "xychart-beta",
    "block-beta",
    "C4Context",
)

_VOID_HTML_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


class _CompleteHTMLDocumentParser(HTMLParser):
    """Validate the nesting required for an interactive HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.error: str = ""
        self.root_seen = False
        self.body_seen = False
        self.body_open = False
        self.body_closed = False
        self.html_closed = False

    def _fail(self, message: str) -> None:
        if not self.error:
            self.error = message

    def handle_decl(self, decl: str) -> None:
        if self.root_seen:
            self._fail("DOCTYPE must appear before the <html> root element.")
        elif decl.strip().lower() != "doctype html":
            self._fail("DOCTYPE must be <!DOCTYPE html>.")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if self.error:
            return
        if tag == "html":
            if self.root_seen or self.stack:
                self._fail("HTML must have one top-level <html> root element.")
                return
            self.root_seen = True
            self.stack.append(tag)
            return
        if not self.root_seen or self.html_closed:
            self._fail("HTML content must be inside one <html> root element.")
            return
        if tag == "head":
            if self.body_seen or self.stack != ["html"]:
                self._fail("<head> must be an unclosed <html> child before <body>.")
                return
            self.stack.append(tag)
            return
        if tag == "body":
            if self.body_seen or self.stack != ["html"]:
                self._fail("<body> must be one direct child of <html>.")
                return
            self.body_seen = True
            self.body_open = True
            self.stack.append(tag)
            return
        if not self.stack:
            self._fail("HTML content must be inside the <html> root element.")
            return
        if not self.body_open and self.stack[-1] != "head":
            self._fail("Visible document content must be inside <body>.")
            return
        if tag not in _VOID_HTML_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _VOID_HTML_TAGS:
            self._fail(f"<{tag} /> is not a valid self-closing HTML element.")
            return
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self.error:
            return
        if tag in _VOID_HTML_TAGS:
            self._fail(f"Void element <{tag}> must not have a closing tag.")
            return
        if not self.stack or self.stack[-1] != tag:
            self._fail(f"Closing tag </{tag}> does not match the open HTML structure.")
            return
        self.stack.pop()
        if tag == "body":
            self.body_open = False
            self.body_closed = True
        elif tag == "html":
            if not self.body_closed:
                self._fail("<html> must contain a complete <body> element.")
                return
            self.html_closed = True

    def handle_data(self, data: str) -> None:
        if data.strip() and (not self.root_seen or self.html_closed):
            self._fail("Text must be inside the HTML document.")

    def document_error(self) -> str:
        """Return the first structural error, if the document is incomplete."""
        if self.error:
            return self.error
        if not self.root_seen:
            return "HTML must contain one <html> root element."
        if not self.body_seen or not self.body_closed:
            return "HTML must contain a complete <body> element."
        if self.stack or not self.html_closed:
            return "HTML document is not properly closed."
        return ""


def extract_code_block(text: str, language: str = "") -> str:
    """Extract a fenced code block from LLM output.

    If *language* is given the block must start with that tag;
    otherwise any triple-backtick fence is accepted.
    """
    if language:
        pattern = rf"```{re.escape(language)}\s*\n([\s\S]*?)\n```"
    else:
        pattern = r"```[A-Za-z]*\s*\n([\s\S]*?)\n```"
    match = re.search(pattern, text or "", re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def is_valid_html_document(html: str) -> bool:
    """Heuristic check that *html* looks like a renderable HTML fragment."""
    if not html:
        return False
    lowered = html.lower()
    return "<html" in lowered or "<!doctype" in lowered or "<body" in lowered or "<div" in lowered


def build_fallback_html(*, title: str, summary: str = "", note: str = "") -> str:
    """Build a minimal, self-contained fallback HTML page.

    Used when the model fails to produce a renderable HTML document, so the
    user still gets *something* shown in the iframe instead of a blank panel.
    """
    safe_title = (title or "Visualization").strip() or "Visualization"
    safe_summary = (summary or "").replace("\n", "<br>") or (
        "The model did not return a renderable HTML document."
    )
    safe_note = (note or "").replace("\n", "<br>")

    note_block = (
        f'<div class="note"><strong>Note:</strong><br>{safe_note}</div>' if safe_note else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{safe_title}</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:linear-gradient(135deg,#F8FAFC 0%,#EFF6FF 100%);
       min-height:100vh;padding:2rem;color:#1E293B;}}
  .card{{max-width:760px;margin:0 auto;background:#fff;border-radius:16px;
        padding:1.75rem 2rem;box-shadow:0 4px 6px -1px rgba(0,0,0,.08);}}
  h1{{color:#1E40AF;font-size:1.4rem;margin-bottom:1rem;}}
  .summary{{line-height:1.7;color:#475569;}}
  .note{{margin-top:1rem;padding:0.9rem 1rem;background:#FEF3C7;
        border-left:4px solid #F59E0B;border-radius:0 8px 8px 0;color:#92400E;}}
</style>
</head>
<body>
  <div class="card">
    <h1>{safe_title}</h1>
    <div class="summary">{safe_summary}</div>
    {note_block}
  </div>
</body>
</html>"""


def _strip_outer_fence(text: str) -> str:
    """Drop a single wrapping triple-backtick fence, if present."""
    stripped = (text or "").strip()
    match = re.match(r"^```[A-Za-z]*\s*\n?([\s\S]*?)\n?```$", stripped)
    return match.group(1).strip() if match else stripped


def validate_self_contained_html(html: str) -> tuple[bool, str]:
    """Validate that interactive output is one complete HTML document.

    Args:
        html: Model-generated HTML, optionally wrapped in one fenced code block.

    Returns:
        A pair containing whether the document is structurally complete and a
        concise error message when it is not.
    """
    text = _strip_outer_fence(html)
    if not text:
        return False, "Generated HTML is empty."

    parser = _CompleteHTMLDocumentParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:  # noqa: BLE001
        return False, f"HTML could not be parsed: {exc}"

    error = parser.document_error()
    return not error, error


def validate_visualization(code: str, render_type: str) -> tuple[bool, str]:
    """Cheap, deterministic, local render-ability check.

    Returns ``(ok, error)``. When ``ok`` is False, ``error`` is a short,
    LLM-actionable message used to drive a single repair pass — none of these
    failures need an LLM call to *discover*. This replaces the generic LLM
    review for the text render types: only when local validation fails do we
    spend a model call (a targeted repair, not an open-ended review).
    """
    text = (code or "").strip()
    if not text:
        return False, "Generated code is empty."

    if render_type == "svg":
        if "<svg" not in text.lower():
            return False, "SVG must contain a root <svg> element."
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            return False, f"SVG is not well-formed XML: {exc}"
        tag = root.tag.split("}")[-1].lower()
        if tag != "svg":
            return False, f"Root element must be <svg>, found <{tag}>."
        # Case-sensitive: SVG only honors the camelCase ``viewBox``; a
        # lowercase ``viewbox`` is ignored by the browser and collapses the
        # figure, so it must NOT pass validation.
        if "viewBox" not in root.attrib:
            return False, (
                "SVG root is missing a viewBox attribute (must be camelCase "
                "`viewBox`, required for responsive scaling)."
            )
        return True, ""

    if render_type == "chartjs":
        candidate = _strip_outer_fence(text)
        try:
            config = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            return False, (
                "Chart.js config must be strict JSON: double-quoted keys, no "
                "function callbacks, no comments, no trailing commas."
            )
        if not isinstance(config, dict):
            return False, "Chart.js config must be a JSON object."
        missing = [field for field in ("type", "data") if field not in config]
        if missing:
            return False, f"Chart.js config is missing required field(s): {', '.join(missing)}."
        return True, ""

    if render_type == "mermaid":
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        # `---` front-matter and `%%{init}` directives are valid lead-ins.
        if (
            first_line.startswith(_MERMAID_KEYWORDS)
            or first_line.startswith("%%")
            or first_line.startswith("---")
        ):
            return True, ""
        return False, (
            "Mermaid code must start with a valid diagram keyword (graph, "
            "flowchart, sequenceDiagram, classDiagram, stateDiagram-v2, "
            "erDiagram, gantt, mindmap, ...)."
        )

    if render_type == "html":
        if is_valid_html_document(text):
            return True, ""
        return False, "Output does not look like a renderable HTML document."

    # Unknown render types are not gated.
    return True, ""


__all__ = [
    "build_fallback_html",
    "extract_code_block",
    "extract_json_object",
    "is_valid_html_document",
    "validate_self_contained_html",
    "validate_visualization",
]
