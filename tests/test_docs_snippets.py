"""The code blocks of the documentation that are marked ``<!-- runnable -->`` must run as written."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
BLOCK = re.compile(r"<!-- runnable -->\s*```python\n(.*?)```", re.S)


def _snippets():
    for page in sorted(DOCS.rglob("*.md")):
        for i, m in enumerate(BLOCK.finditer(page.read_text(encoding="utf-8"))):
            yield pytest.param(m.group(1), id=f"{page.relative_to(DOCS)}#{i}")


@pytest.mark.parametrize("code", list(_snippets()))
def test_documented_snippet_runs(code, capsys):
    exec(compile(code, "<doc snippet>", "exec"), {"__name__": "__doc__"})
    out = capsys.readouterr().out
    assert out.strip(), "a runnable snippet is expected to print something"


def test_there_are_runnable_snippets():
    assert list(_snippets())


def test_every_internal_doc_link_resolves():
    """``{doc}`target``` references point at pages that exist."""
    missing = []
    for page in sorted(DOCS.rglob("*.md")):
        for target in re.findall(r"\{doc\}`(?:[^`<]*<)?([^`>]+)>?`", page.read_text(encoding="utf-8")):
            base = page.parent
            path = (DOCS / target.lstrip("/")) if target.startswith("/") else (base / target)
            if not any(path.with_suffix(s).exists() for s in (".md", ".ipynb", ".rst")):
                missing.append(f"{page.relative_to(DOCS)} -> {target}")
    assert not missing, missing
