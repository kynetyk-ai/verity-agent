"""Sandbox tools & the #6 ToolBinding seam (ROADMAP 5.4) — needs ``pypdf`` (the ``sandbox`` extra).

Proves the registry resolves selected tool names to callables (and refuses unknown ones), and that
``read_pdf`` actually extracts text from a PDF. The agent-binding side (the built agent exposes a
selected tool) is asserted in ``test_sandbox_deepagents.py``; a `@live` run uses it.
"""

from __future__ import annotations

import pytest

from verity.sandbox.tools import SANDBOX_TOOL_REGISTRY, UnknownToolError, resolve_tools

pytest.importorskip("pypdf")

from verity.sandbox.tools import read_pdf  # noqa: E402


def _minimal_pdf(text: str) -> bytes:
    """A tiny, valid single-page PDF whose content stream draws ``text`` (no fixture needed)."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        None,  # the content stream, filled below (needs its own length)
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = b"BT /F1 24 Tf 20 60 Td (" + text.encode("latin-1") + b") Tj ET"
    objects[3] = (
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(i).encode() + b" 0 obj\n" + body + b"\nendobj\n"  # type: ignore[operator]
    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\n"
    out += b"startxref\n" + str(xref_at).encode() + b"\n%%EOF"
    return bytes(out)


def test_read_pdf_extracts_text(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(_minimal_pdf("Hello Verity"))
    assert "Hello Verity" in read_pdf(str(pdf))


def test_registry_resolves_selected_tools() -> None:
    assert "read_pdf" in SANDBOX_TOOL_REGISTRY
    assert resolve_tools(("read_pdf",)) == [read_pdf]
    assert resolve_tools(()) == []


def test_unknown_tool_is_refused() -> None:
    with pytest.raises(UnknownToolError, match="unknown sandbox tool"):
        resolve_tools(("read_pdf", "does_not_exist"))
