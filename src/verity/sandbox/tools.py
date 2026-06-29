"""Sandbox tool extensibility — the harness-bound side of a domain's tools (ROADMAP 5.4, issue #6).

A domain operation is typed once as an ``OperationSignature`` and bound to a *propose* tool by the
sandbox adapter; this module is the parallel for **non-propose, executable** tools the agent may use
while it works (read a PDF, …). The control plane / task wiring declares **which** tools a sandbox
gets (by name); the adapter resolves the names to plain callables and binds them — langchain derives
each tool's schema from its type hints + docstring. Names (not callables) cross the container
boundary via ``CycleInput.tool_names``; ``container_entry`` resolves them from this registry there.

Tools are plain functions on purpose: framework-agnostic, and the same registry serves the
in-process and container drivers.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["SANDBOX_TOOL_REGISTRY", "read_pdf", "resolve_tools", "UnknownToolError"]


class UnknownToolError(KeyError):
    """Raised when a task names a sandbox tool that is not registered."""


def read_pdf(path: str) -> str:
    """Extract the text of a PDF file in the workspace.

    Args:
        path: Path to the PDF file (relative to the working directory, or absolute within it).

    Returns:
        The text content of the PDF, with one page per block separated by blank lines.
    """
    from pypdf import PdfReader  # lazy: only the agent's runtime needs pypdf, not callers/registry

    reader = PdfReader(path)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


# The available non-propose sandbox tools, keyed by the name a task selects.
SANDBOX_TOOL_REGISTRY: dict[str, Callable[..., object]] = {
    "read_pdf": read_pdf,
}


def resolve_tools(names: tuple[str, ...]) -> list[Callable[..., object]]:
    """Resolve selected tool names to their callables (raises on an unknown name)."""
    resolved: list[Callable[..., object]] = []
    for name in names:
        try:
            resolved.append(SANDBOX_TOOL_REGISTRY[name])
        except KeyError:
            raise UnknownToolError(
                f"unknown sandbox tool: {name!r} (registered: {sorted(SANDBOX_TOOL_REGISTRY)})"
            ) from None
    return resolved
