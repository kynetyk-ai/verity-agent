"""Token -> dollar pricing for the eval harness (ROADMAP Phase 6.3).

A small, overridable pricing table and a nullable ``cost_usd``. Cost is deliberately **outside** the
control plane: the CP records token counts in the ``RunReport``; this module prices them. Every path
is nullable (``None`` when telemetry / model / tokens / price are unknown), like the report's own
fields — the consumer never gets a fabricated number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["ModelPrice", "PricingTable", "DEFAULT_PRICING", "cost_usd"]


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per **million** tokens, input and output."""

    input_per_1m: float
    output_per_1m: float


PricingTable = Mapping[str, ModelPrice]

# Keyed by the model name telemetry reports (``response_metadata.model_name``). Rates drift and a
# served model may report an arbitrary/empty name — so this is a *default convenience* only; a
# BenchmarkArm can override with an explicit price. Update against current published rates.
# Local / open-weights models are run at zero marginal API cost.
DEFAULT_PRICING: dict[str, ModelPrice] = {
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0),
    "claude-opus-4-8": ModelPrice(15.0, 75.0),
    "gpt-4o": ModelPrice(2.5, 10.0),
    "gpt-4o-mini": ModelPrice(0.15, 0.6),
}


def cost_usd(
    telemetry: Mapping[str, Any] | None,
    pricing: PricingTable,
    *,
    model_key: str | None = None,
    price: ModelPrice | None = None,
) -> float | None:
    """Dollar cost of one run's token usage, or ``None`` when it can't be known.

    Resolution order for the price: an explicit ``price`` wins; else look up ``model_key`` (or, if
    unset, ``telemetry["model"]``) in ``pricing``. Returns ``None`` if telemetry is absent, no price
    can be resolved, or token counts are missing — never a fabricated zero. Local arms that pass a
    ``ModelPrice(0, 0)`` (or a table entry) correctly yield ``0.0``.
    """
    if telemetry is None:
        return None
    if price is None:
        key = model_key or telemetry.get("model")
        if not key:
            return None
        price = pricing.get(key)
        if price is None:
            return None
    inp, out = telemetry.get("input_tokens"), telemetry.get("output_tokens")
    if inp is None or out is None:
        return None
    return round(float(inp) / 1e6 * price.input_per_1m + float(out) / 1e6 * price.output_per_1m, 6)
