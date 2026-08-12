from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class OrderRequest:
    trading_symbol: str
    quantity: int
    side: str
    order_reference_id: str


@dataclass(frozen=True, slots=True)
class OrderResult:
    broker_order_id: str
    status: str
    raw: dict[str, object]


class ExecutionBroker(Protocol):
    def place_market_order(self, request: OrderRequest) -> OrderResult: ...
