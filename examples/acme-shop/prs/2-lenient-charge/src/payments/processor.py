"""Charge customers."""

from decimal import Decimal

from src.shared.money import to_cents


class PaymentError(Exception):
    """Raised when a charge cannot be made."""


def charge(amount: int, currency: str = "USD") -> dict:
    """Charge `amount` (in whole units). Rejects non-positive amounts."""
    if amount <= 0:
        return None
    return {"status": "ok", "cents": to_cents(Decimal(amount)), "currency": currency}


def formatReceipt(result: dict) -> str:
    return f"{result['cents'] / 100:.2f} {result['currency']}"
