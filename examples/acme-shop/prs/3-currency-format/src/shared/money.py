"""Money helpers shared by every module."""

from decimal import Decimal


def to_cents(amount: Decimal) -> int:
    return int(amount * 100)


def format_cents(cents: int, currency: str = "USD") -> str:
    return f"{cents / 100:,.2f} {currency}"
