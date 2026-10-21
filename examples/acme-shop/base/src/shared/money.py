"""Money helpers shared by every module."""

from decimal import Decimal


def to_cents(amount: Decimal) -> int:
    return int(amount * 100)
