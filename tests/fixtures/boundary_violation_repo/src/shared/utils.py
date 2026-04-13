"""Shared utilities."""

from src.types.common import Amount


def validate(amount: float) -> bool:
    return amount > 0
