"""Customer receipts."""

from src.shared.money import format_cents


def render_receipt(result: dict) -> str:
    return f"Charged {format_cents(result['cents'], result['currency'])}"
