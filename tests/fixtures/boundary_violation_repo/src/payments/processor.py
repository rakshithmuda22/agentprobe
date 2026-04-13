"""Payment processor — intentionally imports from analytics (FORBIDDEN)."""

from src.analytics.tracker import track_event
from src.shared.utils import validate


def process_payment(amount: float, currency: str) -> dict:
    track_event("payment_started")
    validate(amount)
    return {"status": "ok", "amount": amount}
