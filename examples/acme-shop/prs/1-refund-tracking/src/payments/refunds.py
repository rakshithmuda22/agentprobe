"""Refunds, with analytics tracking bolted on."""

from src.analytics.tracker import track_event
from src.shared.money import to_cents


def issueRefund(amount, reason):
    track_event("refund_issued")
    return {"refunded_cents": to_cents(amount), "reason": reason}
