"""Payment processor — after version (returns None on failure instead of raising)."""


class PaymentError(Exception):
    pass


def process_payment(amount, currency="USD"):
    """Process a payment. Returns None on invalid amount."""
    if amount <= 0:
        return None
    if not isinstance(amount, (int, float)):
        return None
    return {"status": "ok", "amount": amount, "currency": currency}
