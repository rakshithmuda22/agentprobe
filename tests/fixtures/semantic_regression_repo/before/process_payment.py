"""Payment processor — before version (raises on failure)."""


class PaymentError(Exception):
    pass


def process_payment(amount, currency="USD"):
    """Process a payment. Raises PaymentError on invalid amount."""
    if amount <= 0:
        raise PaymentError(f"Invalid amount: {amount}")
    if not isinstance(amount, (int, float)):
        raise PaymentError(f"Amount must be numeric, got {type(amount)}")
    return {"status": "ok", "amount": amount, "currency": currency}
