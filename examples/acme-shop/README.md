# acme-shop: sample repo for the end-to-end test

`base/` is the main branch; each folder in `prs/` is a pull request (files
overlaid on `base/`). `scripts/e2e_action.py` turns this into a real git
repository and runs the AgentProbe action against each PR.

| PR | What it does | Expected |
|----|--------------|----------|
| 1-refund-tracking | `payments` imports `analytics` (forbidden) + camelCase `issueRefund` | BLOCK, exit 1 |
| 2-lenient-charge | `charge()` now returns `None` instead of raising + camelCase `formatReceipt` | WARN, exit 0 |
| 3-currency-format | new helper in `shared`, used from `payments` | PASS, exit 0 |
