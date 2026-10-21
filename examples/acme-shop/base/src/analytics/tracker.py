"""Product analytics."""

EVENTS: list[str] = []


def track_event(name: str) -> None:
    EVENTS.append(name)
