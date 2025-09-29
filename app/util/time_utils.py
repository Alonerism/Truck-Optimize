"""Tiny time helpers for traffic binning and simple checks."""

from datetime import datetime, time


def parse_hhmm(s: str) -> int:
    t = time.fromisoformat(s)  # 'HH:MM' -> time
    return t.hour * 60 + t.minute  # minutes since midnight


def in_window(mins: int, window: str) -> bool:
    a, b = window.split("-")  # 'HH:MM-HH:MM'
    start = parse_hhmm(a)  # start minutes
    end = parse_hhmm(b)    # end minutes
    if start <= end:
        return start <= mins < end  # simple range
    return mins >= start or mins < end  # wrap-over-midnight


def minute_of_week(dt: datetime) -> int:
    return dt.weekday() * 24 * 60 + dt.hour * 60 + dt.minute  # 0..10079
