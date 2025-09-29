"""Penalty helpers for solver modeling."""


def drop_penalty(priority: int, base: int, weight: int) -> int:
    return int(base + (priority * weight))  # simple linear
