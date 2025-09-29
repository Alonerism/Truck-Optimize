"""Simple reporting helpers for route summaries and tuning hints."""

from __future__ import annotations

from typing import Optional, Dict, Any

import pandas as pd


def route_summary_df(solution, google_totals: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    rows = []
    for route in solution.routes:
        if not route.assignments:
            continue
        rid = route.truck.name
        rows.append({
            "route_id": rid,
            "stops": len(route.assignments),
            "drive_min": route.total_drive_minutes,
            "service_min": route.total_service_minutes,
            "overtime_min": route.overtime_minutes,
            "offline_total_min": route.total_drive_minutes + route.total_service_minutes,
            "google_total_min": (google_totals.get(rid) if google_totals else None),
        })
    return pd.DataFrame(rows)


def print_priority_tradeoff_hint(disjunction_base: int, priority_weight: int) -> None:
    msg = (
        f"Drop penalty = base({disjunction_base}) + priority*weight({priority_weight}). "
        "Higher weight keeps high-priority jobs but may increase path length."
    )
    print(msg)
