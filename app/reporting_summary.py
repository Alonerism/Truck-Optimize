"""Simple reporting helpers for route summaries and tuning hints."""

from __future__ import annotations

from typing import Optional, Dict, Any


def route_summary_df(solution, google_totals: Optional[Dict[str, float]] = None):
    pd = __import__("pandas")
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


def print_run_summary(solution, google_totals: Optional[Dict[str, float]], penalties: Optional[Dict[str, int]] = None) -> None:
    """Print concise run summary per route + knobs."""
    for r in solution.routes:
        rid = r.truck.name
        off_total = r.total_drive_minutes + r.total_service_minutes
        g_total = google_totals.get(rid) if google_totals else None
        delta = (off_total - g_total) if g_total is not None else None
        print(
            f"[SUMMARY] {rid} stops={len(r.assignments)} offline={off_total:.1f}m"
            + (f" google={g_total:.1f}m delta={delta:+.1f}m" if g_total is not None else "")
            + f" load={r.total_weight_lb:.0f}lb on_time={'Y' if r.overtime_minutes<=0 else 'N'}"
        )
    if penalties:
        print(f"Knobs: priority_weight={penalties.get('priority_weight')} disjunction_base={penalties.get('disjunction_base')}")
