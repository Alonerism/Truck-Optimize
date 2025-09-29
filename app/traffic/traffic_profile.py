"""Traffic profile helpers for offline time matrix adjustments."""

from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Tuple

from ..util.time_utils import parse_hhmm, in_window


def static_factor(start_min_of_day: int, static_profile: List[Dict]) -> float:
    """Return traffic factor from static profile bins.

    Accepts bins as plain dicts or Pydantic models with attributes 'window' and 'factor'.
    """
    for b in static_profile:
        # Support both dict and object-style bins
        window = b["window"] if isinstance(b, dict) else getattr(b, "window", None)
        factor = b.get("factor", 1.0) if isinstance(b, dict) else getattr(b, "factor", 1.0)
        if window and in_window(start_min_of_day, window):
            return float(factor)
    return 1.0  # default


def grid_cell(lat: float, lon: float, grid_km: float) -> Tuple[int, int]:
    # Roughly 1 deg lat ~ 111 km, 1 deg lon ~ 111 km*cos(lat)  # coarse
    scale = grid_km / 111.0  # deg per cell
    gx = int(lon / scale)  # quantize lon
    gy = int(lat / scale)  # quantize lat
    return gx, gy


def learned_factor(start_dt: datetime, o_cell: Tuple[int, int], d_cell: Tuple[int, int], store_path: str | None = None) -> float:
    # Placeholder: look up from CSV/DB in future; for now return neutral 1.0
    # Could be keyed by (dow,hour,o_cell,d_cell) -> ratio
    return 1.0


def load_learned(csv_path: str) -> Dict[int, float]:
    """Load a minimal learned ratio table from csv into a mapping.

    Expected columns: time_bin, ratio. Aggregates by mean per bin.
    This is a placeholder for a richer (time, origin_cell, dest_cell) keyed model.
    """
    try:
        pd = __import__("pandas")
    except Exception:
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "time_bin" not in df.columns or "ratio" not in df.columns:
            return {}
        agg = df.groupby("time_bin")["ratio"].mean()
        return {int(k): float(v) for k, v in agg.to_dict().items()}
    except Exception:
        return {}


def factor_for_leg(start_dt: datetime, o_lat: float, o_lon: float, d_lat: float, d_lon: float, params) -> float:
    mode = getattr(getattr(params, "traffic", {}), "profile_mode", "static")  # 'static'|'learned'
    if mode == "learned":  # dispatch learned
        grid_km = getattr(params.traffic, "grid_size_km", 5)
        oc = grid_cell(o_lat, o_lon, grid_km)
        dc = grid_cell(d_lat, d_lon, grid_km)
        return learned_factor(start_dt, oc, dc, None)
    # static mode
    min_of_day = start_dt.hour * 60 + start_dt.minute  # minutes since midnight
    profile = getattr(params.traffic, "static_profile", [])
    return static_factor(min_of_day, profile)
