"""Compare offline vs Google ETAs and save artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as _pd  # type: ignore


def compare_offline_vs_google(route_legs: List[Dict], offline_minutes: List[float], route_id: str):
    pd = __import__("pandas")
    rows = []
    for i, leg in enumerate(route_legs):
        off = float(offline_minutes[i]) if i < len(offline_minutes) else None  # offline min
        g = float(leg.get("google_minutes")) if leg else None  # google min
        delta = (off - g) if (off is not None and g is not None) else None  # bias
        pct = (abs(delta) / g * 100.0) if (delta is not None and g and g > 0) else None  # % err
        rows.append({
            "route_id": route_id,
            "seq": i,
            "offline_min": off,
            "google_min": g,
            "delta_min": delta,
            "pct_err": pct,
        })
    df = pd.DataFrame(rows)
    # Stats
    valid = df.dropna()
    stats = {}
    if not valid.empty:
        stats = {
            "bias_mean": float(valid["delta_min"].mean()),
            "mape": float((valid["pct_err"].abs().mean()) if not valid["pct_err"].isna().all() else 0.0),
            "p50_abs": float(valid["delta_min"].abs().quantile(0.5)),
            "p90_abs": float(valid["delta_min"].abs().quantile(0.9)),
        }
    return df, stats


def save_audit(df, stats: Dict, out_dir: str, date: str) -> Dict[str, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = str(Path(out_dir) / f"eta_audit_{date}.csv")
    json_path = str(Path(out_dir) / f"eta_audit_{date}_summary.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    return {"csv": csv_path, "json": json_path}


def append_learned_ratios(df, out_csv: str) -> str:
    """Append simple learned ratios to a CSV for later use.

    Expects df with columns: route_id, seq, offline_min, google_min.
    Writes rows: time_bin, ratio where time_bin is 30-min bucket of seq order for simplicity.
    This is a placeholder structure until a richer (o_cell,d_cell,time) model is added.
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    # Compute ratio per-leg where both present
    sub = df.dropna(subset=["offline_min", "google_min"]).copy()
    if sub.empty:
        return out_csv
    sub["ratio"] = (sub["google_min"].astype(float) / sub["offline_min"].astype(float)).clip(lower=0.1, upper=10.0)
    # Bucket by 30-min of seq as a naive proxy (placeholder)
    sub["time_bin"] = (sub["seq"].astype(int) // 1)  # one-bin per step for now
    cols = ["time_bin", "ratio"]
    mode = "a" if Path(out_csv).exists() else "w"
    header = not Path(out_csv).exists()
    sub[cols].to_csv(out_csv, mode=mode, header=header, index=False)
    return out_csv
