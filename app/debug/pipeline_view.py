"""Printable debug pipeline for offline -> Google flow.

Produces concise step-by-step prints and small CSV snapshots to artifacts/debug.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from ..distance import Coordinates
from ..solver_ortools import ORToolsSolver
from ..integrations.google_directions import fetch_route_overview


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _fmt_dt(dt: datetime | None) -> str:
    return dt.strftime("%H:%M") if dt else "-"


async def run_debug_pipeline(service, date: str, limit: int = 5, two_pass: bool = False):
    print(f"[DEBUG] Pipeline for {date} (limit={limit}, two_pass={two_pass})")

    # 1) Load data
    trucks = service.repo.get_trucks()
    jobs = service.repo.get_jobs_by_date(date)
    locations = service.repo.get_locations()
    print(f"[DEBUG] Data: trucks={len(trucks)} jobs={len(jobs)} locations={len(locations)}")

    if not jobs:
        print("[DEBUG] No jobs to optimize.")
        return

    # 2) Depot coords and workday start
    depot_address = service.config.depot.address
    depot_coords = (await service.distance_provider.geocode_locations([depot_address]))[depot_address]
    if not depot_coords:
        raise RuntimeError(f"Could not geocode depot: {depot_address}")
    workday_start = datetime.fromisoformat(f"{date}T{service.config.depot.workday_window.start}:00")
    print(f"[DEBUG] Depot at {depot_coords.lat:.5f},{depot_coords.lon:.5f} | workday_start={workday_start}")

    # 3) Tiny time-matrix preview (first N locations)
    preview_dir = "artifacts/debug"
    _ensure_dir(preview_dir)
    first_jobs = jobs[:limit]
    coords: List[Tuple[float, float]] = [(depot_coords.lat, depot_coords.lon)] + [
        (j.location.lat or depot_coords.lat, j.location.lon or depot_coords.lon) for j in first_jobs
    ]
    try:
        from ..util.haversine import km, minutes_with_traffic
        from ..traffic.traffic_profile import factor_for_leg
        speed = getattr(service.config.solver, "haversine_speed_kmph", 35)
        n = len(coords)
        rows = []
        for i in range(n):
            row = []
            for k in range(n):
                if i == k:
                    row.append(0)
                else:
                    dist_km = km(coords[i][0], coords[i][1], coords[k][0], coords[k][1])
                    factor = factor_for_leg(workday_start, coords[i][0], coords[i][1], coords[k][0], coords[k][1], service.config)
                    row.append(int(round(minutes_with_traffic(dist_km, speed, factor))))
            rows.append(row)
        # Write CSV
        out_csv = os.path.join(preview_dir, f"time_matrix_preview_{date}.csv")
        with open(out_csv, "w") as f:
            headers = ["node_0(depot)"] + [f"job_{j.id}" for j in first_jobs]
            f.write(",".join([" "] + headers) + "\n")
            for idx, r in enumerate(rows):
                name = headers[idx] if idx < len(headers) else f"node_{idx}"
                f.write(",".join([name] + [str(x) for x in r]) + "\n")
        print(f"[DEBUG] Time-matrix preview saved: {out_csv}")
    except Exception as e:
        print(f"[DEBUG] Skipped matrix preview: {e}")

    # 4) Solve with OR-Tools (offline)
    # Force OR-Tools for this debug run, optionally two-pass
    orig_two_pass = bool(getattr(service.config.solver, "two_pass_traffic", False))
    try:
        service.config.solver.two_pass_traffic = bool(two_pass)
    except Exception:
        pass
    solver = ORToolsSolver(service.config)
    sol = solver.solve(
        trucks=trucks,
        jobs=jobs,
        job_items_map={j.id: j.job_items for j in jobs},
        locations=locations,
        distance_matrix=None,
        depot_coords=Coordinates(lat=depot_coords.lat, lon=depot_coords.lon),
        workday_start=workday_start,
    )
    print(f"[DEBUG] Solve: routes={len(sol.routes)} unassigned={len(sol.unassigned_jobs)} time={sol.computation_time_seconds:.2f}s")

    # 5) Print top-N per-route
    for r in sol.routes[:limit]:
        off_total = r.total_drive_minutes + r.total_service_minutes
        print(f"  - {r.truck.name}: stops={len(r.assignments)} offline_total={off_total:.1f}m weight={r.total_weight_lb:.0f}lb overtime={r.overtime_minutes:.1f}m")
        for a in r.assignments[:limit]:
            print(
                f"      [{a.stop_order}] job={a.job.id} loc='{a.job.location.name}' arrive={_fmt_dt(a.estimated_arrival)} depart={_fmt_dt(a.estimated_departure)} drive={a.drive_minutes_from_previous:.1f}m serv={a.service_minutes:.1f}m"
            )

    # 6) Snapshot assignments CSV
    snap_csv = os.path.join(preview_dir, f"route_assignments_{date}.csv")
    with open(snap_csv, "w") as f:
        f.write("route,seq,job_id,location,arrive,depart,drive_min,service_min\n")
        for r in sol.routes:
            for a in r.assignments:
                f.write(
                    f"{r.truck.name},{a.stop_order},{a.job.id},\"{a.job.location.name}\",{_fmt_dt(a.estimated_arrival)},{_fmt_dt(a.estimated_departure)},{a.drive_minutes_from_previous:.1f},{a.service_minutes:.1f}\n"
                )
    print(f"[DEBUG] Assignments snapshot saved: {snap_csv}")

    # 7) Optional Google overview for first route
    if (not service.config.dev.mock_google_api) and service.settings.google_maps_api_key and sol.routes:
        r0 = sol.routes[0]
        ordered_coords = [(depot_coords.lat, depot_coords.lon)] + [
            (a.job.location.lat, a.job.location.lon) for a in r0.assignments
        ] + [(depot_coords.lat, depot_coords.lon)]
        dep_epoch = int(workday_start.timestamp())
        overview = await fetch_route_overview(
            service.settings.google_maps_api_key,
            ordered_coords,
            dep_epoch,
            traffic_model=service.config.google.traffic_model.lower(),
        )
        if overview:
            g_min = float(overview.get("duration_in_traffic_min", 0.0))
            off_total = r0.total_drive_minutes + r0.total_service_minutes
            delta = off_total - g_min
            print(f"[DEBUG] Google overview: offline={off_total:.1f}m google={g_min:.1f}m delta={delta:+.1f}m")

    # restore two-pass
    try:
        service.config.solver.two_pass_traffic = orig_two_pass
    except Exception:
        pass

    print("[DEBUG] Pipeline done.")
