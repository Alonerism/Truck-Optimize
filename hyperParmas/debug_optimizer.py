#!/usr/bin/env python
"""
Lightweight debug harness for route assignments.

Usage:
  poetry run python hypeP/debug_optimizer.py --date 2025-09-18

It will:
- invoke the optimizer for the given date (import data first if needed),
- print a simple table of truck -> ordered stops,
- write hypeP/debug_output_<date>.csv for easy diffing across runs.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import asyncio

from app.service import TruckOptimizerService
from app.schemas import OptimizeRequest


def format_row(truck: str, order: int, loc_name: str, address: str) -> list[str]:
    return [truck, str(order), loc_name or "", address or ""]


async def run(date: str) -> int:
    svc = TruckOptimizerService()

    # Optimize for the given date (do not prompt for overtime changes here)
    # Use a valid policy; choose 'overtime' to accept overtime without prompting
    req = OptimizeRequest(date=date, auto="overtime")
    result = await svc.optimize_routes(req)

    # Print a simple table: truck -> ordered stops
    print("\nRoutes:")
    rows: list[list[str]] = [["truck", "stop_order", "location", "address"]]
    for route in result.routes:
        truck_name = route.truck.name
        if not route.stops:
            print(f"- {truck_name}: <no stops>")
            continue
        print(f"- {truck_name}:")
        for s in sorted(route.stops, key=lambda x: x.stop_order):
            job = s.job
            loc = job.location
            order = s.stop_order
            # Pydantic response models are attribute-based, not subscriptable
            print(f"   {order:>2}  {loc.name}  |  {loc.address}")
            rows.append(format_row(truck_name, order, loc.name, loc.address))

    # Save CSV for diffing
    out_dir = Path("hypeP")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"debug_output_{date}.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")

    await svc.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug route optimizer")
    parser.add_argument("--date", required=True, help="Date YYYY-MM-DD to optimize")
    args = parser.parse_args()
    asyncio.run(run(args.date))


if __name__ == "__main__":
    main()
