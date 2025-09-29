"""Minimal wrapper to fetch Directions overview with traffic after solve."""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


async def fetch_route_overview(
    api_key: str,
    ordered_coords: List[Tuple[float, float]],  # [(lat, lon), ...]
    departure_epoch: int,
    traffic_model: str = "best_guess",
) -> Optional[Dict[str, Any]]:
    # Build a single Directions request per truck, optimize:false  # keep stop order
    if not api_key:
        logger.warning("No API key for Directions; skipping")
        return None
    if len(ordered_coords) < 2:
        return None

    base = "https://maps.googleapis.com/maps/api/directions/json"
    origin = f"{ordered_coords[0][0]},{ordered_coords[0][1]}"
    destination = f"{ordered_coords[-1][0]},{ordered_coords[-1][1]}"
    waypoints = "|".join([f"{lat},{lon}" for (lat, lon) in ordered_coords[1:-1]])

    params = {
        "origin": origin,
        "destination": destination,
        "waypoints": waypoints,
        "departure_time": departure_epoch,
        "traffic_model": traffic_model,
        "key": api_key,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(base, params=params)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK":
            logger.warning(f"Directions status={data.get('status')} msg={data.get('error_message')}")
            return None

        route = data["routes"][0]
        legs = route["legs"]
        total_secs = sum(leg.get("duration_in_traffic", leg["duration"]) ["value"] for leg in legs)
        total_meters = sum(leg["distance"]["value"] for leg in legs)
        polyline = route.get("overview_polyline", {}).get("points")
        return {
            "duration_in_traffic_min": total_secs / 60.0,
            "distance_km": total_meters / 1000.0,
            "polyline": polyline,
        }


async def audit_route(
    api_key: str,
    ordered_coords: List[Tuple[float, float]],  # [(lat, lon), ...]
    departure_epoch: int,
    traffic_model: str = "best_guess",
) -> Optional[List[Dict[str, Any]]]:
    # Returns per-leg audit with Google minutes and km for the given fixed order
    if not api_key or len(ordered_coords) < 2:
        return None
    base = "https://maps.googleapis.com/maps/api/directions/json"
    origin = f"{ordered_coords[0][0]},{ordered_coords[0][1]}"
    destination = f"{ordered_coords[-1][0]},{ordered_coords[-1][1]}"
    waypoints = "|".join([f"{lat},{lon}" for (lat, lon) in ordered_coords[1:-1]])
    params = {
        "origin": origin,
        "destination": destination,
        "waypoints": waypoints,
        "departure_time": departure_epoch,
        "traffic_model": traffic_model,
        "key": api_key,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(base, params=params)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK":
            logger.warning(f"Directions status={data.get('status')} msg={data.get('error_message')}")
            return None
        route = data["routes"][0]
        legs = route["legs"]
        out = []
        for i, leg in enumerate(legs):  # enumerate legs to attach indices
            secs = leg.get("duration_in_traffic", leg["duration"]) ["value"]
            km = leg["distance"]["value"] / 1000.0
            poly = leg.get("steps", [{}])[-1].get("polyline", {}).get("points") if leg.get("steps") else None
            out.append({
                "o_seq": i,  # origin sequence index
                "d_seq": i + 1,  # dest sequence index
                "google_minutes": secs / 60.0,
                "google_km": km,
                "polyline": poly,
            })
        return out
