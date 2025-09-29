"""Build a driver-friendly Google Maps link preserving stop order."""

from urllib.parse import urlencode, quote
from typing import List


def build_driver_link(ordered_addrs: List[str]) -> str:
    if not ordered_addrs:
        return ""
    origin = ordered_addrs[0]
    destination = ordered_addrs[-1]
    waypoints = "|".join(quote(a) for a in ordered_addrs[1:-1])
    params = {
        "api": 1,
        "origin": origin,
        "destination": destination,
        "waypoints": waypoints,
        "travelmode": "driving",
    }
    return "https://www.google.com/maps/dir/?" + urlencode(params)
