"""Tiny haversine helpers for offline VRP matrices."""

from math import radians, sin, cos, asin, sqrt


def km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    R = 6371.0  # Earth radius in km  # simple constant
    lat1, lon1, lat2, lon2 = map(radians, [a_lat, a_lon, b_lat, b_lon])  # deg->rad
    dlat = lat2 - lat1  # delta lat
    dlon = lon2 - lon1  # delta lon
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2  # haversine
    return 2 * R * asin(sqrt(h))  # arc length in km


def minutes_from_km(distance_km: float, speed_kmph: float) -> float:
    if speed_kmph <= 0:
        return 0.0  # guard
    return (distance_km / speed_kmph) * 60.0  # minutes


def minutes_with_traffic(km_val: float, base_speed_kmph: float, factor: float) -> float:
    if base_speed_kmph <= 0:
        return 0.0  # guard
    eff_speed = base_speed_kmph * max(factor, 0.01)  # avoid zero
    return (km_val / eff_speed) * 60.0  # minutes adjusted
