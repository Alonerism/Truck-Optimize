"""
Google Maps integration for geocoding, routing, and distance calculations.
Handles rate limiting, retries, and traffic-aware route optimization.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import httpx
from .schemas import AppConfig, Settings
from .util.haversine import km, minutes_with_traffic
from .traffic.traffic_profile import factor_for_leg


logger = logging.getLogger(__name__)


@dataclass
class Coordinates:
    """Geographic coordinates."""
    lat: float
    lon: float


@dataclass
class RouteMatrix:
    """Distance and duration matrix between locations."""
    origins: List[Coordinates]
    destinations: List[Coordinates]
    durations_minutes: List[List[float]]  # [origin_idx][dest_idx] = minutes
    distances_meters: List[List[float]]   # [origin_idx][dest_idx] = meters
    
    def get_duration(self, origin_idx: int, dest_idx: int) -> float:
        """Get duration in minutes between two points."""
        return self.durations_minutes[origin_idx][dest_idx]
    
    def get_distance(self, origin_idx: int, dest_idx: int) -> float:
        """Get distance in meters between two points."""
        return self.distances_meters[origin_idx][dest_idx]


class GoogleMapsClient:
    """Google Maps API client with rate limiting and retry logic."""
    
    def __init__(self, config: AppConfig, settings: Settings):
        """Initialize Google Maps client."""
        self.config = config
        self.api_key = settings.google_maps_api_key
        self.base_url = "https://maps.googleapis.com/maps/api"
        
        # Rate limiting
        self.last_request_time = 0.0
        self.min_request_interval = 1.0 / config.google.rate_limit_requests_per_second
        
        # HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)
        
        if not self.api_key:
            logger.warning("Google Maps API key not configured - using mock responses")
    
    async def _rate_limited_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a rate-limited HTTP request with retries."""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        # Add API key to params
        params["key"] = self.api_key
        
        # Retry logic
        for attempt in range(self.config.google.max_retries):
            try:
                self.last_request_time = time.time()
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Check Google API status
                if data.get("status") == "OK":
                    return data
                elif data.get("status") in ["ZERO_RESULTS", "NOT_FOUND"]:
                    logger.warning(f"Google API returned {data.get('status')}: {data.get('error_message', '')}")
                    return data
                else:
                    raise Exception(f"Google API error: {data.get('status')} - {data.get('error_message', '')}")
                    
            except Exception as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.config.google.max_retries - 1:
                    await asyncio.sleep(self.config.google.retry_delay_seconds * (2 ** attempt))
                else:
                    raise
    
    async def geocode_address(self, address: str) -> Optional[Coordinates]:
        """Geocode an address to coordinates."""
        if self.config.dev.mock_google_api:
            return self._mock_geocode(address)
        
        if not self.api_key:
            logger.error("Cannot geocode without API key")
            return None
        
        url = f"{self.base_url}/geocode/json"
        params = {"address": address}
        
        try:
            data = await self._rate_limited_request(url, params)
            
            if data.get("status") == "OK" and data.get("results"):
                location = data["results"][0]["geometry"]["location"]
                return Coordinates(lat=location["lat"], lon=location["lng"])
            else:
                logger.error(f"Geocoding failed for '{address}': {data.get('status')}")
                return None
                
        except Exception as e:
            logger.error(f"Geocoding error for '{address}': {e}")
            return None
    
    async def compute_route_matrix(
        self,
        origins: List[Coordinates],
        destinations: List[Coordinates],
        departure_time: Optional[datetime] = None
    ) -> Optional[RouteMatrix]:
        """
        Compute route matrix using Google Distance Matrix API.
        Uses traffic-aware routing with specified departure time.
        """
        if self.config.dev.mock_google_api:
            return self._mock_route_matrix(origins, destinations)
        
        if not self.api_key:
            logger.error("Cannot compute routes without API key")
            return None
        
        # Use Distance Matrix API for efficient bulk calculations
        url = f"{self.base_url}/distancematrix/json"
        
        # Format coordinates
        origins_str = "|".join([f"{coord.lat},{coord.lon}" for coord in origins])
        destinations_str = "|".join([f"{coord.lat},{coord.lon}" for coord in destinations])
        
        params = {
            "origins": origins_str,
            "destinations": destinations_str,
            "units": "metric",
            "traffic_model": self.config.google.traffic_model.lower(),
            "key": self.api_key
        }
        
        # Add departure time for traffic-aware routing
        if departure_time:
            # Convert to Unix timestamp
            params["departure_time"] = int(departure_time.timestamp())
        else:
            # Use current time + offset for traffic estimation
            offset_hours = self.config.google.departure_time_offset_hours
            estimated_departure = datetime.now() + timedelta(hours=offset_hours)
            params["departure_time"] = int(estimated_departure.timestamp())
        
        try:
            logger.info(f"Computing distance matrix for {len(origins)}x{len(destinations)} locations with traffic")
            response = await self.client.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "OK":
                logger.info("Distance matrix computation successful")
                return self._parse_distance_matrix(data, origins, destinations)
            else:
                logger.error(f"Distance Matrix API failed: {data.get('status')} - {data.get('error_message', '')}")
                return None
                
        except Exception as e:
            logger.error(f"Distance matrix request error: {e}")
            return None
    
    async def _compute_routes_matrix_v2(
        self,
        origins: List[Coordinates],
        destinations: List[Coordinates],
        departure_time: Optional[datetime] = None
    ) -> Optional[RouteMatrix]:
        """Compute matrix using the newer Routes API."""
        durations_minutes = []
        distances_meters = []
        
        # For each origin-destination pair, make a Routes API call
        for i, origin in enumerate(origins):
            duration_row = []
            distance_row = []
            
            for j, destination in enumerate(destinations):
                if i == j:
                    # Same location - zero time/distance
                    duration_row.append(0.0)
                    distance_row.append(0.0)
                    continue
                
                try:
                    duration, distance = await self._get_route_duration_distance(origin, destination, departure_time)
                    duration_row.append(duration)
                    distance_row.append(distance)
                except Exception as e:
                    logger.warning(f"Failed to get route from {i} to {j}: {e}")
                    duration_row.append(999999.0)  # Very high fallback
                    distance_row.append(999999.0)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            
            durations_minutes.append(duration_row)
            distances_meters.append(distance_row)
        
        return RouteMatrix(
            origins=origins,
            destinations=destinations,
            durations_minutes=durations_minutes,
            distances_meters=distances_meters
        )
    
    async def _get_route_duration_distance(
        self,
        origin: Coordinates,
        destination: Coordinates,
        departure_time: Optional[datetime] = None
    ) -> Tuple[float, float]:
        """Get duration and distance for a single route using Routes API."""
        url = 'https://routes.googleapis.com/directions/v2:computeRoutes'
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.api_key,
            'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters'
        }
        
        data = {
            'origin': {
                'location': {
                    'latLng': {
                        'latitude': origin.lat,
                        'longitude': origin.lon
                    }
                }
            },
            'destination': {
                'location': {
                    'latLng': {
                        'latitude': destination.lat,
                        'longitude': destination.lon
                    }
                }
            },
            'travelMode': 'DRIVE',
            'routingPreference': 'TRAFFIC_AWARE_OPTIMAL',  # Enable traffic-aware routing
            'computeAlternativeRoutes': False,
            'routeModifiers': {
                'avoidTolls': 'tolls' in self.config.google.maps.avoid,
                'avoidHighways': 'highways' in self.config.google.maps.avoid,
                'avoidFerries': 'ferries' in self.config.google.maps.avoid
            }
        }
        
        # Add departure time if provided
        if departure_time:
            data['departureTime'] = departure_time.isoformat() + 'Z'
        
        response = await self.client.post(url, json=data, headers=headers, timeout=30.0)
        response.raise_for_status()
        
        result = response.json()
        if 'routes' in result and len(result['routes']) > 0:
            route = result['routes'][0]
            duration_seconds = int(route['duration'].rstrip('s'))
            distance_meters = route['distanceMeters']
            return duration_seconds / 60.0, distance_meters
        else:
            raise Exception("No routes found")
    
    def _parse_distance_matrix(
        self,
        data: Dict[str, Any],
        origins: List[Coordinates],
        destinations: List[Coordinates]
    ) -> RouteMatrix:
        """Parse Google Distance Matrix API response."""
        rows = data["rows"]
        
        durations_minutes = []
        distances_meters = []
        
        for i, row in enumerate(rows):
            duration_row = []
            distance_row = []
            
            for j, element in enumerate(row["elements"]):
                if element["status"] == "OK":
                    # Duration in minutes (traffic-aware if departure time was specified)
                    duration_seconds = element["duration_in_traffic"]["value"] \
                        if "duration_in_traffic" in element \
                        else element["duration"]["value"]
                    duration_minutes = duration_seconds / 60.0
                    
                    # Distance in meters
                    distance_meters = element["distance"]["value"]
                else:
                    # Use fallback values for unreachable locations
                    duration_minutes = 999999.0  # Very high value
                    distance_meters = 999999.0
                    logger.warning(f"No route from origin {i} to destination {j}: {element['status']}")
                
                duration_row.append(duration_minutes)
                distance_row.append(distance_meters)
            
            durations_minutes.append(duration_row)
            distances_meters.append(distance_row)
        
        return RouteMatrix(
            origins=origins,
            destinations=destinations,
            durations_minutes=durations_minutes,
            distances_meters=distances_meters
        )
    
    def _mock_geocode(self, address: str) -> Coordinates:
        """Mock geocoding for development/testing."""
        # Simple hash-based mock coordinates in LA area
        hash_val = hash(address.lower()) % 10000
        lat = 34.0 + (hash_val % 100) / 1000.0  # 34.000 to 34.099
        lon = -118.5 + (hash_val % 500) / 1000.0  # -118.500 to -118.000
        
        logger.debug(f"Mock geocoding '{address}' -> ({lat:.6f}, {lon:.6f})")
        return Coordinates(lat=lat, lon=lon)
    
    def _mock_route_matrix(
        self,
        origins: List[Coordinates],
        destinations: List[Coordinates]
    ) -> RouteMatrix:
        """Mock route matrix for development/testing."""
        durations_minutes = []
        distances_meters = []
        
        for origin in origins:
            duration_row = []
            distance_row = []
            
            for dest in destinations:
                if origin == dest:
                    # Same location
                    duration_row.append(0.0)
                    distance_row.append(0.0)
                else:
                    # Estimate based on straight-line distance
                    # Rough calculation: 1 degree ≈ 111 km, average speed 50 km/h in city
                    lat_diff = abs(origin.lat - dest.lat)
                    lon_diff = abs(origin.lon - dest.lon)
                    straight_distance_km = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111
                    
                    # Add some randomness and city driving factor
                    drive_distance_km = straight_distance_km * 1.3  # City routing factor
                    drive_time_hours = drive_distance_km / 40  # 40 km/h average city speed
                    drive_time_minutes = drive_time_hours * 60
                    
                    duration_row.append(max(5.0, drive_time_minutes))  # Minimum 5 minutes
                    distance_row.append(drive_distance_km * 1000)  # Convert to meters
            
            durations_minutes.append(duration_row)
            distances_meters.append(distance_row)
        
        logger.debug(f"Mock route matrix: {len(origins)} origins × {len(destinations)} destinations")
        return RouteMatrix(
            origins=origins,
            destinations=destinations,
            durations_minutes=durations_minutes,
            distances_meters=distances_meters
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class DistanceProvider:
    """High-level interface for distance and routing operations."""
    
    def __init__(self, config: AppConfig, settings: Settings):
        """Initialize distance provider."""
        self.config = config
        self.google_client = GoogleMapsClient(config, settings)
        
        # Geocoding cache
        self._geocoding_cache: Dict[str, Coordinates] = {}
    
    async def geocode_locations(self, addresses: List[str]) -> Dict[str, Optional[Coordinates]]:
        """Geocode multiple addresses, using cache when available."""
        results = {}
        
        for address in addresses:
            if self.config.dev.cache_geocoding and address in self._geocoding_cache:
                results[address] = self._geocoding_cache[address]
                logger.debug(f"Using cached coordinates for '{address}'")
            else:
                coords = await self.google_client.geocode_address(address)
                results[address] = coords
                
                if coords and self.config.dev.cache_geocoding:
                    self._geocoding_cache[address] = coords
        
        return results
    
    async def compute_travel_matrix(
        self,
        locations: List[Coordinates],
        departure_time: Optional[datetime] = None
    ) -> Optional[RouteMatrix]:
        """Compute travel time matrix between all location pairs."""
        if not locations:
            return None
        
        return await self.google_client.compute_route_matrix(
            origins=locations,
            destinations=locations,
            departure_time=departure_time
        )
    
    def travel_time(self, origin: Coordinates, destination: Coordinates, depart_datetime: datetime) -> float:
        """Offline travel time estimator in minutes using haversine + traffic factor.

        Contract: travel_time(origin, destination, depart_datetime) -> minutes (float)
        This provides a single abstraction point to later swap to OSRM/GraphHopper/Google.
        """
        # Short-circuit zero distance
        if origin.lat == destination.lat and origin.lon == destination.lon:
            return 0.0
        speed = getattr(self.config.solver, "haversine_speed_kmph", 35)
        f = factor_for_leg(depart_datetime, origin.lat, origin.lon, destination.lat, destination.lon, self.config)
        d_km = km(origin.lat, origin.lon, destination.lat, destination.lon)
        return minutes_with_traffic(d_km, speed, f)
    
    async def close(self):
        """Clean up resources."""
        await self.google_client.close()
