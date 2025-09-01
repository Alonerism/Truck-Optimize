"""
Google Maps URL builder for route visualization.
Handles waypoint limits and creates segmented URLs for long routes.
"""

import logging
from typing import List, Dict, Any
from urllib.parse import urlencode

from .models import Location, MapsUrlResponse
from .distance import Coordinates
from .schemas import AppConfig
from .solver_greedy import TruckRoute


logger = logging.getLogger(__name__)


class GoogleMapsUrlBuilder:
    """Builder for Google Maps URLs with route waypoints."""
    
    def __init__(self, config: AppConfig):
        """Initialize URL builder with configuration."""
        self.config = config
        self.base_url = "https://www.google.com/maps/dir"
        
    def build_coordinate_urls(
        self,
        coordinates: List[Coordinates],
        truck_name: str
    ) -> MapsUrlResponse:
        """
        Build Google Maps URLs from a list of coordinates.
        Segments into multiple URLs if waypoint limit is exceeded.
        
        Args:
            coordinates: List of coordinates to visit in order
            truck_name: Name of the truck for the response
            
        Returns:
            MapsUrlResponse with segmented URLs
        """
        if len(coordinates) < 2:
            # Not enough points for a route
            return MapsUrlResponse(
                truck_name=truck_name,
                urls=[],
                total_stops=0
            )
        
        max_waypoints = self.config.google.maps.segment_max_waypoints
        
        if len(coordinates) <= max_waypoints:
            # Single URL
            url = self._build_route_url(coordinates)
            return MapsUrlResponse(
                truck_name=truck_name,
                urls=[url],
                total_stops=len(coordinates) - 2  # Exclude start/end depot
            )
        else:
            # Multiple segmented URLs
            urls = self._build_segmented_urls(coordinates, max_waypoints)
            return MapsUrlResponse(
                truck_name=truck_name,
                urls=urls,
                total_stops=len(coordinates) - 2  # Exclude start/end depot
            )
    
    def _build_route_url(self, coordinates: List[Coordinates]) -> str:
        """Build a Google Maps URL for a sequence of coordinates."""
        if len(coordinates) < 2:
            raise ValueError("Need at least 2 coordinates for a route")
        
        # Start and end points
        start = coordinates[0]
        end = coordinates[-1]
        
        # Waypoints (excluding start and end)
        waypoints = coordinates[1:-1]
        
        # Build URL path
        url_path = f"{start.lat},{start.lon}"
        
        # Add waypoints
        for waypoint in waypoints:
            url_path += f"/{waypoint.lat},{waypoint.lon}"
        
        # Add destination
        url_path += f"/{end.lat},{end.lon}"
        
        # Add query parameters
        params = {
            "travelmode": "driving",
            "dir_action": "navigate"
        }
        
        # Add avoid preferences if configured
        if self.config.google.maps.avoid:
            params["avoid"] = ",".join(self.config.google.maps.avoid)
        
        # Construct final URL
        if params:
            query_string = urlencode(params)
            return f"{self.base_url}/{url_path}?{query_string}"
        else:
            return f"{self.base_url}/{url_path}"
    
    def _build_segmented_urls(
        self, 
        coordinates: List[Coordinates], 
        max_waypoints: int
    ) -> List[str]:
        """Build multiple URLs for routes that exceed waypoint limits."""
        urls = []
        
        # Calculate segment size (excluding start/end points)
        segment_size = max_waypoints - 1  # Reserve 1 for start/end overlap
        
        i = 0
        while i < len(coordinates) - 1:
            # Determine segment end
            segment_end = min(i + segment_size + 1, len(coordinates) - 1)
            
            # Create segment
            segment_coords = coordinates[i:segment_end + 1]
            
            # Ensure we have at least start and end
            if len(segment_coords) >= 2:
                urls.append(self._build_route_url(segment_coords))
            
            # Move to next segment (overlap by 1 point)
            i = segment_end
            
            # Avoid infinite loop
            if i >= len(coordinates) - 1:
                break
        
        logger.info(f"Segmented route into {len(urls)} URLs")
        return urls
