"""
Main service layer for truck optimization.
Orchestrates data import, optimization, and result generation.
"""

import logging
import yaml
from datetime import datetime, time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from .models import (
    Truck, Location, Item, Job, JobItem, 
    OptimizationResult, RouteResponse, JobResponse,
    OvertimeDecisionRequest, ActionType, ItemCategory
)
from .schemas import (
    AppConfig, Settings, ImportRequest, JobImportRow,
    ImportStatsResponse, OptimizeRequest
)
from .repo import DatabaseRepository
from .distance import DistanceProvider, Coordinates
from .solver_greedy import GreedySolver, Solution
from .url_builder import GoogleMapsUrlBuilder
from .constraints import ConstraintValidator


logger = logging.getLogger(__name__)


class TruckOptimizerService:
    """Main service for truck route optimization."""
    
    def __init__(self, config_path: str = "config/params.yaml"):
        """Initialize service with configuration."""
        self.config = self._load_config(config_path)
        self.settings = Settings()
        
        # Initialize components
        self.repo = DatabaseRepository(self.config)
        self.distance_provider = DistanceProvider(self.config, self.settings)
        self.url_builder = GoogleMapsUrlBuilder(self.config)
        self.validator = ConstraintValidator(self.config)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize database
        self.repo.create_tables()
        self._initialize_seed_data()
    
    def _load_config(self, config_path: str) -> AppConfig:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return AppConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format
        )
    
    def _initialize_seed_data(self) -> None:
        """Initialize database with seed data from configuration."""
        # Initialize trucks
        trucks_data = [truck.model_dump() for truck in self.config.fleet.trucks]
        self.repo.upsert_trucks(trucks_data)
        
        # Initialize items catalog
        items_data = [item.model_dump() for item in self.config.item_catalog]
        self.repo.upsert_items(items_data)
        
        logger.info("Seed data initialized")
    
    async def import_jobs(self, request: ImportRequest) -> ImportStatsResponse:
        """
        Import jobs from CSV/JSON data.
        
        Args:
            request: Import request with job data
            
        Returns:
            Import statistics
        """
        stats = ImportStatsResponse(
            locations_created=0,
            locations_updated=0,
            items_created=0,
            jobs_created=0,
            total_job_items=0,
            geocoding_requests=0
        )
        
        # Clear existing jobs if requested
        if request.clear_existing:
            deleted_count = self.repo.delete_jobs_by_date(request.date)
            logger.info(f"Cleared {deleted_count} existing jobs for {request.date}")
        
        # Process locations first
        unique_locations = set()
        for row in request.data:
            unique_locations.add(row.location)
        
        location_coords = await self._process_locations(unique_locations, stats)
        
        # Process items
        unique_items = set()
        for row in request.data:
            item_specs = self._parse_items_string(row.items)
            for item_name, _ in item_specs:
                unique_items.add(item_name)
        
        await self._process_items(unique_items, stats)
        
        # Process jobs
        for row in request.data:
            try:
                await self._create_job_from_row(row, request.date, stats)
            except Exception as e:
                error_msg = f"Failed to create job for location '{row.location}': {e}"
                logger.error(error_msg)
                stats.errors.append(error_msg)
        
        logger.info(f"Import completed: {stats.jobs_created} jobs, "
                   f"{stats.locations_created} new locations, "
                   f"{stats.geocoding_requests} geocoding requests")
        
        return stats
    
    async def _process_locations(
        self, 
        location_names: set, 
        stats: ImportStatsResponse
    ) -> Dict[str, Coordinates]:
        """Process and geocode locations."""
        location_coords = {}
        addresses_to_geocode = []
        
        for location_name in location_names:
            existing_location = self.repo.get_location_by_name(location_name)
            
            if existing_location:
                if existing_location.lat and existing_location.lon:
                    # Already has coordinates
                    location_coords[location_name] = Coordinates(
                        lat=existing_location.lat,
                        lon=existing_location.lon
                    )
                else:
                    # Needs geocoding
                    addresses_to_geocode.append(location_name)
                    stats.locations_updated += 1
            else:
                # New location
                new_location = self.repo.create_location({
                    "name": location_name,
                    "address": location_name,  # Use name as address for now
                })
                addresses_to_geocode.append(location_name)
                stats.locations_created += 1
        
        # Geocode addresses
        if addresses_to_geocode:
            geocoding_results = await self.distance_provider.geocode_locations(addresses_to_geocode)
            stats.geocoding_requests += len(addresses_to_geocode)
            
            for address, coords in geocoding_results.items():
                if coords:
                    # Update location with coordinates
                    location = self.repo.get_location_by_name(address)
                    if location:
                        self.repo.update_location_coordinates(location.id, coords.lat, coords.lon)
                        location_coords[address] = coords
                else:
                    error_msg = f"Failed to geocode location: {address}"
                    logger.warning(error_msg)
                    stats.errors.append(error_msg)
        
        return location_coords
    
    async def _process_items(self, item_names: set, stats: ImportStatsResponse) -> None:
        """Process items, creating unknown ones."""
        for item_name in item_names:
            existing_item = self.repo.get_item_by_name(item_name)
            
            if not existing_item:
                # Create unknown item with default properties
                new_item = self.repo.create_item({
                    "name": item_name,
                    "category": "material",  # Default category
                    "weight_lb_per_unit": 50.0,  # Default weight
                    "requires_large_truck": False
                })
                stats.items_created += 1
                logger.warning(f"Created unknown item '{item_name}' with default properties")
    
    async def _create_job_from_row(
        self, 
        row: JobImportRow, 
        date: str, 
        stats: ImportStatsResponse
    ) -> None:
        """Create a job and its items from import row."""
        # Get location
        location = self.repo.get_location_by_name(row.location)
        if not location:
            raise ValueError(f"Location not found: {row.location}")
        
        # Parse times
        earliest = None
        latest = None
        if row.earliest:
            earliest = datetime.fromisoformat(row.earliest)
        if row.latest:
            latest = datetime.fromisoformat(row.latest)
        
        # Create job
        job_data = {
            "location_id": location.id,
            "action": row.action,
            "priority": row.priority,
            "date": date,
            "earliest": earliest,
            "latest": latest,
            "notes": row.notes
        }
        
        job = self.repo.create_job(job_data)
        stats.jobs_created += 1
        
        # Parse and create job items
        item_specs = self._parse_items_string(row.items)
        for item_name, qty in item_specs:
            item = self.repo.get_item_by_name(item_name)
            if not item:
                raise ValueError(f"Item not found: {item_name}")
            
            job_item_data = {
                "job_id": job.id,
                "item_id": item.id,
                "qty": qty
            }
            
            self.repo.create_job_item(job_item_data)
            stats.total_job_items += 1
    
    def _parse_items_string(self, items_str: str) -> List[Tuple[str, float]]:
        """Parse items string like 'big drill:1; rebar:5' into [(name, qty), ...]."""
        items = []
        
        for item_spec in items_str.split(';'):
            item_spec = item_spec.strip()
            if ':' in item_spec:
                name, qty_str = item_spec.rsplit(':', 1)
                try:
                    qty = float(qty_str.strip())
                    items.append((name.strip(), qty))
                except ValueError:
                    logger.warning(f"Invalid quantity in item spec: {item_spec}")
            else:
                # Default quantity of 1
                items.append((item_spec.strip(), 1.0))
        
        return items
    
    async def optimize_routes(self, request: OptimizeRequest) -> OptimizationResult:
        """
        Optimize routes for a given date.
        
        Args:
            request: Optimization parameters
            
        Returns:
            Optimization result or overtime decision request
        """
        start_time = datetime.now()
        
        # Load data
        trucks = self.repo.get_trucks()
        jobs = self.repo.get_jobs_by_date(request.date)
        locations = self.repo.get_locations()
        
        if not jobs:
            logger.warning(f"No jobs found for date {request.date}")
            return OptimizationResult(
                date=request.date,
                routes=[],
                unassigned_jobs=[],
                total_cost=0.0,
                solver_used="greedy",
                computation_time_seconds=0.0
            )
        
        # Build job items mapping
        job_items_map = {}
        for job in jobs:
            job_items_map[job.id] = job.job_items
        
        # Setup depot coordinates
        depot_address = self.config.depot.address
        depot_coords_dict = await self.distance_provider.geocode_locations([depot_address])
        depot_coords = depot_coords_dict[depot_address]
        
        if not depot_coords:
            raise ValueError(f"Could not geocode depot address: {depot_address}")
        
        # Prepare location coordinates for distance matrix
        location_coords = [depot_coords]  # Depot at index 0
        for location in locations:
            if location.lat and location.lon:
                location_coords.append(Coordinates(lat=location.lat, lon=location.lon))
            else:
                logger.warning(f"Missing coordinates for location {location.name}")
                location_coords.append(depot_coords)  # Fallback to depot
        
        # Calculate distance matrix
        workday_start = datetime.fromisoformat(f"{request.date}T{self.config.depot.workday_window.start}:00")
        distance_matrix = await self.distance_provider.compute_travel_matrix(
            location_coords, 
            departure_time=workday_start
        )
        
        if not distance_matrix:
            raise ValueError("Failed to compute distance matrix")
        
        # Run optimization
        if self.config.solver.use_ortools:
            # TODO: Implement OR-Tools solver
            raise NotImplementedError("OR-Tools solver not yet implemented")
        else:
            solver = GreedySolver(self.config)
            solution = solver.solve(
                trucks=trucks,
                jobs=jobs,
                job_items_map=job_items_map,
                locations=locations,
                distance_matrix=distance_matrix,
                depot_coords=depot_coords,
                workday_start=workday_start
            )
        
        # Check overtime policy
        total_overtime = sum(route.overtime_minutes for route in solution.routes)
        
        if (request.auto == "ask" and 
            self.validator.check_overtime_threshold(total_overtime)):
            # Need to present overtime decision
            return await self._handle_overtime_decision(
                solution, request, trucks, jobs, job_items_map,
                locations, distance_matrix, depot_coords, workday_start
            )
        
        # Save results
        await self._save_optimization_results(solution, request.date)
        
        # Convert to API response
        result = self._convert_solution_to_result(solution, request.date, start_time, depot_coords)
        
        logger.info(f"Optimization completed: {len(result.routes)} routes, "
                   f"{len(result.unassigned_jobs)} unassigned jobs")
        
        return result
    
    async def _handle_overtime_decision(
        self,
        original_solution: Solution,
        request: OptimizeRequest,
        trucks: List[Truck],
        jobs: List[Job],
        job_items_map: Dict[int, List[JobItem]],
        locations: List[Location],
        distance_matrix: Any,
        depot_coords: Coordinates,
        workday_start: datetime
    ) -> OvertimeDecisionRequest:
        """Handle overtime decision by computing both alternatives."""
        start_time = datetime.now()
        
        # Original solution is the "overtime" option
        overtime_result = self._convert_solution_to_result(original_solution, request.date, start_time, depot_coords)
        
        # Create defer solution by removing low-priority jobs
        defer_solution = self._create_defer_solution(
            original_solution, trucks, jobs, job_items_map,
            locations, distance_matrix, depot_coords, workday_start
        )
        defer_result = self._convert_solution_to_result(defer_solution, request.date, start_time, depot_coords)
        
        return OvertimeDecisionRequest(
            overtime_plan=overtime_result,
            defer_plan=defer_result,
            overtime_minutes_diff=overtime_result.routes[0].overtime_minutes if overtime_result.routes else 0,
            jobs_deferred_count=len(defer_result.unassigned_jobs) - len(overtime_result.unassigned_jobs)
        )
    
    def _create_defer_solution(
        self,
        original_solution: Solution,
        trucks: List[Truck],
        jobs: List[Job],
        job_items_map: Dict[int, List[JobItem]],
        locations: List[Location],
        distance_matrix: Any,
        depot_coords: Coordinates,
        workday_start: datetime
    ) -> Solution:
        """Create a solution that defers jobs to avoid overtime."""
        # Simple strategy: remove lowest priority jobs until overtime is acceptable
        defer_solution = Solution(
            routes=original_solution.routes.copy(),
            unassigned_jobs=original_solution.unassigned_jobs.copy(),
            total_cost=original_solution.total_cost,
            feasible=original_solution.feasible,
            computation_time_seconds=original_solution.computation_time_seconds
        )
        
        # Identify jobs to defer (lowest priority first)
        all_assignments = []
        for route in defer_solution.routes:
            for assignment in route.assignments:
                all_assignments.append((assignment, route))
        
        # Sort by priority (ascending - lowest first)
        all_assignments.sort(key=lambda x: x[0].job.priority)
        
        # Remove jobs until overtime is acceptable
        while any(route.overtime_minutes > self.config.overtime_deferral.overtime_slack_minutes 
                 for route in defer_solution.routes) and all_assignments:
            
            assignment, route = all_assignments.pop(0)
            
            # Remove assignment from route
            route.assignments.remove(assignment)
            defer_solution.unassigned_jobs.append(assignment.job)
            
            # Recalculate route metrics (simplified)
            # In practice, you'd want to re-optimize the route
            route.overtime_minutes = max(0, route.overtime_minutes - 10)  # Rough estimate
        
        return defer_solution
    
    async def _save_optimization_results(self, solution: Solution, date: str) -> None:
        """Save optimization results to database."""
        # Clear existing results for this date
        self.repo.delete_route_assignments_by_date(date)
        self.repo.delete_unassigned_jobs_by_date(date)
        
        # Save route assignments
        for route in solution.routes:
            assignment_data = {
                "truck_id": route.truck.id,
                "date": date,
                "total_drive_minutes": route.total_drive_minutes,
                "total_service_minutes": route.total_service_minutes,
                "total_weight_lb": route.total_weight_lb,
                "overtime_minutes": route.overtime_minutes
            }
            
            route_assignment = self.repo.create_route_assignment(assignment_data)
            
            # Save route stops
            for assignment in route.assignments:
                stop_data = {
                    "route_assignment_id": route_assignment.id,
                    "job_id": assignment.job.id,
                    "stop_order": assignment.stop_order,
                    "estimated_arrival": assignment.estimated_arrival,
                    "estimated_departure": assignment.estimated_departure,
                    "drive_minutes_from_previous": assignment.drive_minutes_from_previous,
                    "service_minutes": assignment.service_minutes
                }
                
                self.repo.create_route_stop(stop_data)
        
        # Save unassigned jobs
        for job in solution.unassigned_jobs:
            unassigned_data = {
                "job_id": job.id,
                "date": date,
                "reason": "Could not satisfy constraints"
            }
            
            self.repo.create_unassigned_job(unassigned_data)
    
    def _convert_solution_to_result(
        self, 
        solution: Solution, 
        date: str, 
        start_time: datetime,
        depot_coords: Coordinates
    ) -> OptimizationResult:
        """Convert solver solution to API result format."""
        routes = []
        
        for route in solution.routes:
            if not route.assignments:
                continue  # Skip empty routes
            
            route_stops = []
            for assignment in route.assignments:
                # Convert location to response format
                location_response = {
                    "id": assignment.job.location.id,
                    "name": assignment.job.location.name,
                    "address": assignment.job.location.address,
                    "lat": assignment.job.location.lat,
                    "lon": assignment.job.location.lon,
                    "window_start": assignment.job.location.window_start,
                    "window_end": assignment.job.location.window_end
                }
                
                job_response = JobResponse(
                    id=assignment.job.id,
                    location=location_response,
                    action=assignment.job.action,
                    priority=assignment.job.priority,
                    earliest=assignment.job.earliest,
                    latest=assignment.job.latest,
                    notes=assignment.job.notes,
                    items=[{
                        "item_name": item.item.name,
                        "category": item.item.category,
                        "qty": item.qty,
                        "weight_lb_total": item.item.weight_lb_per_unit * item.qty
                    } for item in assignment.job_items]
                )
                
                route_stops.append({
                    "job": job_response,
                    "stop_order": assignment.stop_order,
                    "estimated_arrival": assignment.estimated_arrival,
                    "estimated_departure": assignment.estimated_departure,
                    "drive_minutes_from_previous": assignment.drive_minutes_from_previous,
                    "service_minutes": assignment.service_minutes
                })
            
            # Convert truck to response format
            truck_response = {
                "id": route.truck.id,
                "name": route.truck.name,
                "max_weight_lb": route.truck.max_weight_lb,
                "bed_len_ft": route.truck.bed_len_ft,
                "bed_width_ft": route.truck.bed_width_ft,
                "height_limit_ft": route.truck.height_limit_ft,
                "large_capable": route.truck.large_capable
            }
            
            # Generate Google Maps URL for the route
            coordinates = []
            # Add depot as starting point (use depot_coords from above)
            coordinates.append(depot_coords)
            
            # Add all job locations
            for assignment in route.assignments:
                coordinates.append(Coordinates(
                    lat=assignment.job.location.lat,
                    lon=assignment.job.location.lon
                ))
            
            # Return to depot
            coordinates.append(depot_coords)
            
            # Generate URL
            route_urls = self.url_builder.build_coordinate_urls(
                coordinates,
                route.truck.name
            )
            maps_url = route_urls.urls[0] if route_urls.urls else ""
            
            routes.append(RouteResponse(
                truck=truck_response,
                date=date,
                stops=route_stops,
                total_drive_minutes=route.total_drive_minutes,
                total_service_minutes=route.total_service_minutes,
                total_weight_lb=route.total_weight_lb,
                overtime_minutes=route.overtime_minutes,
                maps_url=maps_url
            ))
        
        # Convert unassigned jobs
        unassigned_jobs = []
        for job in solution.unassigned_jobs:
            # Convert location to response format
            location_response = {
                "id": job.location.id,
                "name": job.location.name,
                "address": job.location.address,
                "lat": job.location.lat,
                "lon": job.location.lon,
                "window_start": job.location.window_start,
                "window_end": job.location.window_end
            }
            
            unassigned_jobs.append(JobResponse(
                id=job.id,
                location=location_response,
                action=job.action,
                priority=job.priority,
                earliest=job.earliest,
                latest=job.latest,
                notes=job.notes,
                items=[]  # Would need to load job items
            ))
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            date=date,
            routes=routes,
            unassigned_jobs=unassigned_jobs,
            total_cost=solution.total_cost,
            solver_used="greedy",
            computation_time_seconds=computation_time
        )
    
    async def get_route_urls(self, date: str) -> List[Dict[str, Any]]:
        """Get Google Maps URLs for routes on a date."""
        route_assignments = self.repo.get_route_assignments_by_date(date)
        
        if not route_assignments:
            return []
        
        # Get depot coordinates
        depot_address = self.config.depot.address
        depot_coords_dict = await self.distance_provider.geocode_locations([depot_address])
        depot_coords = depot_coords_dict[depot_address]
        
        if not depot_coords:
            raise ValueError(f"Could not geocode depot address: {depot_address}")
        
        urls = []
        
        for assignment in route_assignments:
            # Get route stops with their jobs and locations
            route_stops = self.repo.get_route_stops_by_assignment(assignment.id)
            
            if not route_stops:
                # Empty route
                continue
            
            # Build list of coordinates for this route: depot -> stops -> depot
            coordinates = [depot_coords]
            
            # Sort stops by stop order and add their locations
            sorted_stops = sorted(route_stops, key=lambda s: s.stop_order)
            for stop in sorted_stops:
                if stop.job.location.lat and stop.job.location.lon:
                    coordinates.append(Coordinates(
                        lat=stop.job.location.lat,
                        lon=stop.job.location.lon
                    ))
            
            # Return to depot
            coordinates.append(depot_coords)
            
            # Generate URL using simple coordinate-based approach
            route_urls = self.url_builder.build_coordinate_urls(
                coordinates,
                assignment.truck.name
            )
            
            urls.append({
                "truck_name": assignment.truck.name,
                "urls": route_urls.urls,
                "total_stops": route_urls.total_stops
            })
        
        return urls
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration (non-secret)."""
        return self.config.model_dump()
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration parameters."""
        # This would update the YAML file and reload config
        # Implementation depends on specific requirements
        raise NotImplementedError("Config updates not yet implemented")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        return {
            "status": "healthy",
            "database_connected": self.repo.health_check(),
            "google_api_configured": self.settings.google_maps_api_key is not None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.distance_provider.close()
