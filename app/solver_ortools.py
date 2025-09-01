"""
OR-Tools solver for truck route optimization.
Implements Vehicle Routing Problem with Time Windows (VRPTW) and capacity constraints.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from .models import Truck, Job, JobItem, Location
from .distance import RouteMatrix, Coordinates
from .constraints import ConstraintValidator
from .schemas import AppConfig
from .solver_greedy import Solution, TruckRoute, JobAssignment


logger = logging.getLogger(__name__)


class ORToolsSolver:
    """OR-Tools-based VRP solver with time windows and capacity constraints."""
    
    def __init__(self, config: AppConfig):
        """Initialize OR-Tools solver."""
        self.config = config
        self.validator = ConstraintValidator(config)
        
        try:
            from ortools.constraint_solver import routing_enums_pb2
            from ortools.constraint_solver import pywrapcp
            self.routing_enums = routing_enums_pb2
            self.pywrapcp = pywrapcp
            logger.info("OR-Tools available")
        except ImportError:
            logger.warning("OR-Tools not available - install with: pip install ortools")
            self.routing_enums = None
            self.pywrapcp = None
    
    def solve(
        self,
        trucks: List[Truck],
        jobs: List[Job],
        job_items_map: Dict[int, List[JobItem]],
        locations: List[Location],
        distance_matrix: RouteMatrix,
        depot_coords: Coordinates,
        workday_start: datetime
    ) -> Solution:
        """
        Solve using OR-Tools VRP solver.
        
        Args:
            trucks: Available trucks
            jobs: Jobs to assign
            job_items_map: Mapping from job_id to list of JobItems
            locations: All locations (including depot at index 0)
            distance_matrix: Travel time matrix between locations
            depot_coords: Depot coordinates
            workday_start: Start time of workday
            
        Returns:
            Complete solution
        """
        if not self.pywrapcp:
            raise RuntimeError("OR-Tools not available. Install with: pip install ortools")
        
        start_time = datetime.now()
        
        logger.info(f"Starting OR-Tools solver with {len(trucks)} trucks, {len(jobs)} jobs")
        
        # For now, fall back to greedy solver
        # TODO: Implement full OR-Tools VRP solution
        from .solver_greedy import GreedySolver
        
        logger.warning("OR-Tools solver not fully implemented - using greedy fallback")
        greedy_solver = GreedySolver(self.config)
        solution = greedy_solver.solve(
            trucks, jobs, job_items_map, locations,
            distance_matrix, depot_coords, workday_start
        )
        
        # Mark as OR-Tools for identification
        solution.computation_time_seconds = (datetime.now() - start_time).total_seconds()
        
        return solution
    
    def _create_data_model(
        self,
        trucks: List[Truck],
        jobs: List[Job],
        job_items_map: Dict[int, List[JobItem]],
        distance_matrix: RouteMatrix,
        workday_start: datetime
    ) -> Dict:
        """Create OR-Tools data model."""
        
        # Calculate demands (weight) for each job
        demands = [0]  # Depot has no demand
        for job in jobs:
            job_items = job_items_map[job.id]
            total_weight = sum(
                item.item.weight_lb_per_unit * item.qty 
                for item in job_items
            )
            demands.append(int(total_weight))
        
        # Vehicle capacities
        vehicle_capacities = [int(truck.max_weight_lb) for truck in trucks]
        
        # Time matrix (in minutes, converted to integers)
        time_matrix = []
        for i in range(len(distance_matrix.durations_minutes)):
            row = []
            for j in range(len(distance_matrix.durations_minutes[i])):
                # Convert to integer minutes
                time_minutes = int(distance_matrix.durations_minutes[i][j])
                row.append(time_minutes)
            time_matrix.append(row)
        
        # Time windows
        time_windows = [(0, 570)]  # Depot: 0 to 9.5 hours (570 minutes)
        
        for job in jobs:
            # Convert job time windows to minutes from workday start
            if job.earliest:
                earliest_minutes = int((job.earliest - workday_start).total_seconds() / 60)
            else:
                earliest_minutes = 0
            
            if job.latest:
                latest_minutes = int((job.latest - workday_start).total_seconds() / 60)
            else:
                latest_minutes = 570  # End of workday
            
            time_windows.append((earliest_minutes, latest_minutes))
        
        # Service times
        service_times = [0]  # Depot
        for job in jobs:
            job_items = job_items_map[job.id]
            service_time = self.validator.calculate_service_time(job_items)
            service_times.append(service_time)
        
        data = {
            'time_matrix': time_matrix,
            'time_windows': time_windows,
            'service_times': service_times,
            'demands': demands,
            'vehicle_capacities': vehicle_capacities,
            'num_vehicles': len(trucks),
            'depot': 0
        }
        
        return data
    
    def _create_solution_from_routes(
        self,
        manager,
        routing,
        solution,
        trucks: List[Truck],
        jobs: List[Job],
        job_items_map: Dict[int, List[JobItem]],
        workday_start: datetime
    ) -> Solution:
        """Convert OR-Tools solution to our solution format."""
        
        routes = []
        unassigned_jobs = []
        
        # Extract routes for each vehicle
        for vehicle_id in range(len(trucks)):
            truck = trucks[vehicle_id]
            assignments = []
            
            index = routing.Start(vehicle_id)
            route_load = 0
            route_time = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                
                if node_index != 0:  # Not depot
                    job = jobs[node_index - 1]  # Adjust for depot offset
                    job_items = job_items_map[job.id]
                    
                    # Calculate timing
                    time_var = routing.GetDimensionOrDie('Time').CumulVar(index)
                    arrival_time = solution.Value(time_var)
                    
                    service_time = self.validator.calculate_service_time(job_items)
                    
                    assignment = JobAssignment(
                        job=job,
                        job_items=job_items,
                        truck=truck,
                        stop_order=len(assignments),
                        estimated_arrival=workday_start + timedelta(minutes=arrival_time),
                        estimated_departure=workday_start + timedelta(minutes=arrival_time + service_time),
                        drive_minutes_from_previous=0,  # Would need to calculate
                        service_minutes=service_time,
                        location_index=node_index
                    )
                    
                    assignments.append(assignment)
                    
                    # Update load
                    for item in job_items:
                        route_load += item.item.weight_lb_per_unit * item.qty
                
                index = solution.Value(routing.NextVar(index))
            
            if assignments:
                # Calculate route metrics
                total_drive_time = 0  # Would need to calculate from time matrix
                total_service_time = sum(a.service_minutes for a in assignments)
                
                # Check for overtime
                workday_end = workday_start.replace(
                    hour=int(self.config.depot.workday_window.end[:2]),
                    minute=int(self.config.depot.workday_window.end[3:])
                )
                
                last_departure = assignments[-1].estimated_departure if assignments else workday_start
                overtime_minutes = max(0, (last_departure - workday_end).total_seconds() / 60)
                
                route = TruckRoute(
                    truck=truck,
                    assignments=assignments,
                    total_drive_minutes=total_drive_time,
                    total_service_minutes=total_service_time,
                    total_weight_lb=route_load,
                    overtime_minutes=overtime_minutes
                )
                
                routes.append(route)
        
        # Find unassigned jobs
        assigned_job_ids = set()
        for route in routes:
            for assignment in route.assignments:
                assigned_job_ids.add(assignment.job.id)
        
        for job in jobs:
            if job.id not in assigned_job_ids:
                unassigned_jobs.append(job)
        
        total_cost = sum(route.calculate_cost(self.config) for route in routes)
        
        return Solution(
            routes=routes,
            unassigned_jobs=unassigned_jobs,
            total_cost=total_cost,
            feasible=len(unassigned_jobs) == 0,
            computation_time_seconds=0  # Will be set by caller
        )
