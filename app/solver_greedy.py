"""
Greedy solver with local search for truck route optimization.
Implements nearest neighbor with 2-opt improvements and priority weighting.
"""

import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

from .models import Truck, Job, JobItem, Location
from .distance import RouteMatrix, Coordinates
from .constraints import ConstraintValidator, LoadInfo, ConstraintViolation
from .schemas import AppConfig


logger = logging.getLogger(__name__)


@dataclass
class JobAssignment:
    """Represents a job assigned to a truck route."""
    job: Job
    job_items: List[JobItem]
    truck: Truck
    stop_order: int
    estimated_arrival: datetime
    estimated_departure: datetime
    drive_minutes_from_previous: float
    service_minutes: float
    location_index: int  # Index in distance matrix
    
    def __eq__(self, other) -> bool:
        """Compare assignments by job ID to avoid Pydantic comparison issues."""
        if not isinstance(other, JobAssignment):
            return False
        return self.job.id == other.job.id
    
    def __hash__(self) -> int:
        """Hash by job ID."""
        return hash(self.job.id)


@dataclass
class TruckRoute:
    """Complete route for a single truck."""
    truck: Truck
    assignments: List[JobAssignment]
    total_drive_minutes: float
    total_service_minutes: float
    total_weight_lb: float
    overtime_minutes: float
    
    def __eq__(self, other) -> bool:
        """Compare routes by truck ID to avoid Pydantic comparison issues."""
        if not isinstance(other, TruckRoute):
            return False
        return self.truck.id == other.truck.id
    
    def __hash__(self) -> int:
        """Hash by truck ID."""
        return hash(self.truck.id)
    
    @property
    def total_time_minutes(self) -> float:
        """Total route time including drive and service."""
        return self.total_drive_minutes + self.total_service_minutes
    
    def calculate_cost(self, config: AppConfig) -> float:
        """Calculate route cost using multi-objective weighted function."""
        # Use multi-objective weights if defined
        if hasattr(config.solver, "weights"):
            # Multi-objective weighted sum
            drive_cost = self.total_drive_minutes * config.solver.weights.drive_minutes
            service_cost = self.total_service_minutes * config.solver.weights.service_minutes
            overtime_cost = self.overtime_minutes * config.solver.weights.overtime_minutes
            max_route_cost = self.total_time_minutes * config.solver.weights.max_route_minutes
            
            # Priority cost (higher priority jobs earlier = lower cost)
            priority_cost = 0.0
            for i, assignment in enumerate(self.assignments):
                # Later positions get higher cost, weighted by inverse priority
                position_penalty = i + 1
                priority_weight = 1.0 / max(assignment.job.priority, 1)
                priority_cost += position_penalty * priority_weight
            
            priority_cost *= config.solver.weights.priority_soft_cost
            
            return drive_cost + service_cost + overtime_cost + max_route_cost + priority_cost
        else:
            # Legacy cost function for backward compatibility
            efficiency_cost = (self.total_drive_minutes + self.total_service_minutes) * config.solver.efficiency_weight
            overtime_cost = self.overtime_minutes * config.solver.overtime_penalty_per_minute
            
            # Priority cost (higher priority jobs earlier = lower cost)
            priority_cost = 0.0
            for i, assignment in enumerate(self.assignments):
                # Later positions get higher cost, weighted by inverse priority
                position_penalty = i + 1
                priority_weight = 1.0 / max(assignment.job.priority, 1)
                priority_cost += position_penalty * priority_weight
            
            priority_cost *= config.solver.priority_weight
            
            return efficiency_cost + overtime_cost + priority_cost


@dataclass
class Solution:
    """Complete solution with all truck routes."""
    routes: List[TruckRoute]
    unassigned_jobs: List[Job]
    total_cost: float
    feasible: bool
    computation_time_seconds: float
    trace_data: Optional[Dict] = None
    
    def calculate_total_cost(self, config: AppConfig) -> float:
        """Calculate total solution cost including single truck mode penalties."""
        # Sum individual route costs
        base_cost = sum(route.calculate_cost(config) for route in self.routes if route.assignments)
        
        # Add penalty for number of trucks used if in single truck mode
        used_trucks = sum(1 for route in self.routes if route.assignments)
        truck_penalty = 0.0
        
        if getattr(config.solver, "single_truck_mode", 0) == 1 and used_trucks > 1:
            # Apply penalty for each truck beyond the first one
            truck_penalty = (used_trucks - 1) * config.solver.trucks_used_penalty
        
        return base_cost + truck_penalty
    
    @property
    def used_trucks_count(self) -> int:
        """Count the number of trucks used in the solution."""
        return sum(1 for route in self.routes if route.assignments)


class GreedySolver:
    """Greedy solver with local search optimization."""
    
    def __init__(self, config: AppConfig):
        """Initialize solver with configuration."""
        self.config = config
        self.validator = ConstraintValidator(config)
        
        # Set random seed for deterministic results
        random.seed(config.solver.random_seed)
    
    def solve(
        self,
        trucks: List[Truck],
        jobs: List[Job],
        job_items_map: Dict[int, List[JobItem]],
        locations: List[Location],
        distance_matrix: RouteMatrix,
        depot_coords: Coordinates,
        workday_start: datetime,
        trace: bool = False,
        solver_strategy: str = "greedy"
    ) -> Solution:
        """
        Solve the truck routing problem using greedy construction + local search.
        
        Args:
            trucks: Available trucks
            jobs: Jobs to assign
            job_items_map: Mapping from job_id to list of JobItems
            locations: All locations (including depot at index 0)
            distance_matrix: Travel time matrix between locations
            depot_coords: Depot coordinates
            workday_start: Start time of workday
            trace: Whether to record decision trace data
            solver_strategy: Solver strategy to use ("greedy" or "regret2")
            
        Returns:
            Complete solution
        """
        start_time = datetime.now()
        
        logger.info(f"Starting greedy solver with {len(trucks)} trucks, {len(jobs)} jobs")
        
        # Initialize empty routes
        routes = [TruckRoute(
            truck=truck,
            assignments=[],
            total_drive_minutes=0.0,
            total_service_minutes=0.0,
            total_weight_lb=0.0,
            overtime_minutes=0.0
        ) for truck in trucks]
        
        # Create location index mapping
        location_to_index = {loc.id: i for i, loc in enumerate(locations)}
        
        # Initialize trace data if requested
        trace_data = None
        if trace:
            trace_data = {
                "decisions": [],
                "config": {
                    "single_truck_mode": getattr(self.config.solver, "single_truck_mode", 0),
                    "weights": getattr(self.config.solver, "weights", None),
                },
                "truck_count": len(trucks),
                "job_count": len(jobs),
                "timestamp": datetime.now().isoformat(),
            }
        
        # Choose construction method based on solver strategy
        if solver_strategy == "regret2":
            unassigned_jobs = self._build_solution_regret2(
                routes, jobs, job_items_map, location_to_index, 
                distance_matrix, workday_start, trace_data
            )
        else:
            # Default to greedy construction
            unassigned_jobs = self._greedy_construction(
                routes, jobs, job_items_map, location_to_index, 
                distance_matrix, workday_start, trace_data
            )
        
        # Apply local search improvements if enabled
        improve_config = getattr(self.config.solver, "improve", None)
        if improve_config and improve_config.enabled:
            self._local_search_improvement(
                routes, unassigned_jobs, job_items_map, location_to_index,
                distance_matrix, workday_start, trace_data
            )
        
        # Calculate final costs and metrics with the new multi-objective function
        total_cost = sum(route.calculate_cost(self.config) for route in routes if route.assignments)
        
        # Add single truck mode penalty if applicable
        used_trucks = sum(1 for route in routes if route.assignments)
        if getattr(self.config.solver, "single_truck_mode", 0) == 1 and used_trucks > 1:
            truck_penalty = (used_trucks - 1) * self.config.solver.trucks_used_penalty
            total_cost += truck_penalty
            
            if trace_data:
                trace_data["single_truck_penalty"] = {
                    "trucks_used": used_trucks,
                    "penalty_per_truck": self.config.solver.trucks_used_penalty,
                    "total_penalty": truck_penalty
                }
        
        # Check feasibility (no constraint violations)
        feasible = len(unassigned_jobs) == 0
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Solver completed in {computation_time:.2f}s: "
                   f"{len(jobs) - len(unassigned_jobs)}/{len(jobs)} jobs assigned")
        
        return Solution(
            routes=routes,
            unassigned_jobs=unassigned_jobs,
            total_cost=total_cost,
            feasible=feasible,
            computation_time_seconds=computation_time,
            trace_data=trace_data
        )
    
    def _greedy_construction(
        self,
        routes: List[TruckRoute],
        jobs: List[Job],
        job_items_map: Dict[int, List[JobItem]],
        location_to_index: Dict[int, int],
        distance_matrix: RouteMatrix,
        workday_start: datetime,
        trace_data: Optional[Dict] = None
    ) -> List[Job]:
        """Greedy construction phase using nearest neighbor heuristic."""
        unassigned_jobs = []
        remaining_jobs = jobs.copy()
        
        # Sort jobs by priority (descending) as initial preference
        remaining_jobs.sort(key=lambda j: j.priority, reverse=True)
        
        while remaining_jobs:
            job = remaining_jobs.pop(0)
            job_items = job_items_map[job.id]
            
            # Find best truck assignment
            best_truck_idx, best_cost, _ = self._find_best_truck_assignment(
                job, job_items, routes, location_to_index, distance_matrix, workday_start, trace_data
            )
            
            if best_truck_idx is not None:
                # Assign job to best truck
                self._assign_job_to_route(
                    job, job_items, routes[best_truck_idx], 
                    location_to_index, distance_matrix, workday_start
                )
            else:
                # Cannot assign this job
                unassigned_jobs.append(job)
                logger.debug(f"Could not assign job {job.id} - constraint violations")
        
        return unassigned_jobs
        
    def _build_solution_regret2(
        self,
        routes: List[TruckRoute],
        jobs: List[Job],
        job_items_map: Dict[int, List[JobItem]],
        location_to_index: Dict[int, int],
        distance_matrix: RouteMatrix,
        workday_start: datetime,
        trace_data: Optional[Dict] = None
    ) -> List[Job]:
        """
        Build solution using regret-2 insertion algorithm.
        
        The regret-2 algorithm selects jobs based on the difference between their best and second-best
        insertion costs, prioritizing jobs that would be most "regretted" if not inserted immediately.
        This typically produces better solutions than pure greedy insertion.
        """
        unassigned_jobs = []
        remaining_jobs = jobs.copy()
        
        # Sort jobs by priority (descending) as initial preference
        remaining_jobs.sort(key=lambda j: j.priority, reverse=True)
        
        # If tracing is enabled, record regret algorithm selection
        if trace_data is not None:
            trace_data["algorithm"] = "regret2"
        
        # Process jobs until none remain
        while remaining_jobs:
            best_job_idx = -1
            best_regret = -float('inf')
            best_job_truck_idx = None
            best_job_cost = float('inf')
            
            # Calculate regret value for each job
            for i, job in enumerate(remaining_jobs):
                job_items = job_items_map[job.id]
                
                # Get costs for all feasible truck assignments
                costs = []
                truck_assignments = []
                
                for truck_idx, route in enumerate(routes):
                    cost, violations, _ = self._evaluate_job_insertion(
                        job, job_items, route, location_to_index[job.location_id],
                        distance_matrix, workday_start
                    )
                    
                    if not violations:
                        costs.append(cost)
                        truck_assignments.append(truck_idx)
                
                # Calculate regret-2 value
                if len(costs) >= 2:
                    # Sort costs in ascending order
                    sorted_costs = sorted(costs)
                    regret = sorted_costs[1] - sorted_costs[0]
                    
                    # Select job with highest regret
                    if regret > best_regret or (regret == best_regret and sorted_costs[0] < best_job_cost):
                        best_regret = regret
                        best_job_idx = i
                        best_job_truck_idx = truck_assignments[costs.index(sorted_costs[0])]
                        best_job_cost = sorted_costs[0]
                        
                elif len(costs) == 1:
                    # Only one feasible insertion - use a large regret value
                    regret = 1000.0  # Artificially high regret value
                    
                    if regret > best_regret or (regret == best_regret and costs[0] < best_job_cost):
                        best_regret = regret
                        best_job_idx = i
                        best_job_truck_idx = truck_assignments[0]
                        best_job_cost = costs[0]
            
            # If we found a job to insert, do it
            if best_job_idx >= 0:
                job = remaining_jobs.pop(best_job_idx)
                job_items = job_items_map[job.id]
                
                # If tracing is enabled, record regret decision
                if trace_data is not None and "decisions" in trace_data:
                    trace_data["decisions"].append({
                        "job_id": job.id,
                        "action": job.action_type.name if hasattr(job, "action_type") else str(job.action),
                        "algorithm": "regret2",
                        "regret_value": best_regret,
                        "selected_truck_id": routes[best_job_truck_idx].truck.id if best_job_truck_idx is not None else None,
                        "selected_truck_name": routes[best_job_truck_idx].truck.name if best_job_truck_idx is not None else None,
                        "cost": best_job_cost,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Assign job to best truck
                self._assign_job_to_route(
                    job, job_items, routes[best_job_truck_idx],
                    location_to_index, distance_matrix, workday_start
                )
            else:
                # No feasible insertion for any remaining job
                unassigned_jobs.extend(remaining_jobs)
                logger.debug(f"Could not assign {len(remaining_jobs)} remaining jobs - constraint violations")
                break
                
        return unassigned_jobs
    
    def _find_best_truck_assignment(
        self,
        job: Job,
        job_items: List[JobItem],
        routes: List[TruckRoute],
        location_to_index: Dict[int, int],
        distance_matrix: RouteMatrix,
        workday_start: datetime,
        trace_data: Optional[Dict] = None
    ) -> Tuple[Optional[int], float, Optional[Dict]]:
        """Find the best truck to assign a job to."""
        best_truck_idx = None
        best_cost = float('inf')
        best_evaluation = None
        
        job_location_idx = location_to_index[job.location_id]
        
        # Collect truck evaluations for tracing
        truck_evaluations = []
        
        # Get single truck mode setting
        single_truck_mode = getattr(self.config.solver, "single_truck_mode", 0) == 1
        
        for truck_idx, route in enumerate(routes):
            # Check if job can be assigned to this truck
            insertion_cost, violations, evaluation = self._evaluate_job_insertion(
                job, job_items, route, job_location_idx,
                distance_matrix, workday_start,
                return_details=True
            )
            
            # If tracing is enabled, collect evaluation data
            if trace_data is not None:
                truck_eval = {
                    "truck_id": route.truck.id,
                    "truck_name": route.truck.name,
                    "base_cost": insertion_cost,
                    "violations": [str(v) for v in violations],
                    "feasible": len(violations) == 0
                }
                truck_evaluations.append(truck_eval)
            
            # Skip if constraint violations
            if violations:
                continue
                
            # Apply single truck mode preference if enabled
            if single_truck_mode:
                # Heavily prefer trucks that already have assignments
                if len(route.assignments) > 0:
                    # This truck is already in use, prefer it
                    insertion_cost *= 0.5
                elif sum(1 for r in routes if r.assignments) > 0:
                    # Other trucks are already in use, penalize this one
                    insertion_cost *= 2.0
            
            # Consider co-loading policy for big truck
            if route.truck.large_capable and route.assignments:
                # Apply co-loading threshold
                if insertion_cost <= self.config.constraints.big_truck_co_load_threshold_minutes:
                    insertion_cost *= 0.8  # Prefer co-loading
            
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_truck_idx = truck_idx
                best_evaluation = evaluation
        
        # Add evaluation data to trace if enabled
        if trace_data is not None and "decisions" in trace_data:
            trace_decision = {
                "job_id": job.id,
                "action": job.action_type.name if hasattr(job, "action_type") else str(job.action),
                "address": job.location.address if hasattr(job.location, "address") else "unknown",
                "truck_evaluations": truck_evaluations,
                "selected_truck_id": routes[best_truck_idx].truck.id if best_truck_idx is not None else None,
                "selected_truck_name": routes[best_truck_idx].truck.name if best_truck_idx is not None else None,
                "final_cost": best_cost if best_truck_idx is not None else float('inf'),
                "assigned": best_truck_idx is not None,
                "timestamp": datetime.now().isoformat()
            }
            trace_data["decisions"].append(trace_decision)
        
        return best_truck_idx, best_cost, best_evaluation
    
    def _evaluate_job_insertion(
        self,
        job: Job,
        job_items: List[JobItem],
        route: TruckRoute,
        job_location_idx: int,
        distance_matrix: RouteMatrix,
        workday_start: datetime,
        return_details: bool = False
    ) -> Tuple[float, List[ConstraintViolation], Optional[Dict]]:
        """Evaluate the cost of inserting a job into a route."""
        # Calculate current load
        current_load = self._calculate_route_load(route, job_items)
        
        # Find best insertion position
        best_position = len(route.assignments)
        best_cost = float('inf')
        best_arrival_time = None
        best_violations = []
        position_evaluations = []
        
        for position in range(len(route.assignments) + 1):
            # Calculate insertion cost at this position
            cost, arrival_time = self._calculate_insertion_cost(
                route, position, job_location_idx, distance_matrix, workday_start
            )
            
            # Check constraints at this position
            violations = self.validator.validate_job_assignment(
                job, job_items, route.truck, current_load, arrival_time
            )
            
            # Store position evaluation for tracing
            if return_details:
                position_eval = {
                    "position": position,
                    "cost": cost,
                    "arrival_time": arrival_time.isoformat() if arrival_time else None,
                    "violations": [str(v) for v in violations],
                    "feasible": len(violations) == 0
                }
                position_evaluations.append(position_eval)
            
            if cost < best_cost and not violations:
                best_cost = cost
                best_position = position
                best_arrival_time = arrival_time
            
            # Store violations from the best cost position, even if infeasible
            if not best_violations or cost < best_cost:
                best_violations = violations
        
        # Prepare detailed evaluation if requested
        evaluation_details = None
        if return_details:
            evaluation_details = {
                "best_position": best_position,
                "best_cost": best_cost,
                "best_arrival": best_arrival_time.isoformat() if best_arrival_time else None,
                "position_evaluations": position_evaluations,
                "feasible": best_cost < float('inf')
            }
        
        # Return violations from best position
        if best_cost == float('inf'):
            # Could not find valid insertion position
            return best_cost, [ConstraintViolation(
                job_id=job.id,
                truck_id=route.truck.id,
                violation_type="no_valid_position",
                message="No valid insertion position found"
            )], evaluation_details
        
        return best_cost, [], evaluation_details
    
    def _calculate_insertion_cost(
        self,
        route: TruckRoute,
        position: int,
        job_location_idx: int,
        distance_matrix: RouteMatrix,
        workday_start: datetime
    ) -> Tuple[float, datetime]:
        """Calculate the cost of inserting a job at a specific position."""
        depot_idx = 0  # Depot is always at index 0
        
        if not route.assignments:
            # First job in route
            drive_time = distance_matrix.get_duration(depot_idx, job_location_idx)
            arrival_time = workday_start + timedelta(minutes=drive_time)
            return drive_time, arrival_time
        
        if position == 0:
            # Insert at beginning
            prev_location_idx = depot_idx
            next_location_idx = route.assignments[0].location_index
        elif position == len(route.assignments):
            # Insert at end
            prev_location_idx = route.assignments[-1].location_index
            next_location_idx = depot_idx
        else:
            # Insert in middle
            prev_location_idx = route.assignments[position - 1].location_index
            next_location_idx = route.assignments[position].location_index
        
        # Calculate detour cost
        old_direct = distance_matrix.get_duration(prev_location_idx, next_location_idx)
        new_via_job = (
            distance_matrix.get_duration(prev_location_idx, job_location_idx) +
            distance_matrix.get_duration(job_location_idx, next_location_idx)
        )
        
        detour_cost = new_via_job - old_direct
        
        # Calculate arrival time
        if position == 0:
            arrival_time = workday_start + timedelta(
                minutes=distance_matrix.get_duration(depot_idx, job_location_idx)
            )
        else:
            prev_departure = route.assignments[position - 1].estimated_departure
            travel_time = distance_matrix.get_duration(prev_location_idx, job_location_idx)
            arrival_time = prev_departure + timedelta(minutes=travel_time)
        
        return detour_cost, arrival_time
    
    def _assign_job_to_route(
        self,
        job: Job,
        job_items: List[JobItem],
        route: TruckRoute,
        location_to_index: Dict[int, int],
        distance_matrix: RouteMatrix,
        workday_start: datetime
    ) -> None:
        """Assign a job to a route at the best position."""
        job_location_idx = location_to_index[job.location_id]
        
        # Find best insertion position (simplified - insert at end for now)
        position = len(route.assignments)
        
        # Calculate service time
        service_minutes = self.validator.calculate_service_time(job_items)
        
        # Calculate arrival and departure times
        if not route.assignments:
            # First job
            drive_minutes = distance_matrix.get_duration(0, job_location_idx)
            arrival_time = workday_start + timedelta(minutes=drive_minutes)
        else:
            # After previous job
            prev_assignment = route.assignments[-1]
            drive_minutes = distance_matrix.get_duration(
                prev_assignment.location_index, job_location_idx
            )
            arrival_time = prev_assignment.estimated_departure + timedelta(minutes=drive_minutes)
        
        departure_time = arrival_time + timedelta(minutes=service_minutes)
        
        # Create assignment
        assignment = JobAssignment(
            job=job,
            job_items=job_items,
            truck=route.truck,
            stop_order=position,
            estimated_arrival=arrival_time,
            estimated_departure=departure_time,
            drive_minutes_from_previous=drive_minutes,
            service_minutes=service_minutes,
            location_index=job_location_idx
        )
        
        # Add to route
        route.assignments.append(assignment)
        
        # Update route metrics
        route.total_drive_minutes += drive_minutes
        route.total_service_minutes += service_minutes
        
        # Update weight
        job_weight = sum(
            item.item.weight_lb_per_unit * item.qty 
            for item in job_items
        )
        route.total_weight_lb += job_weight
        
        # Calculate overtime
        workday_end = workday_start.replace(
            hour=int(self.config.depot.workday_window.end[:2]),
            minute=int(self.config.depot.workday_window.end[3:]),
            second=0
        )
        
        if departure_time > workday_end:
            overtime_delta = departure_time - workday_end
            route.overtime_minutes = overtime_delta.total_seconds() / 60
    
    def _calculate_route_load(
        self, 
        route: TruckRoute, 
        additional_job_items: List[JobItem]
    ) -> LoadInfo:
        """Calculate current load on a route."""
        load = LoadInfo()
        
        # Add existing assignments
        for assignment in route.assignments:
            for job_item in assignment.job_items:
                load.total_weight_lb += job_item.item.weight_lb_per_unit * job_item.qty
                if job_item.item.requires_large_truck:
                    load.requires_large_truck = True
                if job_item.item.category not in load.item_categories:
                    load.item_categories.append(job_item.item.category)
        
        # Add proposed job items
        for job_item in additional_job_items:
            load.total_weight_lb += job_item.item.weight_lb_per_unit * job_item.qty
            if job_item.item.requires_large_truck:
                load.requires_large_truck = True
            if job_item.item.category not in load.item_categories:
                load.item_categories.append(job_item.item.category)
        
        return load
    
    def _local_search_improvement(
        self,
        routes: List[TruckRoute],
        unassigned_jobs: List[Job],
        job_items_map: Dict[int, List[JobItem]],
        location_to_index: Dict[int, int],
        distance_matrix: RouteMatrix,
        workday_start: datetime,
        trace_data: Optional[Dict] = None
    ) -> None:
        """Apply local search improvements to the solution."""
        improved = True
        iteration = 0
        
        # If tracing is enabled, add local search info
        if trace_data is not None:
            trace_data["local_search"] = {
                "enabled": True,
                "max_iterations": self.config.solver.local_search_iterations,
                "improvement_threshold": self.config.solver.improvement_threshold,
                "operations": [],
            }
        
        # Main local search loop
        while improved and iteration < self.config.solver.local_search_iterations:
            improved = False
            iteration += 1
            iteration_improvements = []
            
            # Try 2-opt improvements within each route
            for route_idx, route in enumerate(routes):
                if len(route.assignments) >= 2:
                    route_cost_before = route.calculate_cost(self.config)
                    if self._two_opt_improve_route(route, distance_matrix, workday_start):
                        route_cost_after = route.calculate_cost(self.config)
                        improved = True
                        
                        if trace_data is not None:
                            iteration_improvements.append({
                                "operation": "two_opt",
                                "truck_id": route.truck.id,
                                "truck_name": route.truck.name,
                                "cost_before": route_cost_before,
                                "cost_after": route_cost_after,
                                "improvement": route_cost_before - route_cost_after
                            })
            
            # Try relocating jobs between routes
            total_cost_before = sum(r.calculate_cost(self.config) for r in routes if r.assignments)
            if self._relocate_jobs_between_routes(
                routes, location_to_index, distance_matrix, workday_start
            ):
                total_cost_after = sum(r.calculate_cost(self.config) for r in routes if r.assignments)
                improved = True
                
                if trace_data is not None:
                    iteration_improvements.append({
                        "operation": "relocate",
                        "cost_before": total_cost_before,
                        "cost_after": total_cost_after,
                        "improvement": total_cost_before - total_cost_after
                    })
            
            # Try assigning unassigned jobs again
            newly_assigned = []
            for job in unassigned_jobs[:]:
                job_items = job_items_map[job.id]
                best_truck_idx, best_cost, _ = self._find_best_truck_assignment(
                    job, job_items, routes, location_to_index, distance_matrix, workday_start, trace_data
                )
                
                if best_truck_idx is not None:
                    self._assign_job_to_route(
                        job, job_items, routes[best_truck_idx],
                        location_to_index, distance_matrix, workday_start
                    )
                    newly_assigned.append(job)
                    improved = True
                    
                    if trace_data is not None:
                        iteration_improvements.append({
                            "operation": "assign_unassigned",
                            "job_id": job.id,
                            "truck_id": routes[best_truck_idx].truck.id,
                            "truck_name": routes[best_truck_idx].truck.name,
                            "insertion_cost": best_cost
                        })
            
            # Remove newly assigned jobs from unassigned list
            for job in newly_assigned:
                unassigned_jobs.remove(job)
                
            # Add iteration results to trace
            if trace_data is not None and "local_search" in trace_data:
                trace_data["local_search"]["operations"].append({
                    "iteration": iteration,
                    "improvements": iteration_improvements,
                    "improved": improved
                })
        
        if trace_data is not None and "local_search" in trace_data:
            trace_data["local_search"]["completed_iterations"] = iteration
            trace_data["local_search"]["finished_due_to"] = "max_iterations" if iteration >= self.config.solver.local_search_iterations else "no_improvement"
            
        logger.debug(f"Local search completed after {iteration} iterations")
    
    def _two_opt_improve_route(
        self,
        route: TruckRoute,
        distance_matrix: RouteMatrix,
        workday_start: datetime
    ) -> bool:
        """Apply 2-opt improvement to a single route."""
        best_improvement = 0.0
        best_i, best_j = -1, -1
        
        n = len(route.assignments)
        
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                improvement = self._calculate_2opt_improvement(
                    route, i, j, distance_matrix
                )
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_i, best_j = i, j
        
        if best_improvement > self.config.solver.improvement_threshold:
            # Apply the best 2-opt move
            self._apply_2opt_move(route, best_i, best_j, distance_matrix, workday_start)
            return True
        
        return False
    
    def _calculate_2opt_improvement(
        self,
        route: TruckRoute,
        i: int,
        j: int,
        distance_matrix: RouteMatrix
    ) -> float:
        """Calculate improvement from 2-opt move."""
        # This is a simplified 2-opt calculation
        # In practice, you'd want to consider time windows and other constraints
        n = len(route.assignments)
        
        if i >= n - 1 or j > n:
            return 0.0
        
        # Calculate old distances
        old_dist = 0.0
        if i > 0:
            old_dist += distance_matrix.get_duration(
                route.assignments[i-1].location_index,
                route.assignments[i].location_index
            )
        
        old_dist += distance_matrix.get_duration(
            route.assignments[j-1].location_index,
            route.assignments[j % n].location_index if j < n else 0
        )
        
        # Calculate new distances (after reversing segment)
        new_dist = 0.0
        if i > 0:
            new_dist += distance_matrix.get_duration(
                route.assignments[i-1].location_index,
                route.assignments[j-1].location_index
            )
        
        new_dist += distance_matrix.get_duration(
            route.assignments[i].location_index,
            route.assignments[j % n].location_index if j < n else 0
        )
        
        return old_dist - new_dist
    
    def _apply_2opt_move(
        self,
        route: TruckRoute,
        i: int,
        j: int,
        distance_matrix: RouteMatrix,
        workday_start: datetime
    ) -> None:
        """Apply 2-opt move to route."""
        # Reverse the segment between i and j-1
        route.assignments[i:j] = reversed(route.assignments[i:j])
        
        # Recalculate route metrics
        self._recalculate_route_metrics(route, distance_matrix, workday_start)
    
    def _relocate_jobs_between_routes(
        self,
        routes: List[TruckRoute],
        location_to_index: Dict[int, int],
        distance_matrix: RouteMatrix,
        workday_start: datetime
    ) -> bool:
        """Try relocating jobs between routes."""
        # This is a simplified version - try moving one job from each route to others
        improved = False
        
        for source_route in routes:
            if not source_route.assignments:
                continue
            
            for assignment in source_route.assignments[:]:
                # Try moving this assignment to other routes
                for target_route in routes:
                    if target_route == source_route:
                        continue
                    
                    # Calculate cost of removing from source
                    removal_savings = self._calculate_removal_cost(source_route, assignment)
                    
                    # Calculate cost of adding to target
                    insertion_cost, violations, _ = self._evaluate_job_insertion(
                        assignment.job, assignment.job_items, target_route,
                        assignment.location_index, distance_matrix, workday_start
                    )
                    
                    if not violations and insertion_cost < removal_savings:
                        # Profitable move
                        self._move_assignment_between_routes(
                            assignment, source_route, target_route,
                            distance_matrix, workday_start
                        )
                        improved = True
                        break
        
        return improved
    
    def _calculate_removal_cost(
        self, 
        route: TruckRoute, 
        assignment: JobAssignment
    ) -> float:
        """Calculate cost savings from removing an assignment."""
        # Simplified - return drive time savings
        return assignment.drive_minutes_from_previous
    
    def _move_assignment_between_routes(
        self,
        assignment: JobAssignment,
        source_route: TruckRoute,
        target_route: TruckRoute,
        distance_matrix: RouteMatrix,
        workday_start: datetime
    ) -> None:
        """Move an assignment between routes."""
        # Remove from source
        source_route.assignments.remove(assignment)
        
        # Add to target
        self._assign_job_to_route(
            assignment.job, assignment.job_items, target_route,
            {assignment.job.location_id: assignment.location_index},
            distance_matrix, workday_start
        )
        
        # Recalculate both routes
        self._recalculate_route_metrics(source_route, distance_matrix, workday_start)
        self._recalculate_route_metrics(target_route, distance_matrix, workday_start)
    
    def _recalculate_route_metrics(
        self,
        route: TruckRoute,
        distance_matrix: RouteMatrix,
        workday_start: datetime
    ) -> None:
        """Recalculate all route metrics after changes."""
        route.total_drive_minutes = 0.0
        route.total_service_minutes = 0.0
        route.total_weight_lb = 0.0
        route.overtime_minutes = 0.0
        
        current_time = workday_start
        prev_location_idx = 0  # Depot
        
        for i, assignment in enumerate(route.assignments):
            # Calculate drive time from previous location
            drive_time = distance_matrix.get_duration(prev_location_idx, assignment.location_index)
            
            # Update assignment timing
            assignment.stop_order = i
            assignment.drive_minutes_from_previous = drive_time
            assignment.estimated_arrival = current_time + timedelta(minutes=drive_time)
            assignment.estimated_departure = assignment.estimated_arrival + timedelta(
                minutes=assignment.service_minutes
            )
            
            # Update route totals
            route.total_drive_minutes += drive_time
            route.total_service_minutes += assignment.service_minutes
            route.total_weight_lb += sum(
                item.item.weight_lb_per_unit * item.qty 
                for item in assignment.job_items
            )
            
            # Update for next iteration
            current_time = assignment.estimated_departure
            prev_location_idx = assignment.location_index
        
        # Calculate overtime
        workday_end = workday_start.replace(
            hour=int(self.config.depot.workday_window.end[:2]),
            minute=int(self.config.depot.workday_window.end[3:]),
            second=0
        )
        
        if route.assignments and route.assignments[-1].estimated_departure > workday_end:
            overtime_delta = route.assignments[-1].estimated_departure - workday_end
            route.overtime_minutes = overtime_delta.total_seconds() / 60
