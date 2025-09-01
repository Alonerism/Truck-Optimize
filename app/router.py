"""
Router module for coordinating route optimization strategies.
Handles solver selection, fallback logic, and solution comparison.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

from .models import Truck, Job, JobItem, Location
from .distance import RouteMatrix, Coordinates
from .solver_greedy import GreedySolver, Solution
from .solver_ortools import ORToolsSolver
from .schemas import AppConfig


logger = logging.getLogger(__name__)


class RouteOptimizer:
    """High-level route optimizer that coordinates different solving strategies."""
    
    def __init__(self, config: AppConfig):
        """Initialize route optimizer with configuration."""
        self.config = config
        
        # Initialize solvers
        self.greedy_solver = GreedySolver(config)
        
        # OR-Tools solver (optional)
        self.ortools_solver = None
        if config.solver.use_ortools:
            try:
                self.ortools_solver = ORToolsSolver(config)
                logger.info("OR-Tools solver initialized")
            except Exception as e:
                logger.warning(f"OR-Tools solver initialization failed: {e}")
                logger.info("Falling back to greedy solver only")
    
    def optimize(
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
        Optimize routes using the configured solver strategy.
        
        Args:
            trucks: Available trucks
            jobs: Jobs to assign
            job_items_map: Mapping from job_id to list of JobItems
            locations: All locations (including depot at index 0)
            distance_matrix: Travel time matrix between locations
            depot_coords: Depot coordinates
            workday_start: Start time of workday
            
        Returns:
            Best solution found
        """
        solutions = []
        
        # Try OR-Tools solver if enabled and available
        if self.config.solver.use_ortools and self.ortools_solver:
            try:
                logger.info("Running OR-Tools solver...")
                ortools_solution = self.ortools_solver.solve(
                    trucks, jobs, job_items_map, locations,
                    distance_matrix, depot_coords, workday_start
                )
                solutions.append(("OR-Tools", ortools_solution))
                logger.info(f"OR-Tools completed: cost={ortools_solution.total_cost:.2f}, "
                           f"time={ortools_solution.computation_time_seconds:.2f}s")
            except Exception as e:
                logger.error(f"OR-Tools solver failed: {e}")
        
        # Always run greedy solver as baseline/fallback
        logger.info("Running greedy solver...")
        greedy_solution = self.greedy_solver.solve(
            trucks, jobs, job_items_map, locations,
            distance_matrix, depot_coords, workday_start
        )
        solutions.append(("Greedy", greedy_solution))
        logger.info(f"Greedy completed: cost={greedy_solution.total_cost:.2f}, "
                   f"time={greedy_solution.computation_time_seconds:.2f}s")
        
        # Select best solution
        if len(solutions) == 1:
            best_solver, best_solution = solutions[0]
        else:
            # Compare solutions and select best
            best_solver, best_solution = self._select_best_solution(solutions)
        
        logger.info(f"Selected solution from {best_solver} solver")
        return best_solution
    
    def _select_best_solution(
        self, 
        solutions: List[tuple[str, Solution]]
    ) -> tuple[str, Solution]:
        """
        Select the best solution from multiple candidates.
        
        Args:
            solutions: List of (solver_name, solution) tuples
            
        Returns:
            (best_solver_name, best_solution)
        """
        if not solutions:
            raise ValueError("No solutions to compare")
        
        if len(solutions) == 1:
            return solutions[0]
        
        # Score each solution
        scored_solutions = []
        
        for solver_name, solution in solutions:
            score = self._score_solution(solution)
            scored_solutions.append((score, solver_name, solution))
            logger.debug(f"{solver_name} solution score: {score:.2f}")
        
        # Sort by score (lower is better)
        scored_solutions.sort(key=lambda x: x[0])
        
        best_score, best_solver, best_solution = scored_solutions[0]
        logger.info(f"Best solution: {best_solver} (score: {best_score:.2f})")
        
        return best_solver, best_solution
    
    def _score_solution(self, solution: Solution) -> float:
        """
        Score a solution for comparison purposes.
        Lower scores are better.
        
        Args:
            solution: Solution to score
            
        Returns:
            Numerical score (lower = better)
        """
        # Base score from solution cost
        score = solution.total_cost
        
        # Penalty for unassigned jobs
        unassigned_penalty = len(solution.unassigned_jobs) * 1000
        score += unassigned_penalty
        
        # Penalty for infeasible solutions
        if not solution.feasible:
            score += 10000
        
        # Small bonus for faster computation (exploration vs exploitation)
        if solution.computation_time_seconds > 0:
            # Normalize computation time bonus (max 100 points)
            time_bonus = min(100, solution.computation_time_seconds / 10)
            score += time_bonus
        
        return score
    
    def compare_solutions(
        self, 
        solution1: Solution, 
        solution2: Solution
    ) -> Dict[str, Any]:
        """
        Compare two solutions and provide detailed analysis.
        
        Args:
            solution1: First solution
            solution2: Second solution
            
        Returns:
            Comparison analysis
        """
        comparison = {
            "solution1": {
                "cost": solution1.total_cost,
                "routes": len(solution1.routes),
                "unassigned": len(solution1.unassigned_jobs),
                "feasible": solution1.feasible,
                "computation_time": solution1.computation_time_seconds,
                "score": self._score_solution(solution1)
            },
            "solution2": {
                "cost": solution2.total_cost,
                "routes": len(solution2.routes),
                "unassigned": len(solution2.unassigned_jobs),
                "feasible": solution2.feasible,
                "computation_time": solution2.computation_time_seconds,
                "score": self._score_solution(solution2)
            }
        }
        
        # Calculate differences
        comparison["differences"] = {
            "cost_diff": solution2.total_cost - solution1.total_cost,
            "routes_diff": len(solution2.routes) - len(solution1.routes),
            "unassigned_diff": len(solution2.unassigned_jobs) - len(solution1.unassigned_jobs),
            "time_diff": solution2.computation_time_seconds - solution1.computation_time_seconds,
            "score_diff": comparison["solution2"]["score"] - comparison["solution1"]["score"]
        }
        
        # Determine winner
        if comparison["solution1"]["score"] < comparison["solution2"]["score"]:
            comparison["winner"] = "solution1"
        elif comparison["solution2"]["score"] < comparison["solution1"]["score"]:
            comparison["winner"] = "solution2"
        else:
            comparison["winner"] = "tie"
        
        return comparison
    
    def validate_solution(self, solution: Solution) -> Dict[str, Any]:
        """
        Validate a solution for constraint violations and consistency.
        
        Args:
            solution: Solution to validate
            
        Returns:
            Validation report
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Check route consistency
        for i, route in enumerate(solution.routes):
            route_prefix = f"Route {i+1} ({route.truck.name})"
            
            # Check weight capacity
            if route.total_weight_lb > route.truck.max_weight_lb:
                validation["errors"].append(
                    f"{route_prefix}: Weight exceeds capacity "
                    f"({route.total_weight_lb:.1f} > {route.truck.max_weight_lb})"
                )
                validation["valid"] = False
            
            # Check assignment ordering
            for j, assignment in enumerate(route.assignments):
                if assignment.stop_order != j:
                    validation["errors"].append(
                        f"{route_prefix}, Stop {j+1}: Incorrect stop order "
                        f"(expected {j}, got {assignment.stop_order})"
                    )
                    validation["valid"] = False
            
            # Check timing consistency
            current_time = None
            for assignment in route.assignments:
                if current_time and assignment.estimated_arrival < current_time:
                    validation["errors"].append(
                        f"{route_prefix}: Arrival time inconsistency at stop {assignment.stop_order + 1}"
                    )
                    validation["valid"] = False
                
                if assignment.estimated_departure < assignment.estimated_arrival:
                    validation["errors"].append(
                        f"{route_prefix}: Departure before arrival at stop {assignment.stop_order + 1}"
                    )
                    validation["valid"] = False
                
                current_time = assignment.estimated_departure
        
        # Check for duplicate job assignments
        assigned_job_ids = set()
        for route in solution.routes:
            for assignment in route.assignments:
                if assignment.job.id in assigned_job_ids:
                    validation["errors"].append(
                        f"Job {assignment.job.id} assigned to multiple routes"
                    )
                    validation["valid"] = False
                assigned_job_ids.add(assignment.job.id)
        
        # Statistics
        total_jobs = len(assigned_job_ids) + len(solution.unassigned_jobs)
        validation["statistics"] = {
            "total_jobs": total_jobs,
            "assigned_jobs": len(assigned_job_ids),
            "unassigned_jobs": len(solution.unassigned_jobs),
            "assignment_rate": len(assigned_job_ids) / total_jobs if total_jobs > 0 else 0,
            "active_routes": len([r for r in solution.routes if r.assignments]),
            "total_routes": len(solution.routes)
        }
        
        return validation
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get information about available solvers."""
        info = {
            "greedy": {
                "available": True,
                "description": "Greedy nearest-neighbor with local search",
                "features": ["fast", "deterministic", "good_baseline"]
            },
            "ortools": {
                "available": self.ortools_solver is not None,
                "description": "Google OR-Tools VRP solver",
                "features": ["optimal", "constraint_handling", "scalable"]
            },
            "current_strategy": "ortools" if self.config.solver.use_ortools else "greedy"
        }
        
        return info
