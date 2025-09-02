"""
Unit tests for the enhanced solver algorithms.
"""
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.solver_greedy import GreedySolver, TruckRoute, Solution
from app.models import Job, Truck, Location, JobItem, Item
from app.distance import RouteMatrix, Coordinates
from app.schemas import AppConfig


class TestRegret2Algorithm(unittest.TestCase):
    """Test cases for the regret-2 insertion algorithm."""
    
    def setUp(self):
        """Set up test case with mock data."""
        # Create mock config with proper structure
        self.config = Mock(spec=AppConfig)
        self.config.solver = Mock()
        self.config.solver.single_truck_mode = 0
        self.config.solver.trucks_used_penalty = 1000.0
        self.config.solver.random_seed = 42
        self.config.solver.weights = Mock()
        self.config.solver.weights.drive_minutes = 1.0
        self.config.solver.weights.service_minutes = 0.5
        self.config.solver.weights.overtime_minutes = 2.0
        self.config.solver.weights.max_route_minutes = 0.1
        self.config.solver.weights.priority_soft_cost = 0.2
        self.config.constraints = Mock()
        self.config.constraints.big_truck_co_load_threshold_minutes = 15
        
        # Create solver
        self.solver = GreedySolver(self.config)
        
        # Mock validator to always return no violations
        self.solver.validator = Mock()
        self.solver.validator.validate_job_assignment.return_value = []
        self.solver.validator.calculate_service_time.return_value = 15  # 15 minutes service time
        
        # Create test trucks
        self.trucks = [
            Truck(id=1, name="Truck A", max_weight_lb=10000, large_capable=True),
            Truck(id=2, name="Truck B", max_weight_lb=5000, large_capable=False)
        ]
        
        # Create test jobs
        self.jobs = [
            Job(id=1, location_id=1, action="delivery", priority=1),
            Job(id=2, location_id=2, action="delivery", priority=2),
            Job(id=3, location_id=3, action="pickup", priority=3),
            Job(id=4, location_id=4, action="delivery", priority=1)
        ]
        
        # Create locations
        self.locations = [
            Location(id=0, name="Depot", address="123 Depot St", latitude=0, longitude=0),
            Location(id=1, name="Loc A", address="123 A St", latitude=1, longitude=1),
            Location(id=2, name="Loc B", address="123 B St", latitude=2, longitude=2),
            Location(id=3, name="Loc C", address="123 C St", latitude=3, longitude=3),
            Location(id=4, name="Loc D", address="123 D St", latitude=4, longitude=4)
        ]
        
        # Create items and job items
        item = Item(id=1, name="concrete", category="material", weight_lb_per_unit=100)
        self.job_items_map = {
            1: [JobItem(job_id=1, item_id=1, item=item, qty=5)],
            2: [JobItem(job_id=2, item_id=1, item=item, qty=10)],
            3: [JobItem(job_id=3, item_id=1, item=item, qty=7)],
            4: [JobItem(job_id=4, item_id=1, item=item, qty=3)]
        }
        
        # Create mock distance matrix
        self.distance_matrix = Mock(spec=RouteMatrix)
        
        # Define distance behavior - all locations are 10 minutes apart
        def mock_get_duration(from_idx, to_idx):
            return 10 if from_idx != to_idx else 0
            
        self.distance_matrix.get_duration.side_effect = mock_get_duration
        
        # Create location to index mapping
        self.location_to_index = {loc.id: i for i, loc in enumerate(self.locations)}
        
        # Set workday start
        self.workday_start = datetime.now().replace(hour=8, minute=0)

    @patch('app.solver_greedy.GreedySolver._evaluate_job_insertion')
    @patch('app.solver_greedy.GreedySolver._assign_job_to_route')
    def test_regret2_construction(self, mock_assign, mock_evaluate):
        """Test that regret2 construction prioritizes high regret jobs."""
        # Setup mock behavior for _evaluate_job_insertion
        # We'll simulate different costs for each job-truck combination
        
        # Job 1: cost=20 for truck A, cost=30 for truck B (regret=10)
        # Job 2: cost=25 for truck A, cost=35 for truck B (regret=10) 
        # Job 3: cost=15 for truck A, cost=40 for truck B (regret=25) - highest regret
        # Job 4: cost=10 for truck A, cost=15 for truck B (regret=5)
        
        def evaluate_side_effect(job, job_items, route, job_loc_idx, matrix, workday_start, return_details=False):
            truck_id = route.truck.id
            job_id = job.id
            
            costs = {
                (1, 1): 20, (1, 2): 30,  # Job 1
                (2, 1): 25, (2, 2): 35,  # Job 2
                (3, 1): 15, (3, 2): 40,  # Job 3
                (4, 1): 10, (4, 2): 15   # Job 4
            }
            
            cost = costs.get((job_id, truck_id), 100)
            return cost, [], {"best_position": 0} if return_details else None
        
        mock_evaluate.side_effect = evaluate_side_effect
        
        # Initialize empty routes
        routes = [
            TruckRoute(truck=self.trucks[0], assignments=[], total_drive_minutes=0,
                      total_service_minutes=0, total_weight_lb=0, overtime_minutes=0),
            TruckRoute(truck=self.trucks[1], assignments=[], total_drive_minutes=0,
                      total_service_minutes=0, total_weight_lb=0, overtime_minutes=0)
        ]
        
        # Enable tracing to verify algorithm choices
        trace_data = {"decisions": []}
        
        # Execute regret2 algorithm
        unassigned = self.solver._build_solution_regret2(
            routes, self.jobs, self.job_items_map,
            self.location_to_index, self.distance_matrix,
            self.workday_start, trace_data
        )
        
        # Job 3 should be assigned first (highest regret)
        first_assignment_call = mock_assign.call_args_list[0]
        self.assertEqual(first_assignment_call[0][0].id, 3)
        self.assertEqual(first_assignment_call[0][2].truck.id, 1)  # Truck A
        
        # Verify all jobs were assigned
        self.assertEqual(len(unassigned), 0)
        
        # Verify correct number of assignments
        self.assertEqual(mock_assign.call_count, 4)
        
        # Verify algorithm was recorded in trace data
        self.assertEqual(trace_data["algorithm"], "regret2")


if __name__ == '__main__':
    unittest.main()
