"""
Tests for constraint validation logic.
"""

import pytest
from datetime import datetime, time
from dataclasses import dataclass
from typing import Optional

from app.constraints import ConstraintValidator, LoadInfo, ConstraintViolation
from app.models import Truck, Job, JobItem, Item, Location, ActionType, ItemCategory
from app.schemas import AppConfig, TruckConfig, DepotConfig, FleetConfig, WorkdayWindow


@dataclass
class MockItem:
    """Mock item for testing."""
    name: str
    category: ItemCategory
    weight_lb_per_unit: float
    requires_large_truck: bool = False
    volume_ft3_per_unit: Optional[float] = None


@dataclass
class MockJobItem:
    """Mock job item for testing."""
    item: MockItem
    qty: float


@dataclass
class MockLocation:
    """Mock location for testing."""
    id: int
    name: str
    window_start: time = None
    window_end: time = None


@dataclass 
class MockJob:
    """Mock job for testing."""
    id: int
    location_id: int
    action: ActionType
    priority: int = 1
    earliest: datetime = None
    latest: datetime = None
    location: MockLocation = None


def create_test_config():
    """Create test configuration."""
    return AppConfig(
        project={"name": "Test", "units": "imperial", "version": "0.1.0"},
        depot=DepotConfig(
            address="123 Test St",
            workday_window=WorkdayWindow(start="08:00", end="17:00")
        ),
        fleet=FleetConfig(trucks=[
            TruckConfig(
                name="Test Truck",
                max_weight_lb=5000,
                bed_len_ft=10,
                bed_width_ft=6,
                large_capable=True
            )
        ]),
        service_times={"by_category": {"machine": 30, "material": 15}, "default_location_service_minutes": 5},
        item_catalog=[],
        constraints={
            "big_truck_co_load_threshold_minutes": 15,
            "default_location_window_start": "08:00",
            "default_location_window_end": "17:00",
            "volume_checking_enabled": False,
            "weight_checking_enabled": True
        },
        overtime_deferral={"default_mode": "ask", "overtime_slack_minutes": 30, "defer_rule": "lowest_priority_first"},
        solver={
            "use_ortools": False,
            "random_seed": 42,
            "efficiency_weight": 1.0,
            "priority_weight": 0.1,
            "overtime_penalty_per_minute": 2.0,
            "local_search_iterations": 10,
            "improvement_threshold": 0.01
        },
        google={
            "traffic_model": "BEST_GUESS",
            "departure_time_offset_hours": 0,
            "max_retries": 3,
            "retry_delay_seconds": 1.0,
            "rate_limit_requests_per_second": 10,
            "maps": {"segment_max_waypoints": 9, "avoid": []}
        },
        database={"url": "sqlite:///:memory:", "echo": False},
        logging={"level": "INFO", "format": "%(message)s"},
        dev={"mock_google_api": True, "cache_geocoding": False}
    )


def test_weight_capacity_constraint():
    """Test weight capacity constraint validation."""
    config = create_test_config()
    validator = ConstraintValidator(config)
    
    # Create test data
    truck = Truck(
        name="Small Truck",
        max_weight_lb=1000,
        bed_len_ft=8,
        bed_width_ft=5,
        large_capable=False
    )
    
    location = MockLocation(id=1, name="Test Location")
    job = MockJob(id=1, location_id=1, action=ActionType.PICKUP, location=location)
    
    heavy_item = MockItem(
        name="heavy_item",
        category=ItemCategory.MATERIAL,
        weight_lb_per_unit=600
    )
    
    job_items = [MockJobItem(item=heavy_item, qty=2)]  # 1200 lbs total
    
    current_load = LoadInfo(total_weight_lb=0)
    estimated_time = datetime(2025, 9, 1, 10, 0)
    
    # Should violate weight constraint
    violations = validator.validate_job_assignment(
        job, job_items, truck, current_load, estimated_time
    )
    
    assert len(violations) == 1
    assert violations[0].violation_type == "weight_capacity"
    assert "1200" in violations[0].message


def test_large_truck_requirement():
    """Test large truck capability constraint."""
    config = create_test_config()
    validator = ConstraintValidator(config)
    
    # Small truck
    small_truck = Truck(
        name="Small Truck",
        max_weight_lb=5000,
        bed_len_ft=8,
        bed_width_ft=5,
        large_capable=False
    )
    
    location = MockLocation(id=1, name="Test Location")
    job = MockJob(id=1, location_id=1, action=ActionType.PICKUP, location=location)
    
    # Item that requires large truck
    big_item = MockItem(
        name="big_drill",
        category=ItemCategory.MACHINE,
        weight_lb_per_unit=100,
        requires_large_truck=True
    )
    
    job_items = [MockJobItem(item=big_item, qty=1)]
    current_load = LoadInfo()
    estimated_time = datetime(2025, 9, 1, 10, 0)
    
    # Should violate large truck requirement
    violations = validator.validate_job_assignment(
        job, job_items, small_truck, current_load, estimated_time
    )
    
    assert len(violations) == 1
    assert violations[0].violation_type == "large_truck_required"


def test_service_time_calculation():
    """Test service time calculation by category."""
    config = create_test_config()
    validator = ConstraintValidator(config)
    
    # Multiple items of different categories
    machine_item = MockItem("drill", ItemCategory.MACHINE, 100)
    material_item = MockItem("rebar", ItemCategory.MATERIAL, 50)
    
    job_items = [
        MockJobItem(item=machine_item, qty=1),
        MockJobItem(item=material_item, qty=5)
    ]
    
    service_time = validator.calculate_service_time(job_items)
    
    # Should be machine (30) + material (15) = 45 minutes
    assert service_time == 45


def test_time_window_validation():
    """Test time window constraint validation."""
    config = create_test_config()
    validator = ConstraintValidator(config)
    
    truck = Truck(name="Test", max_weight_lb=1000, bed_len_ft=8, bed_width_ft=5, large_capable=False)
    
    # Job with time window
    location = MockLocation(id=1, name="Test Location")
    job = MockJob(
        id=1,
        location_id=1,
        action=ActionType.DROP,
        earliest=datetime(2025, 9, 1, 14, 0),  # 2 PM
        latest=datetime(2025, 9, 1, 16, 0),    # 4 PM
        location=location
    )
    
    job_items = []
    current_load = LoadInfo()
    
    # Test arrival too early
    early_time = datetime(2025, 9, 1, 13, 0)  # 1 PM
    violations = validator.validate_job_assignment(
        job, job_items, truck, current_load, early_time
    )
    
    assert any(v.violation_type == "too_early" for v in violations)
    
    # Test arrival too late
    late_time = datetime(2025, 9, 1, 17, 0)  # 5 PM
    violations = validator.validate_job_assignment(
        job, job_items, truck, current_load, late_time
    )
    
    assert any(v.violation_type == "too_late" for v in violations)
    
    # Test valid time
    valid_time = datetime(2025, 9, 1, 15, 0)  # 3 PM
    violations = validator.validate_job_assignment(
        job, job_items, truck, current_load, valid_time
    )
    
    # Should have no time window violations
    time_violations = [v for v in violations if "time" in v.violation_type or "early" in v.violation_type or "late" in v.violation_type]
    assert len(time_violations) == 0


def test_overtime_threshold():
    """Test overtime threshold checking."""
    config = create_test_config()
    validator = ConstraintValidator(config)
    
    # Overtime within threshold
    assert not validator.check_overtime_threshold(20)  # Below 30-minute threshold
    
    # Overtime exceeding threshold
    assert validator.check_overtime_threshold(45)  # Above 30-minute threshold


def test_co_load_policy():
    """Test co-loading policy evaluation."""
    config = create_test_config()
    validator = ConstraintValidator(config)
    
    # Test case where co-loading is beneficial
    big_truck_time = 120  # 2 hours
    small_truck_time = 90  # 1.5 hours
    co_load_time = 130    # 2 hours 10 minutes (+10 minutes)
    
    # Should prefer co-loading (within 15-minute threshold and total time is less)
    assert validator.check_co_load_policy(big_truck_time, small_truck_time, co_load_time)
    
    # Test case where co-loading adds too much time
    co_load_time_high = 150  # 2.5 hours (+30 minutes)
    
    # Should not prefer co-loading (exceeds 15-minute threshold)
    assert not validator.check_co_load_policy(big_truck_time, small_truck_time, co_load_time_high)


if __name__ == "__main__":
    pytest.main([__file__])
