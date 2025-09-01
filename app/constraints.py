"""
Constraint validation for truck routing optimization.
Handles capacity, capability, time window, and policy constraints.
"""

import json
from datetime import datetime, time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .models import Truck, Job, JobItem, Item, Location
from .schemas import AppConfig


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    job_id: int
    truck_id: int
    violation_type: str
    message: str
    severity: str = "error"  # error, warning


@dataclass
class LoadInfo:
    """Information about current truck load."""
    total_weight_lb: float = 0.0
    total_volume_ft3: float = 0.0
    requires_large_truck: bool = False
    item_categories: List[str] = None
    
    def __post_init__(self):
        if self.item_categories is None:
            self.item_categories = []


class ConstraintValidator:
    """Validates routing constraints for truck assignments."""
    
    def __init__(self, config: AppConfig):
        """Initialize with configuration."""
        self.config = config
    
    def validate_job_assignment(
        self,
        job: Job,
        job_items: List[JobItem],
        truck: Truck,
        current_load: LoadInfo,
        estimated_time: datetime
    ) -> List[ConstraintViolation]:
        """
        Validate if a job can be assigned to a truck.
        
        Args:
            job: The job to validate
            job_items: Items associated with the job
            truck: The truck for assignment
            current_load: Current load on the truck
            estimated_time: Estimated arrival time at job location
            
        Returns:
            List of constraint violations (empty if valid)
        """
        violations = []
        
        # Calculate job load requirements
        job_load = self._calculate_job_load(job_items)
        
        # Check capacity constraints
        violations.extend(
            self._check_capacity_constraints(job, truck, current_load, job_load)
        )
        
        # Check capability constraints
        violations.extend(
            self._check_capability_constraints(job, truck, job_load)
        )
        
        # Check time window constraints
        violations.extend(
            self._check_time_window_constraints(job, estimated_time)
        )
        
        return violations
    
    def _calculate_job_load(self, job_items: List[JobItem]) -> LoadInfo:
        """Calculate the load requirements for a job."""
        load = LoadInfo()
        
        for job_item in job_items:
            item = job_item.item
            qty = job_item.qty
            
            # Weight calculation
            load.total_weight_lb += item.weight_lb_per_unit * qty
            
            # Volume calculation (if available)
            if item.volume_ft3_per_unit:
                load.total_volume_ft3 += item.volume_ft3_per_unit * qty
            
            # Check if any item requires large truck
            if item.requires_large_truck:
                load.requires_large_truck = True
            
            # Track item categories
            if item.category not in load.item_categories:
                load.item_categories.append(item.category)
        
        return load
    
    def _check_capacity_constraints(
        self,
        job: Job,
        truck: Truck,
        current_load: LoadInfo,
        job_load: LoadInfo
    ) -> List[ConstraintViolation]:
        """Check weight and volume capacity constraints."""
        violations = []
        
        # Weight constraint
        if self.config.constraints.weight_checking_enabled:
            total_weight = current_load.total_weight_lb + job_load.total_weight_lb
            if total_weight > truck.max_weight_lb:
                violations.append(ConstraintViolation(
                    job_id=job.id,
                    truck_id=truck.id,
                    violation_type="weight_capacity",
                    message=f"Weight capacity exceeded: {total_weight:.1f}lb > {truck.max_weight_lb}lb"
                ))
        
        # Volume constraint (if enabled and data available)
        if self.config.constraints.volume_checking_enabled:
            bed_volume = truck.bed_len_ft * truck.bed_width_ft * (truck.height_limit_ft or 8)
            total_volume = current_load.total_volume_ft3 + job_load.total_volume_ft3
            if total_volume > bed_volume:
                violations.append(ConstraintViolation(
                    job_id=job.id,
                    truck_id=truck.id,
                    violation_type="volume_capacity",
                    message=f"Volume capacity exceeded: {total_volume:.1f}ft³ > {bed_volume:.1f}ft³"
                ))
        
        return violations
    
    def _check_capability_constraints(
        self,
        job: Job,
        truck: Truck,
        job_load: LoadInfo
    ) -> List[ConstraintViolation]:
        """Check truck capability constraints."""
        violations = []
        
        # Large truck requirement
        if job_load.requires_large_truck and not truck.large_capable:
            violations.append(ConstraintViolation(
                job_id=job.id,
                truck_id=truck.id,
                violation_type="large_truck_required",
                message=f"Job requires large truck capability, but {truck.name} is not large_capable"
            ))
        
        return violations
    
    def _check_time_window_constraints(
        self,
        job: Job,
        estimated_time: datetime
    ) -> List[ConstraintViolation]:
        """Check time window constraints."""
        violations = []
        
        # Check job-specific time windows
        if job.earliest and estimated_time < job.earliest:
            violations.append(ConstraintViolation(
                job_id=job.id,
                truck_id=0,  # Not truck-specific
                violation_type="too_early",
                message=f"Arrival {estimated_time} before earliest allowed {job.earliest}",
                severity="warning"  # Might be soft constraint
            ))
        
        if job.latest and estimated_time > job.latest:
            violations.append(ConstraintViolation(
                job_id=job.id,
                truck_id=0,
                violation_type="too_late",
                message=f"Arrival {estimated_time} after latest allowed {job.latest}"
            ))
        
        # Check location time windows (if location has them)
        location = job.location
        if location.window_start or location.window_end:
            arrival_time = estimated_time.time()
            
            window_start = location.window_start or time.fromisoformat(
                self.config.constraints.default_location_window_start
            )
            window_end = location.window_end or time.fromisoformat(
                self.config.constraints.default_location_window_end
            )
            
            if arrival_time < window_start:
                violations.append(ConstraintViolation(
                    job_id=job.id,
                    truck_id=0,
                    violation_type="location_window_early",
                    message=f"Arrival {arrival_time} before location opens at {window_start}"
                ))
            
            if arrival_time > window_end:
                violations.append(ConstraintViolation(
                    job_id=job.id,
                    truck_id=0,
                    violation_type="location_window_late",
                    message=f"Arrival {arrival_time} after location closes at {window_end}"
                ))
        
        return violations
    
    def calculate_service_time(self, job_items: List[JobItem]) -> int:
        """Calculate service time for a job based on item categories."""
        total_service_minutes = 0
        categories_seen = set()
        
        for job_item in job_items:
            category = job_item.item.category
            if category not in categories_seen:
                # Add base service time for this category
                total_service_minutes += self.config.service_times.by_category.get(
                    category, 
                    self.config.service_times.default_location_service_minutes
                )
                categories_seen.add(category)
        
        # Minimum service time
        return max(total_service_minutes, self.config.service_times.default_location_service_minutes)
    
    def check_co_load_policy(
        self,
        big_truck_route_time: float,
        small_truck_route_time: float,
        co_load_route_time: float
    ) -> bool:
        """
        Check if co-loading on big truck is preferable to separate small truck.
        
        Args:
            big_truck_route_time: Current big truck route time (minutes)
            small_truck_route_time: Time for separate small truck route (minutes)
            co_load_route_time: Time for big truck with additional co-load (minutes)
            
        Returns:
            True if co-loading is preferred
        """
        time_increase = co_load_route_time - big_truck_route_time
        threshold = self.config.constraints.big_truck_co_load_threshold_minutes
        
        # Prefer co-load if:
        # 1. Time increase is within threshold, AND
        # 2. Total time is less than using separate truck
        return (
            time_increase <= threshold and 
            co_load_route_time < (big_truck_route_time + small_truck_route_time)
        )
    
    def validate_route_overtime(
        self,
        route_end_time: datetime,
        workday_end: datetime
    ) -> Tuple[bool, float]:
        """
        Check if route results in overtime.
        
        Args:
            route_end_time: When the route is expected to finish
            workday_end: End of normal workday
            
        Returns:
            (is_overtime, overtime_minutes)
        """
        if route_end_time <= workday_end:
            return False, 0.0
        
        overtime_delta = route_end_time - workday_end
        overtime_minutes = overtime_delta.total_seconds() / 60
        
        return True, overtime_minutes
    
    def check_overtime_threshold(self, overtime_minutes: float) -> bool:
        """Check if overtime exceeds the configured threshold."""
        return overtime_minutes > self.config.overtime_deferral.overtime_slack_minutes
