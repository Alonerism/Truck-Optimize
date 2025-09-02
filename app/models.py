"""
Core data models for the Truck Optimizer.
Uses SQLModel for database ORM and Pydantic for validation.
"""

from datetime import datetime, time
from enum import Enum
from typing import List, Optional, Dict
from sqlmodel import SQLModel, Field, Relationship
from pydantic import BaseModel


class ActionType(str, Enum):
    """Job action types."""
    PICKUP = "pickup"
    DROP = "drop"


class ItemCategory(str, Enum):
    """Item categories for service time calculation."""
    MACHINE = "machine"
    EQUIPMENT = "equipment"
    MATERIAL = "material"
    FUEL = "fuel"


# Database Models (SQLModel)
class Truck(SQLModel, table=True):
    """Truck fleet configuration."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    max_weight_lb: float = Field(gt=0)
    bed_len_ft: float = Field(gt=0)
    bed_width_ft: float = Field(gt=0)
    height_limit_ft: Optional[float] = Field(default=None, gt=0)
    large_capable: bool = Field(default=False)
    
    # Relationships
    route_assignments: List["RouteAssignment"] = Relationship(back_populates="truck")


class Location(SQLModel, table=True):
    """Pickup and drop-off locations."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    address: str
    lat: Optional[float] = Field(default=None)
    lon: Optional[float] = Field(default=None)
    window_start: Optional[time] = Field(default=None)
    window_end: Optional[time] = Field(default=None)
    service_minutes_default: Optional[int] = Field(default=None, ge=0)
    
    # Relationships
    jobs: List["Job"] = Relationship(back_populates="location")


class Item(SQLModel, table=True):
    """Items that can be picked up or dropped off."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    category: ItemCategory
    weight_lb_per_unit: float = Field(ge=0)
    volume_ft3_per_unit: Optional[float] = Field(default=None, ge=0)
    dims_lwh_ft: Optional[str] = Field(default=None)  # JSON string: [length, width, height]
    requires_large_truck: bool = Field(default=False)
    
    # Relationships
    job_items: List["JobItem"] = Relationship(back_populates="item")


class Job(SQLModel, table=True):
    """Individual pickup or drop job."""
    id: Optional[int] = Field(default=None, primary_key=True)
    location_id: int = Field(foreign_key="location.id")
    action: ActionType
    priority: int = Field(default=1)  # Higher = more important, but soft
    date: str = Field(index=True)  # YYYY-MM-DD format for scheduled date
    earliest: Optional[datetime] = Field(default=None)
    latest: Optional[datetime] = Field(default=None)
    notes: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    location: Location = Relationship(back_populates="jobs")
    job_items: List["JobItem"] = Relationship(back_populates="job")
    route_stops: List["RouteStop"] = Relationship(back_populates="job")


class JobItem(SQLModel, table=True):
    """Items associated with a specific job."""
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id")
    item_id: int = Field(foreign_key="item.id")
    qty: float = Field(gt=0)
    notes: Optional[str] = Field(default=None)
    
    # Relationships
    job: Job = Relationship(back_populates="job_items")
    item: Item = Relationship(back_populates="job_items")


# Route optimization results
class RouteAssignment(SQLModel, table=True):
    """Optimized route assignment for a specific date."""
    id: Optional[int] = Field(default=None, primary_key=True)
    truck_id: int = Field(foreign_key="truck.id")
    date: str = Field(index=True)  # YYYY-MM-DD format
    total_drive_minutes: float = Field(ge=0)
    total_service_minutes: float = Field(ge=0)
    total_weight_lb: float = Field(ge=0)
    overtime_minutes: float = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    truck: Truck = Relationship(back_populates="route_assignments")
    route_stops: List["RouteStop"] = Relationship(back_populates="route_assignment")


class RouteStop(SQLModel, table=True):
    """Individual stop in an optimized route."""
    id: Optional[int] = Field(default=None, primary_key=True)
    route_assignment_id: int = Field(foreign_key="routeassignment.id")
    job_id: int = Field(foreign_key="job.id")
    stop_order: int = Field(ge=0)  # 0 = first stop after depot
    estimated_arrival: datetime
    estimated_departure: datetime
    drive_minutes_from_previous: float = Field(ge=0)
    service_minutes: float = Field(ge=0)
    
    # Relationships
    route_assignment: RouteAssignment = Relationship(back_populates="route_stops")
    job: Job = Relationship(back_populates="route_stops")


class UnassignedJob(SQLModel, table=True):
    """Jobs that couldn't be assigned due to constraints."""
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id")
    date: str = Field(index=True)  # YYYY-MM-DD format
    reason: str  # Why this job couldn't be assigned
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Pydantic models for API requests/responses
class TruckResponse(BaseModel):
    """Truck information for API responses."""
    id: int
    name: str
    max_weight_lb: float
    bed_len_ft: float
    bed_width_ft: float
    height_limit_ft: Optional[float]
    large_capable: bool


class LocationResponse(BaseModel):
    """Location information for API responses."""
    id: int
    name: str
    address: str
    lat: Optional[float]
    lon: Optional[float]
    window_start: Optional[time]
    window_end: Optional[time]


class JobResponse(BaseModel):
    """Job information for API responses."""
    id: int
    location: LocationResponse
    action: ActionType
    priority: int
    earliest: Optional[datetime]
    latest: Optional[datetime]
    notes: Optional[str]
    items: List[dict]  # [{item_name, category, qty, weight_lb_total}]


class RouteStopResponse(BaseModel):
    """Route stop for API responses."""
    job: JobResponse
    stop_order: int
    estimated_arrival: datetime
    estimated_departure: datetime
    drive_minutes_from_previous: float
    service_minutes: float


class RouteResponse(BaseModel):
    """Complete route for a truck."""
    truck: TruckResponse
    date: str
    stops: List[RouteStopResponse]
    total_drive_minutes: float
    total_service_minutes: float
    total_weight_lb: float
    overtime_minutes: float
    maps_url: str


class OptimizationResult(BaseModel):
    """Complete optimization result."""
    date: str
    routes: List[RouteResponse]
    unassigned_jobs: List[JobResponse]
    total_cost: float
    solver_used: str
    computation_time_seconds: float
    output_files: Optional[Dict[str, str]] = None


class OvertimeDecisionRequest(BaseModel):
    """When overtime exceeds threshold, present both options."""
    overtime_plan: OptimizationResult
    defer_plan: OptimizationResult
    overtime_minutes_diff: float
    jobs_deferred_count: int


# CSV Import models
class JobImportRow(BaseModel):
    """Single row from CSV import."""
    location: str
    action: ActionType
    items: str  # "item1:qty1; item2:qty2"
    priority: int = Field(default=1)
    notes: str = Field(default="")
    earliest: Optional[str] = Field(default=None)  # ISO format or None
    latest: Optional[str] = Field(default=None)    # ISO format or None
    service_minutes_override: Optional[int] = Field(default=None)


class ImportRequest(BaseModel):
    """Import request for CSV or JSON data."""
    data: List[JobImportRow]
    date: str  # YYYY-MM-DD
    clear_existing: bool = Field(default=False)


# Maps URL response
class MapsUrlResponse(BaseModel):
    """Google Maps URLs for route visualization."""
    truck_name: str
    urls: List[str]  # Segmented URLs if route exceeds waypoint limit
    total_stops: int
