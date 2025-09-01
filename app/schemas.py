"""
Pydantic schemas for configuration, settings, and API validation.
"""

from datetime import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class TruckConfig(BaseModel):
    """Truck configuration from params.yaml."""
    name: str
    max_weight_lb: float = Field(gt=0)
    bed_len_ft: float = Field(gt=0)
    bed_width_ft: float = Field(gt=0)
    height_limit_ft: Optional[float] = Field(default=None, gt=0)
    large_capable: bool = Field(default=False)


class WorkdayWindow(BaseModel):
    """Time window for workday operations."""
    start: str = Field(pattern=r"^\d{2}:\d{2}$")  # HH:MM format
    end: str = Field(pattern=r"^\d{2}:\d{2}$")    # HH:MM format
    
    @validator('end')
    def end_after_start(cls, v, values):
        """Ensure end time is after start time."""
        if 'start' in values:
            start_time = time.fromisoformat(values['start'])
            end_time = time.fromisoformat(v)
            if end_time <= start_time:
                raise ValueError('End time must be after start time')
        return v


class DepotConfig(BaseModel):
    """Depot configuration."""
    address: str
    workday_window: WorkdayWindow


class FleetConfig(BaseModel):
    """Fleet configuration."""
    trucks: List[TruckConfig]


class ServiceTimesConfig(BaseModel):
    """Service time configuration."""
    by_category: Dict[str, int] = Field(
        description="Service minutes by item category"
    )
    default_location_service_minutes: int = Field(default=5, ge=0)


class ItemCatalogEntry(BaseModel):
    """Item catalog entry from params.yaml."""
    name: str
    category: str = Field(pattern="^(machine|equipment|material|fuel)$")
    weight_lb_per_unit: float = Field(ge=0)
    dims_lwh_ft: Optional[List[float]] = None
    requires_large_truck: bool = Field(default=False)


class ConstraintsConfig(BaseModel):
    """Constraint configuration."""
    big_truck_co_load_threshold_minutes: int = Field(default=15, ge=0)
    default_location_window_start: str = Field(default="07:00")
    default_location_window_end: str = Field(default="16:30")
    volume_checking_enabled: bool = Field(default=False)
    weight_checking_enabled: bool = Field(default=True)


class OvertimeDeferralConfig(BaseModel):
    """Overtime and deferral policy configuration."""
    default_mode: str = Field(default="ask", pattern="^(ask|overtime|defer)$")
    overtime_slack_minutes: int = Field(default=30, ge=0)
    defer_rule: str = Field(default="lowest_priority_first")


class SolverConfig(BaseModel):
    """Solver configuration."""
    use_ortools: bool = Field(default=False)
    random_seed: int = Field(default=42)
    efficiency_weight: float = Field(default=1.0, gt=0)
    priority_weight: float = Field(default=0.1, ge=0)
    overtime_penalty_per_minute: float = Field(default=2.0, ge=0)
    local_search_iterations: int = Field(default=100, ge=1)
    improvement_threshold: float = Field(default=0.01, gt=0)


class GoogleMapsConfig(BaseModel):
    """Google Maps API configuration."""
    segment_max_waypoints: int = Field(default=9, ge=2, le=25)
    avoid: List[str] = Field(default_factory=list)


class GoogleConfig(BaseModel):
    """Google API configuration."""
    traffic_model: str = Field(
        default="BEST_GUESS", 
        pattern="^(BEST_GUESS|OPTIMISTIC|PESSIMISTIC)$"
    )
    departure_time_offset_hours: int = Field(default=0, ge=0, le=24)
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0, gt=0)
    rate_limit_requests_per_second: int = Field(default=10, ge=1, le=100)
    maps: GoogleMapsConfig = Field(default_factory=GoogleMapsConfig)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = Field(default="sqlite:///./truck_optimizer.db")
    echo: bool = Field(default=False)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(
        default="INFO", 
        pattern="^(DEBUG|INFO|WARNING|ERROR)$"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


class DevConfig(BaseModel):
    """Development and testing configuration."""
    mock_google_api: bool = Field(default=False)
    cache_geocoding: bool = Field(default=True)


class ProjectConfig(BaseModel):
    """Top-level project configuration."""
    name: str = Field(default="Concrete Truck Optimizer")
    units: str = Field(default="imperial")
    version: str = Field(default="0.1.0")


class AppConfig(BaseModel):
    """Complete application configuration loaded from params.yaml."""
    project: ProjectConfig
    depot: DepotConfig
    fleet: FleetConfig
    service_times: ServiceTimesConfig
    item_catalog: List[ItemCatalogEntry]
    constraints: ConstraintsConfig
    overtime_deferral: OvertimeDeferralConfig
    solver: SolverConfig
    google: GoogleConfig
    database: DatabaseConfig
    logging: LoggingConfig
    dev: DevConfig = Field(default_factory=DevConfig)


class Settings(BaseSettings):
    """Environment-based settings (primarily for secrets)."""
    google_maps_api_key: Optional[str] = Field(default=None, env="GOOGLE_MAPS_API_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# API Request/Response Schemas
class JobImportRow(BaseModel):
    """Single row from CSV import."""
    location: str
    action: str  # Will be validated as ActionType in service
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


class OptimizeRequest(BaseModel):
    """Request parameters for optimization endpoint."""
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")  # YYYY-MM-DD
    auto: str = Field(default="ask", pattern="^(ask|overtime|defer)$")
    seed: Optional[int] = Field(default=None)


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration parameters."""
    updates: Dict[str, Any] = Field(
        description="Nested dictionary of configuration updates"
    )


class ImportStatsResponse(BaseModel):
    """Response from import operation."""
    locations_created: int
    locations_updated: int
    items_created: int
    jobs_created: int
    total_job_items: int
    geocoding_requests: int
    errors: List[str] = Field(default_factory=list)


class KPIResponse(BaseModel):
    """Key performance indicators for a route plan."""
    total_drive_minutes: float
    total_service_minutes: float
    total_overtime_minutes: float
    trucks_used: int
    jobs_assigned: int
    jobs_unassigned: int
    efficiency_score: float  # Computed metric
    priority_score: float    # Computed metric


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    database_connected: bool
    google_api_configured: bool
    timestamp: str
