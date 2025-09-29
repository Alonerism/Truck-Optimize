"""
Repository layer for database operations.
Provides clean interface for CRUD operations on all entities.
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any
from sqlmodel import SQLModel, Session, create_engine, select, delete
from sqlalchemy.exc import IntegrityError

from .models import (
    Truck, Location, Item, Job, JobItem, 
    RouteAssignment, RouteStop, UnassignedJob,
    ActionType, ItemCategory
)
from .schemas import AppConfig


class DatabaseRepository:
    """Database repository for all truck optimizer entities."""
    
    def __init__(self, config: AppConfig):
        """Initialize database connection."""
        self.config = config
        self.engine = create_engine(
            config.database.url,
            echo=config.database.echo
        )
        
    def create_tables(self) -> None:
        """Create all database tables."""
        SQLModel.metadata.create_all(self.engine)

    def migrate_schema(self) -> None:
        """Lightweight, idempotent migrations for SQLite to add new columns.

        - Adds RouteAssignment.scenario (TEXT NULL)
        - Adds RouteStop planned_* and batch_* columns (NULLable)
        - Adds Job shipment_id (TEXT), shipment_role (TEXT CHECK), must_same_truck (BOOLEAN)
        - Adds RouteStop extended logging fields (arrival/service_start/depart locals, priority, earliest/latest strings, curfew_window, overtime_flag, shipment fields)
        New tables defined in models.py are created by create_all(); this method
        focuses on altering existing tables safely.
        """
        try:
            with self.engine.connect() as conn:
                # Helper: check if a column exists in a table
                def column_exists(table: str, column: str) -> bool:
                    rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
                    return any(r[1] == column for r in rows)

                # RouteAssignment.scenario
                if not column_exists("routeassignment", "scenario"):
                    conn.exec_driver_sql("ALTER TABLE routeassignment ADD COLUMN scenario TEXT")

                # RouteStop planned_* and batch_* columns
                rs_table = "routestop"
                add_cols = {
                    "planned_travel_minutes": "REAL",
                    "planned_service_minutes": "REAL",
                    "planned_wait_minutes": "REAL",
                    "batch_index": "INTEGER",
                    "batch_seq_in_batch": "INTEGER",
                    # Extended logging snapshot fields
                    "arrival_time_local": "TEXT",
                    "service_start_local": "TEXT",
                    "depart_time_local": "TEXT",
                    "priority": "INTEGER",
                    "earliest_str": "TEXT",
                    "latest_str": "TEXT",
                    "curfew_window": "TEXT",
                    "overtime_flag": "BOOLEAN",
                    "shipment_id": "TEXT",
                    "shipment_role": "TEXT"
                }
                for col, coltype in add_cols.items():
                    if not column_exists(rs_table, col):
                        conn.exec_driver_sql(f"ALTER TABLE {rs_table} ADD COLUMN {col} {coltype}")

                # Job shipment fields
                j_table = "job"
                if not column_exists(j_table, "shipment_id"):
                    conn.exec_driver_sql("ALTER TABLE job ADD COLUMN shipment_id TEXT")
                if not column_exists(j_table, "shipment_role"):
                    conn.exec_driver_sql("ALTER TABLE job ADD COLUMN shipment_role TEXT")
                if not column_exists(j_table, "must_same_truck"):
                    conn.exec_driver_sql("ALTER TABLE job ADD COLUMN must_same_truck BOOLEAN DEFAULT 1")

                # Add index for shipment_id
                try:
                    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_job_shipment ON job(shipment_id)")
                except Exception:
                    pass

                # UnassignedJob scenario column
                if not column_exists("unassignedjob", "scenario"):
                    conn.exec_driver_sql("ALTER TABLE unassignedjob ADD COLUMN scenario TEXT")
        except Exception:
            # Best-effort: avoid crashing app startup; log at DEBUG from caller if needed
            pass
        
    def get_session(self) -> Session:
        """Get database session."""
        return Session(self.engine)
    
    # Truck operations
    def get_trucks(self) -> List[Truck]:
        """Get all trucks."""
        with self.get_session() as session:
            return session.exec(select(Truck)).all()
    
    def get_truck_by_name(self, name: str) -> Optional[Truck]:
        """Get truck by name."""
        with self.get_session() as session:
            return session.exec(
                select(Truck).where(Truck.name == name)
            ).first()
    
    def create_truck(self, truck_data: Dict[str, Any]) -> Truck:
        """Create a new truck."""
        with self.get_session() as session:
            truck = Truck(**truck_data)
            session.add(truck)
            session.commit()
            session.refresh(truck)
            return truck
    
    def upsert_trucks(self, trucks_data: List[Dict[str, Any]]) -> List[Truck]:
        """Insert or update trucks from configuration."""
        trucks = []
        for truck_data in trucks_data:
            existing = self.get_truck_by_name(truck_data["name"])
            if existing:
                # Update existing truck
                with self.get_session() as session:
                    for key, value in truck_data.items():
                        if key != "name":  # Don't update the unique key
                            setattr(existing, key, value)
                    session.add(existing)
                    session.commit()
                    session.refresh(existing)
                    trucks.append(existing)
            else:
                # Create new truck
                trucks.append(self.create_truck(truck_data))
        return trucks
    
    # Location operations
    def get_locations(self) -> List[Location]:
        """Get all locations."""
        with self.get_session() as session:
            return session.exec(select(Location)).all()
    
    def get_location_by_name(self, name: str) -> Optional[Location]:
        """Get location by name."""
        with self.get_session() as session:
            return session.exec(
                select(Location).where(Location.name == name)
            ).first()
    
    def create_location(self, location_data: Dict[str, Any]) -> Location:
        """Create a new location."""
        with self.get_session() as session:
            location = Location(**location_data)
            session.add(location)
            session.commit()
            session.refresh(location)
            return location
    
    def update_location_coordinates(
        self, 
        location_id: int, 
        lat: float, 
        lon: float
    ) -> None:
        """Update location coordinates after geocoding."""
        with self.get_session() as session:
            location = session.get(Location, location_id)
            if location:
                location.lat = lat
                location.lon = lon
                session.add(location)
                session.commit()
    
    # Item operations
    def get_items(self) -> List[Item]:
        """Get all items."""
        with self.get_session() as session:
            return session.exec(select(Item)).all()
    
    def get_item_by_name(self, name: str) -> Optional[Item]:
        """Get item by name."""
        with self.get_session() as session:
            return session.exec(
                select(Item).where(Item.name == name)
            ).first()
    
    def create_item(self, item_data: Dict[str, Any]) -> Item:
        """Create a new item."""
        with self.get_session() as session:
            # Convert dims_lwh_ft list to JSON string if present
            if "dims_lwh_ft" in item_data and item_data["dims_lwh_ft"]:
                import json
                item_data["dims_lwh_ft"] = json.dumps(item_data["dims_lwh_ft"])
            
            item = Item(**item_data)
            session.add(item)
            session.commit()
            session.refresh(item)
            return item
    
    def upsert_items(self, items_data: List[Dict[str, Any]]) -> List[Item]:
        """Insert or update items from catalog."""
        items = []
        for item_data in items_data:
            existing = self.get_item_by_name(item_data["name"])
            if existing:
                # Update existing item
                with self.get_session() as session:
                    for key, value in item_data.items():
                        if key == "dims_lwh_ft" and value:
                            import json
                            value = json.dumps(value)
                        if key != "name":  # Don't update the unique key
                            setattr(existing, key, value)
                    session.add(existing)
                    session.commit()
                    session.refresh(existing)
                    items.append(existing)
            else:
                # Create new item
                items.append(self.create_item(item_data))
        return items
    
    # Job operations
    def get_jobs_by_date(self, target_date: str) -> List[Job]:
        """Get all jobs for a specific date with eager loading of relationships."""
        with self.get_session() as session:
            from sqlmodel import select
            from sqlalchemy.orm import selectinload
            
            # Eager load job_items and their related items and location
            jobs = session.exec(
                select(Job)
                .options(
                    selectinload(Job.job_items).selectinload(JobItem.item),
                    selectinload(Job.location)
                )
                .where(Job.date == target_date)
            ).all()
            
            # Trigger loading of all relationships while session is active
            for job in jobs:
                _ = job.job_items  # Force loading
                _ = job.location   # Force loading
                for job_item in job.job_items:
                    _ = job_item.item  # Force loading
            
            return jobs
    
    def create_job(self, job_data: Dict[str, Any]) -> Job:
        """Create a new job."""
        with self.get_session() as session:
            job = Job(**job_data)
            session.add(job)
            session.commit()
            session.refresh(job)
            return job
    
    def create_job_item(self, job_item_data: Dict[str, Any]) -> JobItem:
        """Create a job item association."""
        with self.get_session() as session:
            job_item = JobItem(**job_item_data)
            session.add(job_item)
            session.commit()
            session.refresh(job_item)
            return job_item
    
    def delete_jobs_by_date(self, target_date: str) -> int:
        """Delete all jobs for a specific date."""
        with self.get_session() as session:
            # Delete job items first (foreign key constraint)
            job_ids = session.exec(
                select(Job.id).where(Job.date == target_date)
            ).all()
            
            if job_ids:
                session.exec(
                    delete(JobItem).where(JobItem.job_id.in_(job_ids))
                )
                
                deleted_count = session.exec(
                    delete(Job).where(Job.date == target_date)
                ).rowcount
                
                session.commit()
                return deleted_count
            return 0
    
    # Route assignment operations
    def get_route_assignments_by_date(self, target_date: str) -> List[RouteAssignment]:
        """Get all route assignments for a date with eager loading."""
        with self.get_session() as session:
            from sqlalchemy.orm import selectinload
            
            assignments = session.exec(
                select(RouteAssignment)
                .options(selectinload(RouteAssignment.truck))
                .where(RouteAssignment.date == target_date)
            ).all()
            
            # Force loading while session is active
            for assignment in assignments:
                _ = assignment.truck
            
            return assignments
    
    def create_route_assignment(
        self, 
        assignment_data: Dict[str, Any]
    ) -> RouteAssignment:
        """Create a route assignment."""
        with self.get_session() as session:
            assignment = RouteAssignment(**assignment_data)
            session.add(assignment)
            session.commit()
            session.refresh(assignment)
            return assignment
    
    def create_route_stop(self, stop_data: Dict[str, Any]) -> RouteStop:
        """Create a route stop."""
        with self.get_session() as session:
            stop = RouteStop(**stop_data)
            session.add(stop)
            session.commit()
            session.refresh(stop)
            return stop
    
    def delete_route_assignments_by_date(self, target_date: str) -> int:
        """Delete all route assignments for a date."""
        with self.get_session() as session:
            # Get assignment IDs
            assignment_ids = session.exec(
                select(RouteAssignment.id).where(
                    RouteAssignment.date == target_date
                )
            ).all()
            
            if assignment_ids:
                # Delete route stops first
                session.exec(
                    delete(RouteStop).where(
                        RouteStop.route_assignment_id.in_(assignment_ids)
                    )
                )
                
                # Delete assignments
                deleted_count = session.exec(
                    delete(RouteAssignment).where(
                        RouteAssignment.date == target_date
                    )
                ).rowcount
                
                session.commit()
                return deleted_count
            return 0
    
    def get_route_stops_by_assignment(self, assignment_id: int) -> List[RouteStop]:
        """Get all route stops for a route assignment with eager loading."""
        with self.get_session() as session:
            from sqlalchemy.orm import selectinload
            
            stops = session.exec(
                select(RouteStop)
                .options(
                    selectinload(RouteStop.job).selectinload(Job.location)
                )
                .where(RouteStop.route_assignment_id == assignment_id)
            ).all()
            
            # Force loading while session is active
            for stop in stops:
                _ = stop.job
                _ = stop.job.location
            
            return stops
    
    # Unassigned jobs operations
    def get_unassigned_jobs_by_date(self, target_date: str) -> List[UnassignedJob]:
        """Get unassigned jobs for a date."""
        with self.get_session() as session:
            return session.exec(
                select(UnassignedJob).where(UnassignedJob.date == target_date)
            ).all()
    
    def create_unassigned_job(self, unassigned_data: Dict[str, Any]) -> UnassignedJob:
        """Create an unassigned job record."""
        with self.get_session() as session:
            unassigned = UnassignedJob(**unassigned_data)
            session.add(unassigned)
            session.commit()
            session.refresh(unassigned)
            return unassigned
    
    def delete_unassigned_jobs_by_date(self, target_date: str) -> int:
        """Delete all unassigned jobs for a date."""
        with self.get_session() as session:
            deleted_count = session.exec(
                delete(UnassignedJob).where(UnassignedJob.date == target_date)
            ).rowcount
            session.commit()
            return deleted_count
    
    # Utility operations
    def health_check(self) -> bool:
        """Check if database is accessible."""
        try:
            with self.get_session() as session:
                session.exec(select(1))
                return True
        except Exception:
            return False
