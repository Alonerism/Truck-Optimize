"""
Main service layer for truck optimization.
Orchestrates data import, optimization, and result generation.
"""

import logging
import json
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
from .solver_priority import PrioritySolver
from .solver_ortools import ORToolsSolver
from .url_builder import GoogleMapsUrlBuilder
from .constraints import ConstraintValidator
from .integrations.google_directions import fetch_route_overview, audit_route
from .maps.google_map import render_map_html, render_day_csv_map_html
from .util.gmaps_link import build_driver_link
from .audit.eta_audit import compare_offline_vs_google, save_audit, append_learned_ratios
from .reporting_summary import print_priority_tradeoff_hint, print_run_summary
from app.schemas import Settings

logger = logging.getLogger(__name__)
settings = Settings()

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
        # Apply idempotent migrations to add new columns safely
        try:
            self.repo.migrate_schema()
        except Exception:
            # Best-effort; continue startup even if migration helper fails
            pass
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

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply dot-path overrides onto loaded config (CLI > YAML)."""
        if not overrides:
            return
        def set_dot(obj, path, val):
            parts = path.split('.')
            cur = obj
            for p in parts[:-1]:
                cur = getattr(cur, p)
            setattr(cur, parts[-1], val)
        for k, v in overrides.items():
            try:
                set_dot(self.config, k, v)
            except Exception as e:
                logger.warning(f"Override failed for {k}: {e}")
    
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
            unique_locations: Dict[str, Optional[str]] = {}
            for row in request.data:
                loc_name = getattr(row, 'location_name', None) or getattr(row, 'location', None)
                addr = getattr(row, 'address', None)
                if loc_name:
                    unique_locations.setdefault(loc_name, addr)
        
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
        locations_map: Dict[str, Optional[str]], 
        stats: ImportStatsResponse
    ) -> Dict[str, Coordinates]:
        """Process and geocode locations."""
        location_coords: Dict[str, Coordinates] = {}
        # Map address string we will geocode -> location_name
        addr_to_name: Dict[str, str] = {}
        
        for location_name, given_address in locations_map.items():
            existing_location = self.repo.get_location_by_name(location_name)
            
            if existing_location:
                if existing_location.lat and existing_location.lon:
                    # Already has coordinates
                    location_coords[location_name] = Coordinates(
                        lat=existing_location.lat,
                        lon=existing_location.lon
                    )
                else:
                    # Needs geocoding; prefer stored address then given address, else name
                    address = existing_location.address or given_address or location_name
                    addr_to_name[address] = location_name
                    stats.locations_updated += 1
            else:
                # New location
                new_location = self.repo.create_location({
                    "name": location_name,
                    "address": given_address or location_name,
                })
                addr_to_name[new_location.address] = location_name
                stats.locations_created += 1
        
        # Geocode addresses
        if addr_to_name:
            geocoding_results = await self.distance_provider.geocode_locations(list(addr_to_name.keys()))
            stats.geocoding_requests += len(addr_to_name)
            
            for address, coords in geocoding_results.items():
                loc_name = addr_to_name.get(address)
                if coords and loc_name:
                    # Update location with coordinates
                    location = self.repo.get_location_by_name(loc_name)
                    if location:
                        self.repo.update_location_coordinates(location.id, coords.lat, coords.lon)
                        location_coords[loc_name] = coords
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
        # Get location (new: location_name; legacy: location)
        loc_name = getattr(row, 'location_name', None) or getattr(row, 'location', None)
        if not loc_name:
            raise ValueError("Missing location_name")
        location = self.repo.get_location_by_name(loc_name)
        if not location:
            raise ValueError(f"Location not found: {loc_name}")
        
        # Parse times
        earliest = None
        latest = None
        if row.earliest:
            earliest = datetime.fromisoformat(row.earliest)
        if row.latest:
            latest = datetime.fromisoformat(row.latest)
        
        # Create job
        # Coerce action to enum value
        action_val = row.action.value if hasattr(row.action, 'value') else (row.action.lower() if isinstance(row.action, str) else row.action)
        job_data = {
            "location_id": location.id,
            "action": ActionType(action_val) if isinstance(action_val, str) else action_val,
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
        # Keep a reference to the request for downstream save hooks (e.g., scenario)
        self.last_request = request
        
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
        
        # Calculate time reference
        workday_start = datetime.fromisoformat(f"{request.date}T{self.config.depot.workday_window.start}:00")

        # Run optimization
        distance_matrix = None  # ensure defined for downstream calls (e.g., overtime decision)
        if request.scenario == "priority":
            # Use PrioritySolver with provided parameters
            params = request.params or {}
            # Build params from known knobs in request if present
            # The user might pass a dict externally; keep this flexible by accepting method arg
            prio_solver = PrioritySolver(self.config, self.distance_provider.travel_time)
            solution = prio_solver.solve(
                trucks=trucks,
                jobs=jobs,
                job_items_map=job_items_map,
                locations=locations,
                depot_coords=depot_coords,
                date=request.date,
                params=params,
            )
        elif self.config.solver.use_ortools:
            # Offline OR-Tools path: do not call Google matrix during solve
            solver = ORToolsSolver(self.config)
            # Build a trivial mock matrix to satisfy interface; OR-Tools will use offline matrices internally
            from .distance import RouteMatrix
            mock_matrix = RouteMatrix(origins=location_coords, destinations=location_coords,
                                      durations_minutes=[[0.0]*len(location_coords) for _ in location_coords],
                                      distances_meters=[[0.0]*len(location_coords) for _ in location_coords])
            distance_matrix = mock_matrix
            solution = solver.solve(
                trucks=trucks,
                jobs=jobs,
                job_items_map=job_items_map,
                locations=locations,
                distance_matrix=mock_matrix,
                depot_coords=depot_coords,
                workday_start=workday_start
            )
        else:
            # Legacy greedy path uses Google distance matrix
            distance_matrix = await self.distance_provider.compute_travel_matrix(
                location_coords,
                departure_time=workday_start
            )
            if not distance_matrix:
                raise ValueError("Failed to compute distance matrix")
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

        # Post-solve: fetch Directions overview (traffic) per route and render map
        polylines: List[str] = []
        google_totals: Dict[str, float] = {}
        if not self.config.dev.mock_google_api and self.settings.google_maps_api_key:
            dep_epoch = int(workday_start.timestamp()) + int(self.config.google.departure_time_offset_hours) * 3600
            for route in solution.routes:
                ordered_coords = []
                # depot -> stops -> depot
                ordered_coords.append((depot_coords.lat, depot_coords.lon))
                for a in route.assignments:
                    if a.job.location.lat and a.job.location.lon:
                        ordered_coords.append((a.job.location.lat, a.job.location.lon))
                ordered_coords.append((depot_coords.lat, depot_coords.lon))
                overview = await fetch_route_overview(
                    self.settings.google_maps_api_key,
                    ordered_coords,
                    dep_epoch,
                    traffic_model=self.config.google.traffic_model.lower(),
                )
                if overview:
                    polylines.append(overview.get("polyline"))
                    google_totals[route.truck.name] = float(overview.get("duration_in_traffic_min", 0.0))
                else:
                    polylines.append(None)

        # Optional ETA audit: compare offline vs Google per-leg and save artifacts
        if getattr(getattr(self.config, "audit", {}), "enable_eta_audit", False) and not self.config.dev.mock_google_api and self.settings.google_maps_api_key:
            dep_epoch = int(workday_start.timestamp())
            all_rows = []
            agg_stats = {}
            for route in solution.routes:
                if not route.assignments:
                    continue
                ordered_coords = [(depot_coords.lat, depot_coords.lon)]
                offline_legs = []
                prev = (depot_coords.lat, depot_coords.lon)
                for a in route.assignments:
                    cur = (a.job.location.lat, a.job.location.lon)
                    ordered_coords.append(cur)
                    offline_legs.append(a.drive_minutes_from_previous)
                    prev = cur
                ordered_coords.append((depot_coords.lat, depot_coords.lon))
                # Get Google per-leg
                legs = await audit_route(
                    self.settings.google_maps_api_key,
                    ordered_coords,
                    dep_epoch,
                    traffic_model=self.config.google.traffic_model.lower(),
                )
                if not legs:
                    continue
                df, stats = compare_offline_vs_google(legs, offline_legs, route.truck.name)
                all_rows.append(df)
                agg_stats[route.truck.name] = stats
                # Print short summary
                off_total = sum(x for x in offline_legs if x is not None)
                g_total = sum(legs[i].get("google_minutes", 0.0) for i in range(min(len(legs), len(offline_legs))))
                delta = off_total - g_total
                logger.info(f"[ETA AUDIT] {route.truck.name}: offline={off_total:.1f}m google={g_total:.1f}m delta={delta:+.1f}m")
                worst = df.dropna().sort_values("pct_err", ascending=False).head(3)
                for _, row in worst.iterrows():
                    logger.info(f"  leg {int(row['seq'])}: offline={row['offline_min']:.1f} google={row['google_min']:.1f} pct_err={row['pct_err']:.1f}%")
            if all_rows:
                big = __import__("pandas").concat(all_rows, ignore_index=True)
                out = save_audit(big, agg_stats, out_dir="artifacts", date=request.date)
                # Append simple learned ratios for future runs
                try:
                    learned_csv = append_learned_ratios(big, str(Path("artifacts") / "traffic_learned.csv"))
                    logger.info(f"Learned ratios appended: {learned_csv}")
                except Exception as e:
                    logger.debug(f"append_learned_ratios skipped: {e}")
                logger.info(f"ETA audit saved: {out}")
        elif getattr(getattr(self.config, "audit", {}), "enable_eta_audit", False):
            # Google audit requested but unavailable; save offline-only summary for traceability
            try:
                pd = __import__("pandas")
                rows = []
                for route in solution.routes:
                    if not route.assignments:
                        continue
                    for i, a in enumerate(route.assignments):
                        rows.append({
                            "route_id": route.truck.name,
                            "seq": i,
                            "offline_min": a.drive_minutes_from_previous,
                            "google_min": None,
                            "delta_min": None,
                            "pct_err": None,
                        })
                df = pd.DataFrame(rows)
                out = save_audit(df, {}, out_dir="artifacts", date=request.date)
                logger.info(f"ETA offline audit placeholder saved: {out}")
            except Exception:
                pass
        
        # Render map html if any polyline
        output_files = {}
        if polylines:
            out_path = f"runs/{request.date}_map.html"
            map_path = render_map_html([p for p in polylines if p], out_path, self.settings.google_maps_api_key or "")
            output_files["map_html"] = map_path
        
        # Convert to API response
        result = self._convert_solution_to_result(solution, request.date, start_time, depot_coords)
        if request.scenario == "priority":
            result.solver_used = "priority"
        else:
            result.solver_used = "ortools" if self.config.solver.use_ortools else "greedy"
        if output_files:
            result.output_files = output_files
        
        # Tuning hint for drop penalty
        pen = getattr(self.config, "penalties", None)
        if pen:
            print_priority_tradeoff_hint(pen.disjunction_base, pen.priority_weight)

        # Concise run summary: offline vs google totals when available
        try:
            pen_dict = None
            if getattr(self.config, "penalties", None):
                pen = self.config.penalties
                pen_dict = {"priority_weight": pen.priority_weight, "disjunction_base": pen.disjunction_base}
            print_run_summary(solution, google_totals, pen_dict)
        except Exception:
            # Fallback to logger summaries if helper fails
            for r in solution.routes:
                rid = r.truck.name
                off_total = r.total_drive_minutes + r.total_service_minutes
                g_total = google_totals.get(rid)
                delta = (off_total - g_total) if g_total is not None else None
                load_lb = r.total_weight_lb
                on_time = (r.overtime_minutes <= 0.0)
                logger.info(f"[SUMMARY] {rid} stops={len(r.assignments)} offline={off_total:.1f}m"
                            + (f" google={g_total:.1f}m delta={delta:+.1f}m" if g_total is not None else "")
                            + f" load={load_lb:.0f}lb on_time={'Y' if on_time else 'N'}")

        logger.info(f"Optimization completed: {len(result.routes)} routes, {len(result.unassigned_jobs)} unassigned jobs")
        # Write JSONL per-route logs (low-risk artifact for debugging/comparison)
        try:
            self._write_jsonl_logs(
                solution,
                date=request.date,
                scenario=request.scenario,
                workday_start=workday_start,
                params=request.params or {},
            )
        except Exception as e:
            logger.debug(f"jsonl logging skipped: {e}")
        
        return result

    async def generate_reports(self, date: str, output_dir: str, format: str = "all") -> Dict[str, str]:
        """Generate visualization artifacts for a given date.

        Currently generates a CSV-based Google Map HTML (day1_map.html) that loads
        the example CSV and renders markers/routes. The Google Maps JS API key is
        read from Settings (.env) and injected into the script tag using async+defer.

        Returns a mapping of artifact type to file path.
        """
        out: Dict[str, str] = {}
        try:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # CSV-based map for demo/sample input
            map_html_path = out_dir / "day1_map.html"
            key = self.settings.google_maps_api_key or ""
            render_day_csv_map_html(str(map_html_path), key)
            out["map_html"] = str(map_html_path)
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            raise
        return out

    def _write_jsonl_logs(self, solution: Solution, date: str, scenario: Optional[str], workday_start: datetime, params: Dict[str, Any]) -> None:
        """Emit per-route JSONL logs with meta and per-stop events.

        Files: runs/logs/{date}/{truck}.jsonl
        Events:
        - route_meta: summary for the route and recommended first-leg departure
        - stop: one per stop with arrival/depart, wait minutes, priority, and shipment info
        """
        out_dir = Path("runs") / "logs" / date
        out_dir.mkdir(parents=True, exist_ok=True)
        first_buf = int(params.get("first_leg_buffer_minutes", 10))

        for route in solution.routes:
            if not route.assignments:
                continue
            truck_name = route.truck.name
            file_path = out_dir / f"{truck_name}.jsonl"
            rec_depart = (workday_start + __import__("datetime").timedelta(minutes=first_buf)).isoformat()
            meta = {
                "type": "route_meta",
                "date": date,
                "scenario": scenario,
                "truck": truck_name,
                "total_drive_minutes": route.total_drive_minutes,
                "total_service_minutes": route.total_service_minutes,
                "overtime_minutes": route.overtime_minutes,
                "first_leg_recommended_departure_local": rec_depart,
            }
            with open(file_path, "w") as f:
                f.write(json.dumps(meta) + "\n")
                prev_depart = workday_start
                for i, a in enumerate(route.assignments, start=1):
                    base_arrival = prev_depart + __import__("datetime").timedelta(minutes=a.drive_minutes_from_previous or 0.0)
                    wait_min = max(0.0, (a.estimated_arrival - base_arrival).total_seconds() / 60.0)
                    row = {
                        "type": "stop",
                        "seq": i,
                        "job_id": a.job.id,
                        "location_name": a.job.location.name,
                        "priority": getattr(a.job, "priority", None),
                        "arrival_local": a.estimated_arrival.isoformat(),
                        "depart_local": a.estimated_departure.isoformat(),
                        "drive_minutes_from_previous": a.drive_minutes_from_previous,
                        "service_minutes": a.service_minutes,
                        "wait_minutes": wait_min,
                        "earliest_str": a.job.earliest.time().isoformat(timespec='minutes') if a.job.earliest else None,
                        "latest_str": a.job.latest.time().isoformat(timespec='minutes') if a.job.latest else None,
                        "shipment_id": getattr(a.job, "shipment_id", None),
                        "shipment_role": getattr(a.job, "shipment_role", None),
                    }
                    if i == 1:
                        row["recommended_departure_local"] = rec_depart
                    f.write(json.dumps(row) + "\n")
                    prev_depart = a.estimated_departure
    
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
                "overtime_minutes": route.overtime_minutes,
                "scenario": getattr(getattr(self, 'last_request', None), 'scenario', None),
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
                    "service_minutes": assignment.service_minutes,
                    # Placeholders; future: set to planned values when we compute waits/windows
                    "planned_travel_minutes": assignment.drive_minutes_from_previous,
                    "planned_service_minutes": assignment.service_minutes,
                    "planned_wait_minutes": max(0.0, (assignment.estimated_arrival - (assignment.estimated_departure - __import__('datetime').timedelta(minutes=assignment.service_minutes))).total_seconds()/60.0) if assignment else None,
                    "batch_index": None,
                    "batch_seq_in_batch": None,
                    # Extended logging snapshot fields
                    "arrival_time_local": assignment.estimated_arrival.isoformat(),
                    "service_start_local": assignment.estimated_arrival.isoformat(),
                    "depart_time_local": assignment.estimated_departure.isoformat(),
                    "priority": getattr(assignment.job, 'priority', None),
                    "earliest_str": assignment.job.earliest.time().isoformat(timespec='minutes') if assignment.job.earliest else None,
                    "latest_str": assignment.job.latest.time().isoformat(timespec='minutes') if assignment.job.latest else None,
                    "curfew_window": None,
                    "overtime_flag": route.overtime_minutes > 0,
                    "shipment_id": getattr(assignment.job, 'shipment_id', None),
                    "shipment_role": getattr(assignment.job, 'shipment_role', None),
                }
                
                self.repo.create_route_stop(stop_data)
        
        # Save unassigned jobs
        for job in solution.unassigned_jobs:
            unassigned_data = {
                "job_id": job.id,
                "date": date,
                "reason": "Could not satisfy constraints",
                "scenario": getattr(getattr(self, 'last_request', None), 'scenario', None),
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
            
            # Build driver link preserving order: depot -> stops -> depot
            ordered_addrs = [self.config.depot.address]
            for assignment in route.assignments:
                ordered_addrs.append(assignment.job.location.address)
            ordered_addrs.append(self.config.depot.address)
            maps_url = build_driver_link(ordered_addrs)
            
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
            solver_used="greedy",  # will be overridden by caller when OR-Tools is used
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
