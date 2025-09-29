"""
OR-Tools offline VRPTW solver using haversine matrices (no Google Matrix during solve).
Keeps weight+volume capacities, hard time windows, simple per-vehicle span.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from .models import Truck, Job, JobItem
from .constraints import ConstraintValidator
from .schemas import AppConfig
from .solver_greedy import Solution, TruckRoute, JobAssignment
from .util.haversine import km, minutes_from_km, minutes_with_traffic
from .traffic.traffic_profile import factor_for_leg
from .opt.penalties import drop_penalty as drop_penalty_fn


logger = logging.getLogger(__name__)


class ORToolsSolver:
    """OR-Tools-based VRP solver with time windows and capacity constraints."""
    
    def __init__(self, config: AppConfig):
        """Initialize OR-Tools solver."""
        self.config = config
        self.validator = ConstraintValidator(config)
        
        try:
            from ortools.constraint_solver import routing_enums_pb2, pywrapcp  # import lazily
            self.routing_enums = routing_enums_pb2  # enum access
            self.pywrapcp = pywrapcp  # solver API
        except ImportError as e:
            logger.error("OR-Tools not installed. Add extra 'ortools'.")
            raise
    
    def solve(
        self,
        trucks: List[Truck],
        jobs: List[Job],
        job_items_map: Dict[int, List[JobItem]],
        locations=None,  # kept for API parity; ignored in offline solver
        distance_matrix=None,  # kept for API parity; ignored in offline solver
        depot_coords=None,
        workday_start: datetime = None,
        **kwargs,
    ) -> Solution:
        """Solve VRPTW offline: build haversine time matrix, add dims, solve, convert."""
        start_ts = datetime.now()
        if not jobs:
            return Solution(routes=[], unassigned_jobs=[], total_cost=0.0, feasible=True, computation_time_seconds=0.0)

        # Build nodes: 0=depot (workday start), 1..N = jobs in given order  # stable mapping
        coords: List[Tuple[float, float]] = [(depot_coords.lat, depot_coords.lon)]  # true depot
        for j in jobs:
            coords.append((j.location.lat or coords[0][0], j.location.lon or coords[0][1]))  # fallback to depot

        # Build time matrix (minutes) using haversine with traffic factor  # offline
        speed = getattr(self.config.solver, "haversine_speed_kmph", 35)
        n = len(coords)
        time_matrix = [[0 for _ in range(n)] for __ in range(n)]
        # Approx start time per leg = vehicle_start for pass-1  # simple
        approx_dt = workday_start
        for i in range(n):
            for k in range(n):
                if i == k:
                    continue
                dist_km = km(coords[i][0], coords[i][1], coords[k][0], coords[k][1])  # km
                factor = factor_for_leg(approx_dt, coords[i][0], coords[i][1], coords[k][0], coords[k][1], self.config)  # traffic factor
                mins = minutes_with_traffic(dist_km, speed, factor)  # minutes adj
                time_matrix[i][k] = int(round(mins))  # minutes int

        # Demands: weight and volume per job (index matches node id)  # caps
        weight_demand = [0]
        volume_demand = [0]
        for j in jobs:
            w = sum(it.item.weight_lb_per_unit * it.qty for it in job_items_map[j.id])
            v = sum((it.item.volume_ft3_per_unit or 0.0) * it.qty for it in job_items_map[j.id])
            weight_demand.append(int(round(w)))
            volume_demand.append(int(round(v)))

        # Create manager/model  # core OR-Tools
        manager = self.pywrapcp.RoutingIndexManager(n, len(trucks), 0)
        routing = self.pywrapcp.RoutingModel(manager)

        # Transit callback = drive time + service at from-node  # compact
        service_by_node = [0]
        for j in jobs:
            mins = self.validator.calculate_service_time(job_items_map[j.id])
            service_by_node.append(int(mins))

        def transit_cb(from_index, to_index):  # noqa: ANN001
            f = manager.IndexToNode(from_index)
            t = manager.IndexToNode(to_index)
            return time_matrix[f][t] + service_by_node[f]

        transit_idx = routing.RegisterTransitCallback(transit_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # Time dimension with hard windows  # VRPTW
        horizon = self._day_horizon_minutes(workday_start)
        routing.AddDimension(transit_idx, 0, horizon, True, "Time")
        time_dim = routing.GetDimensionOrDie("Time")

        # Apply job time windows in minutes from start  # hard
        for node in range(1, n):
            job = jobs[node - 1]
            earliest = int(((job.earliest or workday_start) - workday_start).total_seconds() / 60)
            latest = int(((job.latest or (workday_start + timedelta(minutes=horizon))) - workday_start).total_seconds() / 60)
            index = manager.NodeToIndex(node)
            time_dim.CumulVar(index).SetRange(earliest, latest)

        # Vehicle start/end windows
        for v in range(len(trucks)):
            start_idx = routing.Start(v)
            end_idx = routing.End(v)
            time_dim.CumulVar(start_idx).SetRange(0, horizon)
            time_dim.CumulVar(end_idx).SetRange(0, horizon)

        # Weight capacity dimension  # hard
        def weight_cb(from_index):  # noqa: ANN001
            node = manager.IndexToNode(from_index)
            return weight_demand[node]

        w_idx = routing.RegisterUnaryTransitCallback(weight_cb)
        weight_caps = [int(t.max_weight_lb) for t in trucks]
        routing.AddDimensionWithVehicleCapacity(w_idx, 0, weight_caps, True, "Weight")

        # Volume capacity if present  # optional
        if any(volume_demand):
            def vol_cb(from_index):  # noqa: ANN001
                node = manager.IndexToNode(from_index)
                return volume_demand[node]
            v_idx = routing.RegisterUnaryTransitCallback(vol_cb)
            # Approx truck volume ft3 using bed dims
            caps = [int(t.bed_len_ft * t.bed_width_ft * (t.height_limit_ft or 8)) for t in trucks]
            routing.AddDimensionWithVehicleCapacity(v_idx, 0, caps, True, "Volume")

        # Allow dropping any job node with a penalty so model remains feasible
        # Penalty is scaled by priority: higher priority => higher penalty to drop
        def drop_penalty(job: Job) -> int:
            # New knob: base + priority*weight from config.penalties
            base = int(getattr(getattr(self.config, "penalties", {}), "disjunction_base", 1000))
            weight = int(getattr(getattr(self.config, "penalties", {}), "priority_weight", 5))
            prio = max(1, int(getattr(job, "priority", 1)))
            return drop_penalty_fn(prio, base, weight)

        for node in range(1, n):
            idx = manager.NodeToIndex(node)
            routing.AddDisjunction([idx], drop_penalty(jobs[node - 1]))

        # Optionally prefer using fewer trucks via a fixed cost per vehicle
        single_truck_mode = int(getattr(self.config.solver, "single_truck_mode", 0))
        if single_truck_mode:
            veh_penalty = int(getattr(self.config.solver, "trucks_used_penalty", 1000))
            for v in range(len(trucks)):
                routing.SetFixedCostOfVehicle(veh_penalty, v)

        # First solution + metaheuristic + limit
        search = self.pywrapcp.DefaultRoutingSearchParameters()
        search.first_solution_strategy = getattr(self.routing_enums.FirstSolutionStrategy, getattr(self.config.solver, "first_solution", "PATH_CHEAPEST_ARC"))
        search.local_search_metaheuristic = getattr(self.routing_enums.LocalSearchMetaheuristic, getattr(self.config.solver, "metaheuristic", "GUIDED_LOCAL_SEARCH"))
        search.time_limit.FromSeconds(int(getattr(self.config.solver, "time_limit_sec", 30)))
        # Determinism knob
        try:
            seed = int(getattr(self.config.solver, "random_seed", 42))
            search.random_seed = seed
            search.use_random_number_generator = True
        except Exception:
            pass

        assignment = routing.SolveWithParameters(search)
        if not assignment:
            return Solution(routes=[], unassigned_jobs=jobs, total_cost=0.0, feasible=False, computation_time_seconds=(datetime.now() - start_ts).total_seconds())

        # Optional two-pass refinement: rebuild matrix with time-aware factors and re-solve once
        if bool(getattr(self.config.solver, "two_pass_traffic", False)):
            # Estimate start times at each node as earliest arrival in first solution
            node_start_times = [0 for _ in range(n)]  # minutes since workday_start
            time_dim_first = routing.GetDimensionOrDie("Time")
            for v in range(len(trucks)):
                index = routing.Start(v)
                while not routing.IsEnd(index):
                    node = manager.IndexToNode(index)
                    arr = assignment.Value(time_dim_first.CumulVar(index))
                    node_start_times[node] = arr
                    index = assignment.Value(routing.NextVar(index))
            refined = [[0 for _ in range(n)] for __ in range(n)]
            for i in range(n):
                for k in range(n):
                    if i == k:
                        continue
                    # Use node i arrival as start time for leg i->k
                    start_dt = workday_start + timedelta(minutes=node_start_times[i])
                    dist_km = km(coords[i][0], coords[i][1], coords[k][0], coords[k][1])
                    factor = factor_for_leg(start_dt, coords[i][0], coords[i][1], coords[k][0], coords[k][1], self.config)
                    refined[i][k] = int(round(minutes_with_traffic(dist_km, speed, factor)))
            # Re-register transit with refined matrix
            def transit_cb2(from_index, to_index):  # noqa: ANN001
                f = manager.IndexToNode(from_index)
                t = manager.IndexToNode(to_index)
                return refined[f][t] + service_by_node[f]
            transit2 = routing.RegisterTransitCallback(transit_cb2)
            routing.SetArcCostEvaluatorOfAllVehicles(transit2)
            # Short second pass cap to keep runtime bounded
            second = self.pywrapcp.DefaultRoutingSearchParameters()
            second.CopyFrom(search)
            # Apply a short limit for refinement (e.g., 8s)
            try:
                second.time_limit.FromSeconds(min(8, int(getattr(self.config.solver, "time_limit_sec", 20))))
            except Exception:
                second.time_limit.FromSeconds(8)
            assignment = routing.SolveWithParameters(second)
            if not assignment:
                assignment = routing.SolveWithParameters(search)  # fallback

        # Convert to our solution format  # minimal
        routes: List[TruckRoute] = []
        assigned = set()
        for v, truck in enumerate(trucks):
            index = routing.Start(v)
            prev_node = 0
            assignments: List[JobAssignment] = []
            total_drive = 0
            total_service = 0
            total_weight = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != 0:
                    job = jobs[node - 1]
                    items = job_items_map[job.id]
                    # Times
                    arr_min = assignment.Value(time_dim.CumulVar(index))
                    serv_min = service_by_node[node]
                    # Drive time from previous
                    drive = time_matrix[prev_node][node]
                    # Build assignment
                    ja = JobAssignment(
                        job=job,
                        job_items=items,
                        truck=truck,
                        stop_order=len(assignments),
                        estimated_arrival=workday_start + timedelta(minutes=arr_min),
                        estimated_departure=workday_start + timedelta(minutes=arr_min + serv_min),
                        drive_minutes_from_previous=drive,
                        service_minutes=serv_min,
                        location_index=node,
                    )
                    assignments.append(ja)
                    assigned.add(job.id)
                    total_drive += drive
                    total_service += serv_min
                    total_weight += sum(it.item.weight_lb_per_unit * it.qty for it in items)
                    prev_node = node
                index = assignment.Value(routing.NextVar(index))
            if assignments:
                # Overtime check
                end_dt = assignments[-1].estimated_departure
                work_end = workday_start.replace(hour=int(self.config.depot.workday_window.end[:2]), minute=int(self.config.depot.workday_window.end[3:]), second=0)
                overtime = max(0.0, (end_dt - work_end).total_seconds() / 60.0)
                routes.append(TruckRoute(truck=truck, assignments=assignments, total_drive_minutes=total_drive, total_service_minutes=total_service, total_weight_lb=total_weight, overtime_minutes=overtime))

        unassigned = [j for j in jobs if j.id not in assigned]
        sol = Solution(routes=routes, unassigned_jobs=unassigned, total_cost=sum(r.calculate_cost(self.config) for r in routes), feasible=len(unassigned) == 0, computation_time_seconds=(datetime.now() - start_ts).total_seconds())
        return sol

    def _day_horizon_minutes(self, workday_start: datetime) -> int:
        """Compute horizon from tw.vehicle_* if provided, else depot window length."""
        vehicle_end = None
        if isinstance(getattr(self.config, "tw", None), dict):
            vehicle_end = self.config.tw.get("vehicle_end")
        if vehicle_end:
            end = datetime.fromisoformat(workday_start.date().isoformat() + "T" + vehicle_end + ":00")
            return int((end - workday_start).total_seconds() / 60)
        # Fallback to depot window
        end_h = int(self.config.depot.workday_window.end[:2])
        end_m = int(self.config.depot.workday_window.end[3:])
        end = workday_start.replace(hour=end_h, minute=end_m, second=0)
        return int((end - workday_start).total_seconds() / 60)
    
    def _create_data_model(
        self,
        trucks: List[Truck],
        jobs: List[Job],
        job_items_map: Dict[int, List[JobItem]],
        distance_matrix_unused,
        workday_start: datetime
    ) -> Dict:
        """(Deprecated) Helper left for reference; not used in offline path."""
        return {}
    
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
        """(Deprecated) Kept for reference; inline conversion used in solve()."""
        return Solution(routes=[], unassigned_jobs=jobs, total_cost=0.0, feasible=False, computation_time_seconds=0.0)
