"""
Priority-first solver implementing Design A:
- Stage 1: Build skeleton routes with TRUMP+HIGH only under hard windows/curfew/shipment precedence.
- Stage 2: Opportunistic inserts for MED/LOW if detour <= threshold or ratio <= threshold.
Includes waiting at early arrivals, overtime up to max, shipment precedence and same-truck, and curfew for large trucks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple

from .models import Truck, Job, JobItem
from .distance import Coordinates
from .schemas import AppConfig
from .constraints import ConstraintValidator
from .solver_greedy import TruckRoute, JobAssignment, Solution


@dataclass
class InsertEval:
    position: int
    delta_travel: float
    added_wait: float
    added_overtime: float
    gain: float


class PrioritySolver:
    def __init__(self, config: AppConfig, travel_time_fn):
        self.config = config
        self.validator = ConstraintValidator(config)
        self.travel_time = travel_time_fn

    def solve(
        self,
        trucks: List[Truck],
        jobs: List[Job],
        job_items_map: Dict[int, List[JobItem]],
        locations,  # unused, kept for parity
        depot_coords: Coordinates,
        date: str,
        params: Dict,
    ) -> Solution:
        # Params
        tz = params.get("timezone", "America/Los_Angeles")  # reserved
        day_start_s = params.get("day_start", self.config.depot.workday_window.start)
        day_end_s = params.get("day_end", self.config.depot.workday_window.end)
        first_buf = int(params.get("first_leg_buffer_minutes", 10))
        max_ot = int(params.get("max_overtime_minutes", 0))
        wait_pen = float(params.get("wait_penalty", 0.1))
        ot_pen = float(params.get("overtime_penalty", 2.0))
        detour_thr_min = float(params.get("detour_time_threshold_minutes", 30))
        detour_ratio_thr = float(params.get("detour_ratio_threshold", 0.35))
        prio_w = params.get("priority_weights", {"TRUMP": 1_000_000, "HIGH": 500, "MEDIUM": 120, "LOW": 30})
        seed = int(params.get("seed", self.config.solver.random_seed))
        random.seed(seed)
        # Curfews list
        curfews = params.get("curfews", [])
        sm_curf = next((c for c in curfews if c.get("name") == "SantaMonica"), None)
        sm_window = None
        if sm_curf:
            sm_window = (
                time.fromisoformat(sm_curf["window_start"]),
                time.fromisoformat(sm_curf["window_end"]),
                bool(sm_curf.get("applies_to_large_trucks", True)),
            )

        # Helpers
        def prio_bucket(p: int) -> str:
            return {0: "TRUMP", 1: "HIGH", 2: "MEDIUM", 3: "LOW"}.get(int(p), "MEDIUM")

        day_start = datetime.fromisoformat(f"{date}T{day_start_s}:00")
        day_end = datetime.fromisoformat(f"{date}T{day_end_s}:00")

        # Build initial empty routes
        routes: List[TruckRoute] = [
            TruckRoute(truck=t, assignments=[], total_drive_minutes=0.0, total_service_minutes=0.0, total_weight_lb=0.0, overtime_minutes=0.0)
            for t in trucks
        ]

        # Group shipments and maintain assignment index mapping
        ship_groups: Dict[str, Dict[str, Job]] = {}
        for j in jobs:
            if j.shipment_id:
                grp = ship_groups.setdefault(j.shipment_id, {})
                if j.shipment_role:
                    grp[j.shipment_role] = j
        # job.id -> (route, index)
        assigned_map: Dict[int, Tuple[TruckRoute, int]] = {}

        def refresh_mapping(route: TruckRoute) -> None:
            for idx, a in enumerate(route.assignments):
                assigned_map[a.job.id] = (route, idx)

        # Stage 1: TRUMP+HIGH under hard constraints
        stage1_jobs = [j for j in jobs if prio_bucket(j.priority) in ("TRUMP", "HIGH")]
        # Sort TRUMP first, then earliest window
        stage1_jobs.sort(key=lambda j: (0 if prio_bucket(j.priority) == "TRUMP" else 1, j.earliest or day_start))

        unassigned: List[Job] = []
        # Iterate until no further TRUMP/HIGH can be placed to honor pickup->dropoff precedence
        queue = list(stage1_jobs)
        progress = True
        while progress and queue:
            progress = False
            next_queue: List[Job] = []
            for job in queue:
                # Determine shipment constraints
                earliest_override: Optional[datetime] = None
                min_pos: Optional[int] = None
                # Restrict to same-truck route if counterpart assigned and must_same_truck
                restrict_truck_id: Optional[int] = None
                if job.shipment_id:
                    grp = ship_groups.get(job.shipment_id, {})
                    if (job.shipment_role or '').lower() == 'dropoff':
                        pickup = grp.get('pickup')
                        if not pickup or pickup.id not in assigned_map:
                            # Can't place dropoff before pickup; try later
                            next_queue.append(job)
                            continue
                        pick_route, pick_idx = assigned_map[pickup.id]
                        # Enforce same-truck if any leg requires it
                        if getattr(job, 'must_same_truck', True) or getattr(pickup, 'must_same_truck', True):
                            restrict_truck_id = pick_route.truck.id
                        # Ensure dropoff occurs after pickup depart
                        earliest_override = pick_route.assignments[pick_idx].estimated_departure
                        min_pos = pick_idx + 1
                    else:
                        # pickup: if dropoff already placed and needs same-truck, stick to that truck
                        drop = grp.get('dropoff')
                        if drop and drop.id in assigned_map and (getattr(job, 'must_same_truck', True) or getattr(drop, 'must_same_truck', True)):
                            restrict_truck_id = assigned_map[drop.id][0].truck.id

                assigned, r = self._assign_job_best(
                    routes, job, job_items_map[job.id], depot_coords, day_start, day_end, max_ot, sm_window,
                    earliest_override=earliest_override, min_pos=min_pos, restrict_truck_id=restrict_truck_id,
                )
                if assigned and r is not None:
                    refresh_mapping(r)
                    progress = True
                else:
                    next_queue.append(job)
            queue = next_queue
        # Whatever remains in queue couldn't be placed in stage1
        unassigned.extend(queue)

        # Stage 2: MED/LOW opportunistic inserts by gain
        remaining = [j for j in jobs if j.id not in assigned_map]
        # Deduplicate same-address+window merges handled implicitly by later merges (simplified: skip)
        improved = True
        while improved and remaining:
            improved = False
            best_choice: Optional[Tuple[Job, TruckRoute, InsertEval]] = None
            for job in remaining:
                # Shipment-aware constraints for evaluation
                earliest_override: Optional[datetime] = None
                min_pos: Optional[int] = None
                restrict_truck_id: Optional[int] = None
                if job.shipment_id:
                    grp = ship_groups.get(job.shipment_id, {})
                    if (job.shipment_role or '').lower() == 'dropoff':
                        pickup = grp.get('pickup')
                        if not pickup or pickup.id not in assigned_map:
                            # Can't place dropoff before pickup
                            continue
                        pick_route, pick_idx = assigned_map[pickup.id]
                        if getattr(job, 'must_same_truck', True) or getattr(pickup, 'must_same_truck', True):
                            restrict_truck_id = pick_route.truck.id
                        earliest_override = pick_route.assignments[pick_idx].estimated_departure
                        min_pos = pick_idx + 1
                # Evaluate across eligible routes
                for r in routes:
                    if restrict_truck_id is not None and r.truck.id != restrict_truck_id:
                        continue
                    ev = self._evaluate_insert(
                        r, job, job_items_map[job.id], depot_coords, day_start, day_end, max_ot,
                        wait_pen, ot_pen, prio_w, detour_thr_min, detour_ratio_thr, sm_window,
                        earliest_override=earliest_override, min_pos=min_pos,
                    )
                    if ev and (best_choice is None or ev.gain > best_choice[2].gain):
                        best_choice = (job, r, ev)
            if best_choice and best_choice[2].gain > 0:
                job, route, ev = best_choice
                self._apply_insert(route, job, job_items_map[job.id], ev.position, depot_coords, day_start)
                remaining.remove(job)
                refresh_mapping(route)
                improved = True
            else:
                break

        # Mark leftover remaining as unassigned
        unassigned.extend([j for j in remaining if j not in unassigned])

        # Compute totals
        for r in routes:
            self._recalc_route(r, depot_coords, day_start, day_end)

        total_cost = sum(r.calculate_cost(self.config) for r in routes if r.assignments)
        return Solution(routes=routes, unassigned_jobs=unassigned, total_cost=total_cost, feasible=len(unassigned) == 0, computation_time_seconds=0.0)

    # --- Core mechanics ---
    def _assign_job_best(
        self,
        routes: List[TruckRoute],
        job: Job,
        items: List[JobItem],
        depot: Coordinates,
        day_start: datetime,
        day_end: datetime,
        max_ot: int,
        sm_window,
        *,
        earliest_override: Optional[datetime] = None,
        min_pos: Optional[int] = None,
        restrict_truck_id: Optional[int] = None,
    ) -> Tuple[bool, Optional[TruckRoute]]:
        best = None
        best_r = None
        for r in routes:
            if restrict_truck_id is not None and r.truck.id != restrict_truck_id:
                continue
            ev = self._evaluate_insert(
                r, job, items, depot, day_start, day_end, max_ot,
                0.15, 2.0, {"TRUMP":1_000_000,"HIGH":500,"MEDIUM":120,"LOW":30}, 30.0, 0.35, sm_window,
                earliest_override=earliest_override, min_pos=min_pos,
            )
            if ev and (best is None or ev.gain > best.gain):
                best = ev
                best_r = r
        if best_r and best and best.gain > float("-inf"):
            self._apply_insert(best_r, job, items, best.position, depot, day_start)
            return True, best_r
        return False, None

    def _evaluate_insert(
        self,
        route: TruckRoute,
        job: Job,
        items: List[JobItem],
        depot: Coordinates,
        day_start: datetime,
        day_end: datetime,
        max_ot: int,
        wait_pen: float,
        ot_pen: float,
        prio_w: Dict[str, int],
        detour_thr_min: float,
        detour_ratio_thr: float,
        sm_window,
        *,
        earliest_override: Optional[datetime] = None,
        min_pos: Optional[int] = None,
    ) -> Optional[InsertEval]:
        # Determine base times and locations
        def arrival_at_pos(pos: int) -> Tuple[datetime, Coordinates]:
            if not route.assignments:
                return day_start, depot
            if pos == 0:
                return day_start, depot
            prev = route.assignments[pos-1]
            return prev.estimated_departure, Coordinates(prev.job.location.lat, prev.job.location.lon)

        best: Optional[InsertEval] = None
        prio = {0:"TRUMP",1:"HIGH",2:"MEDIUM",3:"LOW"}.get(int(job.priority),"MEDIUM")
        for pos in range(len(route.assignments)+1):
            if min_pos is not None and pos < min_pos:
                continue
            # Travel from prev -> job -> next
            base_depart, prev_coord = arrival_at_pos(pos)
            job_coord = Coordinates(job.location.lat, job.location.lon)
            leg_prev_job = self.travel_time(prev_coord, job_coord, base_depart)
            arrive = base_depart + timedelta(minutes=leg_prev_job)
            # Wait for earliest
            wait = 0.0
            eff_earliest = earliest_override or job.earliest
            if eff_earliest and arrive < eff_earliest:
                wait = (eff_earliest - arrive).total_seconds()/60.0
                arrive = eff_earliest
            # Curfew for Santa Monica on large trucks only
            if sm_window and route.truck.large_capable and "Santa Monica" in (job.location.address or job.location.name):
                s,e,_ = sm_window
                at = arrive.time()
                if at < s:
                    # wait until curfew start
                    wait += (datetime.combine(arrive.date(), s) - arrive).total_seconds()/60.0
                    arrive = datetime.combine(arrive.date(), s)
                elif at > e:
                    # late beyond curfew: infeasible
                    continue
            # Service
            service = self.validator.calculate_service_time(items)
            depart = arrive + timedelta(minutes=service)
            # Check latest
            if job.latest and arrive > job.latest:
                continue
            # If earliest constraint pushes beyond day_end or violates latest window entirely
            if eff_earliest and job.latest and eff_earliest > job.latest:
                continue
            # Overtime check at route end if inserted at end (rough approximation)
            end_dt = depart
            if route.assignments:
                # approximate end as last depart or this depart if later
                end_dt = max(depart, route.assignments[-1].estimated_departure)
            overtime = max(0.0, (end_dt - day_end).total_seconds()/60.0)
            if overtime > max_ot:
                continue
            # Detour estimation vs base leg
            if pos < len(route.assignments):
                next_coord = Coordinates(route.assignments[pos].job.location.lat, route.assignments[pos].job.location.lon)
                leg_job_next = self.travel_time(job_coord, next_coord, depart)
                base_leg = self.travel_time(prev_coord, next_coord, base_depart)
            else:
                leg_job_next = 0.0
                base_leg = 0.0
            delta = leg_prev_job + leg_job_next - base_leg
            ratio = (delta/base_leg) if base_leg > 0 else 0.0
            # Opportunistic constraints for MED/LOW only
            if prio in ("MEDIUM","LOW"):
                if not (delta <= detour_thr_min or ratio <= detour_ratio_thr):
                    continue
            # Gain function
            gain = prio_w.get(prio, 100) / (max(1.0, delta) + 0.5*wait + 2.0*overtime)
            # Track best
            cand = InsertEval(position=pos, delta_travel=delta, added_wait=wait, added_overtime=overtime, gain=gain)
            if not best or cand.gain > best.gain:
                best = cand
        return best

    def _apply_insert(self, route: TruckRoute, job: Job, items: List[JobItem], pos: int, depot: Coordinates, day_start: datetime) -> None:
        # Build assignment object and insert
        # Recompute arrival/depart using previous element timing
        if pos == 0 and not route.assignments:
            prev_coord = depot
            prev_depart = day_start
        elif pos == 0:
            prev_coord = depot
            prev_depart = day_start
        else:
            prev = route.assignments[pos-1]
            prev_coord = Coordinates(prev.job.location.lat, prev.job.location.lon)
            prev_depart = prev.estimated_departure
        job_coord = Coordinates(job.location.lat, job.location.lon)
        drive_min = self.travel_time(prev_coord, job_coord, prev_depart)
        arrival = prev_depart + timedelta(minutes=drive_min)
        # Respect earliest by waiting
        wait = 0.0
        if job.earliest and arrival < job.earliest:
            wait = (job.earliest - arrival).total_seconds()/60.0
            arrival = job.earliest
        service = self.validator.calculate_service_time(items)
        depart = arrival + timedelta(minutes=service)
        ja = JobAssignment(job=job, job_items=items, truck=route.truck, stop_order=pos, estimated_arrival=arrival, estimated_departure=depart, drive_minutes_from_previous=drive_min, service_minutes=service, location_index=job.location.id)
        route.assignments.insert(pos, ja)
        # Fix subsequent stop_order
        for i,a in enumerate(route.assignments):
            a.stop_order = i
        self._recalc_route(route, depot, day_start, day_start.replace(hour=int(self.config.depot.workday_window.end[:2]), minute=int(self.config.depot.workday_window.end[3:]), second=0))

    def _recalc_route(self, route: TruckRoute, depot: Coordinates, day_start: datetime, day_end: datetime) -> None:
        # Recompute times sequentially
        t = day_start
        prev_coord = depot
        total_drive = 0.0
        total_service = 0.0
        for a in route.assignments:
            drive = self.travel_time(prev_coord, Coordinates(a.job.location.lat, a.job.location.lon), t)
            arrive = t + timedelta(minutes=drive)
            if a.job.earliest and arrive < a.job.earliest:
                arrive = a.job.earliest
            depart = arrive + timedelta(minutes=a.service_minutes)
            a.estimated_arrival = arrive
            a.estimated_departure = depart
            a.drive_minutes_from_previous = drive
            prev_coord = Coordinates(a.job.location.lat, a.job.location.lon)
            t = depart
            total_drive += drive
            total_service += a.service_minutes
        route.total_drive_minutes = total_drive
        route.total_service_minutes = total_service
        route.overtime_minutes = max(0.0, (t - day_end).total_seconds()/60.0) if route.assignments else 0.0
