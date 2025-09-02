"""
Visualization and reporting functionality for the truck optimizer.
Generates Plotly charts and reports based on solution data.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging

# Third-party imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    
from .models import Job, Truck
from .solver_greedy import TruckRoute, Solution


logger = logging.getLogger(__name__)


def check_visualization_available() -> bool:
    """Check if visualization packages are available."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Install it with: pip install plotly")
        return False
    return True


class SolutionVisualizer:
    """Generate visualizations and reports for solutions."""
    
    def __init__(self, output_dir: str = "runs"):
        """Initialize visualizer with output directory."""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate_solution_report(
        self, 
        solution: Solution, 
        filename: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate a comprehensive solution report with visualizations.
        
        Args:
            solution: The solution object to visualize
            filename: Optional base filename (without extension)
            
        Returns:
            Dictionary of filenames and their paths
        """
        if not check_visualization_available():
            return {}
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solution_{timestamp}"
            
        output_files = {}
        
        # Generate route visualization
        routes_fig = self.visualize_routes(solution)
        if routes_fig:
            route_path = os.path.join(self.output_dir, f"{filename}_routes.html")
            routes_fig.write_html(route_path)
            output_files["routes"] = route_path
            
        # Generate truck utilization chart
        util_fig = self.visualize_truck_utilization(solution)
        if util_fig:
            util_path = os.path.join(self.output_dir, f"{filename}_utilization.html")
            util_fig.write_html(util_path)
            output_files["utilization"] = util_path
            
        # Generate timeline chart
        timeline_fig = self.visualize_timeline(solution)
        if timeline_fig:
            timeline_path = os.path.join(self.output_dir, f"{filename}_timeline.html")
            timeline_fig.write_html(timeline_path)
            output_files["timeline"] = timeline_path
            
        # Save trace data if available
        if solution.trace_data:
            trace_path = os.path.join(self.output_dir, f"{filename}_trace.json")
            with open(trace_path, 'w') as f:
                json.dump(solution.trace_data, f, indent=2)
            output_files["trace"] = trace_path
            
        # Generate text report
        report_path = os.path.join(self.output_dir, f"{filename}_report.txt")
        with open(report_path, 'w') as f:
            f.write(self._generate_text_report(solution))
        output_files["report"] = report_path
        
        return output_files
    
    def visualize_routes(self, solution: Solution) -> Optional[go.Figure]:
        """
        Create a visualization of truck routes.
        This is a placeholder that would be replaced with actual route visualization
        using coordinates data when available.
        """
        if not check_visualization_available():
            return None
            
        fig = go.Figure()
        
        # This would be expanded with actual coordinates when available
        # For now, we'll just create a placeholder figure
        
        fig.update_layout(
            title="Truck Routes Visualization",
            xaxis_title="Placeholder - Geographic visualization requires coordinate data",
            yaxis_title="",
            showlegend=True
        )
        
        return fig
    
    def visualize_truck_utilization(self, solution: Solution) -> Optional[go.Figure]:
        """Create a visualization of truck utilization metrics."""
        if not check_visualization_available():
            return None
            
        # Extract data
        truck_names = []
        weight_utilization = []
        time_utilization = []
        drive_minutes = []
        service_minutes = []
        num_jobs = []
        
        for route in solution.routes:
            if route.assignments:
                truck_names.append(route.truck.name)
                
                # Calculate weight utilization percentage
                weight_util_pct = (route.total_weight_lb / route.truck.max_weight_lb) * 100
                weight_utilization.append(weight_util_pct)
                
                # Time breakdown
                drive_minutes.append(route.total_drive_minutes)
                service_minutes.append(route.total_service_minutes)
                
                # Number of jobs
                num_jobs.append(len(route.assignments))
                
                # Calculate time utilization (assuming 8-hour workday)
                workday_minutes = 8 * 60  # 8 hours in minutes
                time_util_pct = (route.total_time_minutes / workday_minutes) * 100
                time_utilization.append(time_util_pct)
        
        if not truck_names:
            return None
            
        # Create subplots
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=("Weight Utilization (%)", 
                                          "Time Breakdown (minutes)",
                                          "Number of Jobs",
                                          "Time Utilization (%)"))
        
        # Weight utilization chart
        fig.add_trace(
            go.Bar(x=truck_names, y=weight_utilization, name="Weight Utilization"),
            row=1, col=1
        )
        
        # Time breakdown chart
        fig.add_trace(
            go.Bar(x=truck_names, y=drive_minutes, name="Drive Time"),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=truck_names, y=service_minutes, name="Service Time"),
            row=1, col=2
        )
        
        # Jobs per truck chart
        fig.add_trace(
            go.Bar(x=truck_names, y=num_jobs, name="Job Count"),
            row=2, col=1
        )
        
        # Time utilization chart
        fig.add_trace(
            go.Bar(x=truck_names, y=time_utilization, name="Time Utilization"),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Truck Utilization Analysis",
            height=800,
            barmode='stack'
        )
        
        return fig
    
    def visualize_timeline(self, solution: Solution) -> Optional[go.Figure]:
        """Create a Gantt chart timeline of truck activities."""
        if not check_visualization_available():
            return None
            
        # Prepare data for Gantt chart
        tasks = []
        colors = {}
        
        for route_idx, route in enumerate(solution.routes):
            if not route.assignments:
                continue
                
            # Assign color to truck
            colors[route.truck.name] = px.colors.qualitative.Plotly[route_idx % len(px.colors.qualitative.Plotly)]
            
            # Add depot to starting location
            prev_location = "Depot"
            current_time = route.assignments[0].estimated_arrival - \
                           route.assignments[0].drive_minutes_from_previous
            
            # Add driving task from depot to first location
            tasks.append({
                "Task": route.truck.name,
                "Start": current_time,
                "Finish": route.assignments[0].estimated_arrival,
                "Resource": "Driving",
                "Description": f"Drive from {prev_location} to {route.assignments[0].job.location.address}"
            })
            
            # Add tasks for each assignment
            for assignment in route.assignments:
                # Service task
                tasks.append({
                    "Task": route.truck.name,
                    "Start": assignment.estimated_arrival,
                    "Finish": assignment.estimated_departure,
                    "Resource": "Service",
                    "Description": f"Service at {assignment.job.location.address}"
                })
                
                # Drive to next location if not the last assignment
                if assignment != route.assignments[-1]:
                    next_assignment = route.assignments[route.assignments.index(assignment) + 1]
                    tasks.append({
                        "Task": route.truck.name,
                        "Start": assignment.estimated_departure,
                        "Finish": next_assignment.estimated_arrival,
                        "Resource": "Driving",
                        "Description": f"Drive from {assignment.job.location.address} to {next_assignment.job.location.address}"
                    })
                else:
                    # Drive back to depot
                    # Estimate return time based on last assignment's departure time
                    # This is a simplification since we don't have the actual drive time back to depot
                    return_time = assignment.estimated_departure.replace(
                        minute=assignment.estimated_departure.minute + 30
                    )
                    tasks.append({
                        "Task": route.truck.name,
                        "Start": assignment.estimated_departure,
                        "Finish": return_time,
                        "Resource": "Driving",
                        "Description": f"Return to Depot from {assignment.job.location.address}"
                    })
        
        if not tasks:
            return None
            
        # Create figure
        fig = px.timeline(
            tasks, 
            x_start="Start", 
            x_end="Finish", 
            y="Task",
            color="Resource",
            hover_data=["Description"]
        )
        
        # Update layout
        fig.update_layout(
            title="Truck Schedule Timeline",
            xaxis_title="Time",
            yaxis_title="Truck",
            height=400 + (100 * len(solution.routes))
        )
        
        return fig
    
    def _generate_text_report(self, solution: Solution) -> str:
        """Generate a text-based report for the solution."""
        report = []
        
        report.append("=" * 60)
        report.append("CONCRETE TRUCK OPTIMIZER - SOLUTION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Computation time: {solution.computation_time_seconds:.2f} seconds")
        report.append(f"Solution feasible: {solution.feasible}")
        report.append(f"Total solution cost: {solution.total_cost:.2f}")
        report.append(f"Trucks used: {solution.used_trucks_count}")
        report.append(f"Unassigned jobs: {len(solution.unassigned_jobs)}")
        report.append("")
        
        # Route summaries
        report.append("ROUTE SUMMARIES")
        report.append("-" * 60)
        
        for route in sorted(solution.routes, key=lambda r: r.truck.name):
            if not route.assignments:
                continue
                
            report.append(f"Truck: {route.truck.name} ({route.truck.id})")
            report.append(f"  Jobs assigned: {len(route.assignments)}")
            report.append(f"  Total drive time: {route.total_drive_minutes:.1f} minutes")
            report.append(f"  Total service time: {route.total_service_minutes:.1f} minutes")
            report.append(f"  Total route time: {route.total_time_minutes:.1f} minutes")
            report.append(f"  Total weight: {route.total_weight_lb:.1f} lbs ({(route.total_weight_lb/route.truck.max_weight_lb)*100:.1f}%)")
            report.append(f"  Overtime: {route.overtime_minutes:.1f} minutes")
            report.append(f"  Route cost: {route.calculate_cost(solution.trace_data['config'] if solution.trace_data else None):.2f}")
            report.append("  Route stops:")
            
            for i, assignment in enumerate(route.assignments, 1):
                report.append(f"    {i}. {assignment.job.location.address} - " +
                             f"Arrival: {assignment.estimated_arrival.strftime('%H:%M')}, " +
                             f"Departure: {assignment.estimated_departure.strftime('%H:%M')}")
                             
            report.append("")
        
        # Unassigned jobs
        if solution.unassigned_jobs:
            report.append("UNASSIGNED JOBS")
            report.append("-" * 60)
            for job in solution.unassigned_jobs:
                report.append(f"Job {job.id}: {job.location.address}")
            report.append("")
            
        # Add trace summary if available
        if solution.trace_data:
            report.append("SOLUTION TRACE")
            report.append("-" * 60)
            report.append(f"Algorithm: {solution.trace_data.get('algorithm', 'greedy')}")
            report.append(f"Decisions recorded: {len(solution.trace_data.get('decisions', []))}")
            report.append(f"Single truck mode: {'Enabled' if solution.trace_data.get('config', {}).get('single_truck_mode') == 1 else 'Disabled'}")
            
            if 'local_search' in solution.trace_data:
                ls = solution.trace_data['local_search']
                report.append(f"Local search iterations: {ls.get('completed_iterations', 0)}")
                report.append(f"Local search stopped due to: {ls.get('finished_due_to', 'unknown')}")
            
        return "\n".join(report)


def export_solution_to_csv(solution: Solution, filename: str) -> str:
    """Export solution to CSV file for spreadsheet analysis."""
    output_path = f"{filename}.csv"
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("truck_id,truck_name,stop_order,job_id,location,address,")
        f.write("estimated_arrival,estimated_departure,drive_minutes,service_minutes\n")
        
        # Write data for each assignment
        for route in solution.routes:
            if not route.assignments:
                continue
                
            for assignment in route.assignments:
                f.write(f"{route.truck.id},{route.truck.name},{assignment.stop_order},")
                f.write(f"{assignment.job.id},{assignment.job.location_id},")
                f.write(f"\"{assignment.job.location.address}\",")
                f.write(f"{assignment.estimated_arrival.strftime('%H:%M:%S')},")
                f.write(f"{assignment.estimated_departure.strftime('%H:%M:%S')},")
                f.write(f"{assignment.drive_minutes_from_previous},{assignment.service_minutes}\n")
    
    return output_path

