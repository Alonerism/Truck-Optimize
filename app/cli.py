"""
Command-line interface for truck route optimization.
Provides commands for import, optimization, and URL generation.
"""

import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import click

from .service import TruckOptimizerService
from .schemas import ImportRequest, JobImportRow, OptimizeRequest
from .models import ActionType


logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', default='config/params.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def main(ctx, config: str, verbose: bool):
    """Truck Route Optimizer CLI."""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Store config path in context
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--date', default=None, help='Date for jobs (YYYY-MM-DD, default: today)')
@click.option('--clear', is_flag=True, help='Clear existing jobs for the date')
@click.pass_context
def import_jobs(ctx, input_file: str, date: str, clear: bool):
    """Import jobs from CSV or JSON file."""
    
    async def _import():
        service = TruckOptimizerService(ctx.obj['config_path'])
        
        try:
            # Determine date
            if not date:
                import_date = datetime.now().strftime('%Y-%m-%d')
            else:
                import_date = date
            
            # Load data from file
            file_path = Path(input_file)
            
            if file_path.suffix.lower() == '.csv':
                job_rows = _load_csv_file(file_path)
            elif file_path.suffix.lower() == '.json':
                job_rows = _load_json_file(file_path)
            else:
                raise click.ClickException(f"Unsupported file format: {file_path.suffix}")
            
            # Create import request
            request = ImportRequest(
                data=job_rows,
                date=import_date,
                clear_existing=clear
            )
            
            # Perform import
            click.echo(f"Importing {len(job_rows)} jobs for date {import_date}...")
            stats = await service.import_jobs(request)
            
            # Display results
            click.echo(f"\nImport completed:")
            click.echo(f"  Jobs created: {stats.jobs_created}")
            click.echo(f"  Job items created: {stats.total_job_items}")
            click.echo(f"  Locations created: {stats.locations_created}")
            click.echo(f"  Locations updated: {stats.locations_updated}")
            click.echo(f"  Items created: {stats.items_created}")
            click.echo(f"  Geocoding requests: {stats.geocoding_requests}")
            
            if stats.errors:
                click.echo(f"\nErrors encountered:")
                for error in stats.errors:
                    click.echo(f"  - {error}")
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            raise click.ClickException(str(e))
        finally:
            await service.close()
    
    asyncio.run(_import())


@main.command()
@click.option('--date', required=True, help='Date to optimize (YYYY-MM-DD)')
@click.option('--auto', type=click.Choice(['ask', 'overtime', 'defer']), 
              default='ask', help='Overtime handling mode')
@click.option('--seed', type=int, help='Random seed for deterministic results')
@click.option('--single-truck', is_flag=True, help='Enable single-truck mode')
@click.option('--solver', type=click.Choice(['greedy', 'regret2']), default='greedy',
              help='Solver strategy to use')
@click.option('--use-ortools', is_flag=True, help='Override to use OR-Tools offline solver')  # CLI > YAML
@click.option('--priority-weight', type=int, help='Override penalties.priority_weight')  # CLI > YAML
@click.option('--disjunction-base', type=int, help='Override penalties.disjunction_base')  # CLI > YAML
@click.option('--trace', is_flag=True, help='Enable decision tracing for debugging')
@click.option('--visualize', is_flag=True, help='Generate visualization reports')
@click.option('--output-dir', default='runs', help='Directory for output files')
@click.option('--scenario', default=None, help='Optional scenario tag to save with results')
@click.pass_context
def optimize(ctx, date: str, auto: str, seed: int, single_truck: bool, 
             solver: str, use_ortools: bool, priority_weight: int, disjunction_base: int,
             trace: bool, visualize: bool, output_dir: str, scenario: str):
    """Run route optimization for a specific date."""
    
    async def _optimize():
        service = TruckOptimizerService(ctx.obj['config_path'])
        
        try:
            # Create optimization request
            request = OptimizeRequest(
                date=date,
                auto=auto,
                seed=seed,
                single_truck_mode=single_truck,
                solver_strategy=solver,
                trace=trace,
                visualize=visualize,
                output_dir=output_dir,
                scenario=scenario
            )
            
            # Apply CLI config overrides (precedence: CLI > YAML)
            overrides = {}
            if use_ortools:
                overrides["solver.use_ortools"] = True  # flip to OR-Tools
            if priority_weight is not None:
                overrides["penalties.priority_weight"] = int(priority_weight)
            if disjunction_base is not None:
                overrides["penalties.disjunction_base"] = int(disjunction_base)

            service.apply_overrides(overrides)  # small, targeted overrides

            solver_label = "ortools" if service.config.solver.use_ortools else solver  # truthy print
            click.echo(f"Optimizing routes for {date}...")
            click.echo(f"  Solver: {solver_label}")
            if scenario:
                click.echo(f"  Scenario: {scenario}")
            if single_truck:
                click.echo("  Mode: Single-truck optimization")
            if trace:
                click.echo("  Tracing: Enabled")
            
            # Run optimization
            result = await service.optimize_routes(request)
            
            # Display results
            click.echo(f"\nOptimization completed:")
            click.echo(f"  Total routes: {len(result.routes)}")
            click.echo(f"  Unassigned jobs: {len(result.unassigned_jobs)}")
            click.echo(f"  Total cost: {result.total_cost:.2f}")
            click.echo(f"  Solver used: {result.solver_used}")
            click.echo(f"  Computation time: {result.computation_time_seconds:.2f}s")
            
            # Route details
            for i, route in enumerate(result.routes):
                click.echo(f"\n  Route {i+1} - {route.truck.name}:")
                click.echo(f"    Stops: {len(route.stops)}")
                click.echo(f"    Drive time: {route.total_drive_minutes:.1f} min")
                click.echo(f"    Service time: {route.total_service_minutes:.1f} min")
                click.echo(f"    Weight: {route.total_weight_lb:.1f} lbs")
                if route.overtime_minutes > 0:
                    click.echo(f"    Overtime: {route.overtime_minutes:.1f} min")
            
            # Unassigned jobs
            if result.unassigned_jobs:
                click.echo(f"\n  Unassigned jobs:")
                for job in result.unassigned_jobs:
                    click.echo(f"    Job {job.id}: {job.location.name} ({job.action})")
            
            # Output files
            if result.output_files:
                click.echo(f"\nOutput files:")
                for file_type, file_path in result.output_files.items():
                    click.echo(f"  {file_type}: {file_path}")
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise click.ClickException(str(e))
        finally:
            await service.close()
    
    asyncio.run(_optimize())


@main.command()
@click.option('--date', required=True, help='Date to debug (YYYY-MM-DD)')
@click.option('--limit', default=5, type=int, help='Limit top-N prints (default 5)')
@click.option('--two-pass', default=0, type=int, help='Enable two-pass refinement (0/1)')
@click.pass_context
def debug_pipeline(ctx, date: str, limit: int, two_pass: int):
    """Run a printable debug pipeline with step-by-step outputs."""

    async def _run():
        service = TruckOptimizerService(ctx.obj['config_path'])
        try:
            # Wire overrides for two-pass if requested
            if two_pass:
                service.apply_overrides({"solver.two_pass_traffic": True})  # force on
            # Lazy import to avoid overhead
            from .debug.pipeline_view import run_debug_pipeline
            await run_debug_pipeline(service, date, limit=limit, two_pass=bool(two_pass))
        finally:
            await service.close()

    asyncio.run(_run())


@main.command()
@click.option('--date', required=True, help='Date to get links for (YYYY-MM-DD)')
@click.option('--output', help='Output file for URLs (default: print to console)')
@click.pass_context
def links(ctx, date: str, output: str):
    """Generate Google Maps URLs for routes."""
    
    async def _get_links():
        service = TruckOptimizerService(ctx.obj['config_path'])
        
        try:
            click.echo(f"Generating route URLs for {date}...")
            
            # Get route URLs
            urls = await service.get_route_urls(date)
            
            if not urls:
                click.echo("No routes found for this date.")
                return
            
            # Format output
            output_lines = []
            output_lines.append(f"Google Maps URLs for {date}")
            output_lines.append("=" * 50)
            
            for truck_data in urls:
                output_lines.append(f"\n{truck_data['truck_name']} ({truck_data['total_stops']} stops):")
                for i, url in enumerate(truck_data['urls']):
                    if len(truck_data['urls']) > 1:
                        output_lines.append(f"  Segment {i+1}: {url}")
                    else:
                        output_lines.append(f"  {url}")
            
            # Output results
            if output:
                with open(output, 'w') as f:
                    f.write('\n'.join(output_lines))
                click.echo(f"URLs saved to {output}")
            else:
                for line in output_lines:
                    click.echo(line)
            
        except Exception as e:
            logger.error(f"Failed to get URLs: {e}")
            raise click.ClickException(str(e))
        finally:
            await service.close()
    
    asyncio.run(_get_links())


@main.command()
@click.option('--date', help='Date to show status for (YYYY-MM-DD, default: today)')
@click.pass_context
def status(ctx, date: str):
    """Show status and statistics."""
    
    async def _status():
        service = TruckOptimizerService(ctx.obj['config_path'])
        
        try:
            # Determine date
            if not date:
                status_date = datetime.now().strftime('%Y-%m-%d')
            else:
                status_date = date
            
            click.echo(f"Status for {status_date}:")
            click.echo("=" * 30)
            
            # Health check
            health = service.health_check()
            click.echo(f"Database: {'✓' if health['database_connected'] else '✗'}")
            click.echo(f"Google API: {'✓' if health['google_api_configured'] else '✗'}")
            
            # Count entities
            trucks = service.repo.get_trucks()
            locations = service.repo.get_locations()
            items = service.repo.get_items()
            jobs = service.repo.get_jobs_by_date(status_date)
            routes = service.repo.get_route_assignments_by_date(status_date)
            unassigned = service.repo.get_unassigned_jobs_by_date(status_date)
            
            click.echo(f"\nData summary:")
            click.echo(f"  Trucks: {len(trucks)}")
            click.echo(f"  Locations: {len(locations)}")
            click.echo(f"  Items: {len(items)}")
            click.echo(f"  Jobs ({status_date}): {len(jobs)}")
            click.echo(f"  Routes ({status_date}): {len(routes)}")
            click.echo(f"  Unassigned ({status_date}): {len(unassigned)}")
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            raise click.ClickException(str(e))
        finally:
            await service.close()
    
    asyncio.run(_status())


@main.command()
@click.pass_context
def serve(ctx):
    """Start the FastAPI server."""
    try:
        import uvicorn
        from .api import app
        
        click.echo("Starting Truck Optimizer API server...")
        click.echo("API documentation: http://localhost:8000/docs")
        
        uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
        
    except ImportError:
        raise click.ClickException("uvicorn not installed. Run: pip install uvicorn")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise click.ClickException(str(e))


def _load_csv_file(file_path: Path) -> List[JobImportRow]:
    """Load jobs from CSV file."""
    jobs = []
    
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        # Detect delimiter
        sample = csvfile.read(1024)
        csvfile.seek(0)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        
        for row_num, row in enumerate(reader, 1):
            try:
                # Clean up row data
                cleaned_row = {k.strip(): v.strip() for k, v in row.items() if k}
                
                # Normalize headers to support new and legacy formats
                loc_name = cleaned_row.get('location_name') or cleaned_row.get('location')
                if not loc_name:
                    raise ValueError("Missing 'location_name' or 'location' column")

                address = cleaned_row.get('address') or None

                # Build JobImportRow using new schema fields
                job_row = JobImportRow(
                    location_name=loc_name,
                    address=address,
                    action=ActionType(cleaned_row['action'].lower()),
                    items=cleaned_row['items'],
                    priority=int(cleaned_row.get('priority', 1)),
                    earliest=cleaned_row.get('earliest') or None,
                    latest=cleaned_row.get('latest') or None,
                    service_minutes_override=int(cleaned_row['service_minutes_override']) 
                        if cleaned_row.get('service_minutes_override') else None,
                    notes=cleaned_row.get('notes', '')
                )
                
                jobs.append(job_row)
                
            except Exception as e:
                logger.warning(f"Skipping invalid row {row_num}: {e}")
                continue
    
    return jobs


def _load_json_file(file_path: Path) -> List[JobImportRow]:
    """Load jobs from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    jobs = []
    
    if isinstance(data, list):
        job_data_list = data
    elif isinstance(data, dict) and 'jobs' in data:
        job_data_list = data['jobs']
    else:
        raise ValueError("JSON file must contain a list or an object with 'jobs' key")
    
    for item in job_data_list:
        try:
            job_row = JobImportRow(**item)
            jobs.append(job_row)
        except Exception as e:
            logger.warning(f"Skipping invalid job data: {e}")
            continue
    
    return jobs


@main.command()
@click.option('--date', required=True, help='Date to visualize (YYYY-MM-DD)')
@click.option('--output-dir', default='runs', help='Directory for output files')
@click.option('--format', type=click.Choice(['html', 'csv', 'all']), default='all',
              help='Output format')
@click.pass_context
def visualize(ctx, date: str, output_dir: str, format: str):
    """Generate visualization reports for existing solution."""
    
    async def _visualize():
        service = TruckOptimizerService(ctx.obj['config_path'])
        
        try:
            click.echo(f"Generating visualizations for {date}...")
            
            # Run visualization
            output_files = await service.generate_reports(date, output_dir, format)
            
            if not output_files:
                click.echo("No solution found for this date or visualization failed.")
                return
                
            # Display results
            click.echo(f"\nVisualizations generated:")
            for file_type, file_path in output_files.items():
                click.echo(f"  {file_type}: {file_path}")
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            raise click.ClickException(str(e))
        finally:
            await service.close()
    
    asyncio.run(_visualize())


if __name__ == '__main__':
    main()
