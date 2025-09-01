"""
FastAPI application for truck route optimization.
Provides REST API endpoints for import, optimization, and route visualization.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware

from .service import TruckOptimizerService
from .schemas import (
    ImportRequest, ImportStatsResponse, OptimizeRequest,
    ConfigUpdateRequest, HealthResponse, KPIResponse
)
from .models import OptimizationResult, OvertimeDecisionRequest


logger = logging.getLogger(__name__)

# Global service instance
service: TruckOptimizerService = None


def get_service() -> TruckOptimizerService:
    """Dependency to get service instance."""
    global service
    if service is None:
        service = TruckOptimizerService()
    return service


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Truck Route Optimizer",
        description="Concrete truck route optimization with live traffic and priority handling",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize application on startup."""
        global service
        service = TruckOptimizerService()
        logger.info("Truck Optimizer API started")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up on shutdown."""
        if service:
            await service.close()
        logger.info("Truck Optimizer API stopped")
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check(svc: TruckOptimizerService = Depends(get_service)):
        """Health check endpoint."""
        health_data = svc.health_check()
        return HealthResponse(
            status=health_data["status"],
            version="0.1.0",
            database_connected=health_data["database_connected"],
            google_api_configured=health_data["google_api_configured"],
            timestamp=health_data["timestamp"]
        )
    
    @app.post("/import", response_model=ImportStatsResponse)
    async def import_jobs(
        request: ImportRequest,
        svc: TruckOptimizerService = Depends(get_service)
    ):
        """
        Import jobs from CSV or JSON data.
        Upserts locations, items, and jobs; geocodes unknown addresses.
        """
        try:
            stats = await svc.import_jobs(request)
            return stats
        except Exception as e:
            logger.error(f"Import failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/optimize")
    async def optimize_routes(
        request: OptimizeRequest,
        svc: TruckOptimizerService = Depends(get_service)
    ):
        """
        Run route optimization for a specific date.
        Returns either the optimized plan or an overtime decision request.
        """
        try:
            result = await svc.optimize_routes(request)
            
            # Check if this is an overtime decision request
            if isinstance(result, OvertimeDecisionRequest):
                # Return 409 Conflict with both options
                raise HTTPException(
                    status_code=409,
                    detail={
                        "type": "overtime_decision_required",
                        "overtime_plan": result.overtime_plan.model_dump(),
                        "defer_plan": result.defer_plan.model_dump(),
                        "overtime_minutes_diff": result.overtime_minutes_diff,
                        "jobs_deferred_count": result.jobs_deferred_count,
                        "message": "Route optimization requires overtime decision. "
                                 "Choose between allowing overtime or deferring jobs."
                    }
                )
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/routes/{date}")
    async def get_routes(
        date: str,
        svc: TruckOptimizerService = Depends(get_service)
    ):
        """Get optimized routes for a specific date."""
        try:
            # This would load saved routes from database
            # For now, return a placeholder
            route_assignments = svc.repo.get_route_assignments_by_date(date)
            
            if not route_assignments:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No routes found for date {date}"
                )
            
            # Convert to API response format
            # This is simplified - full implementation would reconstruct complete route data
            routes = []
            for assignment in route_assignments:
                routes.append({
                    "truck_id": assignment.truck_id,
                    "truck_name": assignment.truck.name,
                    "total_drive_minutes": assignment.total_drive_minutes,
                    "total_service_minutes": assignment.total_service_minutes,
                    "total_weight_lb": assignment.total_weight_lb,
                    "overtime_minutes": assignment.overtime_minutes,
                    "stops": []  # Would load route stops
                })
            
            unassigned_jobs = svc.repo.get_unassigned_jobs_by_date(date)
            
            return {
                "date": date,
                "routes": routes,
                "unassigned_jobs": [{"job_id": uj.job_id, "reason": uj.reason} for uj in unassigned_jobs],
                "total_routes": len(routes),
                "total_unassigned": len(unassigned_jobs)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get routes: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/links/{date}")
    async def get_route_links(
        date: str,
        svc: TruckOptimizerService = Depends(get_service)
    ):
        """Get Google Maps URLs for routes on a specific date."""
        try:
            urls = await svc.get_route_urls(date)
            
            if not urls:
                raise HTTPException(
                    status_code=404,
                    detail=f"No route URLs found for date {date}"
                )
            
            return {
                "date": date,
                "truck_routes": urls
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get route URLs: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/config")
    async def get_config(svc: TruckOptimizerService = Depends(get_service)):
        """Get current configuration (non-secret parameters)."""
        try:
            config = svc.get_config()
            
            # Remove sensitive information
            if "google" in config and "api_key" in config["google"]:
                del config["google"]["api_key"]
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to get config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.put("/config")
    async def update_config(
        request: ConfigUpdateRequest,
        svc: TruckOptimizerService = Depends(get_service)
    ):
        """Update configuration parameters."""
        try:
            svc.update_config(request.updates)
            return {"message": "Configuration updated successfully"}
            
        except NotImplementedError:
            raise HTTPException(
                status_code=501,
                detail="Configuration updates not yet implemented"
            )
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/trucks")
    async def get_trucks(svc: TruckOptimizerService = Depends(get_service)):
        """Get all available trucks."""
        try:
            trucks = svc.repo.get_trucks()
            return [
                {
                    "id": truck.id,
                    "name": truck.name,
                    "max_weight_lb": truck.max_weight_lb,
                    "bed_len_ft": truck.bed_len_ft,
                    "bed_width_ft": truck.bed_width_ft,
                    "height_limit_ft": truck.height_limit_ft,
                    "large_capable": truck.large_capable
                }
                for truck in trucks
            ]
        except Exception as e:
            logger.error(f"Failed to get trucks: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/locations")
    async def get_locations(svc: TruckOptimizerService = Depends(get_service)):
        """Get all locations."""
        try:
            locations = svc.repo.get_locations()
            return [
                {
                    "id": location.id,
                    "name": location.name,
                    "address": location.address,
                    "lat": location.lat,
                    "lon": location.lon,
                    "window_start": location.window_start.isoformat() if location.window_start else None,
                    "window_end": location.window_end.isoformat() if location.window_end else None
                }
                for location in locations
            ]
        except Exception as e:
            logger.error(f"Failed to get locations: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/items")
    async def get_items(svc: TruckOptimizerService = Depends(get_service)):
        """Get all items in catalog."""
        try:
            items = svc.repo.get_items()
            return [
                {
                    "id": item.id,
                    "name": item.name,
                    "category": item.category,
                    "weight_lb_per_unit": item.weight_lb_per_unit,
                    "volume_ft3_per_unit": item.volume_ft3_per_unit,
                    "requires_large_truck": item.requires_large_truck
                }
                for item in items
            ]
        except Exception as e:
            logger.error(f"Failed to get items: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/kpis/{date}", response_model=KPIResponse)
    async def get_kpis(
        date: str,
        svc: TruckOptimizerService = Depends(get_service)
    ):
        """Get KPIs for routes on a specific date."""
        try:
            route_assignments = svc.repo.get_route_assignments_by_date(date)
            unassigned_jobs = svc.repo.get_unassigned_jobs_by_date(date)
            all_jobs = svc.repo.get_jobs_by_date(date)
            
            if not route_assignments and not unassigned_jobs:
                raise HTTPException(
                    status_code=404,
                    detail=f"No optimization results found for date {date}"
                )
            
            # Calculate KPIs
            total_drive_minutes = sum(ra.total_drive_minutes for ra in route_assignments)
            total_service_minutes = sum(ra.total_service_minutes for ra in route_assignments)
            total_overtime_minutes = sum(ra.overtime_minutes for ra in route_assignments)
            trucks_used = len([ra for ra in route_assignments if ra.total_drive_minutes > 0])
            
            # Simple efficiency score (lower is better)
            efficiency_score = total_drive_minutes + total_service_minutes + (total_overtime_minutes * 2)
            
            # Priority score (would need more complex calculation)
            priority_score = 100.0  # Placeholder
            
            return KPIResponse(
                total_drive_minutes=total_drive_minutes,
                total_service_minutes=total_service_minutes,
                total_overtime_minutes=total_overtime_minutes,
                trucks_used=trucks_used,
                jobs_assigned=len(all_jobs) - len(unassigned_jobs),
                jobs_unassigned=len(unassigned_jobs),
                efficiency_score=efficiency_score,
                priority_score=priority_score
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to calculate KPIs: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
