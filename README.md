# Concrete Truck Optimizer

A route optimization system for concrete delivery trucks with live Google Maps traffic integration, priority handling, and constraint-based assignment.

## Features

- **Multi-truck fleet management** with capacity and capability constraints
- **Live traffic integration** using Google Maps Routes API
- **Priority-based job scheduling** with configurable weights
- **Overtime/deferral decision support** with automatic threshold detection
- **Google Maps URL generation** with automatic route segmentation
- **RESTful API** and command-line interface
- **Configurable solver strategies** (Greedy + Local Search, OR-Tools optional)

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry for dependency management
- Google Maps API key (Geocoding, Routes, Maps APIs enabled)

### Installation

1. **Clone and setup**:
```bash
git clone <repository-url>
cd Truck-Optimize
poetry install
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env and add your Google Maps API key
```

3. **Initialize database and test**:
```bash
poetry run python -m app.cli status
```

### Basic Usage

1. **Import jobs from CSV**:
```bash
poetry run python -m app.cli import example_input/day1.csv --date 2025-09-01
```

2. **Optimize routes**:
```bash
poetry run python -m app.cli optimize --date 2025-09-01 --auto ask
```

3. **Generate Google Maps URLs**:
```bash
poetry run python -m app.cli links --date 2025-09-01
```

4. **Start API server**:
```bash
poetry run python -m app.cli serve
# Visit http://localhost:8000/docs for API documentation
```

## Configuration

All tunable parameters are centralized in `config/params.yaml`:

### Fleet Configuration
```yaml
fleet:
  trucks:
    - name: "Big Truck"
      max_weight_lb: 12000
      bed_len_ft: 14
      bed_width_ft: 8
      height_limit_ft: 9
      large_capable: true
```

### Solver Parameters
```yaml
solver:
  use_ortools: false              # Enable OR-Tools solver
  random_seed: 42                 # Deterministic results
  efficiency_weight: 1.0          # Weight for drive/service time
  priority_weight: 0.1            # Weight for job priorities
  overtime_penalty_per_minute: 2.0
```

### Constraints
```yaml
constraints:
  big_truck_co_load_threshold_minutes: 15
  overtime_slack_minutes: 30
```

## API Endpoints

### Core Operations
- `POST /import` - Import jobs from CSV/JSON
- `POST /optimize` - Run route optimization
- `GET /routes/{date}` - Get optimized routes
- `GET /links/{date}` - Get Google Maps URLs

### Configuration
- `GET /config` - View current configuration
- `PUT /config` - Update parameters
- `GET /health` - System health check

### Data Access
- `GET /trucks` - List fleet vehicles
- `GET /locations` - List known locations
- `GET /items` - List item catalog
- `GET /kpis/{date}` - Route performance metrics

## Data Model

### Job Import Format (CSV)
```csv
location,action,items,priority,notes,earliest,latest,service_minutes_override
delfern,pickup,"big drill:1; rebar:5",2,"Need large truck",,,,
construction site,drop,"rebar:10",1,"Standard delivery",,,,
```

### Item Specifications
Items in the format: `"item1:qty1; item2:qty2"`

Available categories:
- **machine** (30 min service time) - requires large truck
- **equipment** (15 min service time)
- **material** (15 min service time) 
- **fuel** (10 min service time)

## Solver Strategies

### Greedy + Local Search (Default)
- Fast nearest-neighbor construction
- 2-opt local search improvements
- Inter-route job relocation
- Deterministic with configurable seed

### OR-Tools VRP (Optional)
```bash
poetry install --extras ortools
```
Enable in config: `solver.use_ortools: true`

## Overtime Decision Handling

When projected overtime exceeds `overtime_slack_minutes`:

1. **ask mode** (default): Returns HTTP 409 with both options
2. **overtime mode**: Allows overtime
3. **defer mode**: Automatically defers low-priority jobs

API Response for overtime decision:
```json
{
  "type": "overtime_decision_required",
  "overtime_plan": { /* full route plan */ },
  "defer_plan": { /* alternative with deferred jobs */ },
  "overtime_minutes_diff": 45.5,
  "jobs_deferred_count": 3
}
```

## Google Maps Integration

### Features Used
- **Geocoding API**: Convert addresses to coordinates
- **Routes API**: Traffic-aware travel time matrix
- **Maps URLs**: Shareable route visualization

### URL Segmentation
Routes automatically split into multiple URLs when exceeding waypoint limits:
```yaml
google:
  maps:
    segment_max_waypoints: 9  # Configurable limit
```

### Rate Limiting
```yaml
google:
  rate_limit_requests_per_second: 10
  max_retries: 3
  retry_delay_seconds: 1.0
```

## Development

### Project Structure
```
app/
├── api.py              # FastAPI application
├── cli.py              # Command-line interface
├── models.py           # SQLModel database models
├── schemas.py          # Pydantic validation schemas
├── service.py          # Main business logic
├── repo.py             # Database repository layer
├── constraints.py      # Constraint validation
├── distance.py         # Google Maps integration
├── solver_greedy.py    # Greedy solver implementation
├── solver_ortools.py   # OR-Tools solver (optional)
├── router.py           # Solver coordination
└── url_builder.py      # Google Maps URL generation

config/
└── params.yaml         # All configuration parameters

tests/
├── test_constraints.py
├── test_solver.py
├── test_url_builder.py
└── test_distance_integration.py
```

### Running Tests
```bash
poetry run pytest
poetry run pytest -m "not integration"  # Skip API-dependent tests
```

### Code Quality
```bash
poetry run black .
poetry run ruff check .
poetry run mypy app/
```

## Deployment

### Environment Variables
```bash
GOOGLE_MAPS_API_KEY=required_for_geocoding_and_routing
DATABASE_URL=sqlite:///./truck_optimizer.db  # Optional override
LOG_LEVEL=INFO  # DEBUG|INFO|WARNING|ERROR
```

### Production Considerations
- Use PostgreSQL instead of SQLite for concurrent access
- Configure CORS origins in `api.py`
- Set up proper logging aggregation
- Monitor Google Maps API usage and costs
- Consider caching geocoding results

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --only=main
COPY . .
CMD ["poetry", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Common Issues

1. **"Import fastapi could not be resolved"**
   ```bash
   poetry install  # Install dependencies
   ```

2. **"Google Maps API key not configured"**
   ```bash
   cp .env.example .env
   # Edit .env with your API key
   ```

3. **"No routes found for date"**
   - Ensure jobs are imported for the correct date
   - Check that locations were successfully geocoded

4. **"Geocoding failed"**
   - Verify Google Maps API key has Geocoding API enabled
   - Check API quota and billing setup
   - Use more specific addresses

### Performance Tuning

- Adjust `solver.local_search_iterations` for solution quality vs speed
- Increase `google.rate_limit_requests_per_second` if API quota allows
- Enable `dev.cache_geocoding` to avoid repeated API calls
- Use `dev.mock_google_api: true` for development without API costs

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check this README and configuration documentation
2. Review API documentation at `/docs` endpoint
3. Enable DEBUG logging for detailed troubleshooting
4. Check Google Maps API quotas and billing
