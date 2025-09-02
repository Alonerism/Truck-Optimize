# Truck Optimizer Enhancements

## Overview
This document summarizes the enhancements made to the Concrete Truck Optimizer MVP.

## Features Implemented

### 1. Single-Truck Mode
- Added `single_truck_mode` configuration option (0=multi-truck, 1=prefer single truck)
- Added `trucks_used_penalty` parameter to penalize solutions using multiple trucks
- Modified cost function to apply penalties for additional trucks when in single-truck mode
- Added CLI option `--single-truck` to enable single-truck mode

### 2. Multi-Objective Optimization
- Added weighted cost function with configurable weights for different objectives:
  - Drive minutes weight
  - Service minutes weight
  - Overtime minutes weight
  - Max route minutes weight
  - Priority soft cost weight
- Added backward compatibility with legacy cost function

### 3. Solver Improvements
- Added regret-2 insertion algorithm as alternative to pure greedy construction
  - Prioritizes jobs based on difference between best and second-best insertion costs
  - Produces better-quality solutions than greedy insertion alone
- Enhanced local search improvements:
  - Configurable iterations and time limit
  - Multiple neighborhood operations (relocate, swap, two_opt)
  - Detailed tracing of improvement operations

### 4. Visualization & Reporting
- Added visualization module using Plotly:
  - Truck utilization charts
  - Timeline/Gantt charts
  - Text-based reports
- Implemented CSV export for spreadsheet analysis
- Added `visualize` command to CLI
- Added `--output-dir` option to specify output directory

### 5. Traceability
- Added comprehensive tracing of solver decisions:
  - Records evaluations of all truck-job combinations
  - Logs position evaluations within routes
  - Captures constraint violations
  - Records algorithm choices and parameters
- Outputs trace data in JSON format for debugging

## Configuration Updates
- Updated `params.yaml` with new configuration sections:
  - Added `single_truck_mode` and `trucks_used_penalty`
  - Added `weights` section for multi-objective optimization
  - Added `improve` section for local search parameters
  - Added `tracing` section for traceability settings

## CLI Enhancements
- Added new options to `optimize` command:
  - `--single-truck`: Enable single-truck mode
  - `--solver`: Choose solver strategy (greedy or regret2)
  - `--trace`: Enable decision tracing
  - `--visualize`: Generate visualization reports
  - `--output-dir`: Specify directory for output files
- Added new `visualize` command to generate reports for existing solutions

## API Updates
- Enhanced `OptimizeRequest` with new parameters
- Added report generation endpoints

## Testing
- Added unit tests for regret-2 algorithm
- Added test cases for cost function with different weights
