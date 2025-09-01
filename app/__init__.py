"""
Truck Route Optimizer package.
Concrete truck routing optimization with live traffic and priority handling.
"""

__version__ = "0.1.0"
__author__ = "Alon Florentin"

from .service import TruckOptimizerService
from .api import app
from .models import *
from .schemas import *

__all__ = [
    "TruckOptimizerService",
    "app"
]
