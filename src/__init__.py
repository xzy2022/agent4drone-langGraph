"""
UAV Control System - API Client Module

This module provides a standardized, stateless HTTP client for interacting
with the UAV Control System API.
"""

from .uav_api_client import UAVAPIClient, UAVAPIError

__all__ = ["UAVAPIClient", "UAVAPIError"]
__version__ = "1.0.0"
