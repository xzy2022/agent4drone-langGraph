# src\uav_api_client.py
"""
UAV API Client - Standardized HTTP Client for UAV Control System

This client provides a clean, stateless interface to the UAV Control System API.
It handles all HTTP communication, authentication, and error handling.

Usage:
    client = UAVAPIClient(base_url="http://localhost:8000", api_key=None)
    drones = client.list_drones()

All methods return structured JSON responses from the API.
All methods raise UAVAPIError for API-related errors.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests


@dataclass
class UAVAPIError(Exception):
    """Base exception for UAV API errors"""

    message: str
    status_code: Optional[int] = None
    detail: Optional[str] = None

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}: {self.detail}"
        return f"{self.message}: {self.detail}"


class UAVAPIClient:
    """
    Stateless HTTP client for the UAV Control System API.

    This client maintains no internal state - all state is managed by the server.
    It can be safely shared across multiple agents and threads.

    Authentication:
        - No api_key: USER role (basic access)
        - api_key provided: Corresponds to the role associated with the key
    """

    # API Base Paths
    BASE_PATH = "/api/v1"

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None, timeout: float = 30.0):
        """
        Initialize the UAV API Client.

        Args:
            base_url: Base URL of the UAV API server (e.g., "http://localhost:8000")
            api_key: Optional API key for authentication. If None, uses USER role.
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json"}

        if self.api_key:
            self._headers["X-API-Key"] = self.api_key

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint"""
        return urljoin(self.base_url + "/", endpoint.lstrip("/"))

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Make HTTP request to the API with standardized error handling.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path (e.g., "/drones")
            params: Query parameters
            json_data: Request body for POST/PUT requests

        Returns:
            Parsed JSON response

        Raises:
            UAVAPIError: For all API errors with detailed message
        """
        url = self._build_url(endpoint)

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self._headers,
                params=params,
                json=json_data,
                timeout=self.timeout,
            )

            # Handle 204 No Content
            if response.status_code == 204:
                return None

            # Try to parse JSON
            try:
                response_data = response.json()
            except ValueError:
                response_data = {"text": response.text}

            # Raise for error status codes
            response.raise_for_status()
            return response_data

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            detail = None

            # Extract error detail from response if available
            try:
                error_data = e.response.json()
                detail = error_data.get("detail", error_data.get("message", str(e)))
            except ValueError:
                detail = e.response.text or str(e)

            # Specific error messages
            if status_code == 401:
                raise UAVAPIError(
                    message="Authentication failed",
                    status_code=status_code,
                    detail="Invalid API key or no API key provided",
                ) from e
            elif status_code == 403:
                raise UAVAPIError(
                    message="Permission denied",
                    status_code=status_code,
                    detail=detail,
                ) from e
            elif status_code == 404:
                raise UAVAPIError(
                    message="Resource not found",
                    status_code=status_code,
                    detail=detail,
                ) from e
            elif status_code == 422:
                raise UAVAPIError(
                    message="Validation error",
                    status_code=status_code,
                    detail=detail,
                ) from e
            else:
                raise UAVAPIError(
                    message=f"HTTP {status_code} error",
                    status_code=status_code,
                    detail=detail,
                ) from e

        except requests.exceptions.Timeout as e:
            raise UAVAPIError(
                message="Request timeout",
                detail=f"Request exceeded {self.timeout}s timeout",
            ) from e

        except requests.exceptions.ConnectionError as e:
            raise UAVAPIError(
                message="Connection failed",
                detail=f"Could not connect to UAV server at {self.base_url}",
            ) from e

        except requests.exceptions.RequestException as e:
            raise UAVAPIError(
                message="Request failed",
                detail=str(e),
            ) from e

    # ========================================================================
    # Connection & Health
    # ========================================================================

    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to the UAV server.

        Returns:
            Server health/status information

        Raises:
            UAVAPIError: If connection fails
        """
        return self._request("GET", "/health")

    def get_api_info(self) -> Dict[str, Any]:
        """
        Get API information and available endpoints.

        Returns:
            API metadata including version, endpoints, etc.
        """
        return self._request("GET", "/")

    # ========================================================================
    # Drone Operations
    # ========================================================================

    def list_drones(self) -> List[Dict[str, Any]]:
        """
        Get all drones in the current session.

        Returns:
            List of drone objects with status, battery, position, etc.
        """
        return self._request("GET", "/drones")

    def get_drone_status(self, drone_id: str) -> Dict[str, Any]:
        """
        Get detailed status of a specific drone.

        Args:
            drone_id: Unique identifier of the drone

        Returns:
            Drone status including position, battery, heading, state, etc.
        """
        return self._request("GET", f"/drones/{drone_id}")

    # ------------------------------------------------------------------------
    # Movement Commands
    # ------------------------------------------------------------------------

    def take_off(self, drone_id: str, altitude: float = 10.0) -> Dict[str, Any]:
        """
        Command drone to take off to specified altitude.

        Args:
            drone_id: Unique identifier of the drone
            altitude: Target altitude in meters (default: 10.0)

        Returns:
            Command execution result
        """
        return self._request("POST", f"/drones/{drone_id}/command/take_off", params={"altitude": altitude})

    def land(self, drone_id: str) -> Dict[str, Any]:
        """
        Command drone to land at current position.

        Args:
            drone_id: Unique identifier of the drone

        Returns:
            Command execution result
        """
        return self._request("POST", f"/drones/{drone_id}/command/land")

    def move_to(self, drone_id: str, x: float, y: float, z: float) -> Dict[str, Any]:
        """
        Move drone to specific 3D coordinates.

        Args:
            drone_id: Unique identifier of the drone
            x: Target X coordinate in meters
            y: Target Y coordinate in meters
            z: Target Z coordinate (altitude) in meters

        Returns:
            Command execution result
        """
        return self._request(
            "POST",
            f"/drones/{drone_id}/command/move_to",
            params={"x": x, "y": y, "z": z},
        )

    def move_along_path(self, drone_id: str, waypoints: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Move drone along a path of waypoints.

        Args:
            drone_id: Unique identifier of the drone
            waypoints: List of waypoints [{"x": 10, "y": 20, "z": 15}, ...]

        Returns:
            Command execution result
        """
        return self._request(
            "POST",
            f"/drones/{drone_id}/command/move_along_path",
            json_data={"waypoints": waypoints},
        )

    def change_altitude(self, drone_id: str, altitude: float) -> Dict[str, Any]:
        """
        Change drone altitude while maintaining X/Y position.

        Args:
            drone_id: Unique identifier of the drone
            altitude: Target altitude in meters

        Returns:
            Command execution result
        """
        return self._request(
            "POST",
            f"/drones/{drone_id}/command/change_altitude",
            params={"altitude": altitude},
        )

    def hover(self, drone_id: str, duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Command drone to hover at current position.

        Args:
            drone_id: Unique identifier of the drone
            duration: Optional duration to hover in seconds. If None, hovers indefinitely.

        Returns:
            Command execution result
        """
        params = {}
        if duration is not None:
            params["duration"] = duration
        return self._request("POST", f"/drones/{drone_id}/command/hover", params=params)

    def rotate(self, drone_id: str, heading: float) -> Dict[str, Any]:
        """
        Rotate drone to face specific direction.

        Args:
            drone_id: Unique identifier of the drone
            heading: Target heading in degrees (0=North, 90=East, 180=South, 270=West)

        Returns:
            Command execution result
        """
        return self._request("POST", f"/drones/{drone_id}/command/rotate", params={"heading": heading})

    def move_towards(
        self,
        drone_id: str,
        distance: float,
        heading: Optional[float] = None,
        dz: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Move drone a specific distance in a direction.

        Args:
            drone_id: Unique identifier of the drone
            distance: Distance to move in meters
            heading: Optional heading direction (0-360). If None, uses current heading.
            dz: Optional vertical component (altitude change in meters)

        Returns:
            Command execution result
        """
        params = {"distance": distance}
        if heading is not None:
            params["heading"] = heading
        if dz is not None:
            params["dz"] = dz
        return self._request("POST", f"/drones/{drone_id}/command/move_towards", params=params)

    def return_home(self, drone_id: str) -> Dict[str, Any]:
        """
        Command drone to return to its home position.

        Args:
            drone_id: Unique identifier of the drone

        Returns:
            Command execution result
        """
        return self._request("POST", f"/drones/{drone_id}/command/return_home")

    def set_home(self, drone_id: str) -> Dict[str, Any]:
        """
        Set drone's current position as its new home position.

        Args:
            drone_id: Unique identifier of the drone

        Returns:
            Command execution result
        """
        return self._request("POST", f"/drones/{drone_id}/command/set_home")

    # ------------------------------------------------------------------------
    # Utility Commands
    # ------------------------------------------------------------------------

    def calibrate(self, drone_id: str) -> Dict[str, Any]:
        """
        Calibrate drone sensors.

        Args:
            drone_id: Unique identifier of the drone

        Returns:
            Command execution result
        """
        return self._request("POST", f"/drones/{drone_id}/command/calibrate")

    def charge(self, drone_id: str, charge_amount: float) -> Dict[str, Any]:
        """
        Charge drone battery (when landed at charging station).

        Args:
            drone_id: Unique identifier of the drone
            charge_amount: Amount to charge in percentage points (0-100)

        Returns:
            Command execution result with new battery level
        """
        return self._request(
            "POST",
            f"/drones/{drone_id}/command/charge",
            params={"charge_amount": charge_amount},
        )

    def take_photo(self, drone_id: str) -> Dict[str, Any]:
        """
        Take a photo with drone camera.

        Args:
            drone_id: Unique identifier of the drone

        Returns:
            Command execution result with photo URL/metadata
        """
        return self._request("POST", f"/drones/{drone_id}/command/take_photo")

    # ------------------------------------------------------------------------
    # Communication Commands
    # ------------------------------------------------------------------------

    def send_message(self, drone_id: str, target_drone_id: str, message: str) -> Dict[str, Any]:
        """
        Send a message from one drone to another.

        Args:
            drone_id: Unique identifier of the sender drone
            target_drone_id: Unique identifier of the recipient drone
            message: Content of the message

        Returns:
            Command execution result
        """
        return self._request(
            "POST",
            f"/drones/{drone_id}/command/send_message",
            params={"target_drone_id": target_drone_id, "message": message},
        )

    def broadcast(self, drone_id: str, message: str) -> Dict[str, Any]:
        """
        Broadcast a message from one drone to all other drones.

        Args:
            drone_id: Unique identifier of the sender drone
            message: Content of the message

        Returns:
            Command execution result
        """
        return self._request(
            "POST",
            f"/drones/{drone_id}/command/broadcast",
            params={"message": message},
        )

    # ========================================================================
    # Session Operations
    # ========================================================================

    def get_current_session(self) -> Dict[str, Any]:
        """
        Get information about the current mission session.

        Returns:
            Session info including name, task type, status, etc.
        """
        return self._request("GET", "/sessions/current")

    def get_session_data(self, session_id: str = "current") -> Dict[str, Any]:
        """
        Get all entities in a session (drones, targets, obstacles, environment).

        Args:
            session_id: Session ID or 'current' for active session

        Returns:
            Session data with all entities
        """
        return self._request("GET", f"/sessions/{session_id}/data")

    def get_task_progress(self, session_id: str = "current") -> Dict[str, Any]:
        """
        Get mission task completion progress.

        Args:
            session_id: Session ID or 'current' for active session

        Returns:
            Task progress including percentage, completed targets, status message
        """
        return self._request("GET", f"/sessions/{session_id}/task-progress")

    # ========================================================================
    # Environment Operations
    # ========================================================================

    def get_weather(self) -> Dict[str, Any]:
        """
        Get current weather conditions.

        Returns:
            Weather data including wind speed, visibility, conditions
        """
        return self._request("GET", "/environments/current")

    def get_targets(self) -> List[Dict[str, Any]]:
        """
        Get all targets in the current session.

        Returns:
            List of targets with positions, types, visit status
        """
        return self._request("GET", "/targets")

    def get_waypoints(self) -> List[Dict[str, Any]]:
        """
        Get all charging station waypoints.

        Returns:
            List of charging station locations
        """
        return self._request("GET", "/targets/waypoints")

    def get_obstacles(self) -> List[Dict[str, Any]]:
        """
        Get all obstacles in the current session.

        Returns:
            List of obstacles with positions, sizes, shapes
        """
        return self._request("GET", "/obstacles")

    def get_nearby_entities(self, drone_id: str) -> Dict[str, Any]:
        """
        Get entities near a specific drone (within perception radius).

        Args:
            drone_id: Unique identifier of the drone

        Returns:
            Nearby entities including drones, targets, obstacles
        """
        return self._request("GET", f"/drones/{drone_id}/nearby")

    # ========================================================================
    # Safety Operations
    # ========================================================================

    def check_point_collision(
        self, x: float, y: float, z: float, safety_margin: float = 0.0
    ) -> Dict[str, Any]:
        """
        Check if a point collides with any obstacle.

        Args:
            x: X coordinate of the point
            y: Y coordinate of the point
            z: Z coordinate (altitude) of the point
            safety_margin: Additional safety margin in meters

        Returns:
            Collision check result with boolean collision status
        """
        return self._request(
            "POST",
            "/obstacles/collision/check",
            json_data={"point": {"x": x, "y": y, "z": z}, "safety_margin": safety_margin},
        )

    def check_path_collision(
        self,
        start_x: float,
        start_y: float,
        start_z: float,
        end_x: float,
        end_y: float,
        end_z: float,
        safety_margin: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Check if a path between two points intersects any obstacle.

        Args:
            start_x, start_y, start_z: Starting coordinates
            end_x, end_y, end_z: Ending coordinates
            safety_margin: Additional safety margin in meters (default: 1.0)

        Returns:
            Collision check result with boolean collision status
        """
        return self._request(
            "POST",
            "/obstacles/collision/path",
            json_data={
                "start": {"x": start_x, "y": start_y, "z": start_z},
                "end": {"x": end_x, "y": end_y, "z": end_z},
                "safety_margin": safety_margin,
            },
        )
