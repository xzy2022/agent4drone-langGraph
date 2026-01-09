"""
Test Suite for UAV API Client

This test suite verifies all API client functionality including:
- Connection and authentication
- Drone operations (list, status, movement, etc.)
- Session and environment queries
- Error handling

Prerequisites:
    - UAV Server running at http://localhost:8000
    - Optional: Set UAV_API_KEY for authenticated requests

Usage:
    python tests/test_api_client.py
"""

import os
import sys
from typing import Any, Dict

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.uav_api_client import UAVAPIClient, UAVAPIError


# Configuration
DEFAULT_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("UAV_API_KEY")  # Optional: set for authenticated requests


def print_section(title: str) -> None:
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(name: str, success: bool, data: Any = None) -> None:
    """Print formatted test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {name}")
    if data and not success:
        print(f"     Error: {data}")
    elif data and success:
        print(f"     Result: {str(data)[:100]}...")


def test_connection(client: UAVAPIClient) -> bool:
    """Test 1: Connection to UAV Server"""
    print_section("Test 1: Server Connection")

    try:
        # Try /health first, fall back to /sessions/current which should always exist
        try:
            result = client.test_connection()
            print_result("Connection successful (via /health)", True, result)
            print(f"     Server response: {result}")
        except UAVAPIError:
            # Health endpoint might not exist, try sessions instead
            result = client.get_current_session()
            print_result("Connection successful (via /sessions)", True)
            print(f"     Server is responding. Session: {result.get('name', 'Unknown')}")
        return True
    except UAVAPIError as e:
        print_result("Connection failed", False, str(e))
        print("\n     ⚠️  Make sure the UAV server is running at http://localhost:8000")
        return False


def test_list_drones(client: UAVAPIClient) -> bool:
    """Test 2: List all drones"""
    print_section("Test 2: List Drones")

    try:
        drones = client.list_drones()
        print_result("List drones", True, f"Found {len(drones)} drone(s)")

        for drone in drones:
            print(f"\n     🚁 Drone: {drone.get('id', 'Unknown')}")
            print(f"        Status: {drone.get('status', 'Unknown')}")
            print(f"        Battery: {drone.get('battery_level', 0):.1f}%")
            print(f"        Position: ({drone.get('x', 0):.1f}, {drone.get('y', 0):.1f}, {drone.get('z', 0):.1f})")

        return True
    except UAVAPIError as e:
        print_result("List drones failed", False, str(e))
        return False


def test_get_drone_status(client: UAVAPIClient) -> bool:
    """Test 3: Get detailed drone status"""
    print_section("Test 3: Get Drone Status")

    try:
        # First get a drone ID
        drones = client.list_drones()
        if not drones:
            print_result("No drones available", False, "Cannot test without drones")
            return False

        drone_id = drones[0]["id"]
        status = client.get_drone_status(drone_id)
        print_result(f"Get status for {drone_id}", True)

        print(f"\n     Detailed Status:")
        for key, value in status.items():
            print(f"        {key}: {value}")

        return True
    except UAVAPIError as e:
        print_result("Get drone status failed", False, str(e))
        return False


def test_session_info(client: UAVAPIClient) -> bool:
    """Test 4: Get session information"""
    print_section("Test 4: Session Information")

    all_passed = True

    try:
        session = client.get_current_session()
        print_result("Get current session", True, session)
        print(f"\n     Session: {session.get('name', 'Unknown')}")
        print(f"     Task: {session.get('task', 'Unknown')}")
        print(f"     Status: {session.get('status', 'Unknown')}")
    except UAVAPIError as e:
        print_result("Get current session failed", False, str(e))
        all_passed = False

    try:
        progress = client.get_task_progress()
        print_result("Get task progress", True)
        print(f"\n     Progress: {progress.get('progress_percentage', 0)}%")
        print(f"     Status: {progress.get('status_message', 'Unknown')}")
    except UAVAPIError as e:
        print_result("Get task progress failed", False, str(e))
        all_passed = False

    return all_passed


def test_environment(client: UAVAPIClient) -> bool:
    """Test 5: Get environment data"""
    print_section("Test 5: Environment Data")

    all_passed = True

    try:
        weather = client.get_weather()
        print_result("Get weather", True)
        print(f"\n     Weather: {weather.get('weather_type', 'Unknown')}")
        print(f"     Wind Speed: {weather.get('wind_speed', 0)} m/s")
        print(f"     Visibility: {weather.get('visibility', 0)} m")
    except UAVAPIError as e:
        print_result("Get weather failed", False, str(e))
        all_passed = False

    try:
        targets = client.get_targets()
        print_result(f"Get targets (found {len(targets)})", True)
    except UAVAPIError as e:
        print_result("Get targets failed", False, str(e))
        all_passed = False

    try:
        obstacles = client.get_obstacles()
        print_result(f"Get obstacles (found {len(obstacles)})", True)
    except UAVAPIError as e:
        print_result("Get obstacles failed", False, str(e))
        all_passed = False

    try:
        waypoints = client.get_waypoints()
        print_result(f"Get waypoints (found {len(waypoints)})", True)
    except UAVAPIError as e:
        print_result("Get waypoints failed", False, str(e))
        all_passed = False

    return all_passed


def test_nearby_entities(client: UAVAPIClient) -> bool:
    """Test 6: Get nearby entities"""
    print_section("Test 6: Nearby Entities")

    try:
        drones = client.list_drones()
        if not drones:
            print_result("No drones available", False, "Cannot test without drones")
            return False

        drone_id = drones[0]["id"]
        nearby = client.get_nearby_entities(drone_id)
        print_result(f"Get nearby entities for {drone_id}", True)

        print(f"\n     Nearby Drones: {len(nearby.get('drones', []))}")
        print(f"     Nearby Targets: {len(nearby.get('targets', []))}")
        print(f"     Nearby Obstacles: {len(nearby.get('obstacles', []))}")

        return True
    except UAVAPIError as e:
        print_result("Get nearby entities failed", False, str(e))
        return False


def test_drone_commands(client: UAVAPIClient) -> bool:
    """Test 7: Drone control commands (READ-ONLY)"""
    print_section("Test 7: Drone Commands (Status Check)")

    try:
        drones = client.list_drones()
        if not drones:
            print_result("No drones available", False, "Cannot test without drones")
            return False

        drone_id = drones[0]["id"]

        # Get current status (safe, read-only operation)
        status = client.get_drone_status(drone_id)
        print_result(f"Check status of {drone_id}", True)
        print(f"\n     Current state: {status.get('state', 'Unknown')}")
        print(f"     Battery: {status.get('battery_level', 0):.1f}%")

        return True
    except UAVAPIError as e:
        print_result("Drone command test failed", False, str(e))
        return False


def test_active_drone_commands(client: UAVAPIClient) -> bool:
    """Test 8: Drone control commands (ACTIVE)"""
    print_section("Test 8: Drone Commands (Active)")

    if os.getenv("RUN_COMMAND_TESTS") != "1":
        print_result("Active command tests", True, "SKIP (set RUN_COMMAND_TESTS=1 to enable)")
        return True

    all_passed = True

    try:
        drones = client.list_drones()
        if not drones:
            print_result("No drones available", False, "Cannot test without drones")
            return False

        drone_id = drones[0]["id"]
        secondary_id = drones[1]["id"] if len(drones) > 1 else None

        try:
            result = client.take_off(drone_id, altitude=3.0)
            print_result("Take off", True, result)
        except UAVAPIError as e:
            print_result("Take off failed", False, str(e))
            all_passed = False

        try:
            result = client.hover(drone_id, duration=2.0)
            print_result("Hover", True, result)
        except UAVAPIError as e:
            print_result("Hover failed", False, str(e))
            all_passed = False

        try:
            result = client.move_to(drone_id, x=5.0, y=5.0, z=3.0)
            print_result("Move to", True, result)
        except UAVAPIError as e:
            print_result("Move to failed", False, str(e))
            all_passed = False

        try:
            result = client.move_towards(drone_id, distance=3.0, heading=90.0, dz=0.5)
            print_result("Move towards", True, result)
        except UAVAPIError as e:
            print_result("Move towards failed", False, str(e))
            all_passed = False

        try:
            result = client.change_altitude(drone_id, altitude=4.0)
            print_result("Change altitude", True, result)
        except UAVAPIError as e:
            print_result("Change altitude failed", False, str(e))
            all_passed = False

        try:
            result = client.rotate(drone_id, heading=180.0)
            print_result("Rotate", True, result)
        except UAVAPIError as e:
            print_result("Rotate failed", False, str(e))
            all_passed = False

        try:
            path = [
                {"x": 6.0, "y": 5.0, "z": 4.0},
                {"x": 6.0, "y": 6.0, "z": 4.0},
            ]
            result = client.move_along_path(drone_id, waypoints=path)
            print_result("Move along path", True, result)
        except UAVAPIError as e:
            print_result("Move along path failed", False, str(e))
            all_passed = False

        try:
            result = client.take_photo(drone_id)
            print_result("Take photo", True, result)
        except UAVAPIError as e:
            print_result("Take photo failed", False, str(e))
            all_passed = False

        try:
            result = client.calibrate(drone_id)
            print_result("Calibrate", True, result)
        except UAVAPIError as e:
            print_result("Calibrate failed", False, str(e))
            all_passed = False

        try:
            result = client.set_home(drone_id)
            print_result("Set home", True, result)
        except UAVAPIError as e:
            print_result("Set home failed", False, str(e))
            all_passed = False

        try:
            result = client.return_home(drone_id)
            print_result("Return home", True, result)
        except UAVAPIError as e:
            print_result("Return home failed", False, str(e))
            all_passed = False

        try:
            result = client.send_message(drone_id, secondary_id, "Test message") if secondary_id else None
            if secondary_id:
                print_result("Send message", True, result)
            else:
                print_result("Send message", True, "SKIP (need at least 2 drones)")
        except UAVAPIError as e:
            print_result("Send message failed", False, str(e))
            all_passed = False

        try:
            result = client.broadcast(drone_id, "Broadcast test")
            print_result("Broadcast", True, result)
        except UAVAPIError as e:
            print_result("Broadcast failed", False, str(e))
            all_passed = False

        try:
            result = client.charge(drone_id, charge_amount=5.0)
            print_result("Charge", True, result)
        except UAVAPIError as e:
            print_result("Charge failed", False, str(e))
            all_passed = False

        try:
            result = client.land(drone_id)
            print_result("Land", True, result)
        except UAVAPIError as e:
            print_result("Land failed", False, str(e))
            all_passed = False

        return all_passed
    except UAVAPIError as e:
        print_result("Active command tests failed", False, str(e))
        return False


def test_error_handling(client: UAVAPIClient) -> bool:
    """Test 9: Error handling"""
    print_section("Test 9: Error Handling")

    all_passed = True

    # Test invalid drone ID
    try:
        client.get_drone_status("invalid-drone-id-999")
        print_result("404 error handling", False, "Should have raised UAVAPIError")
        all_passed = False
    except UAVAPIError as e:
        if e.status_code == 404:
            print_result("404 error handling", True, "Correctly raised 404 error")
        else:
            print_result("404 error handling", False, f"Wrong status code: {e.status_code}")
            all_passed = False

    # Test invalid endpoint
    try:
        client._request("GET", "/invalid/endpoint")
        print_result("Invalid endpoint handling", False, "Should have raised UAVAPIError")
        all_passed = False
    except UAVAPIError as e:
        print_result("Invalid endpoint handling", True, "Correctly raised error")

    return all_passed


def test_collision_detection(client: UAVAPIClient) -> bool:
    """Test 10: Collision detection"""
    print_section("Test 10: Collision Detection")

    all_passed = True

    try:
        # Test point collision
        result = client.check_point_collision(50.0, 50.0, 10.0, safety_margin=2.0)
        print_result("Point collision check", True, f"Collision: {result.get('has_collision', False)}")
    except UAVAPIError as e:
        print_result("Point collision check", False, str(e))
        all_passed = False

    try:
        # Test path collision
        result = client.check_path_collision(0.0, 0.0, 10.0, 100.0, 100.0, 10.0, safety_margin=2.0)
        print_result("Path collision check", True, f"Collision: {result.get('has_collision', False)}")
    except UAVAPIError as e:
        print_result("Path collision check", False, str(e))
        all_passed = False

    return all_passed


def run_all_tests() -> None:
    """Run all tests and report results"""
    print("\n" + "🚁" * 35)
    print("  UAV API Client - Test Suite")
    print("🚁" * 35)

    # Initialize client
    client = UAVAPIClient(base_url=DEFAULT_BASE_URL, api_key=API_KEY)

    print(f"\n📡 Configuration:")
    print(f"   Base URL: {DEFAULT_BASE_URL}")
    print(f"   API Key: {'Set' if API_KEY else 'Not set (USER role)'}")

    # Run tests
    results = {}

    # Only run subsequent tests if connection succeeds
    if not test_connection(client):
        print("\n❌ Cannot proceed without server connection")
        return

    results["list_drones"] = test_list_drones(client)
    results["get_drone_status"] = test_get_drone_status(client)
    results["session_info"] = test_session_info(client)
    results["environment"] = test_environment(client)
    results["nearby_entities"] = test_nearby_entities(client)
    results["drone_commands"] = test_drone_commands(client)
    results["active_drone_commands"] = test_active_drone_commands(client)
    results["error_handling"] = test_error_handling(client)
    results["collision_detection"] = test_collision_detection(client)

    # Summary
    print_section("Test Summary")
    total = len(results)
    passed = sum(results.values())

    print(f"\n   Total Tests: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {total - passed}")
    print(f"   Success Rate: {(passed/total*100):.1f}%")

    if passed == total:
        print("\n   ✅ All tests passed!")
    else:
        print("\n   ⚠️  Some tests failed - check output above")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
