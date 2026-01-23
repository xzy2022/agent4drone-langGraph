import unittest
from typing import List, Optional, Dict
from core.entities.drone import Drone, DroneState
from core.entities.position import Position
from core.interfaces.drone_repository import DroneRepository
from core.use_cases.drone_control import DroneControlUseCase

class InMemoryDroneRepository(DroneRepository):
    def __init__(self):
        self.drones: Dict[str, Drone] = {}

    def get_drone(self, drone_id: str) -> Optional[Drone]:
        return self.drones.get(drone_id)

    def save_drone(self, drone: Drone) -> None:
        self.drones[drone.id] = drone

    def get_all_drones(self) -> List[Drone]:
        return list(self.drones.values())

class TestDroneControlUseCase(unittest.TestCase):
    def setUp(self):
        self.repo = InMemoryDroneRepository()
        self.control = DroneControlUseCase(self.repo)
        self.drone_id = "test_drone"
        self.drone = Drone(id=self.drone_id)
        self.repo.save_drone(self.drone)

    def test_takeoff(self):
        # Initial state is IDLE (default) -> Land first to test takeoff logic correctly if strictly implemented
        # But Drone default is IDLE. takeoff requires LANDED.
        # Let's set it to LANDED.
        self.drone.state = DroneState.LANDED
        self.repo.save_drone(self.drone)
        
        success = self.control.takeoff(self.drone_id, 10.0)
        self.assertTrue(success)
        
        updated_drone = self.repo.get_drone(self.drone_id)
        self.assertEqual(updated_drone.state, DroneState.IDLE) # Takeoff transitions to IDLE in air
        self.assertEqual(updated_drone.position.z, 10.0)

    def test_takeoff_fail_when_already_flying(self):
        self.drone.state = DroneState.FLYING
        self.repo.save_drone(self.drone)
        
        success = self.control.takeoff(self.drone_id, 10.0)
        self.assertFalse(success)

    def test_move(self):
        self.drone.state = DroneState.IDLE # Can move in IDLE
        self.repo.save_drone(self.drone)
        
        target = Position(10, 10, 10)
        success = self.control.move_drone(self.drone_id, target)
        self.assertTrue(success)
        
        updated_drone = self.repo.get_drone(self.drone_id)
        self.assertEqual(updated_drone.position, target)

    def test_land(self):
        self.drone.state = DroneState.IDLE
        self.drone.position.z = 10.0
        self.repo.save_drone(self.drone)
        
        success = self.control.land(self.drone_id)
        self.assertTrue(success)
        
        updated_drone = self.repo.get_drone(self.drone_id)
        self.assertEqual(updated_drone.state, DroneState.LANDED)
        self.assertEqual(updated_drone.position.z, 0.0)

if __name__ == '__main__':
    unittest.main()
