import unittest
from unittest.mock import MagicMock
from core.entities.drone import Drone, DroneState
from core.entities.position import Position
from core.entities.task import Task, TaskStatus
from core.entities.map import GridMap
from core.use_cases.drone_control import DroneControlUseCase
from core.use_cases.navigation import NavigationUseCase
from core.use_cases.navigation_executor import NavigationExecutor
from core.interfaces.drone_repository import DroneRepository

class TestNavigationExecutor(unittest.TestCase):
    def setUp(self):
        self.repo = MagicMock(spec=DroneRepository)
        self.control = MagicMock(spec=DroneControlUseCase)
        self.nav = MagicMock(spec=NavigationUseCase)
        self.map = GridMap(resolution=1.0)
        
        self.executor = NavigationExecutor(self.repo, self.control, self.nav, self.map)
        
        self.drone_id = "test_drone"
        self.drone = Drone(id=self.drone_id)
        self.repo.get_drone.return_value = self.drone

    def test_execute_task_success_land(self):
        # Target Z = 0 means it should land at the end
        task = Task(id="t1", position=Position(10, 10, 0))
        self.drone.state = DroneState.LANDED
        
        # Mock Path
        path = [Position(1,1,5), Position(10,10,5)] # Altitude 5 during flight
        self.nav.compute_path.return_value = path
        
        # Mock Control Success
        self.control.takeoff.return_value = True
        self.control.move_line.return_value = True
        self.control.land.return_value = True
        
        success = self.executor.execute_task(self.drone_id, task, default_altitude=5.0)
        
        self.assertTrue(success)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        
        # Verify calls
        # 1. Takeoff to 5.0m
        self.control.takeoff.assert_called_once_with(self.drone_id, 5.0)
        # 2. Move along path (2 points)
        self.assertEqual(self.control.move_line.call_count, 2)
        # 3. Land
        self.control.land.assert_called_once()
        
    def test_execute_task_fail_path(self):
        task = Task(id="t1", position=Position(10, 10, 10))
        self.drone.state = DroneState.HOVERING # So it doesn't takeoff
        self.nav.compute_path.return_value = None
        
        success = self.executor.execute_task(self.drone_id, task)
        
        self.assertFalse(success)
        self.assertEqual(task.status, TaskStatus.FAILED)
        
    def test_execute_task_fail_takeoff(self):
        task = Task(id="t1", position=Position(10, 10, 10))
        self.drone.state = DroneState.LANDED
        self.control.takeoff.return_value = False # Fail takeoff
        
        success = self.executor.execute_task(self.drone_id, task)
        
        self.assertFalse(success)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.control.move_line.assert_not_called()

if __name__ == '__main__':
    unittest.main(verbosity=2)
