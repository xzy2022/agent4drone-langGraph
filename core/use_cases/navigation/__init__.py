import math
import heapq
from typing import List, Optional, Tuple, Set, Dict
from ...entities.position import Position
from ...entities.map import GridMap
# Optionally implement NavigationService if we want D.I.
# from ..interfaces.navigation_service import NavigationService

class NavigationUseCase:
    """
    Use case for path planning using A* algorithm.
    """
    
    def compute_path(self, start: Position, end: Position, grid_map: GridMap) -> Optional[List[Position]]:
        """
        Find a path from start to end on the grid_map.
        Returns a list of Position objects representing the path.
        """
        # 1. Convert real world coordinates to grid coordinates
        start_gx, start_gy = grid_map.to_grid(start.x), grid_map.to_grid(start.y)
        end_gx, end_gy = grid_map.to_grid(end.x), grid_map.to_grid(end.y)
        
        # 2. A* Algorithm Setup
        open_set: List[Tuple[float, int, int]] = [] # Priority queue: (f_score, gx, gy)
        heapq.heappush(open_set, (0.0, start_gx, start_gy))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        g_score: Dict[Tuple[int, int], float] = {}
        g_score[(start_gx, start_gy)] = 0.0
        
        f_score: Dict[Tuple[int, int], float] = {}
        f_score[(start_gx, start_gy)] = self._heuristic(start_gx, start_gy, end_gx, end_gy)
        
        checked_nodes: Set[Tuple[int, int]] = set()

        # Check if start or end is blocked (allowing start in blocked for "escape" scenarios is complex, assuming free)
        # Assuming flight altitude of start.z for collision check
        flight_z = start.z 
        if grid_map.is_blocked(end_gx, end_gy, flight_z, optimistic=True):
             # Try to find a nearest free neighbor for end point could be an enhancement, but for now fail.
             # Or simply proceed and let it fail if it can't reach.
             pass

        iterations = 0
        max_iterations = 50000 # Safety break for open maps

        while open_set:
            iterations += 1
            if iterations > max_iterations:
                print("A* max iterations reached.")
                return None
                
            current_f, cx, cy = heapq.heappop(open_set)
            
            if (cx, cy) == (end_gx, end_gy):
                return self._reconstruct_path(came_from, (cx, cy), grid_map, start.z)
            
            checked_nodes.add((cx, cy))
            
            # 8-connected grid
            for dx, dy in [
                (0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]:
                nx, ny = cx + dx, cy + dy
                
                # Check collision
                if grid_map.is_blocked(nx, ny, flight_z, optimistic=True):
                    continue
                
                # Distance cost (1.0 for orthogonal, 1.414 for diagonal)
                dist_cost = math.sqrt(dx*dx + dy*dy)
                tentative_g_score = g_score[(cx, cy)] + dist_cost
                
                if (nx, ny) not in g_score or tentative_g_score < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[(nx, ny)] = tentative_g_score
                    f_score[(nx, ny)] = tentative_g_score + self._heuristic(nx, ny, end_gx, end_gy)
                    
                    # If not already in open set (naive check, better to just push as heap handles duplicates usually or check validity)
                    # For simplicity in python heapq, duplicates are fine, we just ignore processed nodes if popped again?
                    # But checked_nodes handles closed set.
                    # We need to distinguish if it's in open_set to update.
                    # Simpler: just push. If we pop a checked node, skip.
                    if (nx, ny) not in checked_nodes:
                        heapq.heappush(open_set, (f_score[(nx, ny)], nx, ny))
                        
        return None # No path found

    def _heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Euclidean distance heuristic."""
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int], grid_map: GridMap, z: float) -> List[Position]:
        path_grid = [current]
        while current in came_from:
            current = came_from[current]
            path_grid.append(current)
        path_grid.reverse()
        
        # Convert back to real coordinates (center of grid cell)
        path_real = []
        for gx, gy in path_grid:
            # Use to_real to get bottom-left, add half resolution for center
            rx = grid_map.to_real(gx) + grid_map.resolution / 2.0
            ry = grid_map.to_real(gy) + grid_map.resolution / 2.0
            path_real.append(Position(rx, ry, z))
            
        return path_real
