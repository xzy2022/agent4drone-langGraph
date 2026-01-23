from typing import List, Optional
import math
from ...entities.position import Position
from ...entities.map import GridMap
from ...entities.drone import Drone
from . import NavigationUseCase

class AdaptiveNavigation:
    """
    自适应特定导航逻辑：
    1. 检查无人机与目标之间的水平距离。
    2. 如果距离 < max_move_dist，返回完整的 A* 路径。
    3. 如果距离 > max_move_dist，返回截断路径（最大长度为 max_move_dist）。
    4. 路径必须仅包含拐点（角点）。
    """

    def __init__(self, navigation_use_case: Optional[NavigationUseCase] = None):
        self.nav_algo = navigation_use_case or NavigationUseCase()

    def navigate(self, drone: Drone, destination: Position, grid_map: GridMap, max_move_dist: float) -> Optional[List[Position]]:
        """
        计算自适应路径。
        """
        start_pos = drone.position
        
        # 计算水平距离
        h_dist = math.sqrt((start_pos.x - destination.x)**2 + (start_pos.y - destination.y)**2)
        
        # 确定搜索的实际目标
        # 即使稍后需要截断，理想情况下我们也希望有一条通往目的地的路径。
        # 因此，我们首先搜索完整路径，然后进行截断。
        # 注意：如果距离非常远，搜索完整路径可能会比较耗时，但对于此任务，我们假设它是可行的。
        # 或者，我们可以向目标投影一个距离为 max_move_dist 的点，但这可能会撞到障碍物
        # 如果我们以真实目标为目标，就可以避免这种情况。
        # 鉴于要求“如果无人机到终点的水平距离 > 最大移动距离，则返回截断路径”，
        # 这暗示我们确实在尝试前往终点，但受到航程限制。
        
        full_path = self.nav_algo.compute_path(start_pos, destination, grid_map)
        
        if not full_path:
            return None
            
        # 来自 NavigationUseCase 的路径是否包含起点？检查表明它返回 [start, ..., end] 或 [p1, ..., end]
        # 让我们检查 navigation.py。
        # 它调用 _reconstruct_path：path_grid = [current]，当 current 在 came_from 中时... 反转。
        # 所以它返回 [Start, ..., End]。
        

        # 将起点和终点对齐到精确位置（A* 返回网格中心）
        # 将起点和终点对齐到精确位置（A* 返回网格中心）
        full_path[0] = start_pos
        full_path[-1] = destination
        
        # 1. 首先简化路径（基于视线优化）
        # 这确保我们测量和截断的是实际飞行的“拉直”路径，而不是锯齿状网格路径。
        simplified_path = self._simplify_path(full_path, grid_map)
        
        # 2. 截断路径
        # _truncate_path 会处理“如果路径短于 max_dist 则保持原样”的逻辑，
        # 所以我们可以始终调用它，或者保留 h_dist 检查作为优化（但 h_dist 是直线距离，路径可能绕弯）。
        # 为了正确处理绕障路径长度，我们直接对简化后的路径进行截断检查。
        final_path = self._truncate_path(simplified_path, max_move_dist)
        
        return final_path

    def _truncate_path(self, path: List[Position], max_dist: float) -> List[Position]:
        """
        截断路径，使其总长度不超过 max_dist。
        实际上，要求是“如果无人机到终点的水平距离 > 最大移动距离，则返回截断路径”。
        这可能指的是单步执行能力。
        截断逻辑通常意味着“沿着路径行驶，直到达到 max_dist”。
        """
        if not path:
            return []
            
        truncated = [path[0]]
        current_dist = 0.0
        
        for i in range(1, len(path)):
            prev = path[i-1]
            curr = path[i]
            
            seg_dist = math.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
            
            if current_dist + seg_dist <= max_dist:
                truncated.append(curr)
                current_dist += seg_dist
            else:
                # 我们需要对最后一个点进行插值
                remaining = max_dist - current_dist
                ratio = remaining / seg_dist
                new_x = prev.x + (curr.x - prev.x) * ratio
                new_y = prev.y + (curr.y - prev.y) * ratio
                # 理论上 Z 轴也应该进行插值或保持不变
                new_z = prev.z # 通常假设 2D 导航的高度恒定
                truncated.append(Position(new_x, new_y, new_z))
                break
                
        return truncated

    def _simplify_path(self, path: List[Position], grid_map: GridMap) -> List[Position]:
        """
        简化路径：使用视线检查（Raycast Smoothing / String Pulling）来移除多余的中间点。
        这将把锯齿状的 A* 路径拉直为直线段。
        """
        if len(path) <= 2:
            return path
            
        points = path
        smoothed_path = [points[0]]
        current_idx = 0
        
        while current_idx < len(points) - 1:
            # 贪婪地寻找从 current_idx 可以直线到达的最远的一个点 next_idx
            next_idx = current_idx + 1
            for i in range(len(points) - 1, current_idx, -1):
                if self._is_line_clear(points[current_idx], points[i], grid_map):
                    next_idx = i
                    break
            
            smoothed_path.append(points[next_idx])
            current_idx = next_idx
            
        return smoothed_path

    def _is_line_clear(self, p1: Position, p2: Position, grid_map: GridMap) -> bool:
        """
        检查两点之间的直线段是否无碰撞。
        使用简单的采样检测。
        """
        dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        if dist < 1e-3:
            return True #Same point
            
        steps = int(dist / (grid_map.resolution * 0.5)) + 1
        
        # Check flight altitude (assuming constant z from start/p1)
        z = p1.z
        
        for i in range(steps + 1):
            t = i / steps
            x = p1.x + (p2.x - p1.x) * t
            y = p1.y + (p2.y - p1.y) * t
            
            gx, gy = grid_map.to_grid(x), grid_map.to_grid(y)
            if grid_map.is_blocked(gx, gy, z, optimistic=True):
                return False
                
        return True
