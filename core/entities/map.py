from typing import Dict, Tuple, Set, Optional, List
import math
from .obstacle import Obstacle
from .position import Position

class GridMap:
    """
    网格地图实体。
    用于世界网格表示的纯逻辑实现。
    """
    def __init__(self, resolution: float = 5.0, inflation: int = 1):
        self.resolution = resolution # 网格分辨率，单位米
        self.inflation = inflation # 膨胀因子，单位网格数
        # (gx, gy) -> 最大高度 (0.0 表示无限高, >0 表示有限高度)
        self.obstacles: Dict[Tuple[int, int], float] = {}
        # (gx, gy) -> 集合, 已探索的空闲空间
        self.explored_free_space: Set[Tuple[int, int]] = set()

    def to_grid(self, val: float) -> int:
        """将实际坐标转换为网格坐标"""
        return int(math.floor(val / self.resolution))

    def to_real(self, idx: int) -> float:
        """将网格坐标转换为实际坐标"""
        return float(idx) * self.resolution

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """获取地图当前的物理边界 [min_x, max_x, min_y, max_y]"""
        if not self.obstacles and not self.explored_free_space:
            return 0.0, 0.0, 0.0, 0.0
        
        all_nodes = list(self.obstacles.keys()) + list(self.explored_free_space)
        min_gx = min(n[0] for n in all_nodes)
        max_gx = max(n[0] for n in all_nodes)
        min_gy = min(n[1] for n in all_nodes)
        max_gy = max(n[1] for n in all_nodes)
        
        # 转换为物理边界
        return (self.to_real(min_gx), self.to_real(max_gx + 1),
                self.to_real(min_gy), self.to_real(max_gy + 1))

    def get_status(self, gx: int, gy: int) -> float:
        """
        获取网格单元状态。
        返回值:
            障碍物高度 (>0) 或 0.0 (无限高)。
            -1.0 表示已探索的空闲空间。
            -2.0 表示未知区域。
        """
        if (gx, gy) in self.obstacles:
            return self.obstacles[(gx, gy)]
        if (gx, gy) in self.explored_free_space:
            return -1.0
        return -2.0

    def is_blocked(self, gx: int, gy: int, z: float, optimistic: bool = False) -> bool:
        """
        检查无人机在高度 z 时，该网格单元是否被阻挡。
        """
        status = self.get_status(gx, gy)
        
        if status == -1.0: return False  # 空闲空间
        if status == -2.0: 
            return not optimistic        # 未知区域：悲观模式下视为阻挡，乐观模式下视为通行
            
        if status == 0.0: return True    # 无限高障碍物
        return z <= status               # 如果飞行高度低于或等于障碍物高度，则视为阻挡

    def _update_grid_cell(self, gx: int, gy: int, height: float):
        """更新单个网格单元的障碍物信息。"""
        node = (gx, gy)
        
        # 1. 维护探索状态：只要是障碍物，就一定不是空闲区域
        self.explored_free_space.discard(node) # discard 不会因为 node 不存在而报错，比 remove 更安全
                
        # 获取当前高度，如果没有记录则视为 -1.0 没有障碍物
        current_val = self.obstacles.get(node, -1.0)
        
        # 2. 处理无限高逻辑 (0.0 是“无限高”的特殊标志)
        if current_val == 0.0:
            return  # 已经是无限高，无需任何更新
        
        if height == 0.0:
            self.obstacles[node] = 0.0  # 新增无限高，直接覆盖
            return

        # 3. 处理有限高度逻辑
        # 如果新障碍物更高，则更新高度
        if height > current_val:
            self.obstacles[node] = height

    def _inflate_point(self, gx: int, gy: int, height: float):
        """对单个障碍物点进行膨胀处理。"""
        for dx in range(-self.inflation, self.inflation + 1):
            for dy in range(-self.inflation, self.inflation + 1):
                self._update_grid_cell(gx + dx, gy + dy, height)

    def add_obstacle(self, obstacle: Obstacle):
        """
        向网格地图添加障碍物实体。
        处理几何障碍物到网格的栅格化。
        """
        height = obstacle.height
        
        # 需要进行类型检查或使用多态。
        # 由于 Obstacle 是抽象基类，我们可以检查类型或添加 'rasterize' 方法。
        # 但 'rasterize' 依赖于地图分辨率，因此该逻辑目前保留在 Map 实体中。
        
        # 实际上，为了避免循环依赖：Obstacle 不应了解地图分辨率。
        # 地图应该知道如何栅格化障碍物。
        
        from .obstacle import PolygonObstacle, CircleObstacle
        
        print(f"DEBUG: Adding obstacle type {type(obstacle)}")
        if isinstance(obstacle, PolygonObstacle):
            self._add_polygon_obstacle(obstacle)
        elif isinstance(obstacle, CircleObstacle):
            self._add_circle_obstacle(obstacle)
            
    def _add_polygon_obstacle(self, obs: 'PolygonObstacle'):
        """添加多边形障碍物。"""
        verts = obs.vertices
        if not verts: return
        
        min_x = min(v.x for v in verts)
        max_x = max(v.x for v in verts)
        min_y = min(v.y for v in verts)
        max_y = max(v.y for v in verts)
        
        start_gx, end_gx = self.to_grid(min_x), self.to_grid(max_x)
        start_gy, end_gy = self.to_grid(min_y), self.to_grid(max_y)
        
        print(f"DEBUG: Rasterizing poly. BoundX: {start_gx}->{end_gx}, BoundY: {start_gy}->{end_gy}")
        
        count = 0
        for gx in range(start_gx, end_gx + 1):
            for gy in range(start_gy, end_gy + 1):
                rx, ry = self.to_real(gx), self.to_real(gy)
                # print(f"Checking point ({rx}, {ry}) for gx={gx}, gy={gy}")
                if obs.contains(rx, ry):
                    self._inflate_point(gx, gy, obs.height)
                    count += 1
        print(f"DEBUG: Added {count} grid cells from polygon.")

    def _add_circle_obstacle(self, obs: 'CircleObstacle'):
        """添加圆形障碍物。"""
        cx, cy = obs.center.x, obs.center.y
        radius = obs.radius
        
        radius_grid = int(radius / self.resolution)
        cgx, cgy = self.to_grid(cx), self.to_grid(cy)
        
        # 圆形的简单栅格化实现
        for dx in range(-radius_grid, radius_grid + 1):
            for dy in range(-radius_grid, radius_grid + 1):
                if dx*dx + dy*dy <= radius_grid**2 + 0.5: # +0.5 用于容差/舍入
                     self._inflate_point(cgx + dx, cgy + dy, obs.height)

        # 确保中心点始终被添加
        if radius_grid == 0:
            self._inflate_point(cgx, cgy, obs.height)

    def mark_explored(self, cx: float, cy: float, radius: float):
        """将 (cx, cy) 周围区域标记为已探索。"""
        cgx, cgy = self.to_grid(cx), self.to_grid(cy)
        rg = int(math.ceil(radius / self.resolution))
        
        for dx in range(-rg, rg + 1):
            for dy in range(-rg, rg + 1):
                if dx*dx + dy*dy <= (radius / self.resolution)**2 + 0.5:
                    gx, gy = cgx + dx, cgy + dy
                    if (gx, gy) not in self.obstacles:
                        self.explored_free_space.add((gx, gy))
