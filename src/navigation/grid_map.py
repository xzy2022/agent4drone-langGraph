from typing import List, Tuple, Optional, Dict
import math
import heapq

class GridMapManager:
    """
    支持多边形和椭圆解析的带高度栅格地图
    """
    def __init__(self, resolution: float = 5.0, inflation: int = 1):
        self.resolution = resolution
        # (gx, gy) -> max_height (-1:空, 0:无限高, >0:有限高)
        self.grid: Dict[Tuple[int, int], float] = {}
        self.inflation = inflation
        self.motions = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)]

    def _to_grid(self, pos: float) -> int:
        return int(round(pos / self.resolution))

    def _to_real(self, idx: int) -> float:
        return float(idx) * self.resolution

    def _update_grid_cell(self, gx: int, gy: int, height: float):
        """核心更新逻辑：无限高(0) 覆盖一切"""
        node = (gx, gy)
        current_val = self.grid.get(node, -1.0)
        
        if current_val == 0.0: return # 已经是无限高，跳过
        if height == 0.0:
            self.grid[node] = 0.0 # 设为无限高
            return
            
        if height > 0:
            if current_val == -1.0:
                self.grid[node] = height
            else:
                self.grid[node] = max(current_val, height)

    def _inflate_point(self, gx: int, gy: int, height: float):
        """对单个点进行膨胀处理"""
        for dx in range(-self.inflation, self.inflation + 1):
            for dy in range(-self.inflation, self.inflation + 1):
                self._update_grid_cell(gx + dx, gy + dy, height)

    def _is_point_in_polygon(self, x: float, y: float, vertices: List[dict]) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(vertices)
        if n < 3: return False
        inside = False
        p1x, p1y = vertices[0]['x'], vertices[0]['y']
        for i in range(n + 1):
            p2x, p2y = vertices[i % n]['x'], vertices[i % n]['y']
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def add_obstacles_from_entities(self, entities: dict):
        """
        全能型障碍物加载器：支持 Polygon, Ellipse, Cylinder
        """
        obs_list = entities.get('obstacles', [])
        
        for obs in obs_list:
            
            height = obs.get('height', 0.0)
            obs_type = obs.get('type', 'unknown')
            pos = obs.get('position', obs)
            cx, cy = pos.get('x', 0), pos.get('y', 0)
            print("感知到障碍物：", obs_type)
            # --- 情况 1: 多边形 (Polygon) ---
            if obs_type == 'polygon' and obs.get('vertices'):
                verts = obs['vertices']
                # 1. 计算包围盒 (Bounding Box)
                min_x = min(v['x'] for v in verts)
                max_x = max(v['x'] for v in verts)
                min_y = min(v['y'] for v in verts)
                max_y = max(v['y'] for v in verts)
                
                # 2. 遍历包围盒内的所有栅格
                start_gx, end_gx = self._to_grid(min_x), self._to_grid(max_x)
                start_gy, end_gy = self._to_grid(min_y), self._to_grid(max_y)
                
                for gx in range(start_gx, end_gx + 1):
                    for gy in range(start_gy, end_gy + 1):
                        # 检查栅格中心点是否在多边形内
                        rx, ry = self._to_real(gx), self._to_real(gy)
                        if self._is_point_in_polygon(rx, ry, verts):
                            self._inflate_point(gx, gy, height)

            # --- 情况 2: 椭圆 (Ellipse) ---
            elif obs_type == 'ellipse' and obs.get('width') and obs.get('length'):
                # width 是 x轴方向全长，length 是 y轴方向全长
                a = obs['width'] / 2.0  # 半长轴 x
                b = obs['length'] / 2.0 # 半短轴 y
                
                # 1. 计算包围盒
                start_gx = self._to_grid(cx - a)
                end_gx = self._to_grid(cx + a)
                start_gy = self._to_grid(cy - b)
                end_gy = self._to_grid(cy + b)
                
                for gx in range(start_gx, end_gx + 1):
                    for gy in range(start_gy, end_gy + 1):
                        rx, ry = self._to_real(gx), self._to_real(gy)
                        # 椭圆方程: (x-cx)^2/a^2 + (y-cy)^2/b^2 <= 1
                        if ((rx - cx)**2 / (a**2)) + ((ry - cy)**2 / (b**2)) <= 1.0:
                            self._inflate_point(gx, gy, height)

            # --- 情况 3: 点/圆柱/圆 (Point/Cylinder/Circle) ---
            elif obs_type == 'point' or obs_type == 'cylinder' or obs_type == 'circle':
                radius = obs.get('radius') or 0.0
                radius_grid = int(radius / self.resolution)
                
                cgx, cgy = self._to_grid(cx), self._to_grid(cy)
                total_range = radius_grid 
                
                for dx in range(-total_range, total_range + 1):
                    for dy in range(-total_range, total_range + 1):
                        # 简单的圆形判定
                        if dx*dx + dy*dy <= total_range*total_range + 0.5:
                             self._inflate_point(cgx + dx, cgy + dy, height)
                
                # 确保中心点一定被添加
                if radius_grid == 0:
                    self._inflate_point(cgx, cgy, height)

            # --- 情况 4: 未知类型 (Fallback) ---
            else:

                print("[GridMap] 警告：未知障碍物类型")
                print(obs)
                radius = obs.get('radius') or 0.0
                # 如果 radius 为 0 或 None，至少占用 1 格
                radius_grid = int(radius / self.resolution)
                
                cgx, cgy = self._to_grid(cx), self._to_grid(cy)
                total_range = radius_grid # 这里不在循环里膨胀，而是把范围算大一点
                
                # 遍历圆形区域
                for dx in range(-total_range, total_range + 1):
                    for dy in range(-total_range, total_range + 1):
                        # 简单的圆形判定
                        if dx*dx + dy*dy <= total_range*total_range + 0.5:
                             self._inflate_point(cgx + dx, cgy + dy, height)
                
                # 确保中心点一定被添加
                if radius_grid == 0:
                    self._inflate_point(cgx, cgy, height)

    def is_blocked(self, gx: int, gy: int, nav_z: float) -> bool:
        """检查点是否被遮挡 (高度冲突)"""
        height = self.grid.get((gx, gy), -1.0)
        if height == -1.0: return False  # 无障碍物
        if height == 0.0: return True   # 无限高
        return nav_z <= height          # 飞行高度低于或等于障碍物高度

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(self, came_from, current, nav_z):
        path = []
        while current in came_from:
            rx = self._to_real(current[0])
            ry = self._to_real(current[1])
            path.append({"x": rx, "y": ry, "z": nav_z})
            current = came_from[current]
        path.reverse()
        return path

    def a_star_search(self, start_pos: dict, target_pos: dict) -> Optional[List[dict]]:
        start_node = (self._to_grid(start_pos['x']), self._to_grid(start_pos['y']))
        end_node = (self._to_grid(target_pos['x']), self._to_grid(target_pos['y']))
        
        # 锁定飞行平面的高度
        nav_z = start_pos.get('z', 0)

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, end_node)}

        max_nodes = 2000
        nodes_explored = 0

        while open_set and nodes_explored < max_nodes:
            nodes_explored += 1
            current = heapq.heappop(open_set)[1]

            if current == end_node:
                return self._reconstruct_path(came_from, current, nav_z)

            for dx, dy, cost in self.motions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 只要 neighbor 不是起点，就检查碰撞
                if neighbor != start_node: 
                    if self.is_blocked(neighbor[0], neighbor[1], nav_z):
                        continue

                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end_node)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
        
        return None
