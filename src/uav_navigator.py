from typing import Dict, Any, List
import math
import networkx as nx
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union

from src.uav_api_client import UAVAPIClient

class UAVNavigator:
    """
    高级导航助手 v3.0 (可见性图导航)
    1. 基于 Shapely 的几何占据与膨胀。
    2. 基于 NetworkX Dijkstra 的最短路径搜索。
    3. 支持无限高与动态重规划。
    """
    def __init__(self, client: UAVAPIClient, inflation: float = 5.0):
        self.client = client
        self.inflation = inflation
        self.known_obstacles = [] # 存储字典: {'poly': Polygon, 'id': str, 'height': float, 'is_infinite': bool}

    def update_map(self, nearby_entities: Dict[str, Any]):
        """解析并聚合已知障碍物"""
        # 只处理真正的障碍物列表
        new_obs_list = nearby_entities.get('obstacles', [])
        
        for obs in new_obs_list:
            if any(o['id'] == obs.get('id') for o in self.known_obstacles):
                continue

            p = obs.get('position', {'x': 0, 'y': 0, 'z': 0})
            h = obs.get('height', 0.0)
            obs_type = obs.get('type', 'box')
            
            poly = None
            if obs_type == 'polygon':
                v = obs.get('vertices', [])
                if v:
                    poly = Polygon([(pt['x'], pt['y']) for pt in v])
            elif obs_type == 'ellipse':
                # 用包围盒或安全外接圆简化椭圆
                w, l = obs.get('width', 10.0), obs.get('length', 10.0)
                # 降低分段数以加速计算
                poly = Point(p['x'], p['y']).buffer(max(w, l) / 2, quad_segs=4)
            else: # Box / Default
                s = obs.get('size', {'x': 10.0, 'y': 10.0})
                min_x, max_x = p['x'] - s['x']/2, p['x'] + s['x']/2
                min_y, max_y = p['y'] - s['y']/2, p['y'] + s['y']/2
                poly = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
            
            if poly:
                self.known_obstacles.append({
                    'id': obs.get('id'),
                    'poly': poly,
                    'height': h,
                    'is_infinite': (h == 0.0),
                    'base_z': p['z']
                })

    def is_point_safe(self, x: float, y: float, z: float) -> bool:
        """检查点是否在任何障碍物的占据范围内"""
        pt = Point(x, y)
        for o in self.known_obstacles:
            # 如果不是无限高且当前高度高于障碍物，视为安全
            if not o['is_infinite'] and z > (o['base_z'] + o['height'] + self.inflation):
                continue
            # 检查 2D 占据 (带分段优化)
            if o['poly'].buffer(self.inflation, quad_segs=4).contains(pt):
                return False
        return True

    def find_path(self, start: Dict[str, float], end: Dict[str, float]) -> List[Dict[str, float]]:
        """构建可见性图并执行最短路径搜索"""
        start_pt = (start['x'], start['y'])
        end_pt = (end['x'], end['y'])
        curr_z = start['z']

        # 1. 过滤当前高度受影响的障碍物并膨胀 (限制精细度以加速)
        relevant_polys = []
        for o in self.known_obstacles:
            if o['is_infinite'] or curr_z <= (o['base_z'] + o['height'] + self.inflation):
                relevant_polys.append(o['poly'].buffer(self.inflation, quad_segs=2))
        
        if not relevant_polys:
            return [end]

        unified_obstacles = unary_union(relevant_polys)

        # 2. 直线连接检查 (优化)
        line = LineString([start_pt, end_pt])
        if not line.intersects(unified_obstacles) or line.touches(unified_obstacles):
            return [end]

        # 3. 收集所有关键节点（起点、终点、障碍物顶点）
        nodes = [start_pt, end_pt]
        if unified_obstacles.geom_type == 'Polygon':
            nodes.extend(list(unified_obstacles.exterior.coords))
        elif unified_obstacles.geom_type == 'MultiPolygon':
            for poly in unified_obstacles.geoms:
                nodes.extend(list(poly.exterior.coords))
        
        # 去重
        nodes = list(set(tuple(n) for n in nodes))

        # 4. 构建可见性图 (Edges)
        G = nx.Graph()
        for i, p1 in enumerate(nodes):
            for j, p2 in enumerate(nodes):
                if i >= j: continue
                l = LineString([p1, p2])
                # 如果连线不穿过障碍区域（允许相切）
                if not l.intersects(unified_obstacles) or l.touches(unified_obstacles):
                    dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                    G.add_edge(p1, p2, weight=dist)

        # 5. Dijkstra 寻路
        try:
            path_coords = nx.shortest_path(G, source=start_pt, target=end_pt, weight='weight')
            # 转换回航点列表 (忽略起点)
            # The path_coords includes start_pt and end_pt. We want intermediate waypoints.
            # If path_coords has only 2 points (start and end), it means direct path.
            if len(path_coords) <= 2:
                return [end]
            return [{"x": p[0], "y": p[1], "z": curr_z} for p in path_coords[1:-1]] + [end]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_bypass_point(self, current: Dict[str, float], target: Dict[str, float]) -> Dict[str, float]:
        """自适应绕行/逃离策略：根据障碍物类型（无限vs有限）决定绕行方案"""
        pt = Point(current['x'], current['y'])
        m = self.inflation
        
        # 寻找当前包围（或即将碰撞）的障碍物
        blocking_obs = [o for o in self.known_obstacles if o['poly'].buffer(m).contains(pt)]
        
        if not blocking_obs:
            # 如果没检测到直接包围，可能只是路径前方被挡
            print("[Navigator] 未发现直接阻挡物，尝试常规小幅爬升以获取视场...")
            return {"x": current['x'], "y": current['y'], "z": current['z'] + 10.0}
        
        # 检查是否有无限高墙 (height=0)
        infinite_obs = [o for o in blocking_obs if o['is_infinite']]
        if infinite_obs:
            print(f"[Navigator] 警告: 检测到被 {len(infinite_obs)} 个无限高障碍物困住，执行水平逃逸...")
            # 水平偏移：向远离质心的方向或简单的斜向位移
            return {
                "x": current['x'] + 30.0, 
                "y": current['y'] + 30.0,
                "z": current['z']
            }
        
        # 全是有限高：寻找最高的一个
        max_h = max(o['base_z'] + o['height'] for o in blocking_obs)
        target_z = max_h + m
        
        # 如果当前已经高于最高点，但还卡着（可能是浮点误差或膨胀问题），继续小幅上升
        if current['z'] >= target_z:
            target_z = current['z'] + 5.0

        print(f"[Navigator] 检测到有限高障碍物 (最高 {max_h:.1f}m)，计划精准爬升至 {target_z:.1f}m")
        return {
            "x": current['x'],
            "y": current['y'],
            "z": target_z
        }
