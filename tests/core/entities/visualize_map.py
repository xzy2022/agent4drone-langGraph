import matplotlib.pyplot as plt
import numpy as np
import math
from core.entities.map import GridMap
from core.entities.position import Position
from core.entities.obstacle import PolygonObstacle, CircleObstacle, EllipseObstacle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def visualize_map(grid_map: GridMap):
    # 1. 获取物理边界
    min_x, max_x, min_y, max_y = grid_map.get_bounds()
    print(min_x, max_x, min_y, max_y)
    
    if min_x == max_x and min_y == max_y:
        print("地图为空，无数据可显示。")
        return

    # 2. 计算栅格范围以构建 numpy 矩阵
    # 利用 to_grid 确保索引对齐
    min_gx = grid_map.to_grid(min_x)
    max_gx = grid_map.to_grid(max_x) 
    min_gy = grid_map.to_grid(min_y)
    max_gy = grid_map.to_grid(max_y)

    width = max_gx - min_gx + 1
    height = max_gy - min_gy + 1
    
    # 初始化矩阵：-2.0 表示未知 (Unknown)
    grid_data = np.full((height, width), -2.0)

    # 3. 填充数据
    # 填充已探索空闲区域
    for gx, gy in grid_map.explored_free_space:
        grid_data[gy - min_gy, gx - min_gx] = -1.0
        
    # 填充障碍物
    for (gx, gy), height_val in grid_map.obstacles.items():
        # 这里可以直接存高度，或者存一个固定值表示障碍物
        # 为了方便显示状态，我们存 1.0 表示有障碍
        grid_data[gy - min_gy, gx - min_gx] = 1.0

    # 4. 绘图配置
    plt.figure(figsize=(10, 8))
    
    # 创建离散颜色表：未知=灰色，空闲=白色，障碍物=黑色
    # 映射顺序：-2.0 (未知), -1.0 (空闲), 1.0 (障碍)
    # 注意：imshow 会根据数据范围自动映射颜色，这里简单处理：
    cmap = ListedColormap(['#7f8c8d', '#ecf0f1', '#2c3e50']) 
    
    # 关键点：使用 get_bounds 返回的物理坐标作为 extent
    # extent = [left, right, bottom, top]
    extent = [min_x, max_x, min_y, max_y]
    
    img = plt.imshow(
        grid_data, 
        origin='lower', 
        extent=extent, 
        cmap=cmap,
        interpolation='nearest' # 必须用 nearest 保证栅格边界清晰
    )

    # 5. 利用 resolution 绘制辅助网格线
    ax = plt.gca()
    res = grid_map.resolution
    
    # 网格线应该对齐到每一个栅格的物理边界
    ax.set_xticks(np.arange(min_x, max_x + res, res), minor=True)
    ax.set_yticks(np.arange(min_y, max_y + res, res), minor=True)
    plt.grid(True, which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    # 6. 装饰
    plt.title(f"GridMap Visualization (Res: {res}m)")
    plt.xlabel("X [meters]")
    plt.ylabel("Y [meters]")
    
    # 颜色条图例
    cbar = plt.colorbar(img, ticks=[-1.5, -0.5, 0.5])
    cbar.ax.set_yticklabels(['Unknown', 'Free', 'Obstacle'])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. 初始化地图 (分辨率 0.5m)
    grid_map = GridMap(resolution=1.0, inflation=0)
    
    # 2. 添加一些障碍物用于测试
    # 圆形障碍物
    grid_map.add_obstacle(CircleObstacle(
        height=10.0, 
        center=Position(5, 5, 0), 
        radius=2.0
    ))
    
    # 多边形障碍物 (三角形)
    grid_map.add_obstacle(PolygonObstacle(
        height=20.0,
        vertices=[
            Position(12, 12, 0),
            Position(18, 12, 0),
            Position(15, 18, 0)
        ]
    ))
    
    # 3. 模拟探索（标记一部分为已知空闲区域）
    # 在 (10, 5) 附近探索半径为 4 的区域
    grid_map.mark_explored(10, 5, radius=4.0)
    
    # 4. 调用可视化函数，传入刚才创建的地图对象
    visualize_map(grid_map)
