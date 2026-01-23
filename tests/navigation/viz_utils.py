
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional
import math
import sys
import os

# Add project root to path to import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.entities.map import GridMap
from core.entities.position import Position

def visualize_navigation_result(
    grid_map: GridMap,
    start_pos: Position,
    end_pos: Position,
    path: Optional[List[Position]],
    max_move_dist: float,
    title: str = "Navigation Result"
):
    """
    Visualizes the navigation result including map, start/end, range, and path.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Draw GridMap (Obstacles)
    # We iterate through the obstacles dictionary
    if grid_map.obstacles:
        for (gx, gy), height in grid_map.obstacles.items():
            if height > 0 or height == 0.0: # 0.0 is infinite height
                # Convert grid coords to real coords for plotting
                # Note: GridMap (gx, gy) corresponds to the bottom-left corner of the grid cell
                rx = grid_map.to_real(gx)
                ry = grid_map.to_real(gy)
                resolution = grid_map.resolution
                
                rect = patches.Rectangle(
                    (rx, ry), resolution, resolution, 
                    linewidth=0, edgecolor='none', facecolor='black', alpha=0.6
                )
                ax.add_patch(rect)

    # 2. Draw Range Circle
    circle = plt.Circle((start_pos.x, start_pos.y), max_move_dist, 
                        color='gray', fill=False, linestyle='--', linewidth=1.5, label='Max Range')
    ax.add_patch(circle)

    # 3. Draw Start and Original Target
    ax.plot(start_pos.x, start_pos.y, 'go', markersize=10, label='Start') # Green Circle
    ax.plot(end_pos.x, end_pos.y, 'rx', markersize=10, markeredgewidth=2, label='Original Target') # Red Cross

    # 4. Draw Path
    path_len = 0.0
    if path and len(path) > 0:
        xs = [p.x for p in path]
        ys = [p.y for p in path]
        
        # Draw lines
        ax.plot(xs, ys, 'b-', linewidth=2.5, label='Actual Path')
        
        # Draw waypoints (Blue Dots) to verify simplification
        ax.plot(xs, ys, 'bo', markersize=5)
        
        # Calculate path length
        for i in range(1, len(path)):
            dx = path[i].x - path[i-1].x
            dy = path[i].y - path[i-1].y
            path_len += math.sqrt(dx*dx + dy*dy)
    else:
        # If no path found/returned
        pass

    # 5. Settings
    ax.set_aspect('equal')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Set plot limits to include everything with some padding
    # Basic limits based on start/end
    all_x = [start_pos.x, end_pos.x]
    all_y = [start_pos.y, end_pos.y]
    if path:
        all_x.extend([p.x for p in path])
        all_y.extend([p.y for p in path])
        
    # Include circle bounds roughly
    all_x.extend([start_pos.x - max_move_dist, start_pos.x + max_move_dist])
    all_y.extend([start_pos.y - max_move_dist, start_pos.y + max_move_dist])
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    padding = 2.0
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    
    # 6. info text
    info_text = f"Max Dist: {max_move_dist:.2f}m\nPath Len: {path_len:.2f}m"
    plt.text(0.02, 0.98, info_text, transform=ax.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(title)
    ax.legend(loc='lower right')
    
    print(f"DEBUG: Showing plot for {title}...")
    plt.show()

