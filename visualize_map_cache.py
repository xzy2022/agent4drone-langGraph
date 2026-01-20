import json
import matplotlib.pyplot as plt
import os

SESSION_ID = "11429499"
MAP_CACHE_DIR = ".map_cache"

def visualize_map_and_targets(session_id):
    map_file = os.path.join(MAP_CACHE_DIR, f"map_{session_id}.json")
    targets_file = os.path.join(MAP_CACHE_DIR, f"targets_{session_id}.json")

    if not os.path.exists(map_file):
        print(f"Map file not found: {map_file}")
        return
    if not os.path.exists(targets_file):
        print(f"Targets file not found: {targets_file}")
        return

    # Load Map
    with open(map_file, 'r') as f:
        map_data = json.load(f)

    resolution = map_data.get("resolution", 1.0)
    obstacles = map_data.get("obstacles", {})

    obs_x = []
    obs_y = []

    print(f"Loading {len(obstacles)} obstacle points...")
    for key in obstacles:
        try:
            gx, gy = map(int, key.split(','))
            # Convert grid coords to world coords (center of grid)
            wx = gx * resolution + resolution / 2.0
            wy = gy * resolution + resolution / 2.0
            obs_x.append(wx)
            obs_y.append(wy)
        except ValueError:
            continue

    # Load Targets
    with open(targets_file, 'r') as f:
        targets_data = json.load(f)
    
    targets = targets_data.get("targets", {})
    
    print(f"Loading {len(targets)} targets...")
    
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot Obstacles
    ax.scatter(obs_x, obs_y, c='black', marker='s', s=10, label='Obstacles', alpha=0.6)

    # Plot Targets
    for t_id, t_info in targets.items():
        pos = t_info['position']
        radius = t_info.get('radius', 5.0)
        is_reached = t_info.get('is_reached', False)
        
        color = 'green' if is_reached else 'red'
        name = t_info.get('name', t_id)
        
        circle = plt.Circle((pos['x'], pos['y']), radius, color=color, alpha=0.5, label='_nolegend_')
        ax.add_patch(circle)
        ax.text(pos['x'], pos['y'], name, fontsize=8, ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Map and Targets - Session {session_id}')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Create a custom legend for targets
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Obstacle', markerfacecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Target (Unreached)', markerfacecolor='red', markersize=10, alpha=0.5),
        Line2D([0], [0], marker='o', color='w', label='Target (Reached)', markerfacecolor='green', markersize=10, alpha=0.5)
    ]
    ax.legend(handles=legend_elements)

    output_file = f"visualization_{session_id}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    visualize_map_and_targets(SESSION_ID)
