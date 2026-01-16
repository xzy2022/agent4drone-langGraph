import os
import sys
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.navigation.grid_map import GridMapManager

def test_persistence():
    print("Testing GridMapManager persistence...")
    
    # 1. Create a map and add some data
    gm = GridMapManager(resolution=2.0, inflation=0)
    gm.obstacles[(10, 20)] = 5.5
    gm.obstacles[(30, 40)] = 0.0  # Infinite
    gm.explored.add((1, 2))
    gm.explored.add((3, 4))
    
    # 2. Save to disk
    test_path = "test_map.json"
    gm.save_to_disk(test_path)
    print(f"Map saved to {test_path}")
    
    # 3. Load from disk
    loaded_gm = GridMapManager.load_from_disk(test_path)
    print("Map loaded from disk")
    
    # 4. Verify data
    assert loaded_gm is not None
    assert loaded_gm.resolution == 2.0
    assert loaded_gm.obstacles[(10, 20)] == 5.5
    assert loaded_gm.obstacles[(30, 40)] == 0.0
    assert (1, 2) in loaded_gm.explored
    assert (3, 4) in loaded_gm.explored
    assert len(loaded_gm.obstacles) == 2
    assert len(loaded_gm.explored) == 2
    
    print("Verification successful!")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)

if __name__ == "__main__":
    try:
        test_persistence()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
