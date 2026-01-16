from typing import Dict, List, Optional, Any
import json
import os

class TargetManager:
    """
    Manages known targets and their status (reached/unreached).
    """
    def __init__(self):
        # target_id -> target_data_dict
        self.targets: Dict[str, Dict[str, Any]] = {}

    def update_from_perception(self, entities: Dict[str, Any]):
        """
        Update knowledge base from perception results.
        Expected input is the dictionary returned by get_nearby_entities,
        which contains a 'targets' list.
        """
        targets_list = entities.get('targets', [])
        for t in targets_list:
            t_id = t.get('id')
            if not t_id:
                continue
            
            # If it's a new target, just add it
            if t_id not in self.targets:
                self.targets[t_id] = t
            else:
                # If it exists, update dynamic fields
                # We merge fields, prioritizing the new perception data
                existing = self.targets[t_id]
                updated = existing.copy()
                updated.update(t)
                
                # Special handling: if we locally marked it as reached, maybe we want to keep it?
                # But usually simulator is ground truth. 
                # For now, simplistic update is fine.
                self.targets[t_id] = updated

    def get_known_targets(self) -> List[Dict[str, Any]]:
        """Return list of all known targets."""
        return list(self.targets.values())
    
    def get_target(self, target_id: str) -> Optional[Dict[str, Any]]:
        return self.targets.get(target_id)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "targets": self.targets
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TargetManager':
        """Deserialize from dictionary."""
        instance = cls()
        instance.targets = data.get("targets", {})
        return instance

    def save_to_disk(self, path: str):
        """Save to disk (useful for session persistence)."""
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_disk(cls, path: str) -> Optional['TargetManager']:
        """Load from disk."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return cls.from_dict(data)
        except Exception as e:
            print(f"[TargetManager] Failed to load from {path}: {e}")
            return None
