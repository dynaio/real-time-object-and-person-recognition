import time
import json
import os
import numpy as np
import config as config
from facenet import cosine_similarity

# ================== Gallery persistence ==================
GALLERY_FILE = os.path.join(os.path.dirname(__file__), "gallery.json")

def load_gallery():
    if os.path.exists(GALLERY_FILE):
        with open(GALLERY_FILE, 'r') as f:
            data = json.load(f)
        gallery = {}
        for pid, info in data.items():
            gallery[int(pid)] = {
                "embedding": np.array(info["embedding"], dtype=np.float32),
                "first_seen": info.get("first_seen"),
                "registered_time": info.get("registered_time")
            }
        return gallery
    return {}

def save_gallery(gallery):
    data = {}
    for pid, info in gallery.items():
        data[str(pid)] = {
            "embedding": info["embedding"].tolist(),
            "first_seen": info["first_seen"],
            "registered_time": info["registered_time"]
        }
    with open(GALLERY_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# ================== Person name manager ==================
class PersonNameManager:
    def __init__(self, path=config.PERSON_NAMES_FILE):
        self.path = path
        self.names = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.names, f, indent=2)

    def get_name(self, person_id):
        return self.names.get(str(person_id), f"Person_{person_id}")

    def set_name(self, person_id, name):
        self.names[str(person_id)] = name
        self.save()

    def is_custom(self, person_id):
        current = self.get_name(person_id)
        default = f"Person_{person_id}"
        return current != default

    def add_new(self, person_id):
        name = f"Person_{person_id}"
        self.names[str(person_id)] = name
        self.save()
        return name

    def delete(self, person_id):
        """Remove a person's name (used when deleting from gallery)."""
        key = str(person_id)
        if key in self.names:
            del self.names[key]
            self.save()

# ================== Person Manager ==================
class PersonManager:
    def __init__(self):
        self.gallery = load_gallery()   # persistent embeddings + timestamps
        self.track_data = {}            # track_id -> {first_seen, embedding, person_id, ...}
        self.next_id = self._find_next_id()
        self.name_manager = PersonNameManager()

    def _find_next_id(self):
        if not self.gallery:
            return 0
        return max(self.gallery.keys()) + 1

    def _find_match(self, embedding):
        best_id = -1
        best_sim = 0.0
        for pid, info in self.gallery.items():
            sim = cosine_similarity(embedding, info["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_id = pid
        return best_id, best_sim

    def update_track(self, track_id, embedding, current_time):
        """Called when a face embedding is obtained for a person track."""
        if track_id not in self.track_data:
            self.track_data[track_id] = {
                "first_seen": current_time,
                "embedding": None,
                "person_id": None,       # None = unknown
            }
        td = self.track_data[track_id]

        if td["person_id"] is None:      # still unknown
            matched_id, sim = self._find_match(embedding)
            if sim > 0.6:
                # Recognised an existing person
                td["person_id"] = matched_id
                self.gallery[matched_id]["embedding"] = embedding
                # Update the gallery timestamp if desired, but keep original first_seen and registered_time
            else:
                td["embedding"] = embedding

        # Auto‑register after 60s if still unknown and we have an embedding
        if td["person_id"] is None and (current_time - td["first_seen"]) > 60:
            if td["embedding"] is not None:
                person_id = self.next_id
                self.next_id += 1
                self.gallery[person_id] = {
                    "embedding": td["embedding"],
                    "first_seen": td["first_seen"],
                    "registered_time": current_time
                }
                self.name_manager.add_new(person_id)
                td["person_id"] = person_id
                save_gallery(self.gallery)

    def manual_register(self, track_id, name):
        """Immediately register a person track with a custom name."""
        if track_id not in self.track_data:
            return False
        td = self.track_data[track_id]
        if td["embedding"] is None:
            return False
        if td["person_id"] is None:
            person_id = self.next_id
            self.next_id += 1
            self.gallery[person_id] = {
                "embedding": td["embedding"],
                "first_seen": td["first_seen"],
                "registered_time": time.time()
            }
            td["person_id"] = person_id
        else:
            person_id = td["person_id"]
        self.name_manager.set_name(person_id, name.strip())
        save_gallery(self.gallery)
        return True

    def delete_person(self, person_id):
        """Remove a person from gallery and name manager."""
        if person_id in self.gallery:
            del self.gallery[person_id]
            save_gallery(self.gallery)
        self.name_manager.delete(person_id)

    def get_display_info(self, track_id, class_id):
        """Returns (BGR_color, text) for the video overlay."""
        if class_id != 0:   # object
            label = config.COCO_NAMES[class_id] if class_id < len(config.COCO_NAMES) else f"cls_{class_id}"
            return (255, 0, 0), label       # blue

        if track_id not in self.track_data:
            return (0, 0, 255), "Unknown"   # red

        td = self.track_data[track_id]
        person_id = td["person_id"]
        if person_id is None:
            return (0, 0, 255), "Unknown"   # red

        name = self.name_manager.get_name(person_id)
        if self.name_manager.is_custom(person_id):
            return (0, 255, 0), f"Recognized: {name}"   # green
        else:
            return (255, 0, 255), f"Known: {name}"      # purple

    def cleanup_tracks(self, active_track_ids):
        for tid in list(self.track_data.keys()):
            if tid not in active_track_ids:
                del self.track_data[tid]

    @property
    def total_registered(self):
        return len(self.gallery)

    def get_registered_list(self):
        """Return list of dicts for the Registered Persons table."""
        registered = []
        for pid, info in self.gallery.items():
            name = self.name_manager.get_name(pid)
            custom = self.name_manager.is_custom(pid)
            # Compute recognition time if available
            if info.get("first_seen") and info.get("registered_time"):
                rec_time = info["registered_time"] - info["first_seen"]
            else:
                rec_time = None
            registered.append({
                "id": pid,
                "name": name,
                "custom": custom,
                "recognition_time": rec_time
            })
        return registered

    def rename_person(self, person_id, new_name):
        if person_id in self.gallery:
            self.name_manager.set_name(person_id, new_name.strip())
            return True
        return False