import cv2
import numpy as np
import time
import os
import json
import yaml
from openvino import Core
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ================== SORT TRACKER ==================
class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:,1]==t)[0], 0]
                if len(d) > 0:
                    d = int(d[0])
                    self.trackers[t].update(dets[d, :4])
        for i in unmatched_dets:
            i = int(i)
            if dets[i, 2] > dets[i, 0] and dets[i, 3] > dets[i, 1]:
                trk = KalmanBoxTracker(dets[i, :])
                self.trackers.append(trk)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1))
        if len(self.trackers) > 0:
            self.trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, dets, trks):
        if len(trks) == 0:
            return np.empty((0,2), dtype=int), np.arange(len(dets)), np.empty((0,), dtype=int)
        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d,t] = iou(det[:4], trk[:4])
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(row_ind, col_ind)), dtype=int)
        unmatched_detections = np.array([d for d in range(len(dets)) if d not in row_ind], dtype=int)
        unmatched_trackers = np.array([t for t in range(len(trks)) if t not in col_ind], dtype=int)
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections = np.append(unmatched_detections, int(m[0]))
                unmatched_trackers = np.append(unmatched_trackers, int(m[1]))
            else:
                matches.append(m.reshape(1,2))
        if len(matches) == 0:
            matches = np.empty((0,2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, unmatched_detections.astype(int), unmatched_trackers.astype(int)

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self.bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z(bbox))

    def predict(self):
        self.age += 1
        if (self.kf.x[6]+self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.x_to_bbox(self.kf.x)

    @staticmethod
    def bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h + 1e-6
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def x_to_bbox(x):
        w = np.sqrt(max(0, x[2] * x[3]))
        h = x[2] / (w + 1e-6)
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ================== OpenVINO FaceNet ==================
class FaceNetOpenVINO:
    def __init__(self, model_dir):
        core = Core()
        model_xml = f"{model_dir}/facenet_vggface2.xml"
        model = core.read_model(model_xml)
        self.compiled_model = core.compile_model(model, "CPU")
        self.output_layer = self.compiled_model.output(0)

    def extract(self, face_img):
        blob = np.expand_dims(face_img.transpose(2,0,1).astype(np.float32)/255.0, axis=0)
        res = self.compiled_model([blob])
        return res[self.output_layer].flatten()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# ================== OpenVINO YOLO (multi‑object) ==================
class YOLOOpenVINO:
    def __init__(self, model_dir, conf_thres=0.25, iou_thres=0.45):
        core = Core()
        model_xml = os.path.join(model_dir, "yolov8n-seg.xml")
        model = core.read_model(model_xml)
        self.compiled_model = core.compile_model(model, "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_boxes = self.compiled_model.output(0)
        self.output_masks = self.compiled_model.output(1)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_h, self.input_w = self.input_layer.shape[2:4]

        meta_path = os.path.join(model_dir, "metadata.yaml")
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f)
        self.strides = metadata.get('strides', [8, 16, 32])
        self.anchor_points, self.stride_tensor = self._generate_anchors()

    def _generate_anchors(self):
        anchor_points = []
        stride_tensor = []
        for stride in self.strides:
            h = self.input_h // stride
            w = self.input_w // stride
            for y in range(h):
                for x in range(w):
                    anchor_points.append([x, y])
                    stride_tensor.append(stride)
        return np.array(anchor_points, dtype=np.float32), np.array(stride_tensor, dtype=np.float32)

    def detect(self, frame):
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(img_rgb.transpose(2,0,1).astype(np.float32)/255.0, axis=0)
        outputs = self.compiled_model([blob])
        out_boxes = outputs[self.output_boxes]
        out_masks = outputs[self.output_masks]
        return self.postprocess(out_boxes, out_masks, frame.shape[:2])

    def postprocess(self, pred, proto_masks, orig_shape):
        pred = pred[0].transpose()
        proto = proto_masks[0]

        boxes_raw = pred[:, :4]
        scores = pred[:, 4:84]
        mask_coeffs_all = pred[:, 84:116]

        max_scores = scores.max(axis=1)
        class_ids = scores.argmax(axis=1)

        mask = max_scores > self.conf_thres
        if not mask.any():
            return []

        boxes_raw = boxes_raw[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]
        mask_coeffs = mask_coeffs_all[mask]
        anchor_pts = self.anchor_points[mask]
        stride_t = self.stride_tensor[mask]

        decoded_boxes = self._dist2bbox(boxes_raw, anchor_pts, stride_t, self.input_w, self.input_h)

        scale_x = orig_shape[1] / self.input_w
        scale_y = orig_shape[0] / self.input_h
        decoded_boxes[:, 0] *= scale_x
        decoded_boxes[:, 1] *= scale_y
        decoded_boxes[:, 2] *= scale_x
        decoded_boxes[:, 3] *= scale_y

        candidates = []
        for i in range(len(decoded_boxes)):
            x1, y1, x2, y2 = decoded_boxes[i]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_shape[1], x2), min(orig_shape[0], y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            candidates.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(max_scores[i]),
                "class": int(class_ids[i]),
                "mask_coeffs": mask_coeffs[i]
            })

        # Per-class NMS
        final_dets = []
        if candidates:
            class_to_dets = {}
            for det in candidates:
                cls = det["class"]
                class_to_dets.setdefault(cls, []).append(det)
            for cls, dets in class_to_dets.items():
                boxes = np.array([d["bbox"] for d in dets])
                scores = np.array([d["score"] for d in dets])
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_thres, self.iou_thres)
                if len(indices) > 0:
                    indices = indices.flatten()
                    for i in indices:
                        det = dets[i]
                        coeffs = det["mask_coeffs"]
                        mask = np.dot(coeffs, proto.reshape(32, -1))
                        mask = 1 / (1 + np.exp(-mask))
                        mask = mask.reshape(80, 80)
                        mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
                        mask = (mask > 0.5).astype(np.uint8)
                        final_dets.append({
                            "bbox": det["bbox"],
                            "mask": mask,
                            "score": det["score"],
                            "class": det["class"]
                        })
        return final_dets

    def _dist2bbox(self, distance, anchor_points, stride_tensor, img_w, img_h):
        lt, rb = distance[:, :2], distance[:, 2:]
        ax = anchor_points[:, 0] * stride_tensor
        ay = anchor_points[:, 1] * stride_tensor
        x1y1 = np.stack([ax, ay], axis=1) - lt * stride_tensor[:, np.newaxis]
        x2y2 = np.stack([ax, ay], axis=1) + rb * stride_tensor[:, np.newaxis]
        return np.concatenate([x1y1, x2y2], axis=1)

# ================== OpenCV Face Detector ==================
class OpenCVFaceDetector:
    def __init__(self, confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        self.proto_path = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
        self.caffemodel_path = os.path.join(os.path.dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")
        if not os.path.exists(self.caffemodel_path):
            self._download_model()
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.caffemodel_path)

    def _download_model(self):
        import urllib.request
        base_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
        print("Downloading face detector model files...")
        urllib.request.urlretrieve(base_url + "deploy.prototxt", self.proto_path)
        urllib.request.urlretrieve("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", self.caffemodel_path)
        print("Done.")

    def detect(self, image_rgb):
        h, w = image_rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image_rgb, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                x1 = int(max(0, detections[0, 0, i, 3] * w))
                y1 = int(max(0, detections[0, 0, i, 4] * h))
                x2 = int(min(w, detections[0, 0, i, 5] * w))
                y2 = int(min(h, detections[0, 0, i, 6] * h))
                faces.append((x1, y1, x2, y2))
        return faces

# ================== COCO class names ==================
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ================== Person Name Manager ==================
class PersonNameManager:
    def __init__(self, path="person_names.json"):
        self.path = os.path.join(os.path.dirname(__file__), path)
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

    def add_new(self, person_id):
        name = f"Person_{person_id}"
        self.names[str(person_id)] = name
        self.save()
        return name

# ================== Main ==================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_model_dir = os.path.join(script_dir, "yolov8n-seg_model")
    facenet_model_dir = os.path.join(script_dir, "facenet_model")

    print("Loading YOLOv8n-seg (OpenVINO)...")
    yolo = YOLOOpenVINO(yolo_model_dir, conf_thres=0.25)

    print("Loading FaceNet...")
    facenet = FaceNetOpenVINO(facenet_model_dir)

    face_detector = OpenCVFaceDetector(confidence_threshold=0.6)

    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # Person memory
    gallery = {}                  # person_id -> embedding
    track_data = {}               # track_id -> {first_seen, embedding, person_id, state}
    next_person_id = 0
    name_manager = PersonNameManager()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fullscreen = True
    window_name = "Real-time Instance Segmentation + Re-ID"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except:
        cv2.resizeWindow(window_name, 1920, 1080)

    frame_count = 0
    last_print = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror
        t0 = time.time()

        detections = yolo.detect(frame)

        # Prepare detections for tracker
        dets_for_sort = np.array([d["bbox"] + [d["score"]] for d in detections]) if detections else np.empty((0,5))
        tracks = tracker.update(dets_for_sort)

        active_track_ids = set(int(t[4]) for t in tracks)

        # Update track data
        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            # Find best matching detection
            best_det = None
            best_iou = 0
            for d in detections:
                iou_val = iou(d["bbox"], [x1, y1, x2, y2])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det = d
            if best_det is None or best_iou < 0.3:
                continue

            class_id = best_det["class"]
            mask = best_det["mask"]
            label = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else f"cls_{class_id}"

            if class_id == 0:   # person
                # Initialize track data if new
                if track_id not in track_data:
                    track_data[track_id] = {
                        "first_seen": time.time(),
                        "embedding": None,
                        "person_id": -1,
                        "state": "check"   # check / new / recognized
                    }
                td = track_data[track_id]
                now = time.time()

                # Attempt face recognition every 3 frames while in "check" state
                if td["state"] == "check" and frame_count % 3 == 0:
                    person_region = frame[y1:y2, x1:x2]
                    if person_region.size > 0:
                        rgb_region = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)
                        faces = face_detector.detect(rgb_region)
                        if faces:
                            fx1, fy1, fx2, fy2 = faces[0]
                            if (fx2-fx1) > 20 and (fy2-fy1) > 20:
                                face_crop = rgb_region[fy1:fy2, fx1:fx2]
                                face_crop = cv2.resize(face_crop, (160, 160))
                                emb = facenet.extract(face_crop)

                                # Search gallery
                                matched_id = -1
                                best_sim = 0
                                for pid, g_emb in gallery.items():
                                    sim = cosine_similarity(emb, g_emb)
                                    if sim > best_sim:
                                        best_sim = sim
                                        matched_id = pid

                                if best_sim > 0.6:
                                    # Recognized!
                                    td["state"] = "recognized"
                                    td["person_id"] = matched_id
                                    # Update gallery with fresh embedding
                                    gallery[matched_id] = emb
                                else:
                                    # Store embedding for possible later registration
                                    td["embedding"] = emb

                # Check for auto‑registration after 60s
                if td["state"] == "check" and (now - td["first_seen"]) > 60:
                    if td["embedding"] is not None:
                        # Register new person
                        person_id = next_person_id
                        next_person_id += 1
                        gallery[person_id] = td["embedding"]
                        name_manager.add_new(person_id)
                        td["person_id"] = person_id
                        td["state"] = "new"
                        print(f"Registered Person {person_id} ({name_manager.get_name(person_id)})")
                    else:
                        # No face captured, keep as "check" (will stay purple)
                        pass

                # Determine colour and label
                if td["state"] == "recognized":
                    color = (0, 255, 0)          # green
                    name = name_manager.get_name(td["person_id"])
                    text = f"{name} (recognized)"
                elif td["state"] == "new":
                    color = (0, 0, 255)          # red
                    name = name_manager.get_name(td["person_id"])
                    text = f"{name} (new)"
                else:  # "check"
                    color = (255, 0, 255)        # purple
                    text = "Person (check)"

            else:
                # Object (blue)
                color = (255, 0, 0)
                text = f"{label}"

            # Draw mask contour
            mask_bool = mask > 0
            contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, 2)

            # Place text near top of mask (or top-left of bbox)
            text_x, text_y = int(x1), int(y1) - 5
            if text_y < 20:
                text_y = 20
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Cleanup stale track data
        for tid in list(track_data.keys()):
            if tid not in active_track_ids:
                del track_data[tid]

        # Statistics
        total_registered = len(gallery)
        recognized_now = sum(1 for td in track_data.values() if td["state"] == "recognized")
        total_objects = len(tracks)

        # Overlay statistics (top‑left corner)
        stats = [
            f"Registered: {total_registered}",
            f"Recognized: {recognized_now}",
            f"Objects: {total_objects}"
        ]
        y0 = 60
        for i, txt in enumerate(stats):
            cv2.putText(frame, txt, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # FPS
        fps = 1.0 / (time.time() - t0 + 1e-5)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Terminal summary (once per second)
        if time.time() - last_print > 1.0:
            last_print = time.time()
            class_counts = {}
            for d in detections:
                cls = d["class"]
                class_counts[cls] = class_counts.get(cls, 0) + 1
            det_str = ", ".join(f"{COCO_NAMES[cls]}:{cnt}" for cls, cnt in class_counts.items())
            if det_str:
                print(f"FPS: {fps:.1f} | Objects: {len(detections)} ({det_str})")

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()