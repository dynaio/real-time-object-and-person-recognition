import cv2
import numpy as np
import time
import os
import config
from sort import Sort, iou
from yolo_detector import YOLOOpenVINO
from facenet import FaceNetOpenVINO
from face_detector import OpenCVFaceDetector
from person_manager import PersonManager

PANEL_WIDTH = 320
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540

class Button:
    def __init__(self, x, y, w, h, text, color):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.color = color

    def draw(self, canvas):
        cv2.rectangle(canvas, (self.x, self.y), (self.x+self.w, self.y+self.h), self.color, -1)
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        cv2.putText(canvas, self.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    def is_inside(self, px, py):
        return self.x <= px <= self.x+self.w and self.y <= py <= self.y+self.h

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_model_dir = config.YOLO_MODEL_DIR
    facenet_model_dir = config.FACENET_MODEL_DIR

    print("Loading YOLOv8n-seg (OpenVINO)...")
    yolo = YOLOOpenVINO(yolo_model_dir, conf_thres=0.25)

    print("Loading FaceNet...")
    facenet = FaceNetOpenVINO(facenet_model_dir)

    face_detector = OpenCVFaceDetector(confidence_threshold=0.6)

    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    person_manager = PersonManager()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Get first frame to determine video dimensions
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)
    video_h, video_w = frame.shape[:2]

    # Buttons placed on the right panel
    panel_x_start = video_w
    stop_btn = Button(panel_x_start + 20, 400, 120, 40, "STOP", (0,0,255))
    quit_btn = Button(panel_x_start + 160, 400, 120, 40, "QUIT", (0,0,255))
    fullscreen_btn = Button(panel_x_start + 20, 460, 260, 40, "TOGGLE FULLSCREEN", (200,200,200))

    # Window setup
    fullscreen = True
    window_name = "Real-time Instance Segmentation + Re-ID"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)

    # Mouse action flags
    actions = {"stop": False, "quit": False, "toggle_fullscreen": False}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            win_w = cv2.getWindowImageRect(window_name)[2]
            win_h = cv2.getWindowImageRect(window_name)[3]
            scale_x = (video_w + PANEL_WIDTH) / win_w
            scale_y = max(video_h, WINDOW_HEIGHT) / win_h
            img_x = int(x * scale_x)
            img_y = int(y * scale_y)
            if stop_btn.is_inside(img_x, img_y):
                actions["stop"] = True
            elif quit_btn.is_inside(img_x, img_y):
                actions["quit"] = True
            elif fullscreen_btn.is_inside(img_x, img_y):
                actions["toggle_fullscreen"] = True

    cv2.setMouseCallback(window_name, mouse_callback)

    frame_count = 0
    running = True
    stop_flag = False

    while running:
        if not stop_flag:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            t0 = time.time()

            detections = yolo.detect(frame)

            dets_for_sort = np.array([d["bbox"] + [d["score"]] for d in detections]) if detections else np.empty((0,5))
            tracks = tracker.update(dets_for_sort)

            # Face recognition on persons every 3 frames
            if frame_count % 3 == 0:
                for det in detections:
                    if det["class"] == 0:
                        x1, y1, x2, y2 = det["bbox"]
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
                                    # Find best track for this detection
                                    best_tid = -1
                                    best_iou = 0
                                    for track in tracks:
                                        if iou(det["bbox"], track[:4]) > best_iou:
                                            best_iou = iou(det["bbox"], track[:4])
                                            best_tid = int(track[4])
                                    if best_tid != -1 and best_iou > 0.3:
                                        person_manager.update_track(best_tid, emb, time.time())

            # Draw every detection
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                mask = det["mask"]
                class_id = det["class"]

                # Find matching track
                best_tid = -1
                best_iou = 0
                for track in tracks:
                    iou_val = iou(det["bbox"], track[:4])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_tid = int(track[4])

                if best_tid != -1 and best_iou > 0.3:
                    color, text = person_manager.get_display_info(best_tid, class_id)
                else:
                    if class_id == 0:
                        color, text = (255, 0, 255), "Person (check)"
                    else:
                        label = config.COCO_NAMES[class_id] if class_id < len(config.COCO_NAMES) else f"cls_{class_id}"
                        color, text = (255, 0, 0), label

                # Draw mask
                mask_bool = mask > 0
                contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, 2)

                text_y = max(y1 - 5, 15)
                cv2.putText(frame, text, (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Cleanup
            active_track_ids = set(int(t[4]) for t in tracks)
            person_manager.cleanup_tracks(active_track_ids)

            # Statistics
            total_registered = person_manager.total_registered
            recognized_now = sum(1 for td in person_manager.track_data.values() if td["state"] == "recognized")
            total_objects = len(tracks)
            class_counts = {}
            for d in detections:
                cls = d["class"]
                class_counts[cls] = class_counts.get(cls, 0) + 1

            # Build combined GUI image
            panel_h = max(video_h, WINDOW_HEIGHT)
            combined = np.zeros((panel_h, video_w + PANEL_WIDTH, 3), dtype=np.uint8)
            combined[:video_h, :video_w] = frame

            # Right panel background
            cv2.rectangle(combined, (video_w, 0), (video_w+PANEL_WIDTH, panel_h), (50,50,50), -1)

            # Stats text
            y0 = 30
            cv2.putText(combined, "STATISTICS", (video_w+20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y0 += 40
            cv2.putText(combined, f"Registered: {total_registered}", (video_w+20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200))
            y0 += 25
            cv2.putText(combined, f"Recognized: {recognized_now}", (video_w+20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200))
            y0 += 25
            cv2.putText(combined, f"Tracked: {total_objects}", (video_w+20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200))
            y0 += 40
            cv2.putText(combined, "Detected:", (video_w+20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200))
            y0 += 20
            for cls, cnt in class_counts.items():
                name = config.COCO_NAMES[cls] if cls < len(config.COCO_NAMES) else f"cls_{cls}"
                cv2.putText(combined, f"{name}: {cnt}", (video_w+30, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200))
                y0 += 18

            # Draw buttons
            stop_btn.draw(combined)
            quit_btn.draw(combined)
            fullscreen_btn.draw(combined)

            fps = 1.0 / (time.time() - t0 + 1e-5)
            cv2.putText(combined, f"FPS: {fps:.1f}", (video_w+20, panel_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

            display = cv2.resize(combined, (WINDOW_WIDTH, WINDOW_HEIGHT))
            cv2.imshow(window_name, display)

        # Handle keys and button actions
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)

        if actions["quit"]:
            running = False
        if actions["toggle_fullscreen"]:
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
            actions["toggle_fullscreen"] = False
        if actions["stop"]:
            stop_flag = not stop_flag
            actions["stop"] = False

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()