import os
os.environ["QT_QPA_PLATFORM"] = "wayland"
import sys
import time
import json
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QPushButton, QFrame,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QSizePolicy, QInputDialog, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

from yolo_detector import YOLOOpenVINO
from facenet import FaceNetOpenVINO
from face_detector import OpenCVFaceDetector
from sort import Sort, iou
from person_manager import PersonManager
import config

# ================== Worker Thread ==================
class CaptureWorker(QThread):
    frame_ready = pyqtSignal(QPixmap)
    stats_ready = pyqtSignal(dict)
    track_data_ready = pyqtSignal(dict)        # aggregated class data
    registered_list_ready = pyqtSignal(list)   # only when the list changes
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.paused = False
        self.recording = False
        self.mutex = QMutex()
        self.record_log = []
        self.record_frame_interval = 10
        self.person_manager = None

    def initialize(self):
        self.yolo = YOLOOpenVINO(config.YOLO_MODEL_DIR, conf_thres=0.2, iou_thres=0.35)   # lower thresholds
        self.facenet = FaceNetOpenVINO(config.FACENET_MODEL_DIR)
        self.face_detector = OpenCVFaceDetector()
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.person_manager = PersonManager()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.error.emit("Cannot open camera.")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True

    def run(self):
        if not self.initialize():
            return

        self.running = True
        frame_count = 0
        prev_registered = self.person_manager.total_registered   # for change detection

        # Emit the initial registered list once
        self.registered_list_ready.emit(self.person_manager.get_registered_list())

        while self.running:
            self.mutex.lock()
            paused = self.paused
            rec = self.recording
            self.mutex.unlock()

            if paused:
                time.sleep(0.01)
                continue

            ret, frame = self.cap.read()
            if not ret:
                self.error.emit("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)
            t0 = time.time()

            detections = self.yolo.detect(frame)
            dets_for_sort = np.array([d["bbox"] + [d["score"]] for d in detections]) if detections else np.empty((0,5))
            tracks = self.tracker.update(dets_for_sort)

            # Map detection index to best track ID
            det_to_track = {}
            for i, det in enumerate(detections):
                best_tid = -1
                best_iou = 0
                for track in tracks:
                    iou_val = iou(det["bbox"], track[:4])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_tid = int(track[4])
                if best_tid != -1 and best_iou > 0.3:
                    det_to_track[i] = best_tid

            # Face recognition every 3 frames
            if frame_count % 3 == 0:
                for i, det in enumerate(detections):
                    if det["class"] == 0 and i in det_to_track:
                        x1, y1, x2, y2 = map(int, det["bbox"])
                        person_region = frame[y1:y2, x1:x2]
                        if person_region.size > 0:
                            rgb_region = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)
                            faces = self.face_detector.detect(rgb_region)
                            if faces:
                                fx1, fy1, fx2, fy2 = faces[0]
                                if (fx2-fx1) > 20 and (fy2-fy1) > 20:
                                    face_crop = rgb_region[fy1:fy2, fx1:fx2]
                                    face_crop = cv2.resize(face_crop, (160, 160))
                                    emb = self.facenet.extract(face_crop)
                                    self.person_manager.update_track(det_to_track[i], emb, time.time())

            # If a new person was auto-registered, emit the registered list
            if self.person_manager.total_registered > prev_registered:
                self.registered_list_ready.emit(self.person_manager.get_registered_list())
                prev_registered = self.person_manager.total_registered

            # Build aggregated class data
            class_data = {}
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = map(int, det["bbox"])
                mask = det["mask"]
                class_id = det["class"]
                tid = det_to_track.get(i, -1)

                # Get display info
                if tid != -1 and tid in self.person_manager.track_data:
                    color, text = self.person_manager.get_display_info(tid, class_id)
                else:
                    if class_id == 0:
                        color, text = (0, 0, 255), "Unknown"
                    else:
                        label = config.COCO_NAMES[class_id] if class_id < len(config.COCO_NAMES) else f"cls_{class_id}"
                        color, text = (255, 0, 0), label

                # Smooth filled mask + thin contour
                overlay = frame.copy()
                mask_bool = mask > 0
                overlay[mask_bool] = color
                frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
                contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, 1)

                text_y = max(y1 - 5, 15)
                cv2.putText(frame, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Aggregate class data
                if class_id not in class_data:
                    class_data[class_id] = {
                        "track_ids": set(),
                        "active_ids": set(),
                        "person_status": {"recognized": 0, "known": 0, "unknown": 0}
                    }
                if tid != -1:
                    class_data[class_id]["track_ids"].add(tid)
                    class_data[class_id]["active_ids"].add(tid)

                if class_id == 0 and tid != -1 and tid in self.person_manager.track_data:
                    td = self.person_manager.track_data[tid]
                    pid = td["person_id"]
                    if pid is None:
                        class_data[class_id]["person_status"]["unknown"] += 1
                    elif self.person_manager.name_manager.is_custom(pid):
                        class_data[class_id]["person_status"]["recognized"] += 1
                    else:
                        class_data[class_id]["person_status"]["known"] += 1

            active_track_ids = set(int(t[4]) for t in tracks)
            self.person_manager.cleanup_tracks(active_track_ids)

            # Statistics
            total_registered = self.person_manager.total_registered
            recognized_now = sum(1 for td in self.person_manager.track_data.values()
                                 if td["person_id"] is not None and self.person_manager.name_manager.is_custom(td["person_id"]))
            total_objects = len(tracks)
            class_counts = {}
            for d in detections:
                cls = d["class"]
                class_counts[cls] = class_counts.get(cls, 0) + 1
            fps = 1.0 / (time.time() - t0 + 1e-5)

            if rec and (frame_count % self.record_frame_interval == 0):
                self.record_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "class_counts": {config.COCO_NAMES[cls]: cnt for cls, cnt in class_counts.items()},
                    "recognized_ids": [td["person_id"] for td in self.person_manager.track_data.values()
                                       if td["person_id"] is not None and self.person_manager.name_manager.is_custom(td["person_id"])]
                })

            # Emit frame and stats
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            self.frame_ready.emit(pixmap)

            self.stats_ready.emit({
                "fps": fps,
                "registered": total_registered,
                "recognized": recognized_now,
                "tracked": total_objects,
                "class_counts": class_counts,
                "recording": rec
            })
            self.track_data_ready.emit(class_data)

            # No longer emitting registered_list_ready periodically – only when content changes

            frame_count += 1

        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def pause_resume(self, state):
        self.mutex.lock()
        self.paused = state
        self.mutex.unlock()

    def set_recording(self, state):
        self.mutex.lock()
        self.recording = state
        if state:
            self.record_log = []
        else:
            log_path = os.path.join(os.path.dirname(__file__), "record_log.json")
            with open(log_path, 'w') as f:
                json.dump(self.record_log, f, indent=2)
        self.mutex.unlock()

    def manual_register_track(self, track_id, name):
        if self.person_manager:
            success = self.person_manager.manual_register(track_id, name)
            if success:
                self.registered_list_ready.emit(self.person_manager.get_registered_list())
            return success
        return False

    def rename_person(self, person_id, new_name):
        if self.person_manager:
            success = self.person_manager.rename_person(person_id, new_name)
            if success:
                self.registered_list_ready.emit(self.person_manager.get_registered_list())
            return success
        return False

    def delete_person(self, person_id):
        if self.person_manager:
            self.person_manager.delete_person(person_id)
            self.registered_list_ready.emit(self.person_manager.get_registered_list())

# ================== Main GUI Window ==================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Instance Segmentation + Person Re-ID")
        self.setMinimumSize(1300, 650)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Video (80% width)
        self.video_label = QLabel()
        self.video_label.setFrameStyle(QFrame.Box)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.video_label, 4)

        # Sidebar (20% min 280, max 420)
        sidebar = QWidget()
        sidebar.setMinimumWidth(280)
        sidebar.setMaximumWidth(420)
        sidebar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(6)

        # ---- Statistics ----
        stats_title = QLabel("STATISTICS")
        stats_title.setFont(QFont("Arial", 14, QFont.Bold))
        sidebar_layout.addWidget(stats_title)

        self.lbl_fps = QLabel("FPS: 0")
        self.lbl_registered = QLabel("Registered: 0")
        self.lbl_recognized = QLabel("Recognized: 0")
        self.lbl_tracked = QLabel("Tracked: 0")
        self.lbl_avg_rec_time = QLabel("Avg. rec. time: –")
        for lbl in [self.lbl_fps, self.lbl_registered, self.lbl_recognized, self.lbl_tracked, self.lbl_avg_rec_time]:
            lbl.setFont(QFont("Arial", 10))
            sidebar_layout.addWidget(lbl)

        self.lbl_classes = QLabel("Detected:\n  None")
        self.lbl_classes.setFont(QFont("Arial", 10))
        self.lbl_classes.setWordWrap(True)
        sidebar_layout.addWidget(self.lbl_classes)

        # ---- Cumulative Classes Table ----
        track_title = QLabel("CUMULATIVE CLASSES")
        track_title.setFont(QFont("Arial", 12, QFont.Bold))
        sidebar_layout.addWidget(track_title)

        self.track_table = QTableWidget()
        self.track_table.setColumnCount(4)
        self.track_table.setHorizontalHeaderLabels(["Class", "Status", "Active", "Count"])
        self.track_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.track_table.verticalHeader().setVisible(False)
        self.track_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.track_table.setSelectionMode(QTableWidget.NoSelection)
        self.track_table.setMaximumHeight(180)
        sidebar_layout.addWidget(self.track_table)

        # ---- Registered Persons ----
        reg_title = QLabel("REGISTERED PERSONS")
        reg_title.setFont(QFont("Arial", 12, QFont.Bold))
        sidebar_layout.addWidget(reg_title)

        self.reg_table = QTableWidget()
        self.reg_table.setColumnCount(4)
        self.reg_table.setHorizontalHeaderLabels(["ID", "Name", "Status", "Rec. Time"])
        self.reg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.reg_table.verticalHeader().setVisible(False)
        self.reg_table.setEditTriggers(QTableWidget.DoubleClicked)
        self.reg_table.setSelectionMode(QTableWidget.SingleSelection)
        self.reg_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.reg_table.setMaximumHeight(150)
        self.reg_table.cellChanged.connect(self.on_registered_name_changed)
        sidebar_layout.addWidget(self.reg_table)

        # Register & delete buttons
        btn_row = QHBoxLayout()
        self.btn_register = QPushButton("Register Person")
        self.btn_register.clicked.connect(self.on_register_clicked)
        btn_row.addWidget(self.btn_register)

        self.btn_delete_person = QPushButton("Delete Selected")
        self.btn_delete_person.clicked.connect(self.on_delete_person)
        btn_row.addWidget(self.btn_delete_person)
        sidebar_layout.addLayout(btn_row)

        sidebar_layout.addStretch()

        # ---- Control Buttons ----
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.toggle_start)
        self.btn_record = QPushButton("Record")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_fullscreen = QPushButton("Fullscreen")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        self.btn_quit = QPushButton("Quit")
        self.btn_quit.clicked.connect(self.close)

        for btn in [self.btn_start, self.btn_record, self.btn_fullscreen, self.btn_quit]:
            btn.setMinimumHeight(35)
            sidebar_layout.addWidget(btn)

        main_layout.addWidget(sidebar, 1)

        # State
        self.worker = None
        self.is_running = False
        self.fullscreen_state = False
        self.reg_table_programmatic_update = False

        # Cumulative class data (persistent across frames)
        self.cumulative_classes = {}   # class_id -> {track_ids: set, class_name: str}

    def toggle_start(self):
        if not self.is_running:
            self.worker = CaptureWorker()
            self.worker.frame_ready.connect(self.update_frame)
            self.worker.stats_ready.connect(self.update_stats)
            self.worker.track_data_ready.connect(self.update_track_table)
            self.worker.registered_list_ready.connect(self.update_registered_table)
            self.worker.start()
            self.btn_start.setText("Stop")
            self.is_running = True
        else:
            if self.worker:
                self.worker.stop()
                self.worker = None
            self.btn_start.setText("Start")
            self.is_running = False
            self.track_table.setRowCount(0)
            self.reg_table.setRowCount(0)
            self.cumulative_classes.clear()

    def toggle_recording(self):
        if self.worker:
            self.worker.set_recording(self.btn_record.isChecked())

    def toggle_fullscreen(self):
        if not self.fullscreen_state:
            self.showFullScreen()
            self.fullscreen_state = True
        else:
            self.showNormal()
            self.fullscreen_state = False

    def update_frame(self, pixmap):
        label_size = self.video_label.size()
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def update_stats(self, stats):
        self.lbl_fps.setText(f"FPS: {stats['fps']:.1f}")
        self.lbl_registered.setText(f"Registered: {stats['registered']}")
        self.lbl_recognized.setText(f"Recognized: {stats['recognized']}")
        self.lbl_tracked.setText(f"Tracked: {stats['tracked']}")
        class_lines = "Detected:\n"
        if stats['class_counts']:
            for cls, cnt in stats['class_counts'].items():
                name = config.COCO_NAMES[cls] if cls < len(config.COCO_NAMES) else f"cls_{cls}"
                class_lines += f"  {name}: {cnt}\n"
        else:
            class_lines += "  None"
        self.lbl_classes.setText(class_lines)

        if stats.get('recording', False):
            self.btn_record.setText("Recording...")
        else:
            self.btn_record.setText("Record")

    def update_track_table(self, class_data):
        """Aggregated by class; rows persist once a class is seen."""
        for class_id, data in class_data.items():
            if class_id not in self.cumulative_classes:
                self.cumulative_classes[class_id] = {
                    "track_ids": set(),
                    "class_name": config.COCO_NAMES[class_id] if class_id < len(config.COCO_NAMES) else f"cls_{class_id}"
                }
            self.cumulative_classes[class_id]["track_ids"].update(data["track_ids"])

        sorted_classes = sorted(self.cumulative_classes.items())
        self.track_table.setRowCount(len(sorted_classes))
        for row, (class_id, cum_info) in enumerate(sorted_classes):
            class_name = cum_info["class_name"]
            count = len(cum_info["track_ids"])
            if class_id in class_data:
                active = len(class_data[class_id]["active_ids"])
                if class_id == 0:
                    ps = class_data[class_id]["person_status"]
                    status = f"R:{ps['recognized']} K:{ps['known']} U:{ps['unknown']}"
                else:
                    status = ""
            else:
                active = 0
                status = ""

            name_item = QTableWidgetItem(class_name)
            name_item.setTextAlignment(Qt.AlignCenter)
            self.track_table.setItem(row, 0, name_item)

            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            self.track_table.setItem(row, 1, status_item)

            active_item = QTableWidgetItem(str(active))
            active_item.setTextAlignment(Qt.AlignCenter)
            self.track_table.setItem(row, 2, active_item)

            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.track_table.setItem(row, 3, count_item)

    def update_registered_table(self, reg_list):
        self.reg_table_programmatic_update = True
        self.reg_table.setRowCount(len(reg_list))
        for row, person in enumerate(reg_list):
            id_item = QTableWidgetItem(str(person["id"]))
            id_item.setTextAlignment(Qt.AlignCenter)
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            self.reg_table.setItem(row, 0, id_item)

            name_item = QTableWidgetItem(person["name"])
            name_item.setTextAlignment(Qt.AlignCenter)
            self.reg_table.setItem(row, 1, name_item)

            status = "Registered" if person["custom"] else "Known (no name)"
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            if person["custom"]:
                status_item.setBackground(QColor(0, 255, 0, 100))
            else:
                status_item.setBackground(QColor(255, 0, 255, 100))
            self.reg_table.setItem(row, 2, status_item)

            rec_time = person.get("recognition_time")
            rec_str = f"{rec_time:.1f} s" if rec_time is not None else "–"
            rec_item = QTableWidgetItem(rec_str)
            rec_item.setTextAlignment(Qt.AlignCenter)
            rec_item.setFlags(rec_item.flags() & ~Qt.ItemIsEditable)
            self.reg_table.setItem(row, 3, rec_item)

            bg = QColor(0, 255, 0, 80) if person["custom"] else QColor(255, 0, 255, 80)
            for col in range(4):
                self.reg_table.item(row, col).setBackground(bg)

        times = [p["recognition_time"] for p in reg_list if p["recognition_time"] is not None]
        if times:
            avg = sum(times) / len(times)
            self.lbl_avg_rec_time.setText(f"Avg. rec. time: {avg:.1f} s")
        else:
            self.lbl_avg_rec_time.setText("Avg. rec. time: –")

        self.reg_table_programmatic_update = False

    def on_registered_name_changed(self, row, col):
        if self.reg_table_programmatic_update or col != 1 or self.worker is None:
            return
        id_item = self.reg_table.item(row, 0)
        name_item = self.reg_table.item(row, 1)
        if id_item is None or name_item is None:
            return
        person_id = int(id_item.text())
        new_name = name_item.text().strip()
        if new_name:
            self.worker.rename_person(person_id, new_name)   # this will re-emit the list (which rebuilds the table)

    def on_register_clicked(self):
        if self.worker is None or not self.is_running:
            QMessageBox.warning(self, "Warning", "No active session.")
            return
        if not self.worker.person_manager:
            return
        for tid, td in self.worker.person_manager.track_data.items():
            if td["person_id"] is None and td["embedding"] is not None:
                name, ok = QInputDialog.getText(self, "Register Person", f"Enter name for track {tid}:")
                if ok and name:
                    success = self.worker.manual_register_track(tid, name.strip())
                    if success:
                        QMessageBox.information(self, "Success", f"Person registered as '{name}'.")
                    else:
                        QMessageBox.warning(self, "Error", "Registration failed.")
                return
        QMessageBox.information(self, "Info", "No unknown person with a face embedding available.")

    def on_delete_person(self):
        if self.worker is None:
            return
        selected = self.reg_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a row first.")
            return
        row = selected[0].row()
        id_item = self.reg_table.item(row, 0)
        if id_item is None:
            return
        person_id = int(id_item.text())
        self.worker.delete_person(person_id)
        QMessageBox.information(self, "Deleted", f"Person {person_id} removed.")
        # worker.delete_person already emits the updated list

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()

# ================== Run ==================
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()