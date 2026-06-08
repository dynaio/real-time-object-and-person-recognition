Here is the full README markdown content without emojis, ready to copy and paste into your `README.md` file.

```markdown
# Real-Time Object & Person Recognition

A complete, real-time computer vision pipeline that performs **instance segmentation**, **multi-object tracking**, and **person re-identification** using a webcam.  
It runs entirely on CPU (OpenVINO-optimised) and features a professional PyQt5 dashboard.

![Demo screenshot](screenshot.png) <!-- add your own screenshot later -->

## Highlights

- **Instance segmentation** on all 80 COCO classes (people, phones, bottles, chairs, …) with semi-transparent masks.
- **Person re-identification** – three states:  
  - Unknown (new face, not stored)  
  - Known (face print saved, no custom name)  
  - Registered (custom name given)
- **Smooth visual tracking** – masks appear only for stable tracks (≥3 consecutive frames) and are temporally blended to eliminate flicker.
- **Fill toggle** – one-click button to turn the transparent mask overlay on/off.
- **Persistent memory** – face embeddings and custom names survive across sessions (`gallery.json` / `person_names.json`).
- **Cumulative class table** – shows every object class ever seen, with active count and total unique track IDs.
- **Recording** – logs per-frame statistics to `record_log.json`.
- **Full PyQt5 GUI** – start/stop, fullscreen, register/delete persons, editable names, live FPS and breakdowns.

## Models & Algorithms

| Component | Model |
|-----------|-------|
| Instance Segmentation | YOLOv8n-seg (Ultralytics) → exported to **OpenVINO** IR |
| Face Embedding | FaceNet (InceptionResnetV1, pretrained on VGGFace2) → OpenVINO |
| Face Detection | OpenCV DNN SSD (ResNet-based) |
| Multi-Object Tracking | SORT (Kalman filter + Hungarian algorithm) |
| Person Memory | Custom state machine with cosine-similarity matching |

All neural networks run via the **OpenVINO inference engine** on the **CPU**, achieving 15-35 FPS on an Intel i5-8365U.

## Project Structure

```
CVi/
├── gui_main.py         ← PyQt5 dashboard and main loop
├── yolo_detector.py    ← YOLOv8 OpenVINO wrapper (with anchor-based decoding)
├── facenet.py          ← FaceNet OpenVINO wrapper
├── face_detector.py    ← OpenCV face detection
├── person_manager.py   ← Person state machine & persistent gallery
├── sort.py             ← SORT tracker
├── config.py           ← Paths & COCO class names
├── check.py            ← Dependency & model file diagnostic
├── gallery.json        ← Face embedding database (auto-created, local only)
├── person_names.json   ← Custom person names (auto-created)
├── record_log.json     ← Recording output (auto-created)
├── yolov8n-seg_model/  ← Exported YOLO OpenVINO model (.xml + .bin)
├── facenet_model/      ← Exported FaceNet OpenVINO model
├── deploy.prototxt     ← OpenCV face detector architecture
└── res10_300x300_ssd_iter_140000.caffemodel ← Face detector weights
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dynaio/real-time-object-and-person-recognition.git
   cd real-time-object-and-person-recognition
   ```

2. **Create & activate a virtual environment** (Python 3.10+)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install openvino opencv-python numpy PyQt5 filterpy scipy pyyaml
   ```

4. **Download required models**  
   The YOLO and FaceNet OpenVINO models are already included in the repository (`yolov8n-seg_model/` and `facenet_model/`).  
   The face detector files (`deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`) are also included.  
   No extra download is needed – everything works out-of-the-box.

## Usage

```bash
python gui_main.py
```

### Controls

| Action | Button / Shortcut |
|--------|-------------------|
| Start / Stop camera | **Start** / **Stop** |
| Toggle mask fill | **Fill ON/OFF** |
| Record session | **Record** (toggle) |
| Fullscreen | **Fullscreen** or `F` key |
| Quit | **Quit** or `Q` key |
| Register a person | Click **Register Person** while the person is in **Unknown** state |
| Rename a person | Double-click the name cell in the **Registered Persons** table |
| Delete a person | Select a row and click **Delete Selected** |

### Person states

- **Unknown** – face not yet stored.
- **Known** – face print is stored but no custom name given.
- **Registered** – custom name assigned; instantly recognised on subsequent appearances.

## Adding new projects

The folder `projects/` contains standalone mini-projects (object-only tracker, heatmap, crowd counter, etc.) that reuse the same modules.  
Each project is self-contained and can be run independently.

## Requirements

- Linux / Windows / macOS
- Python 3.10+
- Webcam
- Intel CPU (OpenVINO works best on Intel, but can run on any CPU)

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FaceNet (facenet-pytorch)](https://github.com/timesler/facenet-pytorch)
- [SORT tracker](https://github.com/abewley/sort)
- [OpenVINO Toolkit](https://docs.openvino.ai/)

## License

This project is for educational purposes. You are free to use, modify, and distribute it.

---

*Built by Abdelkader as part of an AI Engineering practical project.*
```

Copy the entire block above and replace your existing `README.md` with it. Then stage, commit, and push as described in your original instructions.