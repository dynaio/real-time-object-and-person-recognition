import os
import importlib

def file_exists(path):
    return os.path.isfile(path)

def dir_exists(path):
    return os.path.isdir(path)

def module_available(name):
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("PROJECT MODEL & DEPENDENCY CHECK")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Core model files
    print("\n[1] YOLOv8n-seg (OpenVINO)")
    yolo_dir = os.path.join(script_dir, "yolov8n-seg_model")
    yolo_xml = os.path.join(yolo_dir, "yolov8n-seg.xml")
    yolo_bin = os.path.join(yolo_dir, "yolov8n-seg.bin")
    print(f"    XML: {'' if file_exists(yolo_xml) else ''}  ({yolo_xml})")
    print(f"    BIN: {'' if file_exists(yolo_bin) else ''}  ({yolo_bin})")

    print("\n[2] FaceNet (OpenVINO)")
    face_dir = os.path.join(script_dir, "facenet_model")
    face_xml = os.path.join(face_dir, "facenet_vggface2.xml")
    face_bin = os.path.join(face_dir, "facenet_vggface2.bin")
    print(f"    XML: {'' if file_exists(face_xml) else ''}  ({face_xml})")
    print(f"    BIN: {'' if file_exists(face_bin) else ''}  ({face_bin})")

    print("\n[3] OpenCV Face Detector (SSD)")
    proto = os.path.join(script_dir, "deploy.prototxt")
    caffe = os.path.join(script_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    print(f"    deploy.prototxt: {'' if file_exists(proto) else ''}")
    print(f"    caffemodel:      {'' if file_exists(caffe) else ''}")

    # 2. Runtime data files (optional – created automatically)
    print("\n[4] Runtime Data (created on first run)")
    gallery = os.path.join(script_dir, "gallery.json")
    names = os.path.join(script_dir, "person_names.json")
    record = os.path.join(script_dir, "record_log.json")
    print(f"    gallery.json:       {'' if file_exists(gallery) else '⚠️ (will be created)'}")
    print(f"    person_names.json:  {'' if file_exists(names) else '⚠️ (will be created)'}")
    print(f"    record_log.json:    {'' if file_exists(record) else '⚠️ (created when recording)'}")

    # 3. Python packages
    print("\n[5] Required Python Packages")
    packages = {
        "openvino": "OpenVINO runtime",
        "cv2": "OpenCV (opencv-python)",
        "numpy": "NumPy",
        "PyQt5": "PyQt5 GUI",
        "filterpy": "Kalman filters (SORT)",
        "scipy": "Hungarian algorithm",
        "yaml": "YAML parser (metadata)",
        "facenet_pytorch": "FaceNet model (training, optional)",
        "ultralytics": "Ultralytics YOLO (optional – used for export only)",
    }
    for pkg, desc in packages.items():
        ok = module_available(pkg)
        print(f"    {pkg:20s} {'' if ok else ''}  {desc}")

    # 4. Future / optional tools
    print("\n[6] Optional Tools (for hand/face/eye tracking)")
    future_packages = {
        "mediapipe": "MediaPipe (hands, face mesh, iris)",
        "tensorflow": "TensorFlow (TFLite models)",
        "torch": "PyTorch (custom training)",
    }
    for pkg, desc in future_packages.items():
        ok = module_available(pkg)
        status = " available" if ok else " not installed (optional)"
        print(f"    {pkg:20s} {status}  – {desc}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if all([file_exists(yolo_xml), file_exists(yolo_bin),
            file_exists(face_xml), file_exists(face_bin),
            file_exists(proto), file_exists(caffe)]):
        print(" All core models present – project is ready to run.")
    else:
        print(" Some core model files missing. Please download them.")
    
    missing_pkgs = [pkg for pkg, ok in {
        "openvino": module_available("openvino"),
        "cv2": module_available("cv2"),
        "numpy": module_available("numpy"),
        "PyQt5": module_available("PyQt5"),
        "filterpy": module_available("filterpy"),
        "scipy": module_available("scipy"),
        "yaml": module_available("yaml"),
    }.items() if not ok]
    if missing_pkgs:
        print(f" Missing required packages: {', '.join(missing_pkgs)}")
    else:
        print(" All required Python packages installed.")

    print("\nOptional future modules (MediaPipe, etc.) can be installed when needed.")
    print("=" * 60)

if __name__ == "__main__":
    main()