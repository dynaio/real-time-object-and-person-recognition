import cv2
import os

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