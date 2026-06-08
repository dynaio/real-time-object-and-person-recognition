import cv2
import numpy as np
import yaml
from openvino import Core
import config as config
import os
class YOLOOpenVINO:
    def __init__(self, model_dir=config.YOLO_MODEL_DIR, conf_thres=0.25, iou_thres=0.45):
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