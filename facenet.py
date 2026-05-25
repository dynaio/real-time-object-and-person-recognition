import numpy as np
from openvino import Core

class FaceNetOpenVINO:
    def __init__(self, model_dir):
        core = Core()
        model_xml = f"{model_dir}/facenet_vggface2.xml"
        model = core.read_model(model_xml)
        self.compiled_model = core.compile_model(model, "CPU")
        self.output_layer = self.compiled_model.output(0)

    def extract(self, face_img):
        """face_img: (160,160,3) RGB"""
        blob = np.expand_dims(face_img.transpose(2,0,1).astype(np.float32)/255.0, axis=0)
        res = self.compiled_model([blob])
        return res[self.output_layer].flatten()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)