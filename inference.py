
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights", "best.pt")

IMGSZ = 640
CONF = 0.001
IOU = 0.45
MAX_DET = 100


class Model:
    def __init__(self, yolo, device):
        self.yolo = yolo
        self.device = device


def load(device="cuda"):
    from ultralytics import YOLO

    yolo = YOLO(WEIGHTS)
    return Model(yolo, device)


def predict(model, image):
    # Convert RGB input to BGR for Ultralytics
    bgr = np.ascontiguousarray(image[:, :, ::-1])

    result = model.yolo.predict(
        bgr,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        augment=True,
        device=model.device,
        verbose=False
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    h, w = result.orig_shape
    detections = []

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box

        # Clamp coordinates to original image boundaries
        x1 = np.clip(x1, 0, w)
        x2 = np.clip(x2, 0, w)
        y1 = np.clip(y1, 0, h)
        y2 = np.clip(y2, 0, h)

        # Reject invalid predictions
        if not np.all(np.isfinite([x1, y1, x2, y2, score])):
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append((
            int(cls),
            float(score),
            float(x1),
            float(y1),
            float(x2),
            float(y2)
        ))

    return detections
