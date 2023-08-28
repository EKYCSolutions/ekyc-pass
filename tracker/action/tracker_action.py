
from ultralytics import YOLO


model_path = "weights/best.pt"


class TrackerAction():

    def __init__(self):

        self.model = YOLO(model_path)
        self.action_labels = {
            0: "block",
            1: "defense",
            2: "serve",
            3: "set",
            4: "spike"
        }

    def infer(self, frame):
        results = self.model(frame)

        if len(results[0].boxes) > 0:
            x, y, w, h = results[0].boxes.xywh[0]
            cls = results[0].boxes.cls.tolist()[0]
            label = self.action_labels[cls]
            result = f"{x},{y},{w},{h},{label}\n"

        return result
