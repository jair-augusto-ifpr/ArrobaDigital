from ultralytics import YOLO

model_path = 'yolov8m.pt' 

class YoloDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image)
        return results
    
    def filter_cows(self, results):
        cows = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 19:  # cow
                    cows.append(box)

        return cows