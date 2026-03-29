from ultralytics import YOLO

model_path = 'yolov8m-seg.pt'

class YoloDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image)
        return results

    def filter_cows(self, results):
        """
        Retorna lista de dicts com box e máscara de cada boi detectado.
        Requer modelo de segmentação (yolov8m-seg.pt).
        """
        cows = []

        for r in results:
            if r.boxes is None:
                continue

            masks = r.masks  

            for i, box in enumerate(r.boxes):
                cls = int(box.cls[0])

                if cls != 19: 
                    continue

                mask = None
                if masks is not None and i < len(masks):
                    mask = masks[i]

                cows.append({
                    "box": box,
                    "mask": mask,
                })

        return cows