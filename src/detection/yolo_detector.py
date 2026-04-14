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
        Retorna lista de dicts {"box": ..., "mask": ...} para cada boi detectado.
        Usa yolov8-seg, então masks estarão disponíveis quando o modelo suportar.
        """
        cows = []

        for r in results:
            boxes = r.boxes
            masks = r.masks  # None se modelo não for seg

            for i, box in enumerate(boxes):
                cls = int(box.cls[0])

                if cls == 19:  # cow no COCO
                    mask = masks[i] if masks is not None else None
                    cows.append({"box": box, "mask": mask})

        return cows
