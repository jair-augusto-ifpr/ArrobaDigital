from ultralytics import YOLO

model_path = 'yolov8m-seg.pt'

# No dataset COCO, a classe "cow" tem id 19.
COW_CLASS_ID = 19


class YoloDetector:
    def __init__(self, model_path, conf=0.25, cow_class_id=COW_CLASS_ID):
        self.model = YOLO(model_path)
        self.conf = conf
        self.cow_class_id = cow_class_id

    @property
    def names(self):
        return getattr(self.model, "names", {}) or {}

    def detect(self, image):
        # verbose=False evita imprimir cada frame no terminal (resumo + Speed: ...)
        return self.model(image, verbose=False, conf=self.conf)

    def track(self, image, persist: bool = True, tracker: str = "bytetrack.yaml"):
        """Como `detect`, mas mantém IDs estáveis entre frames (ByteTrack embutido).

        Ultralytics já distribui `bytetrack.yaml`. Use `persist=True` para que o
        tracker mantenha o estado entre chamadas sucessivas no mesmo vídeo.
        """
        return self.model.track(
            image, persist=persist, tracker=tracker,
            verbose=False, conf=self.conf,
        )

    def _iter_boxes(self, results):
        for r in results:
            boxes = r.boxes
            masks = r.masks
            if boxes is None:
                continue
            for i, box in enumerate(boxes):
                mask = masks[i] if masks is not None and i < len(masks) else None
                yield box, mask

    @staticmethod
    def _track_id(box):
        tid = getattr(box, "id", None)
        if tid is None:
            return None
        try:
            return int(tid[0])
        except Exception:
            try:
                return int(tid.item())
            except Exception:
                return None

    def detect_all(self, results):
        """Retorna todas as detecções: [{box, mask, cls, name, conf, track_id}].

        `track_id` só é preenchido se `results` vier de `track(...)`.
        """
        out = []
        for box, mask in self._iter_boxes(results):
            cls = int(box.cls[0])
            conf_val = float(box.conf[0]) if box.conf is not None else 0.0
            out.append({
                "box": box,
                "mask": mask,
                "cls": cls,
                "name": self.names.get(cls, str(cls)),
                "conf": conf_val,
                "track_id": self._track_id(box),
            })
        return out

    def filter_cows(self, results):
        """Lista de dicts {box, mask, cls, name, conf} só com a classe de boi."""
        return [d for d in self.detect_all(results) if d["cls"] == self.cow_class_id]
