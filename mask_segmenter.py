import cv2
import numpy as np


def segment_cows(cow_detections, frame):
    """
    Recebe lista de dicts {"box": ..., "mask": ...} gerados pelo YoloDetector.
    Retorna lista de dicts com:
      - "crop":        recorte BGR do boi (fundo zerado pela máscara)
      - "mask":        máscara binária (uint8, 0/255) no tamanho do crop
      - "mask_full":   máscara binária no tamanho do frame original
      - "box":         coordenadas xyxy do bounding box
    """
    segments = []

    frame_h, frame_w = frame.shape[:2]

    for detection in cow_detections:
        box = detection["box"]
        mask_obj = detection["mask"]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)

        crop = frame[y1:y2, x1:x2].copy()

        if mask_obj is not None:
            mask_data = mask_obj.data[0].cpu().numpy()
            mask_full_float = cv2.resize(mask_data, (frame_w, frame_h),
                                         interpolation=cv2.INTER_LINEAR)
            mask_full = (mask_full_float > 0.5).astype(np.uint8) * 255
            mask_crop = mask_full[y1:y2, x1:x2]
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, mask_crop = cv2.threshold(gray, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_full = np.zeros((frame_h, frame_w), dtype=np.uint8)
            mask_full[y1:y2, x1:x2] = mask_crop

        
        masked_crop = cv2.bitwise_and(crop, crop, mask=mask_crop)

        segments.append({
            "crop":      masked_crop,
            "mask":      mask_crop,
            "mask_full": mask_full,
            "box":       (x1, y1, x2, y2),
        })

    return segments