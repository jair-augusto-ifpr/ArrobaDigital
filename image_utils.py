import cv2
import numpy as np


def improve_lighting(frame):
    return cv2.convertScaleAbs(frame, alpha=1.5, beta=30)


def draw_boxes(frame, cow_detections):
    """
    Desenha bounding box e contorno da máscara para cada boi detectado.
    Recebe lista de dicts {"box", "mask"} gerados pelo YoloDetector.
    """
    for detection in cow_detections:
        box = detection["box"]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame


def draw_masks(frame, segments):
    """
    Sobrepõe as máscaras dos segmentos no frame original com transparência.
    Recebe lista de dicts {"mask_full", ...} gerados pelo mask_segmenter.
    """
    overlay = frame.copy()

    for seg in segments:
        mask_full = seg["mask_full"]
        overlay[mask_full > 0] = (0, 200, 100)

    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    return frame