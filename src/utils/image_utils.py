import cv2

def improve_lighting(frame):
    return cv2.convertScaleAbs(frame, alpha=1.5, beta=30)

def draw_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    return frame