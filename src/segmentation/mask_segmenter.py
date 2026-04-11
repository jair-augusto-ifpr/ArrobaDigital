import cv2

def segment_cows(cow_boxes, frame):
    segments = []

    for box in cow_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        segmented = cv2.bitwise_and(crop, crop, mask=thresh)

        segments.append(segmented)

    return segments