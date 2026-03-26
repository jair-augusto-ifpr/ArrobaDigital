def segment_cows(cow_boxes, frame):
    segments = []

    for box in cow_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = frame[y1:y2, x1:x2]
        segments.append(crop)

    return segments