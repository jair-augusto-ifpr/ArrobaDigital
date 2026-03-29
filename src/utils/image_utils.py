import cv2

def improve_lighting(frame):

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    l = cv2.equalizeHist(l)

    lab = cv2.merge((l, a, b))

    improved = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return improved

def draw_boxes(frame, boxes):

    for box in boxes:
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            conf = float(box.conf[0]) if box.conf is not None else 0.0

            if conf > 0.7:
                color = (0, 255, 0)     
            elif conf > 0.4:
                color = (0, 255, 255)   
            else:
                color = (0, 0, 255)      

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"Boi {conf:.2f}"

            (tw, th), _ = cv2.getTextSize(label,
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, 2)

            cv2.rectangle(frame, (x1, y1 - th - 10),
                          (x1 + tw, y1), color, -1)

            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1)

        except Exception as e:
            print(f"[WARN] Erro ao desenhar box: {e}")

    return frame

def draw_fps(frame, fps):

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)

    return frame

def draw_weights(frame, cows, weights):

    for i, box in enumerate(cows):
        if i >= len(weights):
            continue

        try:
            x1, y1, _, _ = map(int, box.xyxy[0])

            peso = weights[i]

            text = f"{peso:.1f} kg"

            cv2.putText(frame, text, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        except Exception as e:
            print(f"[WARN] Erro ao desenhar peso: {e}")

    return frame

def resize_frame(frame, width=640, height=480):

    return cv2.resize(frame, (width, height))