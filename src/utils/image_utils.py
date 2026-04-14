import cv2


def _detection_box(det):
    """Aceita dict {'box': ...} (pipeline seg) ou objeto box do Ultralytics."""
    if isinstance(det, dict) and "box" in det:
        return det["box"]
    return det


def improve_lighting(frame):
    """Equaliza o canal L no espaço LAB para melhorar iluminação variável."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def draw_boxes(frame, detections):
    """
    Desenha bounding boxes no frame.
    detections: lista de dicts com chave 'box' ou objetos Ultralytics com .xyxy.
    """
    h_frame, w_frame, _ = frame.shape

    for det in detections:
        try:
            box = _detection_box(det)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_frame, x2)
            y2 = min(h_frame, y2)

            conf = None
            if getattr(box, "conf", None) is not None:
                conf = float(box.conf[0])

            if conf is None:
                color = (0, 255, 0)
                label = "Boi"
            else:
                if conf > 0.7:
                    color = (0, 255, 0)
                elif conf > 0.4:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                label = f"Boi {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        except Exception as e:
            print(f"[WARN] Erro ao desenhar box: {e}")

    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


def draw_weights(frame, cows, weights):
    for i, det in enumerate(cows):
        if i >= len(weights):
            continue

        try:
            box = _detection_box(det)
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
