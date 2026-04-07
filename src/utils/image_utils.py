import cv2


def improve_lighting(frame):
    """Equaliza o canal L no espaço LAB para melhorar iluminação variável."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def draw_boxes(frame, boxes):
    """
    Desenha bounding boxes no frame.
    boxes: lista de tuplas (x1, y1, x2, y2) em pixels inteiros.
    """
    h_frame, w_frame, _ = frame.shape

    for box in boxes:
        try:
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w_frame, int(x2))
            y2 = min(h_frame, int(y2))

            color = (0, 255, 0)  # verde

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = "Boi"
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


def resize_frame(frame, width=640, height=480):
    return cv2.resize(frame, (width, height))
