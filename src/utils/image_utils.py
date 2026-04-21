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


def _box_xyxy(box, frame_shape):
    h_frame, w_frame = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_frame, x2)
    y2 = min(h_frame, y2)
    return x1, y1, x2, y2


def draw_boxes(frame, detections, cow_class_id=19):
    """
    Desenha bounding boxes. Aceita:
      - lista de dicts {box, name, conf, cls, ...} vindos do YoloDetector.detect_all;
      - lista de objetos `box` (Ultralytics) — modo legado.
    Boi = verde, outras classes = amarelo, baixa conf = vermelho.
    """
    for det in detections:
        try:
            box = _detection_box(det)
            x1, y1, x2, y2 = _box_xyxy(box, frame.shape)

            name = det.get("name") if isinstance(det, dict) else None
            conf = det.get("conf") if isinstance(det, dict) else None
            cls = det.get("cls") if isinstance(det, dict) else None
            if conf is None and getattr(box, "conf", None) is not None:
                conf = float(box.conf[0])
            if cls is None and getattr(box, "cls", None) is not None:
                cls = int(box.cls[0])

            is_cow = cls == cow_class_id
            if conf is not None and conf < 0.4:
                color = (0, 0, 255)
            elif is_cow:
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)

            label_txt = name or ("Boi" if is_cow else "obj")
            if conf is not None:
                label_txt = f"{label_txt} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label_txt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        except Exception as e:
            print(f"[WARN] Erro ao desenhar box: {e}")

    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


def draw_hud(frame, info):
    """
    Desenha um painel com informações de execução no canto superior esquerdo.
    `info` é um dict com chaves opcionais: fps, source, db, detections, cows, scale, paused, show_all.
    """
    lines = []
    if "fps" in info:
        lines.append(f"FPS: {info['fps']:.1f}")
    if "source" in info:
        lines.append(f"Fonte: {info['source']}")
    if "db" in info:
        lines.append(f"DB: {info['db']}")
    if "ia" in info:
        lines.append(f"IA: {info['ia']}")
    if "detections" in info or "cows" in info:
        lines.append(
            f"Deteccoes: {info.get('detections', 0)} (bois: {info.get('cows', 0)})"
        )
    if info.get("tracks") is not None:
        lines.append(f"Tracks ativos: {info['tracks']}")
    if "scale" in info and info["scale"] is not None:
        lines.append(f"Escala: {info['scale']:.3f} cm/px")
    if info.get("show_all"):
        lines.append("Modo: TODAS as classes")
    if info.get("paused"):
        lines.append("PAUSADO (tecla p)")

    if not lines:
        return frame

    pad = 8
    line_h = 22
    box_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        box_w = max(box_w, tw)
    box_w += pad * 2
    box_h = pad * 2 + line_h * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 10 + pad + 16
    for line in lines:
        cv2.putText(frame, line, (10 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h
    return frame


def _wrap_text(text, width=48):
    """Quebra texto em linhas com no máximo `width` chars, respeitando palavras."""
    if text is None:
        return []
    lines = []
    for raw_line in str(text).splitlines() or [""]:
        words = raw_line.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for w in words[1:]:
            if len(current) + 1 + len(w) <= width:
                current += " " + w
            else:
                lines.append(current)
                current = w
        lines.append(current)
    return lines


def draw_ia_panel(frame, title, text, anchor="top-right", color=(255, 255, 255), width_chars=48):
    """Desenha um painel translúcido com a resposta da IA.

    `anchor`: 'top-right' (default) ou 'bottom-right'.
    """
    if not text:
        return frame

    lines = [f"== {title} =="] + _wrap_text(text, width=width_chars)
    if not lines:
        return frame

    pad = 8
    line_h = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1

    box_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        box_w = max(box_w, tw)
    box_w += pad * 2
    box_h = pad * 2 + line_h * len(lines)

    h_frame, w_frame = frame.shape[:2]
    if anchor == "bottom-right":
        x1 = w_frame - box_w - 10
        y1 = h_frame - box_h - 10
    else:
        x1 = w_frame - box_w - 10
        y1 = 10

    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = y1 + pad + 14
    for i, line in enumerate(lines):
        line_color = (0, 220, 255) if i == 0 else color
        cv2.putText(frame, line, (x1 + pad, y), font, scale, line_color, thickness, cv2.LINE_AA)
        y += line_h
    return frame


def draw_weights(frame, cows, weights):
    """Mantido para compatibilidade; desenha peso acima de cada boi."""
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
