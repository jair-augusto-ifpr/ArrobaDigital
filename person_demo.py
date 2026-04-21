"""Modo Pessoas (experimental) — pipeline separado para humanos em pé.

Executa o mesmo esqueleto do `main.py` (YOLOv8-seg + segmentação + medidas)
mas filtrando a classe COCO `person` (id 0) e usando uma fórmula específica
de peso humano (`src/person/weight.py`).

Auto-calibração: passe `--known-height-cm 175` para recalibrar a escala
automaticamente assumindo que a pessoa visível tem essa altura real.
"""

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

import argparse
import sys
import time
from datetime import datetime

import cv2
import yaml

from src.biometrics.measurements import calculate_scale, extract_measurements
from src.camera.capture import Capture
from src.detection.yolo_detector import YoloDetector, model_path
from src.person import estimar_peso_pessoa
from src.segmentation.mask_segmenter import segment_cows  # genérico p/ qualquer classe
from src.tracking import CattleAggregator
from src.utils.image_utils import draw_boxes, draw_hud, improve_lighting

PERSON_CLASS_ID = 0  # COCO


def parse_args():
    p = argparse.ArgumentParser(description="ArrobaDigital — modo Pessoas (experimental).")
    p.add_argument("--source", "-s", default="0",
                   help="Fonte: número (webcam), caminho de vídeo/imagem ou URL. Padrão: 0.")
    p.add_argument("--conf", type=float, default=0.35,
                   help="Confiança mínima do YOLO (padrão 0.35).")
    p.add_argument("--scale", type=float, default=None,
                   help="Escala cm/px manual. Se não informada, usa config.yaml.")
    p.add_argument("--known-height-cm", type=float, default=None,
                   help="Auto-calibra a escala assumindo que a pessoa detectada "
                        "tem essa altura em cm (ex.: 175). Usa o primeiro frame "
                        "válido com 1 pessoa, ou toda vez que você apertar 'c'.")
    p.add_argument("--no-track", action="store_true", help="Desativa tracking entre frames.")
    p.add_argument("--update-every", type=float, default=2.0,
                   help="Debounce (s): o peso exibido so atualiza a cada N segundos. "
                        "0 = atualiza todo frame. Padrao: 2.0s.")
    p.add_argument("--show-all", action="store_true", help="Mostra todas as classes (depuração).")
    p.add_argument("--save-video", default=None, help="Grava mp4 anotado.")
    p.add_argument("--window", default="ArrobaDigital - Pessoas")
    return p.parse_args()


def resolve_source(raw):
    if raw is None:
        return 0
    if raw.isdigit():
        return int(raw)
    return raw


def load_scale_from_config():
    path = _PROJECT_ROOT / "config.yaml"
    if not path.is_file():
        return 0.5
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    sc = cfg.get("scale") or {}
    return calculate_scale(
        dist_real_cm=sc.get("dist_real_cm", 200),
        dist_pixels=sc.get("dist_pixels", 400),
    )


def ensure_captures_dir():
    out = _PROJECT_ROOT / "captures"
    out.mkdir(exist_ok=True)
    return out


def _desenhar_pessoa(frame, box, resultado, track_id=None, amostras=0, restante_s=None):
    x1, y1, _, _ = map(int, box.xyxy[0])

    header = f"{resultado.peso_estimado:.1f} kg"
    if track_id is not None:
        header = f"#{track_id}  " + header
    faixa = f"{resultado.margem_minima:.0f}-{resultado.margem_maxima:.0f} kg"
    if amostras:
        faixa += f"  (n={amostras})"
    if restante_s is not None and restante_s > 0:
        faixa += f"  | prox {restante_s:.1f}s"
    detalhe = (
        f"H {resultado.altura_cm:.0f}cm | W {resultado.largura_cm:.0f}cm | "
        f"IMC {resultado.imc_estimado:.1f}"
    )

    cv2.putText(frame, header, (x1, max(0, y1 - 48)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, faixa, (x1, max(0, y1 - 28)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    cv2.putText(frame, detalhe, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def _maior_pessoa_px(pessoas_medidas):
    """Retorna (altura_px, largura_px) do maior contorno de pessoa na lista."""
    melhor = None
    melhor_altura = 0.0
    for m in pessoas_medidas:
        if m.get("comprimento_px", 0) > melhor_altura:
            melhor_altura = m["comprimento_px"]
            melhor = m
    if melhor is None:
        return None, None
    return melhor["comprimento_px"], melhor["largura_px"]


def _auto_calibrar(scale_atual, pessoas_medidas, known_height_cm, log=True):
    """Retorna nova escala cm/px usando a pessoa mais alta da lista."""
    altura_px, _ = _maior_pessoa_px(pessoas_medidas)
    if not altura_px or altura_px <= 0:
        if log:
            print("[CAL] Nao foi possivel calibrar — nenhuma pessoa medida.")
        return scale_atual
    nova = known_height_cm / altura_px
    if log:
        print(f"[CAL] Recalibrado: altura_px={altura_px:.0f} → scale={nova:.4f} cm/px "
              f"(assumindo {known_height_cm:.0f} cm de altura).")
    return nova


class _ResEMA:
    """Wrapper bare-bones para usar o CattleAggregator com o ResultadoPesoPessoa."""

    def __init__(self, r):
        self.peso_estimado = r.peso_estimado
        self.margem_minima = r.margem_minima
        self.margem_maxima = r.margem_maxima


def processar_frame(frame, detector, scale, show_all, use_tracking, aggregator):
    """Roda detect/track + segmentação + EMA. NÃO desenha peso (isso fica pro debounce)."""
    frame = improve_lighting(frame)
    results = detector.track(frame) if use_tracking else detector.detect(frame)
    todas = detector.detect_all(results)
    pessoas = [d for d in todas if d["cls"] == PERSON_CLASS_ID]

    segments = segment_cows(pessoas, frame)
    medidas = extract_measurements(segments, scale)

    processed = []
    for det, m in zip(pessoas, medidas):
        if m["comprimento_px"] <= 0 or m["largura_px"] <= 0:
            continue
        altura_cm = m["comprimento_cm"]
        largura_cm = m["largura_cm"]
        try:
            resultado = estimar_peso_pessoa(altura_cm=altura_cm, largura_cm=largura_cm)
        except ValueError:
            continue

        track_id = det.get("track_id")
        amostras = 0
        if aggregator is not None and track_id is not None:
            sample = aggregator.atualizar(
                track_id,
                {
                    "largura_cm": resultado.largura_cm,
                    "largura_m": resultado.largura_cm / 100.0,
                    "area_cm2": 0.0, "area_m2": 0.0,
                    "altura_cm": resultado.altura_cm,
                    "altura_m": resultado.altura_m,
                    "comprimento_cm": resultado.altura_cm,
                },
                _ResEMA(resultado),
            )
            amostras = sample.amostras
            if amostras >= 2:
                peso_s = sample.get("peso_estimado", resultado.peso_estimado)
                altura_s = sample.get("altura_cm", resultado.altura_cm)
                largura_s = sample.get("largura_cm", resultado.largura_cm)
                altura_m_s = altura_s / 100.0
                imc_s = peso_s / (altura_m_s ** 2) if altura_m_s > 0 else resultado.imc_estimado
                from src.person.weight import ResultadoPesoPessoa  # local
                resultado = ResultadoPesoPessoa(
                    peso_estimado=round(peso_s, 1),
                    margem_minima=round(peso_s * 0.85, 1),
                    margem_maxima=round(peso_s * 1.15, 1),
                    imc_estimado=round(imc_s, 1),
                    altura_cm=round(altura_s, 1),
                    largura_cm=round(largura_s, 1),
                    razao=round(largura_s / altura_s, 3) if altura_s else 0.0,
                )

        processed.append({
            "box": det["box"], "measurement": m, "resultado": resultado,
            "track_id": track_id, "amostras": amostras,
        })

    # Boxes (YOLO) desenhados em tempo real — só o peso é debounced.
    if show_all:
        outros = [d for d in todas if d["cls"] != PERSON_CLASS_ID]
        draw_boxes(frame, outros, cow_class_id=PERSON_CLASS_ID)
    draw_boxes(frame, pessoas, cow_class_id=PERSON_CLASS_ID)

    return frame, todas, pessoas, processed, medidas


def desenhar_debounced(frame, processed, display_cache, update_every, agora):
    """Atualiza `display_cache` por track_id a cada `update_every` segundos.

    Usa o resultado já suavizado pela EMA como "sample" no momento da atualização.
    Entre duas atualizações, desenha o último valor congelado (sem piscar números).
    `display_cache[track_id] = {"resultado", "atualizado_em", "amostras"}`
    """
    vistos = set()
    for item in processed:
        tid = item["track_id"]
        if tid is None:
            # Sem tracking (--no-track): debounce "global" na chave None.
            tid = "_single"
        vistos.add(tid)

        cache = display_cache.get(tid)
        precisa_atualizar = (
            cache is None
            or update_every <= 0
            or (agora - cache["atualizado_em"]) >= update_every
        )

        if precisa_atualizar:
            display_cache[tid] = {
                "resultado": item["resultado"],
                "atualizado_em": agora,
                "amostras": item["amostras"],
                "box": item["box"],
            }

        # Usa sempre o último cache, mas a BOX atual (pra acompanhar movimento).
        cached = display_cache[tid]
        restante = None
        if update_every > 0:
            restante = max(0.0, update_every - (agora - cached["atualizado_em"]))

        _desenhar_pessoa(
            frame, item["box"], cached["resultado"],
            track_id=item["track_id"], amostras=cached["amostras"],
            restante_s=restante,
        )

    # Limpa cache de tracks que sumiram do frame.
    for tid in list(display_cache.keys()):
        if tid not in vistos:
            del display_cache[tid]


def main():
    args = parse_args()
    source = resolve_source(args.source)

    if args.scale is not None and args.scale > 0:
        scale = float(args.scale)
    else:
        scale = load_scale_from_config()
    print(f"[INFO] Escala inicial: {scale:.4f} cm/px")
    if args.known_height_cm:
        print(f"[INFO] Auto-calibracao pendente: alvo={args.known_height_cm:.0f} cm")

    use_tracking = not args.no_track
    aggregator = CattleAggregator(min_amostras_para_salvar=10**9) if use_tracking else None
    # min_amostras_para_salvar gigante: não queremos persistir no banco aqui.

    print(f"[INFO] Tracking: {'ligado (ByteTrack)' if use_tracking else 'desligado'}")
    if args.update_every > 0:
        print(f"[INFO] Debounce: peso exibido atualiza a cada {args.update_every:.1f}s.")
    else:
        print("[INFO] Debounce: desligado (atualiza todo frame).")
    print(f"[INFO] Carregando modelo {model_path} (conf={args.conf})...")
    detector = YoloDetector(model_path, conf=args.conf, cow_class_id=PERSON_CLASS_ID)

    try:
        camera = Capture(src=source).start()
    except Exception as e:
        print(f"[ERRO] Nao foi possivel abrir a fonte '{source}': {e}")
        return 2

    writer = None
    show_all = args.show_all
    paused = False
    calibrado = False  # auto-calibra 1x no primeiro frame válido
    display_cache = {}  # track_id -> {resultado, atualizado_em, amostras, box}

    fps = 0.0
    fps_start = time.time()
    fps_frames = 0

    frame = None
    todas, pessoas, processed, medidas = [], [], [], []

    try:
        while True:
            if not paused:
                ret, frame = camera.read()
                if not ret or frame is None:
                    print("[INFO] Fim do video / falha na captura.")
                    break

                frame, todas, pessoas, processed, medidas = processar_frame(
                    frame, detector, scale, show_all,
                    use_tracking=use_tracking, aggregator=aggregator,
                )

                # Debounce: só atualiza o peso exibido a cada `update_every` segundos.
                desenhar_debounced(
                    frame, processed, display_cache,
                    update_every=args.update_every, agora=time.time(),
                )

                # Auto-calibração automática: na 1ª vez que aparece exatamente 1 pessoa.
                if (not calibrado) and args.known_height_cm and len(pessoas) == 1 and medidas:
                    nova = _auto_calibrar(scale, medidas, args.known_height_cm)
                    if nova != scale:
                        scale = nova
                        calibrado = True
                        if aggregator:
                            aggregator._tracks.clear()  # zera EMA para não herdar escala antiga
                        display_cache.clear()

                if aggregator is not None:
                    aggregator.limpar_expirados()

                fps_frames += 1
                now = time.time()
                elapsed = now - fps_start
                if elapsed >= 0.5:
                    fps = fps_frames / elapsed
                    fps_start = now
                    fps_frames = 0

                draw_hud(frame, {
                    "fps": fps,
                    "source": f"webcam:{source}" if isinstance(source, int) else str(source),
                    "db": "off (demo)",
                    "detections": len(todas) if show_all else len(pessoas),
                    "cows": len(pessoas),   # 'cows' aqui significa "alvos" (pessoas)
                    "tracks": len(aggregator) if aggregator else None,
                    "scale": scale,
                    "show_all": show_all,
                })

                if args.save_video:
                    if writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (w, h))
                    writer.write(frame)

                display = frame
            else:
                display = frame.copy() if frame is not None else None
                if display is not None:
                    draw_hud(display, {
                        "fps": fps, "source": f"webcam:{source}",
                        "db": "off (demo)",
                        "detections": len(todas) if show_all else len(pessoas),
                        "cows": len(pessoas),
                        "tracks": len(aggregator) if aggregator else None,
                        "scale": scale, "show_all": show_all, "paused": True,
                    })

            if display is not None:
                cv2.imshow(args.window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d"):
                show_all = not show_all
                print(f"[INFO] show_all = {show_all}")
            if key == ord("p"):
                paused = not paused
                print(f"[INFO] {'pausado' if paused else 'retomado'}")
            if key == ord("s") and display is not None:
                out_dir = ensure_captures_dir()
                out_path = out_dir / ("pessoa_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg")
                cv2.imwrite(str(out_path), display)
                print(f"[INFO] Frame salvo em {out_path}")
            if key == ord("c"):
                if not args.known_height_cm:
                    print("[CAL] Use --known-height-cm X para permitir recalibrar com 'c'.")
                elif not medidas:
                    print("[CAL] Nenhuma pessoa detectada para calibrar.")
                else:
                    scale = _auto_calibrar(scale, medidas, args.known_height_cm)
                    if aggregator:
                        aggregator._tracks.clear()
                    display_cache.clear()

    except KeyboardInterrupt:
        print("\nEncerrado pelo usuario (Ctrl+C).", file=sys.stderr)
    finally:
        try:
            camera.stop()
        except Exception:
            pass
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuario (Ctrl+C).", file=sys.stderr)
        raise SystemExit(130) from None
