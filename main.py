from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
# Carrega .env antes de importar database (evita DATABASE_URL vazia no import).
load_dotenv(_PROJECT_ROOT / ".env", override=True)

import argparse
import sys
import threading
import time
from datetime import datetime

import cv2
import yaml

from src.biometrics.measurements import calculate_scale, extract_measurements
from src.camera.capture import Capture
from src.conversao.conversao import estimar_peso
from src.detection.yolo_detector import YoloDetector, model_path
from src.ia import IAClient, IAError, load_ia_config
from src.ia.laudo import gerar_laudo
from src.ia.relatorio import gerar_relatorio_lote
from src.ia.visao import analisar_boi
from src.segmentation.mask_segmenter import segment_cows
from src.tracking import CattleAggregator
from src.utils.image_utils import draw_boxes, draw_hud, draw_ia_panel, improve_lighting
import database
from database import iniciar_banco, salvar_registro, ultimos_registros

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="ArrobaDigital - detecção e estimativa de peso bovino.")
    parser.add_argument(
        "--source", "-s", default="0",
        help="Fonte de vídeo: número (webcam), caminho de vídeo/imagem ou URL. Padrão: 0 (webcam).",
    )
    parser.add_argument("--conf", type=float, default=None, help="Confiança mínima do YOLO (sobrescreve config.yaml).")
    parser.add_argument("--show-all", action="store_true", help="Desenha todas as classes detectadas, não só 'cow'.")
    parser.add_argument("--save-video", default=None, help="Caminho para salvar o vídeo anotado (mp4).")
    parser.add_argument("--no-db", action="store_true", help="Desativa a tentativa de gravar no banco.")
    parser.add_argument("--no-ia", action="store_true", help="Desativa a integração com LLM (OpenRouter).")
    parser.add_argument(
        "--no-track", action="store_true",
        help="Desativa o tracking entre frames (volta ao modo só-detecção).",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Em vez de rodar a câmera, gera um relatório do lote a partir do banco e sai.",
    )
    parser.add_argument(
        "--report-limit", type=int, default=30,
        help="Número de registros considerados pelo --report (padrão: 30).",
    )
    parser.add_argument("--window", default="ArrobaDigital", help="Nome da janela do OpenCV.")
    return parser.parse_args()


def resolve_source(raw):
    if raw is None:
        return 0
    if raw.isdigit():
        return int(raw)
    return raw


def is_image_path(source):
    if not isinstance(source, str):
        return False
    ext = Path(source).suffix.lower()
    return ext in IMG_EXTS and Path(source).is_file()


def source_label(source):
    if isinstance(source, int):
        return f"webcam:{source}"
    return source


def load_config():
    with open(_PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_captures_dir():
    out = _PROJECT_ROOT / "captures"
    out.mkdir(exist_ok=True)
    return out


class IAState:
    """Gerencia chamadas à IA em background e guarda a última resposta visível."""

    def __init__(self, client, enabled, raca_config, aggregator: CattleAggregator = None):
        self.client = client
        self.enabled = enabled
        self.raca_config = raca_config
        self.aggregator = aggregator
        self.lock = threading.Lock()
        self.laudo_text = None
        self.laudo_pending = False
        self.visao_text = None
        self.visao_pending = False

    def _set_laudo(self, txt):
        with self.lock:
            self.laudo_text = txt
            self.laudo_pending = False

    def _set_visao(self, txt):
        with self.lock:
            self.visao_text = txt
            self.visao_pending = False

    def _raca_efetiva(self, track_id):
        if self.aggregator is not None and track_id is not None:
            r = self.aggregator.raca(track_id)
            if r:
                return r
        return self.raca_config

    def solicitar_laudo(self, medida, resultado, track_id=None):
        if not self.enabled:
            self._set_laudo("IA desativada (--no-ia ou API_KEY_IA ausente).")
            return
        if self.laudo_pending:
            return
        with self.lock:
            self.laudo_pending = True
            self.laudo_text = "gerando laudo..."
        raca = self._raca_efetiva(track_id)

        def worker():
            try:
                txt = gerar_laudo(self.client, medida, resultado, raca_config=raca)
            except IAError as e:
                txt = f"[erro IA] {e}"
            except Exception as e:
                txt = f"[erro IA inesperado] {e}"
            self._set_laudo(txt)
            print(f"[IA/laudo] track={track_id} raca={raca}\n{txt}\n")

        threading.Thread(target=worker, daemon=True).start()

    def solicitar_visao(self, crop, medida, resultado, track_id=None):
        if not self.enabled:
            self._set_visao("IA desativada (--no-ia ou API_KEY_IA ausente).")
            return
        if self.visao_pending:
            return
        if crop is None or getattr(crop, "size", 0) == 0:
            self._set_visao("Crop indisponivel para analise visual.")
            return
        with self.lock:
            self.visao_pending = True
            self.visao_text = "analisando imagem..."
        crop_copy = crop.copy()
        raca_fallback = self._raca_efetiva(track_id)

        def worker():
            try:
                analise = analisar_boi(
                    self.client, crop_copy,
                    medida=medida, resultado=resultado, raca_config=raca_fallback,
                )
                # Raça dinâmica: memoriza no aggregator para o próximo laudo/salvamento.
                if self.aggregator is not None and track_id is not None:
                    self.aggregator.registrar_raca(
                        track_id,
                        analise.raca_provavel,
                        confianca=analise.confianca_raca,
                        ecc=analise.ecc,
                    )
                txt = analise.resumo_uma_linha()
                if analise.observacoes:
                    txt += f"\n{analise.observacoes}"
            except IAError as e:
                txt = f"[erro IA] {e}"
            except Exception as e:
                txt = f"[erro IA inesperado] {e}"
            self._set_visao(txt)
            print(f"[IA/visao] track={track_id}\n{txt}\n")

        threading.Thread(target=worker, daemon=True).start()


def _desenhar_peso(frame, box, peso_kg, min_kg, max_kg, medida, track_id=None, raca=None, amostras=0):
    x1, y1, _, _ = map(int, box.xyxy[0])
    header = f"{peso_kg:.1f} kg"
    if track_id is not None:
        header = f"#{track_id}  " + header
    intervalo = f"{min_kg:.0f}-{max_kg:.0f} kg  (n={amostras})" if amostras else f"{min_kg:.0f}-{max_kg:.0f} kg"
    medidas = f"L {medida['largura_cm']:.0f}cm | A {medida['area_cm2']:.0f}cm2"
    if raca:
        medidas += f" | {raca}"

    cv2.putText(frame, header, (x1, max(0, y1 - 42)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, intervalo, (x1, max(0, y1 - 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, medidas, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def processar_frame(
    frame,
    detector,
    scale,
    min_largura_m,
    min_area_m2,
    show_all,
    use_tracking=True,
    aggregator: CattleAggregator = None,
    raca_config_default=None,
):
    """Detecta/tracka bois, calcula medidas e peso (suavizado por track quando disponível).

    Retorna (frame_anotado, todas, cows, processed).
    Cada `processed` inclui: box, measurement, resultado, crop, track_id, raca_usada, amostras.
    """
    frame = improve_lighting(frame)

    results = detector.track(frame) if use_tracking else detector.detect(frame)
    todas = detector.detect_all(results)
    cows = [d for d in todas if d["cls"] == detector.cow_class_id]

    segments = segment_cows(cows, frame)
    measurements = extract_measurements(segments, scale)

    processed_cows = []
    for det, seg, m in zip(cows, segments, measurements):
        try:
            if m["largura_m"] < min_largura_m or m["area_m2"] < min_area_m2:
                continue

            track_id = det.get("track_id")
            # Raça preferencial: a que a análise visual já deu pra este track.
            raca_usada = (
                aggregator.raca(track_id) if (aggregator and track_id is not None) else None
            ) or raca_config_default

            resultado = estimar_peso(
                largura=m["largura_m"],
                area_dorsal=m["area_m2"],
                raca=raca_usada,
            )

            amostras = 0
            if aggregator is not None and track_id is not None:
                sample = aggregator.atualizar(track_id, m, resultado)
                amostras = sample.amostras
                # Substitui por valores suavizados da EMA (somente se já há >=2 amostras).
                if amostras >= 2:
                    class _R:
                        peso_estimado = sample.get("peso_estimado", resultado.peso_estimado)
                        margem_minima = sample.get("peso_min", resultado.margem_minima)
                        margem_maxima = sample.get("peso_max", resultado.margem_maxima)
                        modelo = resultado.modelo
                        formula = resultado.formula
                    resultado = _R()

            processed_cows.append({
                "box": det["box"],
                "measurement": m,
                "resultado": resultado,
                "crop": seg.get("crop"),
                "track_id": track_id,
                "raca_usada": raca_usada,
                "amostras": amostras,
            })
        except Exception as e:
            print(f"[ERRO] {e}")

    if show_all:
        outros = [d for d in todas if d["cls"] != detector.cow_class_id]
        draw_boxes(frame, outros, cow_class_id=detector.cow_class_id)
    draw_boxes(frame, cows, cow_class_id=detector.cow_class_id)

    for item in processed_cows:
        _desenhar_peso(
            frame,
            item["box"],
            item["resultado"].peso_estimado,
            item["resultado"].margem_minima,
            item["resultado"].margem_maxima,
            item["measurement"],
            track_id=item["track_id"],
            raca=item["raca_usada"],
            amostras=item["amostras"],
        )

    return frame, todas, cows, processed_cows


def _overlay_ia(frame, ia_state):
    if ia_state.laudo_text:
        draw_ia_panel(frame, "Laudo (IA)", ia_state.laudo_text, anchor="top-right")
    if ia_state.visao_text:
        draw_ia_panel(frame, "Analise visual (IA)", ia_state.visao_text, anchor="bottom-right")
    return frame


def run_report(args, raca_config):
    if args.no_ia:
        print("[ERRO] --report precisa da IA; não use com --no-ia.")
        return 2

    cfg = load_ia_config()
    if not cfg.disponivel:
        print("[ERRO] API_KEY_IA ausente no .env.")
        return 2

    iniciar_banco()
    if not database.DB_DISPONIVEL:
        print("[ERRO] Banco indisponível — não há registros para resumir.")
        return 2

    registros = ultimos_registros(limit=args.report_limit)
    if not registros:
        print("[INFO] Nenhum registro encontrado no banco.")
        return 0

    print(f"[INFO] Considerando {len(registros)} registros. Chamando {cfg.model}...")
    client = IAClient(cfg)
    try:
        texto = gerar_relatorio_lote(client, registros, raca_config=raca_config)
    except IAError as e:
        print(f"[ERRO IA] {e}")
        return 3
    print("\n===== Relatório do lote (IA) =====\n")
    print(texto)
    print("\n==================================\n")
    return 0


def run_image(
    path, detector, scale, min_largura_m, min_area_m2,
    show_all, window, ia_state, raca_config,
):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERRO] Não consegui abrir a imagem: {path}")
        return 1
    # Em imagem estática não faz sentido ativar tracking.
    frame, todas, cows, processed = processar_frame(
        frame, detector, scale, min_largura_m, min_area_m2, show_all,
        use_tracking=False, aggregator=None, raca_config_default=raca_config,
    )
    draw_hud(frame, {
        "source": path,
        "db": "off",
        "detections": len(todas) if show_all else len(cows),
        "cows": len(cows),
        "scale": scale,
        "show_all": show_all,
    })
    print(f"[INFO] Bois detectados: {len(cows)} | com medida válida: {len(processed)}")

    if processed and ia_state.enabled:
        print("[INFO] Gerando laudo + análise visual (IA)...")
        ia_state.solicitar_laudo(
            processed[0]["measurement"], processed[0]["resultado"],
            track_id=processed[0]["track_id"],
        )
        ia_state.solicitar_visao(
            processed[0]["crop"], processed[0]["measurement"], processed[0]["resultado"],
            track_id=processed[0]["track_id"],
        )

    print("[INFO] Teclas: 'l'=laudo, 'i'=analise visual, 's'=salvar, ESC/q=sair.")
    try:
        while True:
            view = frame.copy()
            _overlay_ia(view, ia_state)
            cv2.imshow(window, view)
            key = cv2.waitKey(50) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                out_dir = ensure_captures_dir()
                out_path = out_dir / (datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg")
                cv2.imwrite(str(out_path), view)
                print(f"[INFO] Frame salvo em {out_path}")
            if processed and key == ord("l"):
                ia_state.solicitar_laudo(
                    processed[0]["measurement"], processed[0]["resultado"],
                    track_id=processed[0]["track_id"],
                )
            if processed and key == ord("i"):
                ia_state.solicitar_visao(
                    processed[0]["crop"], processed[0]["measurement"], processed[0]["resultado"],
                    track_id=processed[0]["track_id"],
                )
    finally:
        cv2.destroyAllWindows()
    return 0


def main():
    args = parse_args()

    config = load_config()
    proc = (config.get("processing") or {})
    scale_cfg = config.get("scale") or {}
    raca_config = proc.get("breed_focus")

    if args.report:
        return run_report(args, raca_config)

    source = resolve_source(args.source)

    conf = args.conf if args.conf is not None else proc.get("conf", 0.25)
    min_largura_m = float(proc.get("min_largura_m", 0.1))
    min_area_m2 = float(proc.get("min_area_m2", 0.02))

    scale = calculate_scale(
        dist_real_cm=scale_cfg.get("dist_real_cm", 200),
        dist_pixels=scale_cfg.get("dist_pixels", 400),
    )
    print(f"[INFO] Escala: {scale:.4f} cm/px (dist_real_cm/dist_pixels em config.yaml)")

    ia_cfg = load_ia_config()
    ia_enabled = (not args.no_ia) and ia_cfg.disponivel
    if args.no_ia:
        ia_status = "off (--no-ia)"
    elif not ia_cfg.disponivel:
        ia_status = "indisponivel (sem API_KEY_IA)"
    else:
        ia_status = f"ok ({ia_cfg.model})"
    print(f"[INFO] IA: {ia_status}")

    use_tracking = not args.no_track
    aggregator = CattleAggregator() if use_tracking else None
    ia_client = IAClient(ia_cfg) if ia_enabled else None
    ia_state = IAState(
        ia_client, enabled=ia_enabled, raca_config=raca_config, aggregator=aggregator,
    )

    print(f"[INFO] Tracking: {'ligado (ByteTrack)' if use_tracking else 'desligado (--no-track)'}")

    print(f"[INFO] Carregando modelo {model_path} (conf={conf})...")
    detector = YoloDetector(model_path, conf=conf)
    print(f"[INFO] Classes do modelo: {len(detector.names)} (ex.: cow id=19)")

    db_ok = False
    if not args.no_db:
        iniciar_banco()
        db_ok = database.DB_DISPONIVEL
    db_status = "ok" if db_ok else ("off" if args.no_db else "indisponivel")

    if is_image_path(source):
        return run_image(
            source, detector, scale, min_largura_m, min_area_m2,
            show_all=args.show_all, window=args.window,
            ia_state=ia_state, raca_config=raca_config,
        )

    try:
        camera = Capture(src=source).start()
    except Exception as e:
        print(f"[ERRO] Não foi possível abrir a fonte '{source}': {e}")
        return 2

    writer = None
    show_all = args.show_all
    paused = False
    last_log = (-1, -1)

    fps = 0.0
    fps_accum_start = time.time()
    fps_frames = 0

    frame = None
    todas, cows, processed = [], [], []

    try:
        while True:
            if not paused:
                ret, frame = camera.read()
                if not ret or frame is None:
                    print("[INFO] Fim do vídeo ou falha ao capturar frame.")
                    break

                frame, todas, cows, processed = processar_frame(
                    frame, detector, scale, min_largura_m, min_area_m2, show_all,
                    use_tracking=use_tracking,
                    aggregator=aggregator,
                    raca_config_default=raca_config,
                )

                # Política de salvamento: 1x por track após EMA madura, com cooldown.
                if aggregator is not None and db_ok:
                    for item in processed:
                        tid = item["track_id"]
                        if tid is None:
                            continue
                        if aggregator.deve_salvar(tid):
                            m = item["measurement"]
                            r = item["resultado"]
                            salvar_registro(
                                peso_kg=r.peso_estimado,
                                area_m2=m["area_m2"],
                                altura_m=m["altura_m"],
                                largura_m=m["largura_m"],
                                area_cm2=m["area_cm2"],
                                altura_cm=m["altura_cm"],
                                largura_cm=m["largura_cm"],
                            )
                            aggregator.marcar_salvo(tid)
                elif db_ok and processed:
                    # Sem tracking: mantém o comportamento antigo (5s global).
                    now = time.time()
                    if now - getattr(main, "_last_save", 0) >= 5.0:
                        for item in processed:
                            m = item["measurement"]
                            r = item["resultado"]
                            salvar_registro(
                                peso_kg=r.peso_estimado, area_m2=m["area_m2"],
                                altura_m=m["altura_m"], largura_m=m["largura_m"],
                                area_cm2=m["area_cm2"], altura_cm=m["altura_cm"],
                                largura_cm=m["largura_cm"],
                            )
                        main._last_save = now

                # Limpa tracks expirados
                if aggregator is not None:
                    aggregator.limpar_expirados()

                fps_frames += 1
                now = time.time()
                elapsed = now - fps_accum_start
                if elapsed >= 0.5:
                    fps = fps_frames / elapsed
                    fps_accum_start = now
                    fps_frames = 0

                sig = (len(todas) if show_all else len(cows), len(cows))
                if sig != last_log:
                    tracks_ativos = len(aggregator) if aggregator else 0
                    print(
                        f"[INFO] Deteccoes={sig[0]} | Bois={sig[1]} | "
                        f"Processados={len(processed)} | Tracks ativos={tracks_ativos}"
                    )
                    last_log = sig

                draw_hud(frame, {
                    "fps": fps,
                    "source": source_label(source),
                    "db": db_status,
                    "ia": ia_status,
                    "detections": len(todas) if show_all else len(cows),
                    "cows": len(cows),
                    "tracks": len(aggregator) if aggregator else None,
                    "scale": scale,
                    "show_all": show_all,
                    "paused": paused,
                })

                if args.save_video:
                    if writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (w, h))
                        print(f"[INFO] Salvando vídeo anotado em {args.save_video}")
                    writer.write(frame)

                display = frame
            else:
                display = frame.copy() if frame is not None else None
                if display is not None:
                    draw_hud(display, {
                        "fps": fps,
                        "source": source_label(source),
                        "db": db_status,
                        "ia": ia_status,
                        "detections": len(todas) if show_all else len(cows),
                        "cows": len(cows),
                        "tracks": len(aggregator) if aggregator else None,
                        "scale": scale,
                        "show_all": show_all,
                        "paused": True,
                    })

            view = None
            if display is not None:
                view = display.copy()
                _overlay_ia(view, ia_state)
                cv2.imshow(args.window, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d"):
                show_all = not show_all
                print(f"[INFO] show_all = {show_all}")
            if key == ord("p"):
                paused = not paused
                print(f"[INFO] {'pausado' if paused else 'retomado'}")
            if key == ord("s") and view is not None:
                out_dir = ensure_captures_dir()
                out_path = out_dir / (datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg")
                cv2.imwrite(str(out_path), view)
                print(f"[INFO] Frame salvo em {out_path}")
            if key == ord("l") and processed:
                ia_state.solicitar_laudo(
                    processed[0]["measurement"], processed[0]["resultado"],
                    track_id=processed[0]["track_id"],
                )
            if key == ord("i") and processed:
                ia_state.solicitar_visao(
                    processed[0]["crop"], processed[0]["measurement"], processed[0]["resultado"],
                    track_id=processed[0]["track_id"],
                )

    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário (Ctrl+C).", file=sys.stderr)
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
        print("\nEncerrado pelo usuário (Ctrl+C).", file=sys.stderr)
        raise SystemExit(130) from None
