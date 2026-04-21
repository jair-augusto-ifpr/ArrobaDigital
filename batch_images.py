"""Processa uma pasta de imagens em modo headless (sem janela OpenCV).

Uso:
    python batch_images.py imgs_tests/                 # pasta inteira
    python batch_images.py imgs_tests/nelore.jpeg ...  # arquivos específicos
    python batch_images.py imgs_tests/ --conf 0.2 --show-all --no-ia

Salva anotadas em `captures/annot_<nome>.jpg` e imprime um resumo por imagem.

Filosofia: o PESO é sempre calculado pelo nosso sistema (regressão ou PT²).
A IA só AUXILIA — raça provável, ECC, pelagem, observações — nunca substitui
o cálculo. Use `--ia-analise` para gerar esse laudo complementar.
"""

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

import argparse
import sys
import time

import cv2
import yaml

from src.biometrics.measurements import calculate_scale
from src.detection.yolo_detector import YoloDetector, model_path
from src.ia import IAClient, IAError, load_ia_config
from src.ia.visao import analisar_boi

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Batch ArrobaDigital — processa imagens sem abrir janela.")
    p.add_argument("paths", nargs="+", help="Pasta(s) ou arquivo(s) de imagem.")
    p.add_argument("--conf", type=float, default=None, help="Confiança mínima YOLO (sobrescreve config.yaml).")
    p.add_argument("--show-all", action="store_true", help="Desenha todas as classes, não só bois.")
    p.add_argument("--out", default="captures", help="Diretório de saída (default: captures/).")
    p.add_argument("--no-ia", action="store_true", help="Não chama a IA (mais rápido, sem internet).")
    p.add_argument("--ia-analise", action="store_true",
                   help="Pede ao LLM multimodal um laudo COMPLEMENTAR (raça provável, ECC, "
                        "pelagem, observações). NÃO calcula peso — isso é sempre feito "
                        "pelo nosso sistema.")
    p.add_argument("--ia-delay", type=float, default=0.0,
                   help="Espera N segundos entre chamadas da IA (útil em tiers com rate-limit). "
                        "Padrão: 0 (gpt-4o-mini não precisa).")
    p.add_argument("--mode", choices=("cow", "person"), default="cow",
                   help="cow = pipeline bovino (main.py); person = pipeline humano (person_demo).")
    p.add_argument("--known-length-cm", type=float, default=None,
                   help="Calibra a escala por imagem assumindo que o maior boi "
                        "tem esse comprimento em cm (ex.: 250 para Nelore adulto). "
                        "Só faz sentido no --mode cow.")
    p.add_argument("--known-height-cm", type=float, default=None,
                   help="Calibra a escala por imagem assumindo que a pessoa "
                        "tem essa altura em cm (ex.: 175). Só no --mode person.")
    p.add_argument("--ground-truth", default=None,
                   help="Caminho para YAML com pesos reais por nome de arquivo "
                        "(chave 'pesos'). Se fornecido, imprime erro por imagem.")
    return p.parse_args()


def load_ground_truth(path):
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        print(f"[WARN] ground_truth {path} nao encontrado — ignorando.")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(k): float(v) for k, v in (data.get("pesos") or {}).items()}


def collect_images(paths):
    out = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in IMG_EXTS:
                    out.append(f)
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
        else:
            print(f"[WARN] ignorando {raw} (nao e imagem valida)")
    return out


def load_config():
    with open(_PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _process_cow(frame, detector, scale, min_largura_m, min_area_m2, show_all):
    from main import processar_frame
    return processar_frame(
        frame, detector, scale, min_largura_m, min_area_m2, show_all,
        use_tracking=False, aggregator=None, raca_config_default=None,
    )


def _process_person(frame, detector, scale, show_all):
    from person_demo import processar_frame, desenhar_debounced
    frame, todas, pessoas, processed, medidas = processar_frame(
        frame, detector, scale, show_all,
        use_tracking=False, aggregator=None,
    )
    desenhar_debounced(frame, processed, {}, update_every=0.0, agora=time.time())
    return frame, todas, pessoas, processed


def _calibrar_por_imagem(frame, detector, mode, known_cm):
    """Mede o maior alvo em pixels e recalcula cm/px pra bater com `known_cm`."""
    from src.biometrics.measurements import extract_measurements
    from src.segmentation.mask_segmenter import segment_cows
    target_cls = 19 if mode == "cow" else 0
    results = detector.detect(frame)
    todas = detector.detect_all(results)
    alvos = [d for d in todas if d["cls"] == target_cls]
    if not alvos:
        return None
    segs = segment_cows(alvos, frame)
    medidas = extract_measurements(segs, scale=1.0)
    if not medidas:
        return None
    maior_px = max((m["comprimento_cm"] for m in medidas), default=0)
    if maior_px <= 0:
        return None
    return known_cm / maior_px


def main():
    args = parse_args()
    imgs = collect_images(args.paths)
    if not imgs:
        print("[ERRO] Nenhuma imagem valida encontrada.")
        return 2

    cfg = load_config()
    proc = (cfg.get("processing") or {})
    sc = cfg.get("scale") or {}
    scale = calculate_scale(
        dist_real_cm=sc.get("dist_real_cm", 200),
        dist_pixels=sc.get("dist_pixels", 400),
    )
    conf = args.conf if args.conf is not None else proc.get("conf", 0.25)
    min_largura_m = float(proc.get("min_largura_m", 0.1))
    min_area_m2 = float(proc.get("min_area_m2", 0.02))

    if args.mode == "cow":
        from main import processar_frame  # noqa: F401 (pré-valida import)
        detector = YoloDetector(model_path, conf=conf)
    else:
        detector = YoloDetector(model_path, conf=conf, cow_class_id=0)

    ia_client = None
    raca_focus = proc.get("breed_focus")
    if args.ia_analise and not args.no_ia:
        if args.mode != "cow":
            print("[WARN] --ia-analise so e suportado no --mode cow — ignorando.")
        else:
            cfg_ia = load_ia_config()
            if not cfg_ia.disponivel:
                print("[WARN] --ia-analise pedido, mas API_KEY_IA ausente — desativando.")
            else:
                ia_client = IAClient(cfg_ia)
                print(f"[INFO] IA (auxiliar): ligada ({cfg_ia.model})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    known_cm = args.known_length_cm if args.mode == "cow" else args.known_height_cm
    ref_nome = "comprimento" if args.mode == "cow" else "altura"

    gt = load_ground_truth(args.ground_truth)

    print(f"[INFO] Modo: {args.mode} | conf={conf} | scale inicial={scale:.4f} cm/px")
    if known_cm:
        print(f"[INFO] Calibragem por imagem: {ref_nome} assumido = {known_cm:.0f} cm")
    if gt:
        print(f"[INFO] Ground truth carregado: {len(gt)} pesos reais.")
    print(f"[INFO] Processando {len(imgs)} imagem(ns). Saida em '{out_dir}/'")

    resumo = []
    for i, path in enumerate(imgs, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"\n[{i}/{len(imgs)}] {path.name}: falha ao ler.")
            continue
        h, w = frame.shape[:2]

        scale_img = scale
        if known_cm:
            novo = _calibrar_por_imagem(frame, detector, args.mode, known_cm)
            if novo:
                scale_img = novo

        if args.mode == "cow":
            frame_out, todas, cows, processed = _process_cow(
                frame, detector, scale_img, min_largura_m, min_area_m2, args.show_all,
            )
            alvo_nome = "bois"
            alvos = cows
        else:
            frame_out, todas, pessoas, processed = _process_person(
                frame, detector, scale_img, args.show_all,
            )
            alvo_nome = "pessoas"
            alvos = pessoas

        destino = out_dir / f"annot_{path.stem}.jpg"
        cv2.imwrite(str(destino), frame_out)

        print(f"\n[{i}/{len(imgs)}] {path.name} ({w}x{h})  scale={scale_img:.4f} cm/px"
              + ("  [calibrada]" if known_cm else ""))
        print(f"  - Deteccoes: {len(todas)} | {alvo_nome} detectados: {len(alvos)} | "
              f"com medida valida: {len(processed)}")
        for j, item in enumerate(processed, 1):
            r = item["resultado"]
            m = item["measurement"]
            if args.mode == "cow":
                print(
                    f"    #{j}  peso={r.peso_estimado:6.1f} kg  "
                    f"({r.margem_minima:.0f}-{r.margem_maxima:.0f} kg)  "
                    f"L={m['largura_cm']:.0f}cm  A={m['area_cm2']:.0f}cm2  "
                    f"Comp={m['comprimento_cm']:.0f}cm"
                )
            else:
                print(
                    f"    #{j}  peso={r.peso_estimado:6.1f} kg  "
                    f"({r.margem_minima:.0f}-{r.margem_maxima:.0f} kg)  "
                    f"H={r.altura_cm:.0f}cm  W={r.largura_cm:.0f}cm  IMC={r.imc_estimado:.1f}"
                )
        print(f"  -> {destino}")

        pesos_pred = [round(it["resultado"].peso_estimado, 1) for it in processed]
        peso_real = gt.get(path.name)

        analise = None
        if ia_client is not None and processed:
            primeiro = processed[0]
            try:
                analise = analisar_boi(
                    ia_client, frame,  # imagem inteira = mais contexto
                    medida=primeiro["measurement"],
                    resultado=primeiro["resultado"],
                    raca_config=raca_focus,
                )
                print(f"    [IA] {analise.resumo_uma_linha()}")
                if analise.observacoes:
                    print(f"         obs: {analise.observacoes}")
            except IAError as e:
                print(f"    [IA] erro: {e}")
            except Exception as e:
                print(f"    [IA] erro inesperado: {e}")
            if args.ia_delay > 0 and i < len(imgs):
                time.sleep(args.ia_delay)

        resumo.append({
            "nome": path.name,
            "alvos": len(alvos),
            "processed": len(processed),
            "pesos": pesos_pred,
            "peso_real": peso_real,
            "analise": analise,
        })

    print("\n===== Resumo =====")
    if gt:
        print(f"  {'imagem':<22}{'real':>8}{'pred':>10}{'err':>10}  raca_IA")
        erros = []
        for r in resumo:
            real = r["peso_real"]
            pred = r["pesos"][0] if r["pesos"] else None
            linha = f"  {r['nome']:<22}"
            linha += f"{real:>6.0f}kg" if real is not None else f"{'?':>8}"
            linha += f"{pred:>8.0f}kg" if pred is not None else f"{'-':>10}"
            if real is not None and pred is not None:
                pct = 100 * (pred - real) / real
                erros.append(abs(pct))
                linha += f"{pct:>+8.1f}%"
            else:
                linha += f"{'-':>10}"
            a = r.get("analise")
            if a is not None:
                linha += f"  {a.raca_provavel} ({a.confianca_raca:.0%})"
            print(linha)

        if erros:
            mape = sum(erros) / len(erros)
            print(f"\n  MAPE (nosso sistema) = {mape:.1f} %  (n={len(erros)})")
            print("  OBS: o sistema usa regressao linear com coeficientes placeholder.")
            print("       Para melhorar, calibrar com dados reais do rebanho ou usar PT².")
    else:
        for r in resumo:
            pesos = ", ".join(f"{p:.1f}kg" for p in r["pesos"]) or "-"
            linha = f"  {r['nome']}: {r['alvos']} alvo(s), {r['processed']} processado(s) -> {pesos}"
            a = r.get("analise")
            if a is not None:
                linha += f"  | {a.resumo_uma_linha()}"
            print(linha)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuario (Ctrl+C).", file=sys.stderr)
        raise SystemExit(130) from None
