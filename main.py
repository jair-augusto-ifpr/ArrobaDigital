import cv2
import time
import yaml

from src.camera.capture import Capture
from src.detection.yolo_detector import YoloDetector, model_path
from src.segmentation.mask_segmenter import segment_cows
from src.biometrics.measurements import extract_measurements, calculate_scale
from src.utils.image_utils import improve_lighting, draw_boxes
from src.conversao.conversao import modelo_regressao
from database import iniciar_banco, salvar_registro

def main():
    camera = Capture().start()
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    scale = calculate_scale(dist_real_cm=config["scale"]["dist_real_cm"], dist_pixels=config["scale"]["dist_pixels"])

    iniciar_banco()
    ultimo_salvamento = 0
    intervalo_salvamento = 5
    
    detector = YoloDetector(model_path)

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Erro ao capturar frame")
            break

        frame = improve_lighting(frame)

        results = detector.detect(frame)

        cows = detector.filter_cows(results)

        segments = segment_cows(cows, frame)

        measurements = extract_measurements(segments,scale)

        processed_cows = []

        for box, m in zip(cows, measurements):
            try:
                if m["largura_m"] < 0.2 or m["area_m2"] < 0.1:
                    continue
                
                resultado = modelo_regressao(
                    largura=m["largura_m"],
                    area_dorsal=m["area_m2"]
                )

                processed_cows.append({
                    "box": box,
                    "measurement": m,
                    "resultado": resultado,
                })

            except Exception as e:
                print(f"[ERRO] {e}")

        agora = time.time()

        if processed_cows and (agora - ultimo_salvamento) >= intervalo_salvamento:
            for item in processed_cows:
                m = item["measurement"]
                resultado = item["resultado"]

                salvar_registro(
                    peso_kg=resultado.peso_estimado,
                    area_m2=m["area_m2"],
                    altura_m=m["altura_m"],
                    largura_m=m["largura_m"],
                    area_cm2=m["area_cm2"],
                    altura_cm=m["altura_cm"],
                    largura_cm=m["largura_cm"],
                )

            ultimo_salvamento = agora

        frame = draw_boxes(frame, cows)

        for item in processed_cows:
            box = item["box"]
            resultado = item["resultado"]
            x1, y1, _, _ = map(int, box.xyxy[0])

            text = f"{resultado.peso_estimado:.1f} kg"
            intervalo = f"{resultado.margem_minima:.0f}-{resultado.margem_maxima:.0f}"

            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.putText(frame, intervalo, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        cv2.imshow("Detecção de Bois", frame)

        if cv2.waitKey(1) == 27:
            break

    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
