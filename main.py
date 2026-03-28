import cv2
import yaml

from src.camera.capture import Capture
from src.detection.yolo_detector import YoloDetector, model_path
from src.segmentation.mask_segmenter import segment_cows
from src.biometrics.measurements import extract_measurements, calculate_scale
from src.weight_estimation.predictor import predict_weight
from src.utils.image_utils import improve_lighting, draw_boxes
from src.conversao.conversao import modelo_regressao

def main():
    camera = Capture().start()
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    scale = calculate_scale(dist_real_cm=config["scale"]["dist_real_cm"], dist_pixels=config["scale"]["dist_pixels"])

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

        weights = []

        for m in measurements:
            resultado = modelo_regressao(
                largura=m["largura_m"],
                area_dorsal=m["area_m2"]
            )
            weights.append(resultado.peso_estimado)

        frame = draw_boxes(frame, cows)

        for i, box in enumerate(cows):
            x1, y1, _, _ = map(int, box.xyxy[0])

            if i < len(weights):
                text = f"{weights[i]:.1f} kg"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Detecção de Bois", frame)

        if cv2.waitKey(1) == 27:
            break

    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
