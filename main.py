import cv2

from camera.capture import Capture
from yolo_detector import YoloDetector, model_path
from mask_segmenter import segment_cows
from measurements import extract_measurements
from predictor import predict_weight
from image_utils import improve_lighting, draw_boxes, draw_masks
from database import iniciar_banco, salvar_registro


def main():
    camera = Capture()
    scale = 0.5
    detector = YoloDetector(model_path)

    iniciar_banco()

    while True:
        frame = camera.get_frame()

        if frame is None:
            print("Erro ao capturar frame")
            break

        frame = improve_lighting(frame)
        results = detector.detect(frame)
        cows = detector.filter_cows(results)
        segments = segment_cows(cows, frame)
        measurements = extract_measurements(segments, scale)
        weights = predict_weight(measurements)

        frame = draw_boxes(frame, cows)
        frame = draw_masks(frame, segments)

        for i, seg in enumerate(segments):
            x1, y1, _, _ = seg["box"]

            if i < len(weights) and i < len(measurements):
                text = f"{weights[i]:.1f} kg"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                salvar_registro(
                    peso=weights[i],
                    area=measurements[i]["area_cm2"],
                    comprimento=measurements[i]["comprimento_cm"]
                )

        cv2.imshow("Detecção de Bois", frame)

        if cv2.waitKey(1) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()