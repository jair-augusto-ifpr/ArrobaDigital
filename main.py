import cv2

from src.camera.capture import Capture
from src.detection.yolo_detector import YoloDetector, model_path
from src.segmentation.mask_segmenter import segment_cows
from src.biometrics.measurements import extract_measurements
from src.weight_estimation.predictor import predict_weight
from src.utils.image_utils import improve_lighting, draw_boxes


def main():
    camera = Capture()

    detector = YoloDetector(model_path)

    while True:
        frame = camera.get_frame()

        if frame is None:
            print("Erro ao capturar frame")
            break

        frame = improve_lighting(frame)

        results = detector.detect(frame)

        cows = detector.filter_cows(results)

        segments = segment_cows(cows, frame)

        measurements = extract_measurements(segments)

        weights = predict_weight(measurements)
        
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

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()