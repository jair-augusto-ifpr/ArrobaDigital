import cv2


def calculate_scale(dist_real_cm, dist_pixels):
    """
    Retorna quantos centímetros equivalem a 1 pixel.
    Usar um objeto de referência de tamanho conhecido na cena para calibrar.
    """
    if dist_pixels <= 0:
        raise ValueError("dist_pixels deve ser maior que zero para calcular a escala.")
    return dist_real_cm / dist_pixels


def extract_measurements(cow_segments, scale):
    """
    Recebe lista de dicts gerados pelo mask_segmenter:
      {"crop", "mask", "mask_full", "box"}
    Para cada boi calcula:
      - area_px / area_cm2        : área dorsal (pixels do animal)
      - comprimento_px / cm       : eixo longo do retângulo orientado
      - largura_px / cm           : eixo curto (largura do dorso/quadril)
      - contour                   : contorno para debug/visualização
    Conversão: valor_cm = valor_px * scale
    Inclui também largura_m, area_m2, altura_cm/m (comprimento como proxy) para o pipeline legado.
    """
    measurements = []

    for seg in cow_segments:
        mask = seg["mask"]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            measurements.append({
                "area_px": 0,
                "comprimento_px": 0,
                "largura_px": 0,
                "area_cm2": 0,
                "comprimento_cm": 0,
                "largura_cm": 0,
                "largura_m": 0,
                "area_m2": 0,
                "altura_cm": 0,
                "altura_m": 0,
                "contour": None,
            })
            continue

        contour = max(contours, key=cv2.contourArea)

        area_px = cv2.contourArea(contour)

        rect = cv2.minAreaRect(contour)
        (_, _), (w_px, h_px), _ = rect
        comprimento_px = max(w_px, h_px)
        largura_px     = min(w_px, h_px)

        area_cm2       = area_px        * (scale ** 2)
        comprimento_cm = comprimento_px * scale
        largura_cm     = largura_px     * scale

        largura_m = largura_cm / 100.0
        area_m2 = area_cm2 / 10000.0
        altura_cm = comprimento_cm
        altura_m = comprimento_cm / 100.0

        print(
            f"[Medidas] "
            f"Area: {area_px:.0f} px | {area_cm2:.2f} cm²  |  "
            f"Comprimento: {comprimento_px:.0f} px | {comprimento_cm:.2f} cm  |  "
            f"Largura: {largura_px:.0f} px | {largura_cm:.2f} cm"
        )

        measurements.append({
            "area_px":        area_px,
            "comprimento_px": comprimento_px,
            "largura_px":     largura_px,

            "area_cm2":       area_cm2,
            "comprimento_cm": comprimento_cm,
            "largura_cm":     largura_cm,

            "largura_m":      largura_m,
            "area_m2":        area_m2,
            "altura_cm":      altura_cm,
            "altura_m":       altura_m,

            "contour":        contour,
        })

    return measurements
