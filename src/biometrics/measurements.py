def calculate_scale(dist_real_cm, dist_pixels):
    if dist_pixels <= 0:
        raise ValueError("dist_pixels deve ser maior que zero para calcular a escala.")

    return dist_real_cm / dist_pixels

def extract_measurements(cow_segments, scale):
    measurements = []

    for seg in cow_segments:
        h, w, _ = seg.shape

        altura_cm = h * scale
        largura_cm = w * scale
        area_cm2 = altura_cm * largura_cm

        altura_m = altura_cm / 100
        largura_m = largura_cm / 100
        area_m2 = area_cm2 / 10000

        data = {
            "altura_cm": altura_cm,
            "largura_cm": largura_cm,
            "area_cm2": area_cm2,
            "altura_m": altura_m,
            "largura_m": largura_m,
            "area_m2": area_m2,
        }

        measurements.append(data)

    return measurements