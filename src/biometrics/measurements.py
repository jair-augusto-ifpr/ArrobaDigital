def calculate_scale(dist_real_cm, dist_pixels):
    return dist_real_cm / dist_pixels

def extract_measurements(cow_segments, scale):
    measurements = []

    for seg in cow_segments:
        h, w, _ = seg.shape

        altura_real = h * scale
        largura_real = w * scale
        area_real = altura_real * largura_real

        data = {
            "altura_cm": altura_real,
            "largura_cm": largura_real,
            "area_cm2": area_real
        }

        measurements.append(data)

    return measurements