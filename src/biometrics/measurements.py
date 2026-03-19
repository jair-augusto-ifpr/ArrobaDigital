def extract_measurements(cow_segments):
    measurements = []

    for seg in cow_segments:
        h, w, _ = seg.shape

        data = {
            "altura": h,
            "largura": w,
            "area": h * w
        }

        measurements.append(data)

    return measurements