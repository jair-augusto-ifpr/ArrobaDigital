def predict_weight(measurements_list):
    weights = []

    for m in measurements_list:
        peso = (m["area_cm2"] * 0.005)  # fórmula simples (ajustável)
        weights.append(peso)

    return weights