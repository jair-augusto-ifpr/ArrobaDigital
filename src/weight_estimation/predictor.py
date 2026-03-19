def predict_weight(measurements_list):
    weights = []

    for m in measurements_list:
        peso = (m["area"] * 0.002)  # fórmula simples (ajustável)
        weights.append(peso)

    return weights