import numpy as np
import pytest

from src.biometrics.measurements import calculate_scale, extract_measurements


def _segment_from_mask(mask):
    """Monta um dict compatível com o que `segment_cows` geraria."""
    h, w = mask.shape
    return {
        "crop": np.zeros((h, w, 3), dtype=np.uint8),
        "mask": mask,
        "mask_full": mask,
        "box": (0, 0, w, h),
    }


def test_calculate_scale_ok():
    assert calculate_scale(dist_real_cm=200, dist_pixels=400) == pytest.approx(0.5)


def test_calculate_scale_rejeita_zero():
    with pytest.raises(ValueError):
        calculate_scale(dist_real_cm=200, dist_pixels=0)


def test_extract_measurements_sem_contorno():
    mask = np.zeros((50, 50), dtype=np.uint8)  # máscara vazia
    out = extract_measurements([_segment_from_mask(mask)], scale=0.5)
    assert len(out) == 1
    m = out[0]
    assert m["area_px"] == 0
    assert m["largura_cm"] == 0
    assert m["comprimento_cm"] == 0


def test_extract_measurements_retangulo_conhecido():
    # Retângulo 100x200 px preenchido de brancos em uma máscara 300x400.
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[50:250, 150:250] = 255  # altura=200, largura=100
    scale = 0.5  # 0.5 cm/px
    m = extract_measurements([_segment_from_mask(mask)], scale=scale)[0]

    # Área: 200*100 = 20_000 px -> 20_000 * 0.25 = 5_000 cm²
    assert m["area_px"] == pytest.approx(200 * 100, rel=0.02)
    assert m["area_cm2"] == pytest.approx(m["area_px"] * scale ** 2, rel=0.02)
    # Comprimento (eixo longo) = 200 px * 0.5 = 100 cm
    assert m["comprimento_cm"] == pytest.approx(100.0, rel=0.02)
    # Largura (eixo curto) = 100 px * 0.5 = 50 cm
    assert m["largura_cm"] == pytest.approx(50.0, rel=0.02)
    # Conversões para metros
    assert m["largura_m"] == pytest.approx(m["largura_cm"] / 100.0)
    assert m["area_m2"] == pytest.approx(m["area_cm2"] / 10000.0)
