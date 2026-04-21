import pytest

from src.conversao.conversao import (
    Raca,
    _raca_from_name,
    estimar_peso,
    modelo_biometrico,
    modelo_regressao,
)


def test_regressao_valor_esperado():
    # Fórmula: 6.15*L + 0.019*A + 70.8
    r = modelo_regressao(largura=0.65, area_dorsal=0.9)
    esperado = 6.15 * 0.65 + 0.019 * 0.9 + 70.8
    assert r.peso_estimado == pytest.approx(round(esperado, 2), abs=0.01)
    assert r.margem_minima == pytest.approx(round(esperado * 0.95, 2), abs=0.01)
    assert r.margem_maxima == pytest.approx(round(esperado * 1.05, 2), abs=0.01)
    assert r.modelo == "Regressão Linear"


def test_regressao_rejeita_negativos():
    with pytest.raises(ValueError):
        modelo_regressao(largura=0, area_dorsal=1)
    with pytest.raises(ValueError):
        modelo_regressao(largura=1, area_dorsal=-0.1)


def test_biometrico_nelore():
    r = modelo_biometrico(comprimento=1.6, perimetro_toracico=1.85, raca=Raca.NELORE)
    esperado = 1.6 * (1.85 ** 2) * Raca.NELORE.value
    assert r.peso_estimado == pytest.approx(round(esperado, 2), abs=0.01)
    assert "NELORE" in r.modelo


def test_raca_from_name_mapeia_aliases():
    assert _raca_from_name("Nelore") is Raca.NELORE
    assert _raca_from_name("nelore") is Raca.NELORE
    assert _raca_from_name("Brahma") is Raca.BRAHMAN
    assert _raca_from_name("Cruzado") is Raca.CRUZAMENTO
    # Raça desconhecida cai no default (Nelore).
    assert _raca_from_name("Holandesa") is Raca.NELORE
    assert _raca_from_name(None) is Raca.NELORE


def test_estimar_peso_usa_pt2_quando_tem_perimetro():
    r = estimar_peso(
        largura=0.7, area_dorsal=1.0,
        comprimento=1.6, perimetro_toracico=1.85,
        raca="Nelore",
    )
    assert "PT" in r.modelo  # PT² foi escolhido
    esperado = 1.6 * (1.85 ** 2) * Raca.NELORE.value
    assert r.peso_estimado == pytest.approx(round(esperado, 2), abs=0.01)


def test_estimar_peso_cai_na_regressao_sem_perimetro():
    r = estimar_peso(largura=0.65, area_dorsal=0.9, raca="Angus")
    assert r.modelo == "Regressão Linear"
