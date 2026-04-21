import time

import pytest

from src.tracking.aggregator import CattleAggregator


class _Resultado:
    def __init__(self, peso, minp, maxp):
        self.peso_estimado = peso
        self.margem_minima = minp
        self.margem_maxima = maxp


def _medida(largura_cm=60, area_cm2=8000):
    return {
        "largura_cm": largura_cm,
        "largura_m": largura_cm / 100.0,
        "area_cm2": area_cm2,
        "area_m2": area_cm2 / 10000.0,
        "comprimento_cm": 150,
        "altura_cm": 150,
        "altura_m": 1.5,
    }


def test_primeira_amostra_copia_valores():
    agg = CattleAggregator()
    s = agg.atualizar(1, _medida(60, 8000), _Resultado(400, 380, 420))
    assert s.amostras == 1
    assert s.get("peso_estimado") == pytest.approx(400.0)
    assert s.get("largura_cm") == pytest.approx(60.0)


def test_ema_suaviza_variacao():
    agg = CattleAggregator(ema_alpha=0.3)
    agg.atualizar(1, _medida(60, 8000), _Resultado(400, 380, 420))
    agg.atualizar(1, _medida(80, 10000), _Resultado(500, 475, 525))
    s = agg.get(1)
    # EMA: 0.7 * 400 + 0.3 * 500 = 430
    assert s.get("peso_estimado") == pytest.approx(430.0, rel=1e-3)
    # 0.7 * 60 + 0.3 * 80 = 66
    assert s.get("largura_cm") == pytest.approx(66.0, rel=1e-3)
    assert s.amostras == 2


def test_deve_salvar_respeita_minimo_amostras_e_cooldown():
    agg = CattleAggregator(min_amostras_para_salvar=3, cooldown_salvar_s=10.0)
    for _ in range(2):
        agg.atualizar(1, _medida(), _Resultado(400, 380, 420))
    assert agg.deve_salvar(1) is False  # ainda não tem 3 amostras
    agg.atualizar(1, _medida(), _Resultado(400, 380, 420))
    assert agg.deve_salvar(1) is True
    agg.marcar_salvo(1)
    # Cooldown: logo em seguida não deve aceitar.
    assert agg.deve_salvar(1) is False


def test_registrar_raca_ignora_indefinido():
    agg = CattleAggregator()
    agg.atualizar(1, _medida(), _Resultado(400, 380, 420))
    agg.registrar_raca(1, "indefinido", confianca=0.1)
    assert agg.raca(1) is None
    agg.registrar_raca(1, "Nelore", confianca=0.9, ecc=3.5)
    assert agg.raca(1) == "Nelore"
    assert agg.get(1).ecc == pytest.approx(3.5)


def test_expiracao_remove_tracks_inativos():
    agg = CattleAggregator(expiracao_s=0.05)
    agg.atualizar(1, _medida(), _Resultado(400, 380, 420))
    agg.atualizar(2, _medida(), _Resultado(420, 400, 440))
    assert len(agg) == 2
    time.sleep(0.1)
    # Ao atualizar só o 1, o 2 deve expirar no sweep.
    agg.atualizar(1, _medida(), _Resultado(400, 380, 420))
    removidos = agg.limpar_expirados()
    assert removidos == 1
    assert len(agg) == 1


def test_construtor_rejeita_alpha_invalido():
    with pytest.raises(ValueError):
        CattleAggregator(ema_alpha=0)
    with pytest.raises(ValueError):
        CattleAggregator(ema_alpha=1.5)
