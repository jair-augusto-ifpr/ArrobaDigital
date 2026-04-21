import pytest

from src.person.weight import _imc_por_razao, estimar_peso_pessoa


def test_imc_por_razao_pontos_ancora():
    assert _imc_por_razao(0.18) == pytest.approx(18.5, abs=0.01)
    assert _imc_por_razao(0.23) == pytest.approx(22.5, abs=0.01)
    assert _imc_por_razao(0.30) == pytest.approx(30.0, abs=0.01)
    assert _imc_por_razao(0.38) == pytest.approx(38.0, abs=0.01)


def test_imc_por_razao_clamp():
    assert _imc_por_razao(0.05) == pytest.approx(18.5)  # abaixo do 1º ponto -> fica nele
    assert _imc_por_razao(2.0) == pytest.approx(38.0)   # acima do último -> fica nele
    # Clamp duro em [16, 40] é garantido pelo código, embora os âncoras já estejam dentro.
    assert 16.0 <= _imc_por_razao(0.01) <= 40.0
    assert 16.0 <= _imc_por_razao(5.0) <= 40.0


def test_imc_por_razao_interpolacao_linear():
    # Entre (0.18, 18.5) e (0.23, 22.5), meio da curva:
    meio = _imc_por_razao(0.205)
    esperado = 18.5 + 0.5 * (22.5 - 18.5)
    assert meio == pytest.approx(esperado, abs=1e-6)


def test_estimar_peso_pessoa_perfil_normal():
    # 175 cm, 40 cm de largura (ombro ~) -> razão 0.2286 -> IMC ~22.4 -> peso ~68.6
    r = estimar_peso_pessoa(altura_cm=175, largura_cm=40)
    assert r.altura_cm == pytest.approx(175.0)
    assert r.largura_cm == pytest.approx(40.0)
    assert r.razao == pytest.approx(40.0 / 175.0, abs=1e-3)
    assert 20 <= r.imc_estimado <= 24
    assert 60 <= r.peso_estimado <= 75
    # Margens ± 15 %.
    assert r.margem_minima == pytest.approx(r.peso_estimado * 0.85, rel=0.01)
    assert r.margem_maxima == pytest.approx(r.peso_estimado * 1.15, rel=0.01)


def test_estimar_peso_pessoa_mais_largo_pesa_mais():
    fino = estimar_peso_pessoa(altura_cm=175, largura_cm=32)
    normal = estimar_peso_pessoa(altura_cm=175, largura_cm=40)
    largo = estimar_peso_pessoa(altura_cm=175, largura_cm=55)
    assert fino.peso_estimado < normal.peso_estimado < largo.peso_estimado
    assert fino.imc_estimado < normal.imc_estimado < largo.imc_estimado


def test_estimar_peso_pessoa_valida_entrada():
    with pytest.raises(ValueError):
        estimar_peso_pessoa(altura_cm=0, largura_cm=40)
    with pytest.raises(ValueError):
        estimar_peso_pessoa(altura_cm=175, largura_cm=-1)
