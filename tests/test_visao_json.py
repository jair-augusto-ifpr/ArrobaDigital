import pytest

from src.ia.client import IAError
from src.ia.visao import _as_float, _extract_json


def test_extract_json_fenced():
    txt = "```json\n{\"raca_provavel\": \"Nelore\", \"ecc\": 3}\n```"
    assert _extract_json(txt) == {"raca_provavel": "Nelore", "ecc": 3}


def test_extract_json_com_prefixo_e_sufixo():
    txt = "claro! \n```json\n{\"x\":1}\n```\nmais algo."
    assert _extract_json(txt) == {"x": 1}


def test_extract_json_sem_fence():
    txt = '{"a":1, "b":"dois"}'
    assert _extract_json(txt) == {"a": 1, "b": "dois"}


def test_extract_json_em_meio_de_texto():
    txt = 'Analise: {"raca_provavel": "Angus", "ecc": 4} fim.'
    assert _extract_json(txt) == {"raca_provavel": "Angus", "ecc": 4}


def test_extract_json_invalido_levanta():
    with pytest.raises(IAError):
        _extract_json("sem json aqui")
    with pytest.raises(IAError):
        _extract_json("{ invalido }")


def test_as_float_limites():
    assert _as_float(None) is None
    assert _as_float("3.14") == pytest.approx(3.14)
    assert _as_float("nao eh numero") is None
    # Limites
    assert _as_float(0.5, lo=0.0, hi=1.0) == pytest.approx(0.5)
    assert _as_float(-0.1, lo=0.0, hi=1.0) is None
    assert _as_float(1.1, lo=0.0, hi=1.0) is None
