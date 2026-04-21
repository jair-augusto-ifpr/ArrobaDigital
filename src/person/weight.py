"""
Estimativa de peso humano a partir da máscara de segmentação.

⚠️ AVISO (IMPORTANTE)
----------------------
Este módulo é um PROTÓTIPO didático para validar o pipeline com um sujeito
disponível (você mesmo em frente à câmera). A estimativa é GROSSEIRA:

1. Pegamos altura (cm) e largura (cm) da máscara da pessoa em pé.
2. Calculamos um IMC presumido a partir da razão largura/altura (proxy de
   compleição corporal, desde magro até sobrepeso).
3. Peso = IMC * altura_m² (fórmula clássica do IMC invertida).

Margens de erro reais são GRANDES (± 15–25 %). A fórmula não substitui
bioimpedância nem pesagem em balança. Serve apenas como demonstração
visual. NÃO use em decisões médicas ou nutricionais.

Intuições que calibram a razão `k = largura_cm / altura_cm`:

    k ≈ 0.18  → pessoa magra  → IMC ≈ 18.5
    k ≈ 0.23  → normal        → IMC ≈ 22.5
    k ≈ 0.30  → sobrepeso     → IMC ≈ 30.0
    k ≈ 0.38  → obesidade     → IMC ≈ 38.0

Interpolação linear nesses pontos, com clamp em [16, 40].
"""

from __future__ import annotations

from dataclasses import dataclass


_IMC_MIN, _IMC_MAX = 16.0, 40.0


@dataclass
class ResultadoPesoPessoa:
    peso_estimado: float
    margem_minima: float
    margem_maxima: float
    imc_estimado: float
    altura_cm: float
    largura_cm: float
    razao: float
    modelo: str = "IMC presumido (razão largura/altura)"

    @property
    def altura_m(self) -> float:
        return self.altura_cm / 100.0


def _imc_por_razao(razao: float) -> float:
    """Interpola IMC pela razão largura/altura (clamp em [16, 40])."""
    # Pontos âncora definidos na docstring do módulo.
    pontos = ((0.18, 18.5), (0.23, 22.5), (0.30, 30.0), (0.38, 38.0))
    if razao <= pontos[0][0]:
        imc = pontos[0][1]
    elif razao >= pontos[-1][0]:
        imc = pontos[-1][1]
    else:
        imc = pontos[-1][1]
        for (r0, i0), (r1, i1) in zip(pontos, pontos[1:]):
            if r0 <= razao <= r1:
                t = (razao - r0) / (r1 - r0)
                imc = i0 + t * (i1 - i0)
                break
    return max(_IMC_MIN, min(_IMC_MAX, imc))


def estimar_peso_pessoa(altura_cm: float, largura_cm: float) -> ResultadoPesoPessoa:
    """Estima peso humano a partir de altura e largura (ambas em cm).

    Lança `ValueError` se os valores forem inválidos.
    """
    if altura_cm <= 0:
        raise ValueError("altura_cm precisa ser > 0")
    if largura_cm <= 0:
        raise ValueError("largura_cm precisa ser > 0")

    razao = largura_cm / altura_cm
    imc = _imc_por_razao(razao)
    altura_m = altura_cm / 100.0
    peso = imc * altura_m * altura_m

    return ResultadoPesoPessoa(
        peso_estimado=round(peso, 1),
        # Margem ± 15 % refletindo a imprecisão honesta do método.
        margem_minima=round(peso * 0.85, 1),
        margem_maxima=round(peso * 1.15, 1),
        imc_estimado=round(imc, 1),
        altura_cm=round(altura_cm, 1),
        largura_cm=round(largura_cm, 1),
        razao=round(razao, 3),
    )
