"""
Estimador de Peso Corporal Animal
==================================
Dois modelos disponíveis:
  1. Regressão Linear  → Peso = 6.15 * largura + 0.019 * area + 70.8
  2. Biométrico (PT²)  → Peso = comprimento * perimetro_toracico² * K

⚠️ SOBRE OS COEFICIENTES DA REGRESSÃO LINEAR
-------------------------------------------
Os coeficientes (6.15, 0.019, 70.8) são um PLACEHOLDER didático:
foram ajustados grosseiramente para um lote pequeno de Nelore adultos
no escopo do TCC ArrobaDigital (IFPR). Não devem ser usados como
referência científica sem recalibração com dados do próprio rebanho.

O modelo biométrico PT² é bibliograficamente mais robusto (veja Santos &
Boin, 1996; Heinrichs et al., 1992 para gado leiteiro) e deve ser
preferido SEMPRE que for possível medir perímetro torácico.

Erro esperado (após calibração): 3% a 10% dependendo da qualidade das
medidas extraídas por visão computacional.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Constantes de raça para o modelo biométrico
# ---------------------------------------------------------------------------
class Raca(Enum):
    NELORE       = 80
    ANGUS        = 90
    HEREFORD     = 90
    CRUZAMENTO   = 85
    SIMENTAL     = 88
    BRAHMAN      = 78


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------
@dataclass
class ResultadoPeso:
    peso_estimado: float          # kg
    margem_minima: float          # kg  (−5 %)
    margem_maxima: float          # kg  (+5 %)
    modelo: str
    formula: str

    def __str__(self) -> str:
        linha = "-" * 44
        return (
            f"\n{linha}\n"
            f"  Modelo          : {self.modelo}\n"
            f"  Fórmula         : {self.formula}\n"
            f"  Peso estimado   : {self.peso_estimado:.1f} kg\n"
            f"  Intervalo (±5%) : {self.margem_minima:.1f} – {self.margem_maxima:.1f} kg\n"
            f"{linha}"
        )


# ---------------------------------------------------------------------------
# Funções de cálculo
# ---------------------------------------------------------------------------
def modelo_regressao(largura: float, area_dorsal: float) -> ResultadoPeso:
    """
    Modelo de Regressão Linear.

    Parâmetros
    ----------
    largura      : largura do quadril em metros (m)
    area_dorsal  : área dorsal em metros quadrados (m²)

    Retorna
    -------
    ResultadoPeso com peso estimado e intervalo de confiança.
    """
    if largura <= 0 or area_dorsal <= 0:
        raise ValueError("Largura e área dorsal devem ser maiores que zero.")

    peso = 6.15 * largura + 0.019 * area_dorsal + 70.8

    formula = (
        f"6.15 × {largura} + 0.019 × {area_dorsal} + 70.8"
        f"  =  {peso:.2f} kg"
    )

    return ResultadoPeso(
        peso_estimado=round(peso, 2),
        margem_minima=round(peso * 0.95, 2),
        margem_maxima=round(peso * 1.05, 2),
        modelo="Regressão Linear",
        formula=formula,
    )


def modelo_biometrico(
    comprimento: float,
    perimetro_toracico: float,
    raca: Raca = Raca.NELORE,
) -> ResultadoPeso:
    """
    Modelo Biométrico (PT²).

    Parâmetros
    ----------
    comprimento          : comprimento corporal em metros (m)
    perimetro_toracico   : perímetro torácico em metros (m)
    raca                 : constante K da raça (enum Raca)

    Retorna
    -------
    ResultadoPeso com peso estimado e intervalo de confiança.
    """
    if comprimento <= 0 or perimetro_toracico <= 0:
        raise ValueError("Comprimento e perímetro torácico devem ser maiores que zero.")

    k = raca.value
    peso = comprimento * (perimetro_toracico ** 2) * k

    formula = (
        f"{comprimento} × {perimetro_toracico}² × {k} ({raca.name})"
        f"  =  {peso:.2f} kg"
    )

    return ResultadoPeso(
        peso_estimado=round(peso, 2),
        margem_minima=round(peso * 0.95, 2),
        margem_maxima=round(peso * 1.05, 2),
        modelo=f"Biométrico PT² ({raca.name})",
        formula=formula,
    )


# ---------------------------------------------------------------------------
# Dispatcher: escolhe o melhor modelo disponível
# ---------------------------------------------------------------------------
def _raca_from_name(nome: Optional[str]) -> Raca:
    """Converte nome (case-insensitive, ignora acentos básicos) em enum Raca."""
    if not nome:
        return Raca.NELORE
    chave = nome.strip().upper().replace("Ç", "C").replace("Ê", "E")
    mapping = {r.name: r for r in Raca}
    if chave in mapping:
        return mapping[chave]
    # Aliases comuns que a LLM pode devolver:
    alias = {
        "CRUZADO": Raca.CRUZAMENTO,
        "ANELORADO": Raca.NELORE,
        "BRAHMA": Raca.BRAHMAN,
    }
    return alias.get(chave, Raca.NELORE)


def estimar_peso(
    largura: float,
    area_dorsal: float,
    comprimento: Optional[float] = None,
    perimetro_toracico: Optional[float] = None,
    raca: Optional[str] = None,
) -> ResultadoPeso:
    """Escolhe automaticamente o melhor modelo.

    - Se `comprimento` e `perimetro_toracico` forem fornecidos (>0) usa PT².
    - Caso contrário, cai no modelo de regressão linear (placeholder).
    - `raca` é string (como vem da análise visual). Default: Nelore.
    """
    if comprimento and perimetro_toracico and comprimento > 0 and perimetro_toracico > 0:
        return modelo_biometrico(
            comprimento=comprimento,
            perimetro_toracico=perimetro_toracico,
            raca=_raca_from_name(raca),
        )
    return modelo_regressao(largura=largura, area_dorsal=area_dorsal)


# ---------------------------------------------------------------------------
# Interface de linha de comando (opcional)
# ---------------------------------------------------------------------------
def _menu_interativo() -> None:
    print("\n========================================")
    print("   Estimador de Peso Corporal Animal   ")
    print("========================================")
    print("Escolha o modelo:")
    print("  1 → Regressão Linear (largura + área dorsal)")
    print("  2 → Biométrico PT²   (comprimento + perímetro)")

    opcao = input("\nOpção [1/2]: ").strip()

    if opcao == "1":
        largura     = float(input("Largura do quadril (m)  : "))
        area_dorsal = float(input("Área dorsal (m²)        : "))
        resultado   = modelo_regressao(largura, area_dorsal)

    elif opcao == "2":
        comprimento  = float(input("Comprimento corporal (m)   : "))
        pt           = float(input("Perímetro torácico (m)     : "))
        print("\nRaças disponíveis:")
        for i, r in enumerate(Raca, 1):
            print(f"  {i} → {r.name}  (K={r.value})")
        idx  = int(input("Escolha a raça [número]: ")) - 1
        raca = list(Raca)[idx]
        resultado = modelo_biometrico(comprimento, pt, raca)

    else:
        print("Opção inválida.")
        return

    print(resultado)


# ---------------------------------------------------------------------------
# Demonstração automática (executado com: python estimador_peso.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("\n=== EXEMPLO 1 — Regressão Linear ===")
    r1 = modelo_regressao(largura=0.65, area_dorsal=0.92)
    print(r1)

    print("\n=== EXEMPLO 2 — Biométrico PT² (Nelore) ===")
    r2 = modelo_biometrico(
        comprimento=1.60,
        perimetro_toracico=1.85,
        raca=Raca.NELORE,
    )
    print(r2)

    print("\n=== EXEMPLO 3 — Biométrico PT² (Angus) ===")
    r3 = modelo_biometrico(
        comprimento=1.55,
        perimetro_toracico=1.90,
        raca=Raca.ANGUS,
    )
    print(r3)

    # -----------------------------------------------------------------------
    # Descomente para usar o menu interativo:
    # _menu_interativo()