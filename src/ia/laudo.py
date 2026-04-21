"""Laudo textual por boi — usa medidas + resultado da regressão + config."""

from __future__ import annotations

from typing import Optional

from .client import IAClient, IAError

_SYSTEM = (
    "Voce e um assistente tecnico de pecuaria de precisao. "
    "Recebe medidas biometricas de um bovino obtidas por visao computacional "
    "(podem conter imprecisao de +/- 5 a 10%) e gera um laudo curto em portugues, "
    "objetivo e sem floreios. Nao invente valores: use apenas os dados fornecidos."
)


def _formatar_contexto(medida: dict, resultado, raca_config: Optional[str]) -> str:
    raca = raca_config or "desconhecida"
    return (
        f"Dados do boi detectado:\n"
        f"- Peso estimado: {resultado.peso_estimado:.1f} kg "
        f"(intervalo {resultado.margem_minima:.0f}-{resultado.margem_maxima:.0f} kg, modelo {resultado.modelo})\n"
        f"- Largura do quadril: {medida.get('largura_cm', 0):.1f} cm "
        f"({medida.get('largura_m', 0):.2f} m)\n"
        f"- Comprimento (proxy): {medida.get('comprimento_cm', 0):.1f} cm "
        f"({medida.get('comprimento_cm', 0) / 100:.2f} m)\n"
        f"- Area dorsal: {medida.get('area_cm2', 0):.0f} cm2 "
        f"({medida.get('area_m2', 0):.3f} m2)\n"
        f"- Raca declarada em config: {raca}\n"
    )


def gerar_laudo(
    client: IAClient,
    medida: dict,
    resultado,
    raca_config: Optional[str] = None,
    historico_resumo: Optional[str] = None,
) -> str:
    """Retorna um laudo de 2-4 linhas em portugues sobre o boi medido.

    Lanca IAError em falhas (para o chamador decidir se loga ou silencia).
    """
    contexto = _formatar_contexto(medida, resultado, raca_config)
    if historico_resumo:
        contexto += f"\nHistorico recente do lote:\n{historico_resumo}\n"

    prompt = (
        contexto
        + "\nGere um laudo curto (ate 4 linhas, sem listas) cobrindo: "
        "(1) classificacao aproximada do animal (bezerro/novilho/adulto) baseado nas medidas; "
        "(2) se o peso parece coerente com a largura/area; "
        "(3) uma recomendacao pratica curta para o pecuarista."
    )
    try:
        return client.chat(prompt, system_prompt=_SYSTEM, max_tokens=320)
    except IAError:
        raise
