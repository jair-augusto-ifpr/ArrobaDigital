"""Relatorio do lote a partir dos ultimos registros do banco."""

from __future__ import annotations

from statistics import mean, median, pstdev
from typing import Iterable, List, Optional

from .client import IAClient, IAError

_SYSTEM = (
    "Voce e um consultor de pecuaria de precisao. Recebe estatisticas agregadas "
    "de pesagem por visao computacional (margem de erro ~5 a 10%). "
    "Escreva um resumo objetivo e util para o pecuarista, em portugues, "
    "em no maximo 6 linhas, sem listas numeradas, sem floreios."
)


def _num(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def resumir_estatisticas(registros: Iterable[dict]) -> dict:
    pesos: List[float] = []
    larguras: List[float] = []
    areas: List[float] = []
    for r in registros:
        p = _num(r.get("peso_kg"))
        w = _num(r.get("largura_m"))
        a = _num(r.get("area_m2"))
        if p is not None:
            pesos.append(p)
        if w is not None:
            larguras.append(w)
        if a is not None:
            areas.append(a)

    def stats(values: List[float]) -> Optional[dict]:
        if not values:
            return None
        return {
            "n": len(values),
            "media": mean(values),
            "mediana": median(values),
            "min": min(values),
            "max": max(values),
            "desvio": pstdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "peso_kg": stats(pesos),
        "largura_m": stats(larguras),
        "area_m2": stats(areas),
        "total_registros": len(pesos),
    }


def _formatar_stats(s: dict) -> str:
    if not s:
        return "sem dados"
    return (
        f"n={s['n']}, media={s['media']:.2f}, mediana={s['mediana']:.2f}, "
        f"min={s['min']:.2f}, max={s['max']:.2f}, desvio={s['desvio']:.2f}"
    )


def gerar_relatorio_lote(
    client: IAClient,
    registros: List[dict],
    raca_config: Optional[str] = None,
) -> str:
    """Gera relatorio em linguagem natural a partir dos registros recentes."""
    if not registros:
        return "Sem registros no banco para gerar relatorio."

    stats = resumir_estatisticas(registros)
    primeiro = registros[-1].get("timestamp")
    ultimo = registros[0].get("timestamp")

    contexto = (
        f"Total de registros considerados: {stats['total_registros']}\n"
        f"Periodo: {primeiro} ate {ultimo}\n"
        f"Raca declarada em config: {raca_config or 'desconhecida'}\n"
        f"Peso (kg): {_formatar_stats(stats['peso_kg'])}\n"
        f"Largura quadril (m): {_formatar_stats(stats['largura_m'])}\n"
        f"Area dorsal (m2): {_formatar_stats(stats['area_m2'])}\n"
    )

    prompt = (
        contexto
        + "\nEscreva um resumo tecnico do lote, comente dispersao, animais fora da curva "
        "(se o desvio sugerir), e de uma recomendacao pratica curta."
    )

    try:
        return client.chat(prompt, system_prompt=_SYSTEM, max_tokens=500)
    except IAError:
        raise
