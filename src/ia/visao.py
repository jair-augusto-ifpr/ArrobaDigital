"""Analise visual do boi via LLM multimodal: raca provavel + escore de condicao corporal."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .client import IAClient, IAError

_SYSTEM = (
    "Voce e um zootecnista avaliando bovinos a partir de uma unica foto. "
    "O peso ja foi calculado pelo sistema do usuario (nao estime peso). "
    "Sua tarefa e identificar caracteristicas visuais complementares. "
    "Seja honesto: se a imagem nao permite concluir algo, diga 'indefinido'. "
    "Nao invente numeros. Responda ESTRITAMENTE em JSON valido."
)

_USER = (
    "Analise este bovino e retorne um JSON com as chaves exatas:\n"
    "  raca_provavel   (string: nome em portugues ou 'indefinido')\n"
    "  confianca_raca  (numero de 0 a 1)\n"
    "  ecc             (numero de 1 a 5, escore de condicao corporal, ou null)\n"
    "  cor_pelagem     (string curta)\n"
    "  observacoes     (string com no maximo 2 frases curtas em portugues)\n"
    "Dados ja calculados pelo sistema (apenas contexto — NAO recalcule):\n"
    "  peso = {peso} kg, largura quadril = {largura} m, "
    "area dorsal = {area} m2, raca declarada em config = {raca_config}.\n"
    "Nao inclua texto fora do JSON."
)


@dataclass
class AnaliseVisual:
    raca_provavel: str
    confianca_raca: float
    ecc: Optional[float]
    cor_pelagem: str
    observacoes: str
    raw: str

    def resumo_uma_linha(self) -> str:
        ecc_s = f"{self.ecc:.1f}/5" if self.ecc is not None else "n/d"
        return (
            f"Raca: {self.raca_provavel} ({self.confianca_raca:.0%}) | "
            f"ECC: {ecc_s} | Pelagem: {self.cor_pelagem}"
        )


def _extract_json(text: str) -> dict:
    # Remove fences ```json ... ```
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    # Pega o primeiro bloco {...}
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise IAError(f"IA nao retornou JSON: {text[:200]}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise IAError(f"JSON invalido da IA: {e}: {m.group(0)[:200]}") from e


def _as_float(v, lo=None, hi=None):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if lo is not None and f < lo:
        return None
    if hi is not None and f > hi:
        return None
    return f


def analisar_boi(
    client: IAClient,
    image_bgr,
    medida: Optional[dict] = None,
    resultado=None,
    raca_config: Optional[str] = None,
) -> AnaliseVisual:
    """Chama o modelo de visao e devolve uma AnaliseVisual estruturada.

    `image_bgr` e o crop do boi (idealmente `segment['crop']` com fundo zerado).
    """
    medida = medida or {}
    peso = getattr(resultado, "peso_estimado", None)
    prompt = _USER.format(
        peso=f"{peso:.1f}" if peso is not None else "desconhecido",
        largura=f"{medida.get('largura_m', 0):.2f}",
        area=f"{medida.get('area_m2', 0):.3f}",
        raca_config=raca_config or "desconhecida",
    )

    raw = client.chat_with_image(
        prompt, image_bgr, system_prompt=_SYSTEM, temperature=0.1, max_tokens=350
    )
    data = _extract_json(raw)

    return AnaliseVisual(
        raca_provavel=str(data.get("raca_provavel") or "indefinido").strip(),
        confianca_raca=_as_float(data.get("confianca_raca"), 0.0, 1.0) or 0.0,
        ecc=_as_float(data.get("ecc"), 1.0, 5.0),
        cor_pelagem=str(data.get("cor_pelagem") or "n/d").strip(),
        observacoes=str(data.get("observacoes") or "").strip(),
        raw=raw,
    )
