"""Agregador de medidas por track_id.

Mantém uma média exponencial (EMA) das medidas de cada boi ao longo dos
frames. Evita:
  - salvar o mesmo animal N vezes por segundo;
  - oscilações grandes no peso exibido (cada frame tem ruído).

Também guarda a raça vinda da análise visual por track_id, pra que o
próximo laudo / salvamento use essa raça dinâmica em vez do config.yaml.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


EMA_CAMPOS = (
    "peso_estimado",
    "peso_min",
    "peso_max",
    "largura_m",
    "largura_cm",
    "area_m2",
    "area_cm2",
    "comprimento_cm",
    "altura_m",
    "altura_cm",
)


@dataclass
class TrackSample:
    """Estado acumulado de um boi (por track_id)."""

    track_id: int
    amostras: int = 0
    valores: Dict[str, float] = field(default_factory=dict)
    raca: Optional[str] = None
    confianca_raca: float = 0.0
    ecc: Optional[float] = None
    primeira_vez_em: float = field(default_factory=time.time)
    ultima_vez_em: float = field(default_factory=time.time)
    ultimo_salvo_em: Optional[float] = None

    def get(self, campo: str, default: float = 0.0) -> float:
        return float(self.valores.get(campo, default))


class CattleAggregator:
    """Gerencia `TrackSample` por track_id.

    Parâmetros:
        ema_alpha: peso da nova amostra na EMA (0–1). Default 0.3 = histórico domina.
        min_amostras_para_salvar: só considera "maduro" após N amostras.
        cooldown_salvar_s: tempo mínimo entre dois salvamentos do mesmo track.
        expiracao_s: track sem update há mais que isso é removido.
    """

    def __init__(
        self,
        ema_alpha: float = 0.3,
        min_amostras_para_salvar: int = 8,
        cooldown_salvar_s: float = 120.0,
        expiracao_s: float = 10.0,
    ):
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha precisa estar em (0, 1].")
        self.ema_alpha = ema_alpha
        self.min_amostras_para_salvar = min_amostras_para_salvar
        self.cooldown_salvar_s = cooldown_salvar_s
        self.expiracao_s = expiracao_s
        self._tracks: Dict[int, TrackSample] = {}

    def __len__(self) -> int:
        return len(self._tracks)

    def get(self, track_id: int) -> Optional[TrackSample]:
        return self._tracks.get(track_id)

    def tracks_ativos(self):
        return list(self._tracks.values())

    def _ema(self, antigo: float, novo: float) -> float:
        return (1 - self.ema_alpha) * antigo + self.ema_alpha * novo

    def atualizar(self, track_id: int, medida: Dict[str, Any], resultado) -> TrackSample:
        """Aplica uma nova amostra (vinda do frame atual) na EMA do track."""
        agora = time.time()
        sample = self._tracks.get(track_id)
        if sample is None:
            sample = TrackSample(track_id=track_id)
            self._tracks[track_id] = sample

        novo_valores = dict(medida)
        novo_valores["peso_estimado"] = float(resultado.peso_estimado)
        novo_valores["peso_min"] = float(resultado.margem_minima)
        novo_valores["peso_max"] = float(resultado.margem_maxima)

        if sample.amostras == 0:
            for k in EMA_CAMPOS:
                if k in novo_valores:
                    sample.valores[k] = float(novo_valores[k])
        else:
            for k in EMA_CAMPOS:
                if k in novo_valores:
                    sample.valores[k] = self._ema(
                        sample.valores.get(k, float(novo_valores[k])),
                        float(novo_valores[k]),
                    )
        sample.amostras += 1
        sample.ultima_vez_em = agora
        return sample

    def registrar_raca(
        self,
        track_id: int,
        raca: str,
        confianca: float = 0.0,
        ecc: Optional[float] = None,
    ) -> None:
        sample = self._tracks.get(track_id)
        if sample is None:
            sample = TrackSample(track_id=track_id)
            self._tracks[track_id] = sample
        if raca and raca.lower() not in ("indefinido", "desconhecida", "n/d", ""):
            sample.raca = raca
            sample.confianca_raca = float(confianca or 0.0)
        if ecc is not None:
            sample.ecc = float(ecc)

    def raca(self, track_id: int, default: Optional[str] = None) -> Optional[str]:
        sample = self._tracks.get(track_id)
        if sample and sample.raca:
            return sample.raca
        return default

    def deve_salvar(self, track_id: int) -> bool:
        sample = self._tracks.get(track_id)
        if sample is None:
            return False
        if sample.amostras < self.min_amostras_para_salvar:
            return False
        if sample.ultimo_salvo_em is None:
            return True
        return (time.time() - sample.ultimo_salvo_em) >= self.cooldown_salvar_s

    def marcar_salvo(self, track_id: int) -> None:
        sample = self._tracks.get(track_id)
        if sample is not None:
            sample.ultimo_salvo_em = time.time()

    def limpar_expirados(self, agora: Optional[float] = None) -> int:
        agora = agora if agora is not None else time.time()
        mortos = [
            tid for tid, s in self._tracks.items()
            if (agora - s.ultima_vez_em) > self.expiracao_s
        ]
        for tid in mortos:
            del self._tracks[tid]
        return len(mortos)
