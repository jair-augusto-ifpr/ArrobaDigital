"""Cliente HTTP fino para a API OpenRouter (compatível com OpenAI Chat Completions).

Uso típico:

    cfg = load_ia_config()
    client = IAClient(cfg)
    texto = client.chat("Explique em 2 linhas o que é peso vivo.")
    texto = client.chat_with_image("Descreva este boi", frame_bgr)
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import requests
from dotenv import dotenv_values, load_dotenv

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"


class IAError(RuntimeError):
    """Erro amigável para falhas de chamada à LLM."""


@dataclass
class IAConfig:
    api_key: Optional[str]
    model: str
    vision_model: str
    timeout: float = 30.0
    site_url: str = "https://github.com/IFPR/ArrobaDigital"
    site_title: str = "ArrobaDigital"

    @property
    def disponivel(self) -> bool:
        return bool(self.api_key)


def _read_env(key: str) -> Optional[str]:
    vals = dotenv_values(_ENV_PATH) or {}
    raw = vals.get(key)
    if raw is None:
        load_dotenv(_ENV_PATH, override=False)
        raw = os.getenv(key)
    if raw is None:
        return None
    s = str(raw).strip().strip('"').strip("'")
    return s or None


def load_ia_config() -> IAConfig:
    """Lê API_KEY_IA, IA_MODEL e IA_VISION_MODEL do .env (com fallbacks seguros)."""
    api_key = _read_env("API_KEY_IA") or _read_env("OPENROUTER_API_KEY")
    # IA_MODEL é a fonte de verdade. Fallback só entra se .env estiver ausente.
    model = _read_env("IA_MODEL") or "openai/gpt-4o-mini"
    vision_model = _read_env("IA_VISION_MODEL") or model
    return IAConfig(api_key=api_key, model=model, vision_model=vision_model)


def _encode_image_bgr(image, max_side: int = 512, quality: int = 85) -> str:
    """Converte ndarray BGR em data URL JPEG (reduz lado maior para controlar payload)."""
    h, w = image.shape[:2]
    m = max(h, w)
    if m > max_side:
        scale = max_side / float(m)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise IAError("Falha ao codificar imagem em JPEG para envio à IA.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class IAClient:
    def __init__(self, config: IAConfig):
        self.config = config

    def _headers(self) -> dict:
        if not self.config.api_key:
            raise IAError(
                "API_KEY_IA ausente no .env — não é possível chamar a IA. "
                "Rode com --no-ia ou preencha API_KEY_IA."
            )
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": self.config.site_url,
            "X-Title": self.config.site_title,
            "Content-Type": "application/json",
        }

    def _post(self, payload: dict, max_retries: int = 5) -> str:
        """Chama o endpoint com retry exponencial para 429/5xx (comum em modelos :free).

        Backoff: 2, 4, 8, 16, 32 s (total ate ~62 s numa cadeia de 429).
        """
        last_text = ""
        last_status = 0
        headers = self._headers()
        for attempt in range(max_retries):
            try:
                r = requests.post(
                    _ENDPOINT, json=payload, headers=headers, timeout=self.config.timeout,
                )
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise IAError(f"Falha de rede ao chamar OpenRouter: {e}") from e
                time.sleep(2 ** (attempt + 1))
                continue

            last_status = r.status_code
            last_text = r.text

            if r.status_code == 200:
                try:
                    data = r.json()
                    return data["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError, ValueError) as e:
                    raise IAError(f"Resposta inesperada da IA: {e}: {r.text[:200]}") from e

            # Retry para 429 (rate-limit) e 5xx (instabilidade do provider)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt < max_retries - 1:
                    espera = 2 ** (attempt + 1)
                    print(f"[IA] {r.status_code} — aguardando {espera}s e tentando novamente "
                          f"(tentativa {attempt + 2}/{max_retries})...")
                    time.sleep(espera)
                    continue

            break

        raise IAError(f"OpenRouter respondeu {last_status}: {last_text[:300]}")

    @staticmethod
    def _supports_system(model: str) -> bool:
        """Gemma (Google AI Studio) não aceita role=system. Outros normalmente sim."""
        m = (model or "").lower()
        return "gemma" not in m

    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 400,
        model: Optional[str] = None,
    ) -> str:
        model_name = model or self.config.model
        messages = []
        if system_prompt and self._supports_system(model_name):
            messages.append({"role": "system", "content": system_prompt})
            user_text = user_prompt
        else:
            user_text = (system_prompt + "\n\n" + user_prompt) if system_prompt else user_prompt
        messages.append({"role": "user", "content": user_text})
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return self._post(payload)

    def chat_with_image(
        self,
        user_prompt: str,
        image_bgr,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 400,
        max_side: int = 512,
        model: Optional[str] = None,
    ) -> str:
        model_name = model or self.config.vision_model
        image_url = _encode_image_bgr(image_bgr, max_side=max_side)

        messages = []
        if system_prompt and self._supports_system(model_name):
            messages.append({"role": "system", "content": system_prompt})
            user_text = user_prompt
        else:
            user_text = (system_prompt + "\n\n" + user_prompt) if system_prompt else user_prompt

        content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        messages.append({"role": "user", "content": content})
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return self._post(payload)
