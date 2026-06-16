import os
import requests
import logging

logger = logging.getLogger(__name__)

HF_TOKEN       = os.environ.get("HF_TOKEN", "")
REQUEST_TIMEOUT = 20

_URL_PT_EN = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-tc-big-pt-en"
_URL_EN_PT = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-tc-big-en-pt"


def _build_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def _traduzir(texto: str, url: str) -> str:
    if not texto or not texto.strip():
        return texto

    try:
        response = requests.post(
            url,
            headers=_build_headers(),
            json={"inputs": texto},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and data:
            return data[0].get("translation_text", texto).strip()

        if isinstance(data, dict) and "error" in data:
            logger.warning("[Tradução] Erro do modelo: %s", data["error"])
            return texto

    except requests.exceptions.Timeout:
        logger.warning("[Tradução] Timeout.")
    except requests.exceptions.RequestException as e:
        logger.error("[Tradução] Erro de conexão: %s", e)
    except ValueError:
        logger.error("[Tradução] Resposta inválida.")

    return texto


def pt_para_en(texto: str) -> str:
    logger.info("[Tradução] PT→EN: '%s'", texto)
    traduzido = _traduzir(texto, _URL_PT_EN)
    logger.info("[Tradução] Resultado: '%s'", traduzido)
    return traduzido


def en_para_pt(texto: str) -> str:
    logger.info("[Tradução] EN→PT: '%s'", texto)
    traduzido = _traduzir(texto, _URL_EN_PT)
    logger.info("[Tradução] Resultado: '%s'", traduzido)
    return traduzido