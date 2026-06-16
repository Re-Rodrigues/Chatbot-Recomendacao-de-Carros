import os
import requests
import logging

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
LLM_MODEL = os.environ.get("HF_LLM_MODEL", "google/flan-t5-small")
HF_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
REQUEST_TIMEOUT = 20


def _build_headers() -> dict:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def gerar_resposta_llm(pergunta: str) -> str:
    if not HF_TOKEN:
        logger.warning("[LLM] HF_TOKEN não definido.")
        return None

    prompt = (
        "Você é um assistente especializado em recomendação de carros. "
        "Responda de forma clara e útil à pergunta do usuário, mantendo o foco em carros e recomendações práticas.\n\n"
        f"Pergunta: {pergunta}\n"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "return_full_text": False,
        },
    }

    try:
        response = requests.post(
            HF_API_URL,
            headers=_build_headers(),
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

    except requests.exceptions.Timeout:
        logger.warning("[LLM] Timeout ao chamar o modelo.")
        return "Desculpe, o serviço de IA demorou demais para responder. Tente novamente em alguns instantes."
    except requests.exceptions.HTTPError as e:
        logger.error("[LLM] Erro HTTP: %s", e)
        return "Desculpe, não consegui obter a resposta da IA agora."
    except requests.exceptions.RequestException as e:
        logger.error("[LLM] Erro de conexão: %s", e)
        return "Desculpe, houve um problema ao conectar ao serviço de IA."
    except ValueError:
        logger.error("[LLM] Resposta inválida (não é JSON).")
        return "Desculpe, recebi uma resposta inválida do serviço de IA."

    if isinstance(data, dict) and data.get("error"):
        logger.warning("[LLM] Erro do modelo: %s", data["error"])
        return "Desculpe, não consegui gerar uma resposta agora."

    if isinstance(data, list) and data:
        texto_gerado = data[0].get("generated_text") or data[0].get("content") or ""
        return texto_gerado.strip() or "Desculpe, não consegui gerar uma resposta agora."

    return "Desculpe, não consegui gerar uma resposta agora."
