import os
import requests
import logging

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

CARFIT_API_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=Mayab2%2Fcarfit-ai-synthetic"
    "&config=cars"
    "&split=train"
)

MAX_RESULTADOS  = 5
FETCH_LIMIT     = 100
REQUEST_TIMEOUT = 15


def _build_headers() -> dict:
    headers = {"Accept": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def _extrair_palavras_chave(texto: str) -> list:
    stop_words = {
        "um", "uma", "de", "do", "da", "dos", "das", "para", "com", "que",
        "me", "meu", "minha", "eu", "quero", "preciso", "busco", "tem", "ter",
        "qual", "quais", "como", "onde", "por", "mais", "menos", "carro",
        "carros", "veiculo", "veiculos", "seria", "favor",
    }
    tokens = texto.lower().split()
    return [t for t in tokens if len(t) >= 3 and t not in stop_words]


def _carro_bate_filtros(carro: dict, palavras_chave: list) -> bool:
    valores = " ".join(str(v).lower() for v in carro.values() if v)
    return any(palavra in valores for palavra in palavras_chave)


def buscar_no_carfit(pergunta: str) -> list:
    palavras_chave = _extrair_palavras_chave(pergunta)

    if not palavras_chave:
        logger.info("[CarFitAI] Nenhuma palavra-chave extraída da pergunta.")
        return []

    params  = {"offset": 0, "length": FETCH_LIMIT}
    headers = _build_headers()

    try:
        response = requests.get(
            CARFIT_API_URL,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

    except requests.exceptions.Timeout:
        logger.warning("[CarFitAI] Timeout ao consultar o dataset.")
        return []
    except requests.exceptions.HTTPError as e:
        logger.warning("[CarFitAI] Erro HTTP: %s", e)
        return []
    except requests.exceptions.RequestException as e:
        logger.error("[CarFitAI] Erro de conexão: %s", e)
        return []
    except ValueError:
        logger.error("[CarFitAI] Resposta inválida (não é JSON).")
        return []

    rows = data.get("rows", [])
    if not rows:
        logger.info("[CarFitAI] Nenhum registro retornado pela API.")
        return []

    carros_filtrados = []
    for row in rows:
        carro = row.get("row", {})
        if _carro_bate_filtros(carro, palavras_chave):
            carros_filtrados.append(carro)
        if len(carros_filtrados) >= MAX_RESULTADOS:
            break

    logger.info(
        "[CarFitAI] %d carro(s) encontrado(s) para: '%s'",
        len(carros_filtrados),
        pergunta,
    )
    return carros_filtrados


def formatar_resposta_carfit(carros: list) -> str:
    if not carros:
        return "Não encontrei carros correspondentes para essa consulta."

    combustivel_map = {
        "gas":      "Gasolina",
        "diesel":   "Diesel",
        "hybrid":   "Híbrido",
        "electric": "Elétrico",
        "flex":     "Flex",
    }

    linhas = ["🚗 Opções encontradas no CarFitAI:\n"]

    for i, carro in enumerate(carros, start=1):
        make        = carro.get("make", "")
        model       = carro.get("model", "")
        segment     = carro.get("segment", "")
        price       = carro.get("price")
        fuel        = carro.get("fuel_type", "")
        seats       = carro.get("seats")
        km_per_l    = carro.get("km_per_l")
        safety      = carro.get("safety_score")
        maintenance = carro.get("maintenance_monthly_est")

        linhas.append(f"{i}. {make} {model}".strip())

        if segment:
            linhas.append(f"   Segmento: {segment}")

        if price:
            linhas.append(f"   Preço: R$ {price:,.0f}".replace(",", "."))

        combustivel = combustivel_map.get(fuel, fuel)
        if combustivel:
            linhas.append(f"   Combustível: {combustivel}")

        if seats:
            linhas.append(f"   Lugares: {seats}")

        if km_per_l:
            linhas.append(f"   Eficiência: {km_per_l:.1f} km/l")

        if safety:
            linhas.append(f"   Segurança: {safety:.1f}/5.0")

        if maintenance:
            linhas.append(f"   Manutenção estimada: R$ {maintenance}/mês")

        linhas.append("")

    return "\n".join(linhas).strip()