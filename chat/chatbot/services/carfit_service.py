import os
import random
import requests
import logging
import re
import unicodedata

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

SEARCH_SYNONYMS = {
    "eletrico": "electric",
    "eletricos": "electric",
    "eletrica": "electric",
    "eletricas": "electric",
    "hibrido": "hybrid",
    "hibrida": "hybrid",
    "hibridos": "hybrid",
    "hibridas": "hybrid",
    "gasolina": "gas",
    "diesel": "diesel",
    "flex": "flex",
    "autonomo": "autonomy",
}


def _remover_acentos(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")


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
    texto_normalizado = _remover_acentos(texto.lower())
    tokens = re.findall(r"\w+", texto_normalizado)
    palavras = []
    for t in tokens:
        if len(t) < 3 or t in stop_words:
            continue
        if t in {"eletrico", "eletrica", "eletricos", "eletricas", "electric"}:
            palavras.extend(["electric", "hybrid"])
        elif t in {"hibrido", "hibrida", "hibridos", "hibridas"}:
            palavras.append("hybrid")
        else:
            palavras.append(SEARCH_SYNONYMS.get(t, t))
    return palavras


def _carro_bate_filtros(carro: dict, palavras_chave: list) -> bool:
    valores = " ".join(str(v).lower() for v in carro.values() if v)
    valores_normalizados = _remover_acentos(valores)
    return any(palavra in valores_normalizados for palavra in palavras_chave)


def buscar_no_carfit(pergunta: str, offset: int = 0, exclude_ids: list | None = None) -> list:
    palavras_chave = _extrair_palavras_chave(pergunta)
    if exclude_ids is None:
        exclude_ids = []

    if not palavras_chave:
        logger.info("[CarFitAI] Nenhuma palavra-chave extraída da pergunta.")
        return []

    params  = {"offset": offset, "length": FETCH_LIMIT}
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
        car_id = row.get("id") or carro.get("car_id")
        if car_id and car_id in exclude_ids:
            continue
        if _carro_bate_filtros(carro, palavras_chave):
            
            if car_id:
                carro["car_id"] = car_id
            carros_filtrados.append(carro)

    random.shuffle(carros_filtrados)
    carros_filtrados = carros_filtrados[:MAX_RESULTADOS]

    logger.info(
        "[CarFitAI] %d carro(s) encontrado(s) para: '%s'",
        len(carros_filtrados),
        pergunta,
    )
    return carros_filtrados


def formatar_resposta_carfit(carros: list) -> str:
    if not carros:
        return "Não encontrei carros correspondentes para essa consulta."

    
    entries = []
    for carro in carros:
        make = carro.get("make", "").strip()
        model = carro.get("model", "").strip()
        combined = f"{make} {model}".strip()
        if combined:
            entries.append(combined)

    if not entries:
        return ""

    
    return ", ".join(entries) + ". Se quiser detalhes de algum modelo, diga o nome exato (ex: 'Quero o Kia Sorento')."


def formatar_detalhes_carro_carfit(carro: dict) -> str:
    """Formata uma resposta curta e limpa com todos os detalhes de um único carro."""
    if not carro:
        return "Informações do carro não encontradas."

    make = carro.get("make", "").strip()
    model = carro.get("model", "").strip()
    segment = carro.get("segment")
    price = carro.get("price")
    fuel = carro.get("fuel_type")
    seats = carro.get("seats")
    km_per_l = carro.get("km_per_l")
    safety = carro.get("safety_score")
    maintenance = carro.get("maintenance_monthly_est")
    combustivel_map = {
        "gas": "Gasolina",
        "diesel": "Diesel",
        "hybrid": "Híbrido",
        "electric": "Elétrico",
        "flex": "Flex",
    }

    
    segment_map = {
        "7-seater": "7 lugares",
        "7 seater": "7 lugares",
        "mini": "Mini",
        "sedan": "Sedã",
        "crossover": "Crossover",
        "suv": "SUV",
        "hatch": "Hatch",
        "compact": "Compacto",
        "coupe": "Cupê",
    }

    
    cabecalho = f"{make} {model}".strip()
    linhas = [cabecalho, "" ]

    
    campos = []
    if price:
        campos.append(("Preço", f"R$ {price:,.0f}".replace(",", ".")))
    if fuel:
        campos.append(("Combustível", combustivel_map.get(fuel, fuel).capitalize()))
    if seats:
        campos.append(("Lugares", str(seats)))
    if km_per_l:
        try:
            campos.append(("Eficiência", f"{km_per_l:.1f} km/l"))
        except Exception:
            campos.append(("Eficiência", str(km_per_l)))
    if safety:
        campos.append(("Segurança", f"{safety:.1f}/5.0"))
    if maintenance:
        campos.append(("Manutenção estimada", f"R$ {maintenance}/mês"))

    # formatar com vírgulas entre os campos
    if campos:
        campos_str = ", ".join(f"{lbl}: {val}" for lbl, val in campos)
        linhas.append(campos_str)

    linhas.append("")
    linhas.append("Se quiser, posso te mostrar modelos parecidos com esse.")

    return "\n".join(linhas)