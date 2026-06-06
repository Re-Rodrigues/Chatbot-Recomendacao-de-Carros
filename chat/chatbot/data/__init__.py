from .intents import FRAMES, INTENCOES, LABELS
from .respostas import RESPOSTAS

import json
from pathlib import Path

arquivo_json = Path(__file__).parent / "carros.json"

with open(arquivo_json, "r", encoding="utf-8") as arquivo:
    carros = json.load(arquivo)

DADOS_CARROS = {
    carro["nome"]: {
        "consumo": carro["consumo"],
        "cambio": carro["cambio"],
        "potencia": carro["potencia"],
        "tipo": carro["tipo"],
        "marca": carro["marca"],
        "ano": str(carro["ano"])
    }
    for carro in carros
}

__all__ = ["DADOS_CARROS", "FRAMES", "INTENCOES", "LABELS", "RESPOSTAS"]
