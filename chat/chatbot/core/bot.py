import random

from .contexto import Contexto
from ..data import DADOS_CARROS
from ..data.respostas import RESPOSTAS
from ..nlp.preprocess import normalizar
from ..nlp.intent_model import detectar_intencao
from ..services.carro_service import (
    responder_marca,
    responder_carro,
    pedir_outros,
    frase_recomendacao,
    gancho
)

contexto = Contexto()


def responder(texto):
    texto = normalizar(texto)

    resposta_marca = responder_marca(texto, contexto)
    if resposta_marca:
        return resposta_marca

    resposta_carro = responder_carro(texto, contexto)
    if resposta_carro:
        return resposta_carro

    if pedir_outros(texto) and contexto.ultima_intencao in [
        "preco", "tipo_suv", "tipo_sedan", "tipo_hatch", "economia", "potencia", "completo"
    ]:
        intencao = contexto.ultima_intencao
    else:
        intencao = detectar_intencao(texto)

    if intencao != contexto.ultima_intencao:
        contexto.reset(intencao)

    resp, carros_pool = RESPOSTAS.get(intencao, ("Não entendi.", []))
    if carros_pool:
        if "outros" in texto and contexto.previous_carros:
            available = [c for c in carros_pool if c not in contexto.previous_carros]
            pool = available if available else carros_pool
        else:
            pool = carros_pool

        selected = random.sample(pool, min(3, len(pool)))
        carros_info = ", ".join(
            f"{c.upper()} ({DADOS_CARROS[c]['marca']})"
            for c in selected
        )
        resp = frase_recomendacao(intencao, carros_info)
        contexto.carros = selected
        contexto.previous_carros = selected
    else:
        contexto.carros = []

    if intencao not in ["despedida", "saudacao", "nao_entendi", "sobre", "opcoes", "agradecimento"]:
        resp += " " + gancho()

    return resp


if __name__ == "__main__":
    print("Chatbot Recomendacao de Carros (digite 'sair' para encerrar)\n")

    while True:
        user = input("Você: ")
        if user.lower() == "sair":
            break

        resposta = responder(user)
        print("Bot:", resposta)
