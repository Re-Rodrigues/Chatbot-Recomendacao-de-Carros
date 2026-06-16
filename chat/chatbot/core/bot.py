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
    gancho,
    gancho_carro
)
from ..services.carfit_service import (
    buscar_no_carfit,
    formatar_resposta_carfit,
    formatar_detalhes_carro_carfit,
)
from ..services.llm_service import gerar_resposta_llm

contexto = Contexto()


def responder(texto):
    texto = normalizar(texto)

    resposta_carro = responder_carro(texto, contexto)
    if resposta_carro:
        return resposta_carro

    
    carros_carfit_match = buscar_no_carfit(texto)
    if carros_carfit_match:
        texto_lower = texto.lower()
        for carro in carros_carfit_match:
            modelo = (carro.get("model") or "").lower()
            marca = (carro.get("make") or "").lower()

            
            STOP_MODEL_TOKENS = {"base", "limited", "plus", "urban", "sport", "edition", "premium", "lx", "ex", "se", "gt"}
            modelo_tokens = [t for t in modelo.split() if len(t) >= 3 and t not in STOP_MODEL_TOKENS]
            if modelo_tokens and any(token in texto_lower for token in modelo_tokens):
                return formatar_detalhes_carro_carfit(carro)

            
            if marca and modelo and f"{marca} {modelo}" in texto_lower:
                return formatar_detalhes_carro_carfit(carro)

    resposta_marca = responder_marca(texto, contexto)
    if resposta_marca:
        return resposta_marca

    if pedir_outros(texto) and contexto.ultima_intencao in [
        "preco", "tipo_suv", "tipo_sedan", "tipo_hatch", "economia", "potencia", "completo", "tipo_eletrico"
    ]:
        intencao = contexto.ultima_intencao
    else:
        intencao = detectar_intencao(texto)

    if intencao != contexto.ultima_intencao:
        contexto.reset(intencao)

    if intencao == "tipo_eletrico":
        
        if pedir_outros(texto) and contexto.ultima_intencao == "tipo_eletrico":
            offset = contexto.carfit_offset
            exclude = contexto.carfit_previous_ids
            query = contexto.carfit_query or texto
        else:
            offset = 0
            exclude = []
            query = texto

        
        if offset == 0:
            contexto.carfit_query = query

        carros_carfit = buscar_no_carfit(query, offset=offset, exclude_ids=exclude)
        if carros_carfit:
            
            for c in carros_carfit:
                cid = c.get("car_id")
                if cid and cid not in contexto.carfit_previous_ids:
                    contexto.carfit_previous_ids.append(cid)
            contexto.carfit_offset += len(carros_carfit)
            return formatar_resposta_carfit(carros_carfit)

        
        resposta_llm = gerar_resposta_llm(texto)
        if resposta_llm:
            return f"{resposta_llm}\n\n(uma API foi usada nesta pesquisa)"

    resp, carros_pool = RESPOSTAS.get(intencao, ("Não entendi.", []))
    if carros_pool:
        if "outros" in texto and contexto.previous_carros:
            available = [c for c in carros_pool if c not in contexto.previous_carros]
            pool = available if available else carros_pool
        else:
            pool = carros_pool

        selected = random.sample(pool, min(3, len(pool)))
        
        # Formatar a lista de carros
        lista_carros = ", ".join(
            f"{c.upper()} ({DADOS_CARROS[c]['marca']})"
            for c in selected
        )
        
        # Usar frase_recomendacao para intencoes com gancho melhor
        if intencao in ["preco", "economia", "potencia", "tipo_suv", "tipo_sedan", "tipo_hatch", "completo"]:
            resp = frase_recomendacao(intencao, lista_carros) + gancho()
        else:
            # Para outros tipos, usar formato padrão
            titulos = {
                "tipo_eletrico": "Carros elétricos",
            }
            titulo = titulos.get(intencao, "Carros recomendados")
            resp = f"{titulo}:\n\n{lista_carros}\n\nGostou desses carros?"
        
        contexto.carros = selected
        contexto.previous_carros = selected
    else:
        contexto.carros = []

    if intencao not in ["despedida", "saudacao", "nao_entendi", "sobre", "opcoes", "agradecimento"]:
        if intencao not in ["preco", "tipo_suv", "tipo_sedan", "tipo_hatch", "economia", "potencia", "completo", "tipo_eletrico"]:
            resp += "\n\n" + gancho()

    if intencao == "nao_entendi":
        carros_carfit = buscar_no_carfit(texto)
        if carros_carfit:
            return formatar_resposta_carfit(carros_carfit)

        resposta_llm = gerar_resposta_llm(texto)
        if resposta_llm:
            return f"{resposta_llm}\n\n(uma API foi usada nesta pesquisa)"
        return resp

    return resp


if __name__ == "__main__":
    print("Chatbot Recomendacao de Carros (digite 'sair' para encerrar)\n")

    while True:
        user = input("Você: ")
        if user.lower() == "sair":
            break

        resposta = responder(user)
        print("Bot:", resposta)
