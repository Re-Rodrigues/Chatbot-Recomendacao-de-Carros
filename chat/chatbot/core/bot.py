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


def _montar_contexto_para_llm(carros_local: list, carros_carfit: list) -> str:
    linhas = []

    if carros_local:
        linhas.append("Carros encontrados na base local:")
        for c in carros_local[:3]:
            d = DADOS_CARROS.get(c, {})
            linhas.append(
                f"- {c.upper()} ({d.get('marca','')}, {d.get('ano','')}): "
                f"tipo {d.get('tipo','')}, consumo {d.get('consumo','')}, "
                f"câmbio {d.get('cambio','')}, potência {d.get('potencia','')}"
            )

    if carros_carfit:
        linhas.append("\nCarros encontrados no CarFitAI:")
        for c in carros_carfit[:3]:
            linhas.append(
                f"- {c.get('make','')} {c.get('model','')} ({c.get('segment','')}): "
                f"combustível {c.get('fuel_type','')}, "
                f"preço R$ {c.get('price','')}, "
                f"eficiência {round(c.get('km_per_l') or 0, 1)} km/l, "
                f"segurança {c.get('safety_score','')}/5"
            )

    return "\n".join(linhas)


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
            marca  = (carro.get("make")  or "").lower()

            STOP_MODEL_TOKENS = {"base","limited","plus","urban","sport","edition","premium","lx","ex","se","gt"}
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

    _, carros_pool = RESPOSTAS.get(intencao, ("", []))

    if carros_pool:
        if "outros" in texto and contexto.previous_carros:
            available = [c for c in carros_pool if c not in contexto.previous_carros]
            pool = available if available else carros_pool
        else:
            pool = carros_pool
        carros_local_selected = random.sample(pool, min(3, len(pool)))
        contexto.carros = carros_local_selected
        contexto.previous_carros = carros_local_selected
    else:
        carros_local_selected = []
        contexto.carros = []

    carros_carfit_selected = []
    if intencao == "tipo_eletrico":
        if pedir_outros(texto) and contexto.ultima_intencao == "tipo_eletrico":
            offset  = contexto.carfit_offset
            exclude = contexto.carfit_previous_ids
            query   = contexto.carfit_query or texto
        else:
            offset  = 0
            exclude = []
            query   = texto
            contexto.carfit_query = query

        carros_carfit_selected = buscar_no_carfit(query, offset=offset, exclude_ids=exclude)
        if carros_carfit_selected:
            for c in carros_carfit_selected:
                cid = c.get("car_id")
                if cid and cid not in contexto.carfit_previous_ids:
                    contexto.carfit_previous_ids.append(cid)
            contexto.carfit_offset += len(carros_carfit_selected)

    INTENCOES_CARROS = [
        "preco", "tipo_suv", "tipo_sedan", "tipo_hatch",
        "economia", "potencia", "completo", "tipo_eletrico"
    ]

    if intencao in INTENCOES_CARROS and (carros_local_selected or carros_carfit_selected):
        contexto_llm = _montar_contexto_para_llm(carros_local_selected, carros_carfit_selected)
        prompt_llm = (
            f"O usuário perguntou: '{texto}'\n\n"
            f"Aqui estão os carros encontrados nas bases de dados:\n{contexto_llm}\n\n"
            "Com base apenas nesses carros, recomende em português o mais adequado para o usuário, "
            "explicando brevemente o motivo da escolha. Seja direto e amigável."
        )
        resposta_llm = gerar_resposta_llm(prompt_llm)
        if resposta_llm:
            return f"{resposta_llm}\n\n(recomendação gerada por IA)"

        if carros_local_selected:
            lista = ", ".join(f"{c.upper()} ({DADOS_CARROS[c]['marca']})" for c in carros_local_selected)
            return frase_recomendacao(intencao, lista) + gancho()
        if carros_carfit_selected:
            return formatar_resposta_carfit(carros_carfit_selected)

    resp, _ = RESPOSTAS.get(intencao, ("Não entendi.", []))

    if intencao not in ["despedida", "saudacao", "nao_entendi", "sobre", "opcoes", "agradecimento"]:
        if intencao not in INTENCOES_CARROS:
            resp += "\n\n" + gancho()

    if intencao == "nao_entendi":
        carros_carfit = buscar_no_carfit(texto)
        if carros_carfit:
            contexto_llm = _montar_contexto_para_llm([], carros_carfit)
            prompt_llm = (
                f"O usuário perguntou: '{texto}'\n\n"
                f"Aqui estão os carros encontrados:\n{contexto_llm}\n\n"
                "Recomende em português o mais adequado, explicando brevemente. Seja direto e amigável."
            )
            resposta_llm = gerar_resposta_llm(prompt_llm)
            if resposta_llm:
                return f"{resposta_llm}\n\n(recomendação gerada por IA)"
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