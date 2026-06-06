import random
from ..data import DADOS_CARROS
from ..data.respostas import RESPOSTAS

MARCAS = sorted(set(dados["marca"] for dados in DADOS_CARROS.values()))


def obter_carros_por_tipo(tipo):
    return [carro for carro, dados in DADOS_CARROS.items() if dados["tipo"] == tipo]


def obter_carros_por_marca(marca):
    return [carro for carro, dados in DADOS_CARROS.items() if dados["marca"] == marca]


def detectar_marca(texto):
    for marca in MARCAS:
        if marca in texto:
            return marca
    return None


def detectar_carro(texto):
    for carro in DADOS_CARROS.keys():
        if carro in texto:
            return carro
    return None


def responder_marca(texto, contexto):
    marca = detectar_marca(texto)
    if marca:
        if contexto.marca_foco != marca:
            contexto.reset_marca(marca)
            contexto.carros_marca_pool = obter_carros_por_marca(marca)

        carros_marca = contexto.carros_marca_pool
        if carros_marca:
            if "outros" in texto and contexto.previous_carros:
                available = [c for c in carros_marca if c not in contexto.previous_carros]
                pool = available if available else carros_marca
            else:
                pool = carros_marca

            selected = pool[: min(4, len(pool))]
            carros_info = ", ".join(
                f"{c.upper()} ({DADOS_CARROS[c]['ano']})"
                for c in selected
            )
            contexto.previous_carros = selected
            return f"Carros {marca.upper()}: {carros_info}."
    return None


def responder_carro(texto, contexto):
    carro = detectar_carro(texto)
    if carro:
        contexto.carro_foco = carro
        d = DADOS_CARROS[carro]

        return (
            f"O {carro.upper()} é um ótimo carro da {d['marca']}, ano {d['ano']}. "
            f"Ele se destaca por ter consumo {d['consumo']}, câmbio {d['cambio']}, "
            f"potência {d['potencia']} e é do tipo {d['tipo']}. "
            f"É uma opção interessante dependendo do que você procura."
            + gancho_carro()
        )

    return None


def gancho():
    return "\n\n" + random.choice([
        "O que achou das opções? Se quiser posso te mostrar outros modelos, é só me falar o que busca.",
        "Curtiu algum desses? Posso te mostrar mais detalhes ou sugerir outros parecidos.",
        "Algum te chamou atenção? Posso trazer mais informações ou outras opções.",
        "Quer ver mais opções ou prefere detalhes de algum desses modelos?",
        "Se quiser algo mais específico, me diz o que você procura que eu refino as sugestões.",
        "Posso te mostrar mais modelos ou detalhar algum desses, o que prefere?",
        "Tá buscando algo mais específico? Me fala que eu ajusto as opções pra você.",
        "Se nenhum desses te agradou, posso sugerir outros com base no que você quer.",
        "Quer comparar algum desses modelos ou ver mais alternativas?",
        "Posso te ajudar a escolher melhor — quer mais opções ou detalhes de algum?",
        "Se quiser, posso filtrar melhor as opções pra você. O que é mais importante: preço, consumo ou potência?",
        "Quer seguir vendo opções ou prefere analisar melhor algum desses?",
        "Me diz o que você prioriza que eu tento achar algo mais certeiro pra você.",
        "Quer ver mais sugestões ou explorar melhor algum desses modelos?",
        "Se quiser algo diferente, me fala o tipo de carro que você quer que eu busco pra você."
    ])


def gancho_carro():
    return "\n\n" + random.choice([
        "O que achou desse modelo? Se quiser posso te mostrar outras opções!",
        "Se quiser, posso te mostrar modelos parecidos com esse.",
        "Posso te sugerir alternativas parecidas, só dizer o que gostou nele!",
    ])


def pedir_outros(texto):
    if any(term in texto for term in ["outros tipos", "outro tipo", "outros tipo", "outros tipos de carro", "outros tipos de"]):
        return False
    return "outros" in texto or "outro" in texto


def frase_recomendacao(intencao, carros_info):
    frases = {
        "preco": [
            f"Se a ideia é economizar, essas são boas opções: {carros_info}.",
            f"Modelos com bom custo-benefício: {carros_info}."
        ],
        "economia": [
            f"Boas opções com baixo consumo são: {carros_info}.",
            f"Se você busca economia, esses modelos se destacam: {carros_info}."
        ],
        "potencia": [
            f"Se você quer mais desempenho, olha esses modelos: {carros_info}.",
            f"Esses carros se destacam pela potência: {carros_info}."
        ],
        "tipo_suv": [
            f"Se você curte SUV, essas são ótimas opções: {carros_info}.",
            f"Esses SUVs podem te interessar: {carros_info}."
        ],
        "tipo_sedan": [
            f"Se a ideia é um sedan, vale dar uma olhada nesses: {carros_info}.",
            f"Esses sedans são boas opções: {carros_info}."
        ],
        "tipo_hatch": [
            f"Se você procura um hatch, esses aqui são boas escolhas: {carros_info}.",
            f"Esses hatches podem te agradar: {carros_info}."
        ],
        "completo": [
            f"Se você quer um carro mais completo, olha essas opções: {carros_info}.",
            f"Esses modelos se destacam pelo nível de equipamentos: {carros_info}."
        ]
    }

    if intencao in frases:
        return random.choice(frases[intencao])

    return f"Aqui vão algumas opções: {carros_info}."
