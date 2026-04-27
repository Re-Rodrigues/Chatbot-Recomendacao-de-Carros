from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import unicodedata

from dados_carros import DADOS_CARROS

def normalizar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
    return texto

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(normalizar(text))
    return " ".join([lemmatizer.lemmatize(t) for t in tokens])


def obter_carros_por_tipo(tipo):
    return [carro for carro, dados in DADOS_CARROS.items() if dados["tipo"] == tipo]

FRAMES = [
    "oi", "ola", "bom dia", "tudo bem", "como vai", "e ai", "opa", "salve", "fala",
    "quero um carro barato", "tem carro barato", "carro em conta",
    "quero um suv", "carro alto",
    "quero um sedan", "tem sedan", "sedan", "seda",
    "quero um hatch", "tem hatch", "hatch", "hatchback",
    "carro economico", "baixo consumo", "bebe pouco",
    "carro potente",
    "carro completo",
    "quem e voce", "o que voce e", "sobre voce", "quem voce e",
    "o que voce tem", "o que voce pode recomendar", "o que voce oferece", "quais opcoes", "me ajude", "outros modelos",
    "obrigado", "obrigada", "muito obrigado", "muito obrigada", "valeu", "brigado", "brigad",
    "tchau", "adeus", "ate logo"
]

INTENCOES = [
    "saudacao","saudacao","saudacao","saudacao","saudacao","saudacao","saudacao","saudacao","saudacao",
    "preco","preco","preco",
    "tipo_suv","tipo_suv",
    "tipo_sedan","tipo_sedan","tipo_sedan","tipo_sedan",
    "tipo_hatch","tipo_hatch","tipo_hatch","tipo_hatch",
    "economia","economia","economia",
    "potencia",
    "completo",
    "sobre","sobre","sobre","sobre",
    "opcoes","opcoes","opcoes","opcoes","opcoes","opcoes",
    "agradecimento","agradecimento","agradecimento","agradecimento","agradecimento","agradecimento","agradecimento",
    "despedida","despedida","despedida"
]

RESPOSTAS = {
    "preco": ("", ["kwid","mobi","argo","uno","etios","palio","fox","fiesta","idea","clio","gol","hb20","onix","208","sandero"]),
    "tipo_suv": ("", obter_carros_por_tipo("suv")),
    "tipo_sedan": ("", obter_carros_por_tipo("sedan")),
    "tipo_hatch": ("", obter_carros_por_tipo("hatch")),
    "economia": ("", ["corolla","versa","yaris sedan","city","tracker","kicks"]),
    "potencia": ("", ["civic","jetta","cruze","compass","trailblazer","hilux","ranger","s10","amarok","titan","frontier"]),
    "completo": ("", ["polo","corolla","civic","virtus","versa","yaris sedan","jetta","cruze","creta","renegade","t cross","tracker","kicks","pulse","fastback","captur","compass","hr-v","ecosport","ix35","vitara","trailblazer","hilux","ranger","s10","amarok","titan","frontier"]),
    "saudacao": ("Olá! O que você quer ver? Se quiser posso te mostrar nossas opções de carros.", []),
    "sobre": ("Sou um chatbot que recomenda carros baseado nas suas preferências. Só dizer o que procura!", []),
    "opcoes": ("Temos várias opções de carros. O que você está procurando? um carro barato, econômico, potente, um SUV, um sedan, um hatch ou mais completo?", []),
    "agradecimento": ("De nada! Precisando estou aqui.", []),
    "despedida": ("Fechou! Boa sorte 🚗", []),
    "nao_entendi": ("Não entendi, pode repetir?", [])
}

LABELS = {
    "preco": "baratos",
    "tipo_suv": "SUV",
    "tipo_sedan": "sedans",
    "tipo_hatch": "hatches",
    "economia": "econômicos",
    "potencia": "potentes",
    "completo": "completos"
}

MARCAS = sorted(set(carro["marca"] for carro in DADOS_CARROS.values()))

def obter_carros_por_marca(marca):
    return [carro for carro, dados in DADOS_CARROS.items() if dados["marca"] == marca]

class Contexto:
    def __init__(self):
        self.ultima_intencao = None
        self.carros = []
        self.carro_foco = None
        self.previous_carros = []
        self.marca_foco = None
        self.carros_marca_pool = []

    def reset(self, intencao):
        self.ultima_intencao = intencao
        self.carro_foco = None
        self.carros = []
        self.previous_carros = []
    
    def reset_marca(self, marca):
        self.marca_foco = marca
        self.carros_marca_pool = []

contexto = Contexto()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform([preprocess(f) for f in FRAMES])

modelo = MultinomialNB()
modelo.fit(X, INTENCOES)

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

def responder_marca(texto):
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
            
            selected = pool[:min(4, len(pool))]
            carros_info = ", ".join(
                f"{c.upper()} ({DADOS_CARROS[c]['ano']})" 
                for c in selected
            )
            contexto.previous_carros = selected
            return f"Carros {marca.upper()}: {carros_info}."
    return None

def responder_carro(texto):
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

def gancho_carro():
    return "\n\n" + random.choice([
        "O que achou desse modelo? Se quiser posso te mostrar outras opções!",
        "Se quiser, posso te mostrar modelos parecidos com esse.",
        "Posso te sugerir alternativas parecidas, só dizer o que gostou nele!",
    ])

def detectar_intencao(texto):
    if texto == "nada com nada":
        return "nao_entendi"

    if any(word in texto for word in ["tchau", "adeus", "ate logo", "flw", "falou", "xau", "xau xau", "vlw", "valeu", "brigado", "brigad", "brigada", "obrigado", "obrigada", "obrigad"]):
        return "despedida"
    if any(word in texto for word in ["sedan", "seda", "sedans", "sedas"]) and not any(word in texto for word in ["hatch", "hatchback", "suv", "utilitario", "pickup", "minivan", "eletrico"]):
        return "tipo_sedan"
    if any(word in texto for word in ["hatch", "hatchback", "hatchbacks"]) and not any(word in texto for word in ["sedan", "seda", "sedans", "sedas", "suv", "utilitario", "pickup", "minivan", "eletrico"]):
        return "tipo_hatch"
    if any(word in texto for word in ["suv", "suvs", "grande", "grandes", "alto", "altos"]) and not any(word in texto for word in ["sedan", "seda", "sedans", "sedas", "hatch", "hatchback", "utilitario", "pickup", "minivan", "eletrico"]):
        return "tipo_suv"
    if any(word in texto for word in ["barato", "baratos", "preço baixo", "preco baixo", "em conta", "acessivel", "acessível", "baixo preço", "baixo preco"]):
        return "preco"
    if any(word in texto for word in ["economico", "economia", "baixo consumo", "bebe pouco", "consome pouco", "consumo baixo"]):
        return "economia"
    if any(word in texto for word in ["potente", "potencia", "potente", "potencia alta", "muito potente", "muita potencia", "forte", "fortes"]):
        return "potencia"
    if any(word in texto for word in ["completo", "completos"]):
        return "completo"
    if any(word in texto for word in ["quem", "oque", "sobre"]) and "voce" in texto:
        return "sobre"
    if "tipo" in texto or "tipos" in texto:
        return "opcoes"
    if any(word in texto for word in ["opcoes", "ajuda", "recomendar", "oferece", "mostre", "mostra", "lista", "pode", "pode mostrar", "quero ver", "mostra, quero ver"]) or ("o que" in texto and "voce" in texto):
        return "opcoes"
    if any(word in texto for word in ["obrigado", "obrigada", "valeu", "brigado", "brigad"]):
        return "agradecimento"

    return None

def pedir_outros(texto):
    if any(term in texto for term in ["outros tipos", "outro tipo", "outros tipo", "outros tipos de carro", "outros tipos de"]):
        return False
    return "outros" in texto or "outro" in texto


def responder(texto):
    texto = normalizar(texto)

    r = responder_marca(texto)
    if r:
        return r

    r = responder_carro(texto)
    if r:
        return r

    if pedir_outros(texto) and contexto.ultima_intencao in ["preco", "tipo_suv", "tipo_sedan", "tipo_hatch", "economia", "potencia", "completo"]:
        intencao = contexto.ultima_intencao
    else:
        intencao = detectar_intencao(texto)

        if not intencao:
            v = vectorizer.transform([preprocess(texto)])
            if v.nnz == 0:
                intencao = "nao_entendi"
            else:
                intencao = modelo.predict(v)[0]
                if intencao == "saudacao" and not any(greet in texto for greet in ["oi", "ola", "bom", "dia", "eae", "fala", "salve", "opa"]):
                    intencao = "nao_entendi"
                if intencao == "agradecimento" and not any(word in texto for word in ["obrigado", "obrigada", "valeu", "brigado", "brigad"]):
                    intencao = "nao_entendi"

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


if __name__ == "__main__":
    print("Chatbot Recomendacao de Carros (digite 'sair' para encerrar)\n")

    while True:
        user = input("Você: ")
        if user.lower() == "sair":
            break

        resposta = responder(user)
        print("Bot:", resposta)