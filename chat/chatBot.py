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
    "oi", "ola", "bom dia", "tudo bem", "como vai", "e ai", "opa",
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
    "saudacao","saudacao","saudacao","saudacao","saudacao","saudacao","saudacao",
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
    return random.choice([
        "Não gostou das opções? Se quiser posso te mostrar outros modelos, só dizer o que procura.",
        "O que achou desses? Se quiser posso te mostrar outros modelos, só dizer o que está procurando.",
        "Gostou de algum? Se quiser posso te mostrar as especificações de algum deles, só dizer o modelo que gostou."
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
            f"{carro.upper()} ({d['marca']} {d['ano']}): "
            f"consumo {d['consumo']}, câmbio {d['cambio']}, "
            f"potência {d['potencia']}, tipo {d['tipo']}."
        )

    return None

def detectar_intencao(texto):
    if texto == "nada com nada":
        return "nao_entendi"

    if any(word in texto for word in ["tchau", "adeus", "ate logo", "flw", "falou"]):
        return "despedida"

    if any(word in texto for word in ["sedan", "seda"]):
        return "tipo_sedan"
    if "hatch" in texto:
        return "tipo_hatch"
    if "suv" in texto:
        return "tipo_suv"
    if "barato" in texto:
        return "preco"
    if "economico" in texto:
        return "economia"
    if "potente" in texto:
        return "potencia"
    if "completo" in texto:
        return "completo"
    if any(word in texto for word in ["quem", "oque", "sobre"]) and "voce" in texto:
        return "sobre"
    if "tipo" in texto or "tipos" in texto:
        return "opcoes"
    if any(word in texto for word in ["opcoes", "ajuda", "recomendar", "oferece", "mostre", "mostra", "lista", "pode", "pode mostrar"]) or ("o que" in texto and "voce" in texto):
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
        resp = f"Carros {LABELS.get(intencao, intencao)}: {carros_info}."
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