from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from .preprocess import normalizar, preprocess
from chatbot.data.intents import FRAMES, INTENCOES

vectorizer = CountVectorizer()
X = vectorizer.fit_transform([preprocess(f) for f in FRAMES])
modelo = MultinomialNB()
modelo.fit(X, INTENCOES)

PALAVRAS_CARRO = {
    "carro", "carros", "veiculo", "veiculos", "automovel", "automoveis",
    "modelo", "modelos", "marca", "marcas", "motor", "cambio", "potencia",
    "consumo", "preco", "barato", "economico", "suv", "sedan", "hatch",
    "eletrico", "hibrido", "pickup", "minivan", "utilitario", "completo",
    "recomend", "opcoes", "opcao", "tipo", "tipos", "comprar", "escolher",
    "qual", "quais", "melhor", "bom", "boa", "indica", "indique", "sugere",
    "toyota", "honda", "ford", "chevrolet", "volkswagen", "fiat", "nissan",
    "hyundai", "kia", "renault", "peugeot", "jeep", "bmw", "mercedes",
    "audi", "volvo", "mitsubishi", "subaru", "mazda", "suzuki",
}


def _tem_contexto_carro(texto: str) -> bool:
    tokens = texto.lower().split()
    return any(
        any(token.startswith(palavra) for palavra in PALAVRAS_CARRO)
        for token in tokens
    )


def detectar_intencao(texto):
    texto = normalizar(texto)

    if texto == "nada com nada":
        return "nao_entendi"

    if any(word in texto for word in ["tchau", "adeus", "ate logo", "flw", "falou", "xau", "xau xau", "vlw", "valeu", "brigado", "brigad", "brigada", "obrigado", "obrigada", "obrigad"]):
        return "despedida"
    if any(word in texto for word in ["eletrico", "eletricos", "eletrica", "eletricas", "hibrido", "hibridos", "hibrida", "hibridas"]):
        return "tipo_eletrico"
    if any(word in texto for word in ["sedan", "seda", "sedans", "sedas"]) and not any(word in texto for word in ["hatch", "hatchback", "suv", "utilitario", "pickup", "minivan", "eletrico"]):
        return "tipo_sedan"
    if any(word in texto for word in ["hatch", "hatchback", "hatchbacks"]) and not any(word in texto for word in ["sedan", "seda", "sedans", "sedas", "suv", "utilitario", "pickup", "minivan", "eletrico"]):
        return "tipo_hatch"
    if any(word in texto for word in ["suv", "suvs", "grande", "grandes", "alto", "altos"]) and not any(word in texto for word in ["sedan", "seda", "sedans", "sedas", "hatch", "hatchback", "utilitario", "pickup", "minivan", "eletrico"]):
        return "tipo_suv"
    if any(word in texto for word in ["barato", "baratos", "preco baixo", "preço baixo", "em conta", "acessivel", "acessível", "baixo preço", "baixo preco"]):
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

    if not _tem_contexto_carro(texto):
        return "nao_entendi"

    v = vectorizer.transform([preprocess(texto)])
    if v.nnz == 0:
        return "nao_entendi"

    intencao = modelo.predict(v)[0]
    if intencao == "saudacao" and not any(greet in texto for greet in ["oi", "ola", "bom", "dia", "eae", "fala", "salve", "opa"]):
        return "nao_entendi"
    if intencao == "agradecimento" and not any(word in texto for word in ["obrigado", "obrigada", "valeu", "brigado", "brigad"]):
        return "nao_entendi"

    return intencao