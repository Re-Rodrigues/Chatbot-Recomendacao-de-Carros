from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Dataset de treino
frases = [
    "oi", "olá", "bom dia",
    "quero um carro barato", "tem carro barato", "mais barato",
    "quero um suv", "preciso de um suv", "carro alto",
    "quero carro econômico", "que gaste pouco", "baixo consumo",
    "quero carro potente", "carro forte", "muito potente",
    "tchau", "até logo"
]

intencoes = [
    "saudacao", "saudacao", "saudacao",
    "preco", "preco", "preco",
    "tipo_suv", "tipo_suv", "tipo_suv",
    "economia", "economia", "economia",
    "potencia", "potencia", "potencia",
    "despedida", "despedida"
]

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    lemas = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemas)


frases_proc = [preprocess(f) for f in frases]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(frases_proc)

modelo = MultinomialNB()
modelo.fit(X, intencoes)

carros = {
    "preco": "Sugestão: Renault Kwid ou Fiat Mobi (mais baratos)",
    "tipo_suv": "Sugestão: Hyundai Creta ou Jeep Renegade",
    "economia": "Sugestão: Toyota Corolla Hybrid ou Onix Plus",
    "potencia": "Sugestão: Honda Civic Touring ou Jetta GLI",
    "saudacao": "Olá! Posso recomendar um carro. O que você procura?",
    "despedida": "Até mais! Boa sorte na escolha do carro!"
}

print("Chatbot de recomendação de carros (digite 'sair' para encerrar)\n")

while True:
    entrada = input("Você: ")

    if entrada.lower() == "sair":
        break

    entrada_proc = preprocess(entrada)
    entrada_vector = vectorizer.transform([entrada_proc])
    intencao_prevista = modelo.predict(entrada_vector)[0]

    resposta = carros.get(
        intencao_prevista, "Não entendi, pode explicar melhor?")
    print("Bot:", resposta)
