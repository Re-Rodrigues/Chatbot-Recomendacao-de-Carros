import unicodedata
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def normalizar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
    return texto


def preprocess(texto):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(normalizar(texto))
    return " ".join([lemmatizer.lemmatize(t) for t in tokens])
