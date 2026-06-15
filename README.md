# Chatbot de Recomendação de Carros

> Assistente inteligente para recomendação de veículos — desenvolvido para a disciplina de **Inteligência Artificial e Machine Learning**

---

## Equipe

| Nome |
|------|
| Augusto Cézar de Almeida Pinto |
| Gabriel Mazilao Ferreira da Silva |
| Renan Rodrigues da Silva |

---

## Como executar

```bash
winget install Python.Python.3.11
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
python chat/app.py
