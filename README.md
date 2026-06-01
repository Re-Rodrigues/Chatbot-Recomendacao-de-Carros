# 🚗 Chatbot de Recomendação de Carros

> Assistente inteligente para recomendação de veículos — desenvolvido para a disciplina de **Inteligência Artificial e Machine Learning**

---

## 👥 Equipe

| Nome |
|------|
| Augusto Cézar de Almeida Pinto |
| Gabriel Mazilao Ferreira da Silva |
| Renan Rodrigues da Silva |

---

## 🚀 Como executar

**1. Ative o ambiente virtual**

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**2. Instale as dependências**

```bash
pip install -r requirements.txt
```

**3. Baixe os recursos do NLTK**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

**4. Inicie a aplicação**

```bash
python chat/app.py
```

Acesse em: `http://localhost:5000`

---

## 📁 Estrutura do Projeto

```
chat/
├── app.py                       # Aplicação Flask principal
├── chatBot.py                   # Lógica do chatbot
├── chatbot/                     # Módulo principal
│   ├── core/                    # Núcleo do chatbot
│   │   ├── bot.py               # Respostas e lógica
│   │   └── contexto.py          # Gerenciamento de contexto
│   ├── data/                    # Dados do chatbot
│   │   ├── carros.py            # Base de dados de carros
│   │   ├── intents.py           # Intenções e frames
│   │   └── respostas.py         # Templates de respostas
│   ├── nlp/                     # Processamento de linguagem natural
│   │   ├── preprocess.py        # Normalização de texto
│   │   └── intent_model.py      # Modelo de detecção de intenção
│   └── services/                # Serviços
│       └── carro_service.py     # Lógica de recomendação de carros
├── static/                      # Arquivos estáticos
│   ├── script.js                # JavaScript frontend
│   └── style.css                # Estilos CSS
└── templates/                   # Templates HTML
    └── index.html               # Interface do chatbot
```

---
