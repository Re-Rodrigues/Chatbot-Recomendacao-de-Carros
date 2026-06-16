from flask import Flask, render_template, request, jsonify
from chatbot import responder
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    mensagem = request.json["mensagem"]
    resposta = responder(mensagem)
    return jsonify({"resposta": resposta})

if __name__ == "__main__":
    app.run(debug=True)