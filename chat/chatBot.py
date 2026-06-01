from chatbot import responder

if __name__ == "__main__":
    print("Chatbot Recomendacao de Carros (digite 'sair' para encerrar)\n")

    while True:
        user = input("Voce: ")
        if user.lower() == "sair":
            break

        resposta = responder(user)
        print("Bot:", resposta)
