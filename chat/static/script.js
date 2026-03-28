async function enviar() {
    const input = document.getElementById("input");
    const mensagem = input.value;

    if (!mensagem.trim()) return;

    const chat = document.getElementById("chat");

    chat.innerHTML += `<div class="msg user">Você: ${mensagem}</div>`;

    const res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ mensagem })
    });

    const data = await res.json();

    chat.innerHTML += `<div class="msg bot">Bot: ${data.resposta}</div>`;

    input.value = "";
    chat.scrollTop = chat.scrollHeight;
}

document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("input");

    input.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            enviar();
        }
    });
});