import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"


def test_llm():
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: Enterprise AI assistant is working."
            }
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()
        print("Model:", result.get("model", MODEL))
        print("Response:", result["message"]["content"])

    except requests.exceptions.RequestException as error:
        print("Ollama connection failed:", error)


if __name__ == "__main__":
    test_llm()
