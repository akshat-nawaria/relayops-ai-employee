
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# This is the local address where Ollama accepts AI requests.
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

# Use the model we downloaded for RelayOps.
MODEL_NAME = "qwen3:4b"

# Ask the model a simple question so we can check the connection.
payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": "Reply with exactly this sentence: RelayOps AI connection works."
        }
    ],
    "stream": False
}

# Convert our request into JSON and prepare it for Ollama.
request = Request(
    OLLAMA_URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    # Send the request and wait for the model's response.
    with urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    # Print only the model's final answer.
    print("AI model response:")
    print(result["message"]["content"])

except HTTPError as error:
    # Show the error returned by Ollama if the request fails.
    details = error.read().decode("utf-8", errors="replace")
    print(f"Ollama returned HTTP {error.code}: {details}")

except URLError as error:
    # Explain when Python cannot connect to Ollama.
    print(f"Could not connect to Ollama: {error.reason}")
