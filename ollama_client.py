
import requests


class OllamaClient:
    def __init__(
        self,
        base_url="http://127.0.0.1:11434",
        embed_model="nomic-embed-text",
        gen_model="llama3.2",
    ):
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.gen_model = gen_model
        self.session = requests.Session()

    def is_available(self):
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=3,
            )
            return response.ok
        except requests.RequestException:
            return False

    def embed(self, text):
        """Return an embedding using Ollama's current /api/embed API."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.embed_model,
                    "input": text,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])

            if not embeddings:
                return []

            return embeddings[0]
        except (requests.RequestException, ValueError):
            return []

    def generate(self, prompt):
        """Generate a non-streaming answer with the configured model."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.gen_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=(5, 180),
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except (requests.RequestException, ValueError):
            return ""
