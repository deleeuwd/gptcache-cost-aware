import os, requests

class DummyLLM:
    def __init__(self, **kw): pass
    def generate(self, prompt: str) -> str:
        return f"Answer: {hash(prompt) % 100000}"

class OllamaLLM:
    def __init__(self, model="llama3", base_url=None, **kw):
        self.model = model
        self.base = base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    def generate(self, prompt: str) -> str:
        r = requests.post(f"{self.base}/api/generate",
                          json={"model": self.model, "prompt": prompt, "stream": False}, timeout=120)
        r.raise_for_status()
        return r.json().get("response","")

def get_provider(name: str, **kw):
    return DummyLLM(**kw) if name == "dummy" else OllamaLLM(**kw)


