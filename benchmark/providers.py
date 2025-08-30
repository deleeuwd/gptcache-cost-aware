import os, requests, random, time

class DummyLLM:
    def __init__(self, simulate_latency: bool = True, random_response: bool = True, multiplier: float = 5, seed: int = None, **kw):
        # If simulate_latency is False, generate instantly (useful for exact timing tests)
        # random_response controls whether the answer length multiplier is drawn randomly.
        # If multiplier is provided it will be used as a fixed multiplier (overrides random_response).
        self.simulate_latency = simulate_latency
        self.random_response = random_response
        # Use a local RNG if seed provided for deterministic behavior
        self._rng = random.Random(seed) if seed is not None else random
        self.multiplier = (self._rng.uniform(2, 10) if random_response else multiplier)

    def generate(self, prompt: str) -> str:
        # Determine answer length based on multiplier
        answer_length = int(len(prompt) * self.multiplier)

        # Create a realistic answer by repeating/extending content
        base_answer = f"This is a comprehensive answer to your question about: {prompt[:50]}..."
        answer = (base_answer * ((answer_length // len(base_answer)) + 1))[:answer_length]

        if self.simulate_latency:
            # Fast simulation: wait len(answer)/10 milliseconds
            simulated_ms = len(answer) / 10.0
            time.sleep(simulated_ms / 1000.0)

        return answer

class OllamaLLM:
    def __init__(self, model="llama3", base_url=None, **kw):
        self.model = model
        self.base = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    def generate(self, prompt: str) -> str:
        r = requests.post(f"{self.base}/api/generate",
                          json={"model": self.model, "prompt": prompt, "stream": False}, timeout=300)
        r.raise_for_status()
        return r.json().get("response","")

def get_provider(name: str, **kw):
    return DummyLLM(**kw) if name == "dummy" else OllamaLLM(**kw)


