#LLM provider abstraction: makes the generator backend (Ollama local / Groq cloud) interchangeable.

from abc import ABC, abstractmethod
import os
import ollama
import config

#Send a prompt to the backend, return its text answer.
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, images: list[str] = None) -> str:
        raise NotImplementedError

#Local, private inference via Ollama.
class OllamaProvider(LLMProvider):

    def __init__(self, model: str = None):
        self.model = model or config.GENERATOR_MODEL

    def generate(self, prompt: str, images: list[str] = None) -> str:
        message = {"role": "user", "content": prompt}
        if images:
            message["images"] = images

        response = ollama.chat(
            model=self.model,
            messages=[message],
            options={"num_ctx": config.CONTEXT_WINDOW}
        )
        return response["message"]["content"]

#Fast cloud inference via Groq.
class GroqProvider(LLMProvider):

    def __init__(self, model: str = None, vision_model: str = None):
        from groq import Groq  #Ollama-only users don't need the package installed

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set. Get one at console.groq.com")

        self.client = Groq(api_key=api_key)
        self.model = model or config.GROQ_MODEL
        self.vision_model = vision_model or config.GROQ_VISION_MODEL   # <-- was missing entirely

    def generate(self, prompt: str, images: list[str] = None) -> str:
        if images:
            return self._generate_with_images(prompt, images)
        return self._generate_text_only(prompt)

    def _generate_text_only(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content

    def _generate_with_images(self, prompt: str, images: list[str]) -> str:
        content = [{"type": "text", "text": prompt}]
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })

        completion = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[{"role": "user", "content": content}],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content
        


def get_llm_provider(backend: str = None) -> LLMProvider:

    backend = (backend or config.LLM_BACKEND).lower()
    print(f"[DEBUG] Using LLM backend: {backend}")   # temporary — confirms routing

    if backend == "ollama":
        return OllamaProvider()
    elif backend == "groq":
        return GroqProvider()
    else:
        raise ValueError(f"Unknown LLM backend: {backend}. Use 'ollama' or 'groq'.")