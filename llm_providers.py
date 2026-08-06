#LLM provider abstraction: makes the generator backend (Ollama local / Groq cloud) interchangeable.

from abc import ABC, abstractmethod
import os
import ollama
import config
import base64
import logging

logger = logging.getLogger(__name__)

#Send a prompt to the backend, return its text answer.
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, images: list[str] = None) -> str:
        raise NotImplementedError
    @abstractmethod
    def generate_stream(self, prompt: str, images: list[str] = None):
        """Yield tokens one at a time."""
        raise NotImplementedError

class OllamaProvider(LLMProvider):
#Local, private inference via Ollama.

    def __init__(self, model: str = None):
        self.model = model or config.GENERATOR_MODEL

    def generate(self, prompt: str, images: list[str] = None) -> str:
        message = {"role": "user", "content": prompt}
        if images:
            message["images"] = images

        try:
            response = ollama.chat(
                model=self.model,
                messages=[message],
                options={"num_ctx": config.CONTEXT_WINDOW}
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Ollama error: {e}")
    def generate_stream(self, prompt: str, images: list[str] = None):
        message = {"role": "user", "content": prompt}
        if images:
            message["images"] = images
        try:
            stream = ollama.chat(
                model=self.model,
                messages=[message],
                options={"num_ctx": config.CONTEXT_WINDOW},
                stream=True
            )
            for chunk in stream:
                yield chunk["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise RuntimeError(f"Ollama stream error: {e}")

class GroqProvider(LLMProvider):
#Fast cloud inference via Groq.

    def __init__(self, model: str = None, vision_model: str = None):
        from groq import Groq  #Ollama-only users don't need the package installed

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set. Get one at console.groq.com")

        self.client = Groq(api_key=api_key)
        self.model = model or config.GROQ_MODEL
        self.vision_model = vision_model or config.GROQ_VISION_MODEL

    def generate(self, prompt: str, images: list[str] = None) -> str:
        if images:
            return self._generate_with_images(prompt, images)
        return self._generate_text_only(prompt)

    def _generate_text_only(self, prompt: str) -> str:
        try:
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
        except Exception as e:
            logger.error(f"Groq text generation failed: {e}")
            raise RuntimeError(f"Groq error: {e}")

    def _generate_with_images(self, prompt: str, images: list[str]) -> str:
        content = [{"type": "text", "text": prompt}]
        for img_path in images:
            # Auto-detect if it's already base64 or a file path
            if len(img_path) > 200 and not os.path.exists(img_path):
                b64 = img_path
            else:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
            
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
        try:
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
        except Exception as e:
            logger.error(f"Groq vision generation failed: {e}")
            raise RuntimeError(f"Groq vision error: {e}")
    def generate_stream(self, prompt: str, images: list[str] = None):
        if images:
            content = [{"type": "text", "text": prompt}]
            for img_path in images:
                if len(img_path) > 200 and not os.path.exists(img_path):
                    b64 = img_path
                else:
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
            model = self.vision_model
            messages = [{"role": "user", "content": content}]
        else:
            model = self.model
            messages = [{"role": "user", "content": prompt}]

        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                max_completion_tokens=2048,
                top_p=1,
                stream=True,
                stop=None
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            raise RuntimeError(f"Groq stream error: {e}")


def get_llm_provider(backend: str = None) -> LLMProvider:

    backend = (backend or config.LLM_BACKEND).lower()
    logger.info(f"[DEBUG] Using LLM backend: {backend}")
    if backend == "ollama":
        return OllamaProvider()
    elif backend == "groq":
        return GroqProvider()
    else:
        raise ValueError(f"Unknown LLM backend: {backend}. Use 'ollama' or 'groq'.")