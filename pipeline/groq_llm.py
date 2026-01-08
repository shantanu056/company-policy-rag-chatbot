import os
from groq import Groq
from dotenv import load_dotenv


class GroqLLM:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful company policy assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()



if __name__ == "__main__":
    # Simple usage note - do not auto-run tests here.
    print("GroqLLM module loaded. Use GroqRAGChain or import GroqLLM in your scripts.")
