import os
import sys
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Ensure src modules are in Python search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

try:
    from generator import RAGGenerator
except ImportError:
    try:
        from generator.generator import RAGGenerator
    except ImportError:
        from .generator import RAGGenerator

class GroqGenerator:
    def __init__(self, fallback_model="Qwen/Qwen2.5-0.5B-Instruct"):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.fallback_generator = None
        self.fallback_model = fallback_model
        
        if not self.api_key:
            print("[WARNING] GROQ_API_KEY not found in environment. Falling back to local model.")
            # Lazy load fallback model to save memory if key is present
            self.fallback_generator = RAGGenerator(model_name=fallback_model)
        else:
            print("GroqGenerator initialized using model: llama-3.3-70b-versatile.")

    def generate_answer(self, query: str, contexts: list[str]) -> str:
        """
        Generates an answer using Groq's Llama 3.3 model if API key is present.
        Falls back to local Qwen model otherwise.
        """
        if not self.api_key:
            return self.fallback_generator.generate_answer(query, contexts)

        context_text = "\n\n---\n\n".join(contexts)
        
        system_prompt = (
            "You are a helpful technical support assistant for FastAPI. "
            "Answer the user's question using ONLY the provided documentation context. "
            "Be highly precise and direct. If the context does not contain the answer, "
            "explicitly say: 'I cannot find the answer in the provided documentation.'"
        )
        
        user_prompt = f"Documentation Context:\n{context_text}\n\nQuestion: {query}"
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            return answer.strip()
        except Exception as e:
            print(f"[ERROR] Groq API call failed: {e}. Falling back to local model.")
            if self.fallback_generator is None:
                self.fallback_generator = RAGGenerator(model_name=self.fallback_model)
            return self.fallback_generator.generate_answer(query, contexts)

if __name__ == "__main__":
    # Test with dummy key to force fallback test or real key if loaded
    os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")
    generator = GroqGenerator()
    ans = generator.generate_answer(
        "What is FastAPI?",
        ["FastAPI is a modern, fast, high-performance web framework for Python."]
    )
    print("Answer:", ans)
