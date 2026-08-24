import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class RAGGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        print(f"Initializing RAGGenerator with model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Use CPU explicitly since CUDA is not available
        self.device = torch.device("cpu")
        self.model.to(self.device)
        print("Model loaded successfully.")

    def generate_answer(self, query: str, contexts: list[str]) -> str:
        """
        Generates an answer to the query based on the provided context chunks.
        """
        context_text = "\n\n---\n\n".join(contexts)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful technical support assistant for FastAPI. "
                    "Answer the user's question using ONLY the provided documentation context. "
                    "Be highly precise and direct. If the context does not contain the answer, "
                    "explicitly say: 'I cannot find the answer in the provided documentation.'"
                )
            },
            {
                "role": "user",
                "content": f"Documentation Context:\n{context_text}\n\nQuestion: {query}"
            }
        ]

        # Apply standard instruction chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,  # Low temperature for factual consistency
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Retrieve only the newly generated tokens
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return answer.strip()

if __name__ == "__main__":
    # Quick test
    generator = RAGGenerator()
    ans = generator.generate_answer(
        "How do you declare a path parameter?",
        ["You declare path parameters using python format string syntax like /items/{item_id}."]
    )
    print("Test Answer:", ans)
