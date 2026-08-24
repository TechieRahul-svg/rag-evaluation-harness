import json
import os
import sys
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Ensure src modules are in Python search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

class RagasEvaluator:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            print("[WARNING] GROQ_API_KEY not found. LLM-as-a-judge metrics will be simulated.")
        else:
            print("RagasEvaluator initialized using Groq Llama 3.3.")

    def _query_groq_judge(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Sends an evaluation prompt to Llama 3.3 on Groq and parses JSON response.
        """
        if not self.api_key:
            # Simulation mode
            return {"score": 5, "reasoning": "Simulated evaluation (no Groq key provided)."}

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 512
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
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            print(f"[ERROR] Judge evaluation failed: {e}")
            return {"score": 0, "reasoning": f"Failed to get score from LLM judge: {e}"}

    def evaluate_faithfulness(self, context: str, answer: str) -> dict:
        """
        Checks if the answer is strictly grounded in the context. Score is 0 or 1.
        """
        system_prompt = (
            "You are a rigorous technical RAG auditor. Evaluate if the generated answer is fully grounded "
            "in the provided context and contains absolutely NO statements or claims that are not present "
            "in the context. Output a JSON object with two fields: 'score' (either 1 for fully faithful or "
            "0 for contains ungrounded claims/hallucinations) and 'reasoning' (a brief explanation)."
        )
        user_prompt = f"CONTEXT:\n{context}\n\nGENERATED ANSWER:\n{answer}"
        return self._query_groq_judge(system_prompt, user_prompt)

    def evaluate_relevance(self, question: str, answer: str) -> dict:
        """
        Evaluates how well the answer addresses the question. Score is 1 to 5.
        """
        system_prompt = (
            "You are a technical RAG auditor. Evaluate how well the generated answer addresses and answers "
            "the user's question. Rate it on a scale of 1 to 5: "
            "1: completely irrelevant, "
            "3: partially addresses it but misses key parts, "
            "5: fully and directly answers the question. "
            "Output a JSON object with two fields: 'score' (an integer from 1 to 5) and 'reasoning' (a brief explanation)."
        )
        user_prompt = f"QUESTION:\n{question}\n\nGENERATED ANSWER:\n{answer}"
        return self._query_groq_judge(system_prompt, user_prompt)

    def evaluate_results_file(self, file_path: Path, output_path: Path):
        """
        Loads E2E results, evaluates each query, and saves results.
        """
        if not file_path.exists():
            print(f"[ERROR] Input file {file_path} does not exist.")
            return

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        print(f"Evaluating {len(data)} items from {file_path.name}...")
        evaluated_data = []
        
        total_faithfulness = 0
        total_relevance = 0
        valid_items = 0

        for idx, item in enumerate(data, 1):
            q = item["question"]
            ans = item["generated_answer"]
            context = "\n\n---\n\n".join([chunk["text"] for chunk in item["retrieved_chunks"]])
            
            print(f"Auditing item {idx}/{len(data)}...")
            
            faith = self.evaluate_faithfulness(context, ans)
            rel = self.evaluate_relevance(q, ans)
            
            # Simulated scores override if no key
            if not self.api_key:
                faith = {"score": 1, "reasoning": "Simulated faithfulness (no API key)"}
                rel = {"score": 5, "reasoning": "Simulated relevance (no API key)"}

            item["faithfulness"] = faith
            item["relevance"] = rel
            evaluated_data.append(item)
            
            total_faithfulness += faith.get("score", 0)
            total_relevance += rel.get("score", 0)
            valid_items += 1

        avg_faithfulness = (total_faithfulness / valid_items) if valid_items else 0
        avg_relevance = (total_relevance / valid_items) if valid_items else 0

        summary = {
            "metrics": {
                "average_faithfulness": avg_faithfulness,
                "average_relevance": avg_relevance,
                "simulated": not bool(self.api_key),
            },
            "evaluations": evaluated_data
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print(f"Evaluation complete.")
        print(f"Average Faithfulness: {avg_faithfulness:.2%}")
        print(f"Average Relevance:    {avg_relevance:.2f}/5")
        print(f"Saved results to:     {output_path}")

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    INPUT_FILE = ROOT / "evaluation" / "e2e_results_semantic.json"
    OUTPUT_FILE = ROOT / "evaluation" / "ragas_eval_results.json"
    
    evaluator = RagasEvaluator()
    evaluator.evaluate_results_file(INPUT_FILE, OUTPUT_FILE)
