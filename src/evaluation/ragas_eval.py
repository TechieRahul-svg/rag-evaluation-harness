
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Ensure src modules are in Python search path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"


class RagasEvaluator:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            print(
                "[WARNING] GROQ_API_KEY not found. "
                "LLM-as-a-judge evaluation will be skipped."
            )
        else:
            print(f"RagasEvaluator initialized using Groq model {GROQ_MODEL}.")

    def _query_groq_judge(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """Query the Groq judge and validate its JSON response."""

        if not self.api_key:
            return {
                "score": None,
                "reasoning": "Evaluation skipped: no Groq API key configured.",
                "status": "skipped",
            }

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 512,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                GROQ_API_URL,
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)

            if not isinstance(result, dict):
                raise ValueError("Judge response must be a JSON object.")

            if "reasoning" not in result:
                raise ValueError("Judge response is missing 'reasoning'.")

            score = result.get("score")

            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("Judge score must be numeric.")

            return {
                "score": score,
                "reasoning": str(result["reasoning"]),
                "status": "success",
            }

        except (
            httpx.HTTPError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            print(f"[ERROR] Judge evaluation failed: {exc}")
            return {
                "score": None,
                "reasoning": f"Judge evaluation failed: {exc}",
                "status": "failed",
            }

    def evaluate_faithfulness(self, context: str, answer: str) -> dict:
        """Score whether the answer is grounded in the retrieved context."""

        system_prompt = (
            "You are a rigorous technical RAG auditor. Evaluate whether the "
            "generated answer is fully grounded in the provided context and "
            "contains no unsupported claims. Return a JSON object with "
            "'score' (1 for fully faithful, 0 for ungrounded claims or "
            "hallucinations) and 'reasoning' (a brief explanation)."
        )

        user_prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"GENERATED ANSWER:\n{answer}"
        )

        result = self._query_groq_judge(system_prompt, user_prompt)

        if result["status"] == "success":
            score = result["score"]
            if isinstance(score, bool) or score not in (0, 1):
                return {
                    "score": None,
                    "reasoning": "Invalid faithfulness score; expected 0 or 1.",
                    "status": "failed",
                }

        return result

    def evaluate_relevance(self, question: str, answer: str) -> dict:
        """Score how well the answer addresses the question, from 1 to 5."""

        system_prompt = (
            "You are a technical RAG auditor. Evaluate how well the generated "
            "answer addresses the user's question. Rate it from 1 to 5: "
            "1 means completely irrelevant, 3 means partially addresses the "
            "question but misses key parts, and 5 means fully and directly "
            "answers the question. Return a JSON object with 'score' "
            "(an integer from 1 to 5) and 'reasoning' (a brief explanation)."
        )

        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"GENERATED ANSWER:\n{answer}"
        )

        result = self._query_groq_judge(system_prompt, user_prompt)

        if result["status"] == "success":
            score = result["score"]
            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or not 1 <= score <= 5
            ):
                return {
                    "score": None,
                    "reasoning": "Invalid relevance score; expected an integer from 1 to 5.",
                    "status": "failed",
                }

        return result

    @staticmethod
    def _average(results: list[dict], metric: str):
        """Average successful numeric scores only; return None if none exist."""

        scores = [
            item[metric]["score"]
            for item in results
            if item[metric]["status"] == "success"
            and item[metric]["score"] is not None
        ]

        return sum(scores) / len(scores) if scores else None

    def evaluate_results_file(self, file_path: Path, output_path: Path) -> int:
        """Evaluate E2E results and save scores plus evaluation status."""

        if not file_path.is_file():
            print(f"[ERROR] Input file does not exist: {file_path}")
            return 1

        try:
            with file_path.open(encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[ERROR] Could not read input JSON: {exc}")
            return 1

        if not isinstance(data, list):
            print("[ERROR] Input JSON must contain a list of evaluation items.")
            return 1

        evaluated_data = []
        print(f"Evaluating {len(data)} items from {file_path.name}...")

        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                print(f"[ERROR] Item {idx} must be a JSON object.")
                return 1

            required_fields = {"question", "generated_answer", "retrieved_chunks"}
            missing_fields = required_fields - item.keys()

            if missing_fields:
                print(
                    f"[ERROR] Item {idx} is missing fields: "
                    f"{', '.join(sorted(missing_fields))}"
                )
                return 1

            chunks = item["retrieved_chunks"]
            if not isinstance(chunks, list) or any(
                not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str)
                for chunk in chunks
            ):
                print(f"[ERROR] Item {idx} has invalid retrieved_chunks.")
                return 1

            question = item["question"]
            answer = item["generated_answer"]
            if not isinstance(question, str) or not isinstance(answer, str):
                print(f"[ERROR] Item {idx} question and answer must be strings.")
                return 1

            context = "\n\n---\n\n".join(
                chunk["text"] for chunk in chunks
            )

            print(f"Auditing item {idx}/{len(data)}...")

            faithfulness = self.evaluate_faithfulness(context, answer)
            relevance = self.evaluate_relevance(question, answer)

            evaluated_item = dict(item)
            evaluated_item["faithfulness"] = faithfulness
            evaluated_item["relevance"] = relevance
            evaluated_data.append(evaluated_item)

        average_faithfulness = self._average(
            evaluated_data, "faithfulness"
        )
        average_relevance = self._average(
            evaluated_data, "relevance"
        )

        statuses = [
            evaluation[metric]["status"]
            for evaluation in evaluated_data
            for metric in ("faithfulness", "relevance")
        ]

        successful = statuses.count("success")
        skipped = statuses.count("skipped")
        failed = statuses.count("failed")

        if failed:
            overall_status = "partial_failure"
        elif skipped:
            overall_status = "skipped"
        else:
            overall_status = "success"

        summary = {
            "metrics": {
                "average_faithfulness": average_faithfulness,
                "average_relevance": average_relevance,
                "simulated": False,
                "evaluation_status": overall_status,
                "total_items": len(evaluated_data),
                "successful_judgments": successful,
                "skipped_judgments": skipped,
                "failed_judgments": failed,
            },
            "evaluations": evaluated_data,
        }

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(summary, file, indent=4, ensure_ascii=False)
        except OSError as exc:
            print(f"[ERROR] Could not write output JSON: {exc}")
            return 1

        print("\nEvaluation complete.")
        print(f"Evaluation status:    {overall_status}")
        print(f"Successful judgments: {successful}")
        print(f"Skipped judgments:    {skipped}")
        print(f"Failed judgments:     {failed}")

        if average_faithfulness is None:
            print("Average Faithfulness: N/A")
        else:
            print(f"Average Faithfulness: {average_faithfulness:.2%}")

        if average_relevance is None:
            print("Average Relevance:    N/A")
        else:
            print(f"Average Relevance:    {average_relevance:.2f}/5")

        print(f"Saved results to:     {output_path}")

        # Fail the process when judge requests failed, so CI does not
        # silently treat incomplete evaluation as a successful evaluation.
        return 1 if failed else 0


if __name__ == "__main__":
    INPUT_FILE = ROOT / "evaluation" / "e2e_results_semantic.json"
    OUTPUT_FILE = ROOT / "evaluation" / "ragas_eval_results.json"

    evaluator = RagasEvaluator()
    sys.exit(evaluator.evaluate_results_file(INPUT_FILE, OUTPUT_FILE))
