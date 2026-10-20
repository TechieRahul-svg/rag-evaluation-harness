
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("src/evaluation/ci_evaluate.py").resolve()
RESULT_FILE = Path("evaluation/retrieval_results.json").resolve()


def run_quality_gate(tmp_path, results):
    """Run the gate against temporary results without changing real benchmarks."""
    result_file = tmp_path / "retrieval_results.json"
    result_file.write_text(json.dumps(results), encoding="utf-8")

    # Use a temporary copy of the script so its result path points
    # to the temporary test file, not the repository benchmark.
    script_text = SCRIPT.read_text(encoding="utf-8")
    script_text = script_text.replace(
        'Path("evaluation/retrieval_results.json")',
        f'Path(r"{result_file}")',
    )
    test_script = tmp_path / "ci_evaluate.py"
    test_script.write_text(script_text, encoding="utf-8")

    return subprocess.run(
        [sys.executable, str(test_script)],
        capture_output=True,
        text=True,
        check=False,
    )


def valid_results(hit5=0.90):
    return {
        "fixed": {
            "hit_at_5": 0.90,
            "total_questions": 50,
        },
        "recursive": {
            "hit_at_5": hit5,
            "total_questions": 50,
        },
        "semantic": {
            "hit_at_5": 0.90,
            "total_questions": 50,
        },
    }


def test_quality_gate_passes_at_threshold(tmp_path):
    result = run_quality_gate(tmp_path, valid_results(0.90))
    assert result.returncode == 0
    assert "RAG Evaluation Passed" in result.stdout


def test_quality_gate_fails_below_threshold(tmp_path):
    result = run_quality_gate(tmp_path, valid_results(0.88))
    assert result.returncode != 0
    assert "RAG Evaluation Failed" in result.stdout


def test_quality_gate_fails_when_strategy_is_missing(tmp_path):
    results = valid_results()
    del results["recursive"]

    result = run_quality_gate(tmp_path, results)
    assert result.returncode != 0
    assert "Missing strategies" in result.stdout


def test_quality_gate_fails_when_hit5_is_invalid(tmp_path):
    result = run_quality_gate(tmp_path, valid_results(1.20))
    assert result.returncode != 0
    assert "Invalid Hit@5" in result.stdout
