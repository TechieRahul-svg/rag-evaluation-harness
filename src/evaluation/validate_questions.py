import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_FILE = ROOT / "evaluation" / "evaluation.json"
RAW_DATA_DIR = ROOT / "data" / "raw"

# --------------------------------------------------
# Header
# --------------------------------------------------

print("=" * 60)
print("GROUND TRUTH VALIDATION")
print("=" * 60)

errors = []

# --------------------------------------------------
# 1. Load JSON
# --------------------------------------------------

try:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)
except json.JSONDecodeError as e:
    print(f"FATAL: evaluation.json is not valid JSON: {e}")
    raise SystemExit(1)
except FileNotFoundError:
    print(f"FATAL: File not found: {QUESTIONS_FILE}")
    raise SystemExit(1)

print(f"Total questions: {len(questions)}")
print()

# --------------------------------------------------
# 2. Must be a list
# --------------------------------------------------

if not isinstance(questions, list):
    print("FATAL: evaluation.json must contain a JSON array.")
    raise SystemExit(1)

# --------------------------------------------------
# 3. Required fields present
# --------------------------------------------------

REQUIRED_FIELDS = ["id", "question", "expected_sources", "expected_answer", "category"]

for question in questions:
    question_id = question.get("id", "<missing id>")
    for field in REQUIRED_FIELDS:
        if field not in question:
            errors.append(
                f"Question {question_id}: missing required field '{field}'"
            )

# --------------------------------------------------
# 4. Unique IDs
# --------------------------------------------------

ids = [q.get("id") for q in questions]
seen = set()
for qid in ids:
    if qid in seen:
        errors.append(f"Duplicate question id: {qid}")
    seen.add(qid)

# --------------------------------------------------
# 5. Non-empty answers and questions
# --------------------------------------------------

for question in questions:
    question_id = question.get("id")

    if not question.get("question", "").strip():
        errors.append(f"Question {question_id}: 'question' field is empty")

    answer = question.get("expected_answer", "")
    if not answer.strip() or answer.strip().upper().startswith("YOUR VERIFIED"):
        errors.append(
            f"Question {question_id}: 'expected_answer' is empty or still a placeholder"
        )

    if not question.get("category", "").strip():
        errors.append(f"Question {question_id}: 'category' field is empty")

# --------------------------------------------------
# 6. expected_sources is a non-empty list
# --------------------------------------------------

for question in questions:
    question_id = question.get("id")
    sources = question.get("expected_sources", [])

    if not isinstance(sources, list) or not sources:
        errors.append(
            f"Question {question_id}: 'expected_sources' must be a non-empty list"
        )

# --------------------------------------------------
# 7. Verify source files actually exist on disk
# --------------------------------------------------

for question in questions:
    question_id = question.get("id")
    sources = question.get("expected_sources", [])

    for source in sources:
        source_path = RAW_DATA_DIR / source
        if not source_path.exists():
            errors.append(
                f"Question {question_id}: source file does not exist: {source}"
            )

# --------------------------------------------------
# Final result
# --------------------------------------------------

print()

if errors:
    print("VALIDATION FAILED ❌")
    print("\nErrors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("All questions are valid.")
    print("All expected sources exist.")
    print("VALIDATION PASSED \u2705")

    from collections import Counter
    cats = Counter(q.get("category") for q in questions)
    print("\nCategory breakdown:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
