import json
import re
from pathlib import Path


PROJECT_ROOT = Path.home() / "VersionDiff-DocVQA"

QA_FILE = PROJECT_ROOT / "data/processed/qa_jsonl/funsd_revision_qa.jsonl"
RESULTS_FILE = PROJECT_ROOT / "results/experiment1/funsd_annotation_diff_results.jsonl"
SUMMARY_FILE = PROJECT_ROOT / "results/experiment1/funsd_annotation_diff_summary.json"

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s./$-]", "", text)
    return text


def exact_match(prediction, ground_truth):
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def token_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common_tokens = set(pred_tokens) & set(gold_tokens)
    overlap = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common_tokens)

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def levenshtein_distance(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i

    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]


def anls(prediction, ground_truth):
    prediction = normalize_text(prediction)
    ground_truth = normalize_text(ground_truth)

    if not prediction and not ground_truth:
        return 1.0

    if not prediction or not ground_truth:
        return 0.0

    dist = levenshtein_distance(prediction, ground_truth)
    max_len = max(len(prediction), len(ground_truth))
    score = 1 - (dist / max_len)

    if score < 0.5:
        return 0.0

    return score


def revised_only_baseline(row):
    """
    Simulates a model that only sees the revised document.
    It can identify the new value, but not the old value.
    """
    return f"The current value is {row['new_value']}."


def dual_document_naive_baseline(row):
    """
    Simulates a weak dual-document baseline.
    It sees both values but does not explicitly reason in the answer format.
    """
    return f"Original value: {row['old_value']}. Revised value: {row['new_value']}."


def annotation_diff_method(row):
    """
    Annotation-based diff method.

    Since OCR is unavailable, we use FUNSD's annotation-derived text and
    our synthetic revised value to simulate detected original/revised text.
    This is the next step after the metadata-only sanity check.
    """
    old_value = row["old_value"]
    new_value = row["new_value"]

    if old_value != new_value:
        return f"The value changed from {old_value} to {new_value}."

    return "No value changed."


def evaluate_method(rows, method_name, method_fn):
    exact_scores = []
    f1_scores = []
    anls_scores = []
    predictions = []

    for row in rows:
        prediction = method_fn(row)
        gold = row["answer"]

        em = exact_match(prediction, gold)
        f1 = token_f1(prediction, gold)
        anls_score = anls(prediction, gold)

        exact_scores.append(em)
        f1_scores.append(f1)
        anls_scores.append(anls_score)

        predictions.append({
            "doc_id": row["doc_id"],
            "method": method_name,
            "question": row["question"],
            "prediction": prediction,
            "ground_truth": gold,
            "exact_match": em,
            "f1": f1,
            "anls": anls_score,
            "change_type": row["change_type"],
            "field_name": row["field_name"],
            "old_value": row["old_value"],
            "new_value": row["new_value"]
        })

    summary = {
        "method": method_name,
        "num_examples": len(rows),
        "exact_match": sum(exact_scores) / len(exact_scores),
        "f1": sum(f1_scores) / len(f1_scores),
        "anls": sum(anls_scores) / len(anls_scores)
    }

    return summary, predictions


def main():
    if not QA_FILE.exists():
        raise FileNotFoundError(f"Missing QA file: {QA_FILE}")

    rows = []

    with open(QA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} QA examples.", flush=True)

    methods = [
        ("revised_only_baseline", revised_only_baseline),
        ("dual_document_naive_baseline", dual_document_naive_baseline),
        ("annotation_diff_method", annotation_diff_method),
    ]

    all_summaries = []
    all_predictions = []

    for method_name, method_fn in methods:
        summary, predictions = evaluate_method(rows, method_name, method_fn)
        all_summaries.append(summary)
        all_predictions.extend(predictions)

        print("\n" + method_name, flush=True)
        print(f"Exact Match: {summary['exact_match']:.4f}", flush=True)
        print(f"F1:          {summary['f1']:.4f}", flush=True)
        print(f"ANLS:        {summary['anls']:.4f}", flush=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred) + "\n")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nSaved detailed results to: {RESULTS_FILE}", flush=True)
    print(f"Saved summary to: {SUMMARY_FILE}", flush=True)


if __name__ == "__main__":
    main()
