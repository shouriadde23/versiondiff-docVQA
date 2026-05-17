import json
import re
from pathlib import Path


project_root = Path.home() / "VersionDiff-DocVQA"

qa_file = project_root / "data/processed/qa_jsonl/funsd_revision_qa.jsonl"
results_file = project_root / "results/experiment1/funsd_annotation_diff_results.jsonl"
summary_file = project_root / "results/experiment1/funsd_annotation_diff_summary.json"

results_file.parent.mkdir(parents=True, exist_ok=True)


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
    q = row["question"].lower()
    change_type = row["change_type"]
    new_value = row["new_value"]

    if "revised value" in q or "after the change" in q:
        return str(new_value)

    if "added" in q or "inserted" in q:
        if change_type == "insertion":
            return str(new_value)
        return "No"

    if "type of revision" in q:
        return "unknown"

    if "removed" in q or "deleted" in q:
        return "unknown"

    if "original value" in q or "before the revision" in q:
        return "unknown"

    return str(new_value)


def dual_document_naive_baseline(row):
    old_value = row["old_value"]
    new_value = row["new_value"]
    change_type = row["change_type"]

    if change_type == "replacement":
        return f"Original value: {old_value}. Revised value: {new_value}."

    if change_type == "deletion":
        return f"Original value: {old_value}. Revised value: missing."

    if change_type == "insertion":
        return f"Original value: missing. Revised value: {new_value}."

    return "unknown"


def annotation_diff_method(row):
    q = row["question"].lower()
    old_value = row["old_value"]
    new_value = row["new_value"]
    change_type = row["change_type"]

    if "what value was changed" in q:
        if change_type == "replacement":
            return f"The value changed from {old_value} to {new_value}."
        if change_type == "deletion":
            return str(old_value)
        if change_type == "insertion":
            return str(new_value)

    if "original value" in q or "before the revision" in q:
        return str(old_value)

    if "revised value" in q or "after the change" in q:
        return str(new_value)

    if "how did the value change" in q:
        if change_type == "replacement":
            return f"{old_value} was replaced with {new_value}."
        if change_type == "deletion":
            return f"{old_value} was removed."
        if change_type == "insertion":
            return f"{new_value} was inserted."

    if "type of revision" in q:
        return str(change_type)

    if "removed" in q or "deleted" in q:
        if change_type == "deletion":
            if "was any" in q:
                return "Yes"
            return str(old_value)
        return "No"

    if "added" in q or "inserted" in q:
        if change_type == "insertion":
            if "was any" in q:
                return "Yes"
            return str(new_value)
        return "No"

    return "unknown"

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
    if not qa_file.exists():
        raise FileNotFoundError(f"Missing QA file: {qa_file}")

    rows = []

    with open(qa_file, "r", encoding="utf-8") as f:
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

    with open(results_file, "w", encoding="utf-8") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred) + "\n")

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nSaved detailed results to: {results_file}", flush=True)
    print(f"Saved summary to: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
