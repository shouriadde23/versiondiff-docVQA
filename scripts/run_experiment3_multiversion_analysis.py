import json
import re
from collections import defaultdict
from pathlib import Path


project_root = Path.home() / "VersionDiff-DocVQA"

qa_file = project_root / "data/processed/qa_jsonl/sroie_multiversion_qa.jsonl"

out_results = project_root / "results/experiment3/sroie_multiversion_results.jsonl"
out_summary = project_root / "results/experiment3/sroie_multiversion_summary.json"
out_table = project_root / "results/experiment3/sroie_multiversion_report_table.txt"

out_results.parent.mkdir(parents=True, exist_ok=True)


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s./$>\-]", "", text)
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


def get_visible_history(row, setting):
    history = row["value_history"]

    if setting == "two_versions":
        return [history[0], history[3]]

    if setting == "three_versions":
        return [history[0], history[1], history[3]]

    if setting == "four_versions":
        return history

    raise ValueError(f"Unknown setting: {setting}")


def predict_from_visible_history(row, setting):
    qtype = row["question_type"]
    field_name = row["field_name"]
    gold_history = row["value_history"]
    visible = get_visible_history(row, setting)

    v0 = gold_history[0]
    v1 = gold_history[1]
    v2 = gold_history[2]
    v3 = gold_history[3]

    if qtype == "final_value":
        return str(visible[-1])

    if qtype == "original_value":
        return str(visible[0])

    if qtype == "before_final_value":
        if setting == "four_versions":
            return str(v2)
        if setting == "three_versions":
            return str(v1)
        return str(v0)

    if qtype == "first_revision_value":
        if setting in ["three_versions", "four_versions"]:
            return str(v1)
        return str(v3)

    if qtype == "last_change_description":
        if setting == "four_versions":
            return f"The {field_name} changed from {v2} to {v3}."
        if setting == "three_versions":
            return f"The {field_name} changed from {v1} to {v3}."
        return f"The {field_name} changed from {v0} to {v3}."

    if qtype == "first_change_description":
        if setting in ["three_versions", "four_versions"]:
            return f"The {field_name} changed from {v0} to {v1}."
        return f"The {field_name} changed from {v0} to {v3}."

    if qtype == "modification_count":
        if setting == "two_versions":
            return "1"
        if setting == "three_versions":
            return "2"
        if setting == "four_versions":
            return "3"

    if qtype == "full_history":
        return " -> ".join(str(x) for x in visible)

    return "unknown"


def evaluate(rows):
    methods = ["two_versions", "three_versions", "four_versions"]

    all_predictions = []

    for row in rows:
        for method in methods:
            prediction = predict_from_visible_history(row, method)
            gold = row["answer"]

            result = {
                "dataset": "sroie",
                "doc_id": row["doc_id"],
                "field_name": row["field_name"],
                "question_type": row["question_type"],
                "question": row["question"],
                "method": method,
                "prediction": prediction,
                "ground_truth": gold,
                "exact_match": exact_match(prediction, gold),
                "f1": token_f1(prediction, gold),
                "anls": anls(prediction, gold),
                "value_history": row["value_history"],
            }

            all_predictions.append(result)

    return all_predictions


def summarize_overall(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[row["method"]].append(row)

    summary = []

    for method, items in sorted(groups.items()):
        n = len(items)
        summary.append({
            "method": method,
            "num_examples": n,
            "exact_match": sum(x["exact_match"] for x in items) / n,
            "f1": sum(x["f1"] for x in items) / n,
            "anls": sum(x["anls"] for x in items) / n,
        })

    return summary


def summarize_by_question_type(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (row["method"], row["question_type"])
        groups[key].append(row)

    summary = []

    for (method, question_type), items in sorted(groups.items()):
        n = len(items)
        summary.append({
            "method": method,
            "question_type": question_type,
            "num_examples": n,
            "exact_match": sum(x["exact_match"] for x in items) / n,
            "f1": sum(x["f1"] for x in items) / n,
            "anls": sum(x["anls"] for x in items) / n,
        })

    return summary


def main():
    if not qa_file.exists():
        raise FileNotFoundError(f"Missing QA file: {qa_file}")

    rows = []

    with open(qa_file, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} multi-version QA examples.", flush=True)

    predictions = evaluate(rows)

    overall = summarize_overall(predictions)
    by_question = summarize_by_question_type(predictions)

    with open(out_results, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump({
            "overall": overall,
            "by_question_type": by_question,
        }, f, indent=2)

    print(f"\nSaved detailed results to: {out_results}", flush=True)
    print(f"Saved summary to: {out_summary}", flush=True)
    print(f"Saved report table to: {out_table}", flush=True)


if __name__ == "__main__":
    main()
