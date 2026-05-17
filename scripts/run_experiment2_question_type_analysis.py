import json
from collections import defaultdict
from pathlib import Path


project_root = Path.home() / "VersionDiff-DocVQA"

input_files = {
    "funsd": project_root / "results/experiment1/funsd_annotation_diff_results.jsonl",
    "sroie": project_root / "results/experiment1/sroie_annotation_diff_results.jsonl",
}

out_summary = project_root / "results/experiment2/question_type_summary.json"
out_table = project_root / "results/experiment2/question_type_report_table.txt"
out_combined = project_root / "results/experiment2/question_type_combined_results.jsonl"

out_summary.parent.mkdir(parents=True, exist_ok=True)


def classify_question_type(question):
    q = question.lower()

    if "changed to" in q or "revised value" in q or "after the change" in q:
        return "revised_value_change"

    if "original" in q or "before the revision" in q:
        return "original_value_retrieval"

    if "how did" in q and "change" in q:
        return "change_description"

    if "type of revision" in q:
        return "revision_type_classification"

    if "removed" in q:
        return "removed_value_detection"

    if "deleted" in q:
        return "deletion_yes_no"

    if "added" in q:
        return "added_value_detection"

    if "inserted" in q:
        return "insertion_yes_no"

    if "what value was changed" in q:
        return "generic_change_identification"

    return "other_question"


def load_results():
    rows = []

    for dataset_name, path in input_files.items():
        if not path.exists():
            print(f"Missing result file: {path}", flush=True)
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                row["dataset"] = dataset_name
                row["question_type"] = classify_question_type(row["question"])
                rows.append(row)

    return rows


def summarize_by_question_type(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (row["dataset"], row["method"], row["question_type"])
        groups[key].append(row)

    summary = []

    for (dataset, method, question_type), items in sorted(groups.items()):
        n = len(items)
        summary.append({
            "dataset": dataset,
            "method": method,
            "question_type": question_type,
            "num_examples": n,
            "exact_match": sum(x["exact_match"] for x in items) / n,
            "f1": sum(x["f1"] for x in items) / n,
            "anls": sum(x["anls"] for x in items) / n,
        })

    return summary


def summarize_combined(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (row["method"], row["question_type"])
        groups[key].append(row)

    summary = []

    for (method, question_type), items in sorted(groups.items()):
        n = len(items)
        summary.append({
            "dataset": "combined_funsd_sroie",
            "method": method,
            "question_type": question_type,
            "num_examples": n,
            "exact_match": sum(x["exact_match"] for x in items) / n,
            "f1": sum(x["f1"] for x in items) / n,
            "anls": sum(x["anls"] for x in items) / n,
        })

    return summary


def main():
    rows = load_results()

    print(f"Loaded {len(rows)} total prediction rows.", flush=True)

    if not rows:
        raise RuntimeError("No results loaded. Run Experiment 1 scripts first.")

    by_dataset = summarize_by_question_type(rows)
    combined = summarize_combined(rows)

    output = {
        "by_dataset_method_question_type": by_dataset,
        "combined_by_method_question_type": combined,
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    with open(out_combined, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    lines = []
    lines.append("Experiment 2: Combined FUNSD + SROIE by Question Type")
    lines.append("=" * 90)
    lines.append(
        f"{'Method':30s} | {'Question Type':35s} | {'N':>5s} | {'EM':>7s} | {'F1':>7s} | {'ANLS':>7s}"
    )
    lines.append("-" * 90)

    for item in combined:
        line = (
            f"{item['method']:30s} | "
            f"{item['question_type']:35s} | "
            f"{item['num_examples']:5d} | "
            f"{item['exact_match']:7.4f} | "
            f"{item['f1']:7.4f} | "
            f"{item['anls']:7.4f}"
        )
        lines.append(line)

    table_text = "\n".join(lines)

    with open(out_table, "w", encoding="utf-8") as f:
        f.write(table_text)

    print(table_text, flush=True)
    print(f"\nSaved Experiment 2 summary to: {out_summary}", flush=True)
    print(f"Saved report table to: {out_table}", flush=True)
    print(f"Saved combined prediction rows to: {out_combined}", flush=True)


if __name__ == "__main__":
    main()
