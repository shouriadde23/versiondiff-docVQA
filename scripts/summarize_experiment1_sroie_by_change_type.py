import json
from collections import defaultdict
from pathlib import Path


project_root = Path.home() / "VersionDiff-DocVQA"

results_file = project_root / "results/experiment1/sroie_annotation_diff_results.jsonl"
out_file = project_root / "results/experiment1/sroie_annotation_diff_by_change_type.json"

def main():
    rows = []

    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    groups = defaultdict(list)

    for row in rows:
        key = (row["method"], row["change_type"])
        groups[key].append(row)

    summary = []

    for (method, change_type), items in sorted(groups.items()):
        n = len(items)
        em = sum(x["exact_match"] for x in items) / n
        f1 = sum(x["f1"] for x in items) / n
        anls = sum(x["anls"] for x in items) / n

        summary.append({
            "method": method,
            "change_type": change_type,
            "num_examples": n,
            "exact_match": em,
            "f1": f1,
            "anls": anls
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nSaved breakdown to: {out_file}", flush=True)


if __name__ == "__main__":
    main()

