import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path.home() / "VersionDiff-DocVQA"

RESULTS_FILE = PROJECT_ROOT / "results/experiment1/sroie_annotation_diff_results.jsonl"
OUT_FILE = PROJECT_ROOT / "results/experiment1/sroie_annotation_diff_by_change_type.json"

def main():
    rows = []

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
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

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nSaved breakdown to: {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()

