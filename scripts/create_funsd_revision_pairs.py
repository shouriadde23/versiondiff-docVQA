import json
import random
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        if desc:
            print(desc, flush=True)
        return iterable


project_root = Path.home() / "VersionDiff-DocVQA"

funsd_root = project_root / "data/raw/FUNSD/dataset"
image_dir = funsd_root / "training_data/images"
ann_dir = funsd_root / "training_data/annotations"

out_orig = project_root / "data/processed/funsd/original"
out_rev = project_root / "data/processed/funsd/revised"
out_meta = project_root / "data/processed/funsd/metadata"
out_qa = project_root / "data/processed/qa_jsonl/funsd_revision_qa.jsonl"

out_orig.mkdir(parents=True, exist_ok=True)
out_rev.mkdir(parents=True, exist_ok=True)
out_meta.mkdir(parents=True, exist_ok=True)
out_qa.parent.mkdir(parents=True, exist_ok=True)


def load_annotation(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_candidate_words(annotation):
    candidates = []

    for item in annotation.get("form", []):
        label = item.get("label", "unknown")

        for word in item.get("words", []):
            text = word.get("text", "").strip()
            box = word.get("box")

            if not text or box is None:
                continue

            if len(text) < 2:
                continue

            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                continue

            candidates.append({
                "text": text,
                "box": box,
                "label": label
            })

    return candidates


def make_replacement_value(old_value):
    clean = old_value.replace(",", "").replace(".", "").replace("$", "")

    if clean.isdigit():
        try:
            num = int(float(clean))
            return str(num + random.choice([5, 10, 25, 100, 500]))
        except Exception:
            return "updated"

    if "/" in old_value or "-" in old_value:
        return "04/27/2026"

    return random.choice(["updated", "revised", "changed", "newvalue"])


def draw_white_box(draw, box):
    x0, y0, x1, y1 = box
    padding = 2
    draw.rectangle(
        [x0 - padding, y0 - padding, x1 + padding, y1 + padding],
        fill="white"
    )


def get_font(box):
    x0, y0, x1, y1 = box
    font_size = max(10, min(22, y1 - y0 + 2))

    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def edit_image(image_path, box, change_type, old_value, new_value, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if change_type == "replacement":
        draw_white_box(draw, box)
        draw.text((box[0], box[1]), new_value, fill="black", font=get_font(box))

    elif change_type == "deletion":
        draw_white_box(draw, box)

    elif change_type == "insertion":
        x0, y0, x1, y1 = box
        insert_x = x1 + 8
        insert_y = y0
        draw.text((insert_x, insert_y), new_value, fill="black", font=get_font(box))

    else:
        raise ValueError(f"Unknown change_type: {change_type}")

    img.save(output_path)


def build_questions(doc_id, original_out, revised_out, change_type, field_name, old_value, new_value, box):
    base = {
        "doc_id": doc_id,
        "original_image": str(original_out.relative_to(project_root)),
        "revised_image": str(revised_out.relative_to(project_root)),
        "change_type": change_type,
        "field_name": field_name,
        "old_value": old_value,
        "new_value": new_value,
        "box": box
    }

    rows = []

    if change_type == "replacement":
        question_answer_pairs = [
            (
                "What value was changed in the document?",
                f"The value changed from {old_value} to {new_value}."
            ),
            (
                "What was the original value before the revision?",
                str(old_value)
            ),
            (
                "What is the revised value after the change?",
                str(new_value)
            ),
            (
                "How did the value change?",
                f"{old_value} was replaced with {new_value}."
            ),
            (
                "What type of revision occurred?",
                "replacement"
            )
        ]

    elif change_type == "deletion":
        question_answer_pairs = [
            (
                "What value was removed from the document?",
                str(old_value)
            ),
            (
                "What type of revision occurred?",
                "deletion"
            ),
            (
                "Was any value deleted?",
                "Yes"
            )
        ]

    elif change_type == "insertion":
        question_answer_pairs = [
            (
                "What value was added to the document?",
                str(new_value)
            ),
            (
                "What type of revision occurred?",
                "insertion"
            ),
            (
                "Was any new value inserted?",
                "Yes"
            )
        ]

    else:
        question_answer_pairs = []

    for idx, (question, answer) in enumerate(question_answer_pairs):
        row = dict(base)
        row["qa_id"] = f"{doc_id}_{change_type}_{idx}"
        row["question"] = question
        row["answer"] = answer
        rows.append(row)

    return rows


def main(max_docs=100):
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    if not ann_dir.exists():
        raise FileNotFoundError(f"Missing annotation directory: {ann_dir}")

    image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))

    random.seed(42)
    qa_rows = []

    if out_qa.exists():
        out_qa.unlink()

    for image_path in tqdm(image_files[:max_docs], desc="Creating FUNSD revision pairs"):
        doc_id = image_path.stem
        ann_path = ann_dir / f"{doc_id}.json"

        if not ann_path.exists():
            continue

        annotation = load_annotation(ann_path)
        candidates = extract_candidate_words(annotation)

        if not candidates:
            continue

        chosen = random.choice(candidates)

        old_value = chosen["text"]
        box = chosen["box"]
        field_name = chosen["label"]

        change_type = random.choice(["replacement", "deletion", "insertion"])

        if change_type == "replacement":
            new_value = make_replacement_value(old_value)
        elif change_type == "deletion":
            new_value = ""
        elif change_type == "insertion":
            new_value = random.choice(["approved", "updated", "paid", "reviewed", "new"])
        else:
            new_value = "updated"

        original_out = out_orig / f"{doc_id}.png"
        revised_out = out_rev / f"{doc_id}_v1.png"
        metadata_out = out_meta / f"{doc_id}_v1.json"

        shutil.copy(image_path, original_out)

        edit_image(
            image_path=image_path,
            box=box,
            change_type=change_type,
            old_value=old_value,
            new_value=new_value,
            output_path=revised_out
        )

        doc_qa_rows = build_questions(
            doc_id=doc_id,
            original_out=original_out,
            revised_out=revised_out,
            change_type=change_type,
            field_name=field_name,
            old_value=old_value,
            new_value=new_value,
            box=box
        )

        metadata = {
            "doc_id": doc_id,
            "original_image": str(original_out.relative_to(project_root)),
            "revised_image": str(revised_out.relative_to(project_root)),
            "change_type": change_type,
            "field_name": field_name,
            "old_value": old_value,
            "new_value": new_value,
            "box": box,
            "qa_pairs": [
                {
                    "qa_id": row["qa_id"],
                    "question": row["question"],
                    "answer": row["answer"]
                }
                for row in doc_qa_rows
            ]
        }

        with open(metadata_out, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        qa_rows.extend(doc_qa_rows)

    with open(out_qa, "w", encoding="utf-8") as f:
        for row in qa_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Saved QA file to: {out_qa}", flush=True)
    print(f"Created {len(qa_rows)} revision QA examples.", flush=True)

    counts = {}
    for row in qa_rows:
        counts[row["change_type"]] = counts.get(row["change_type"], 0) + 1

    print("QA count by change type:", counts, flush=True)


if __name__ == "__main__":
    main(max_docs=100)
