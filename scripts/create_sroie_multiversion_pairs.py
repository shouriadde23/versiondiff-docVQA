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

sroie_root = project_root / "data/raw/SROIE/SROIE2019"
img_dir = sroie_root / "train/img"
box_dir = sroie_root / "train/box"
ent_dir = sroie_root / "train/entities"

out_dir = project_root / "data/processed/sroie_multiversion"
out_img = out_dir / "images"
out_meta = out_dir / "metadata"
out_qa = project_root / "data/processed/qa_jsonl/sroie_multiversion_qa.jsonl"

out_img.mkdir(parents=True, exist_ok=True)
out_meta.mkdir(parents=True, exist_ok=True)
out_qa.parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_image_for_doc(doc_id):
    for ext in [".jpg", ".png", ".jpeg"]:
        p = img_dir / f"{doc_id}{ext}"
        if p.exists():
            return p
    return None


def parse_box_file(path):
    rows = []

    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 9:
                continue

            try:
                coords = list(map(int, parts[:8]))
            except Exception:
                continue

            text = ",".join(parts[8:]).strip()

            xs = coords[0::2]
            ys = coords[1::2]

            box = [min(xs), min(ys), max(xs), max(ys)]

            rows.append({
                "text": text,
                "box": box
            })

    return rows


def normalize(s):
    return str(s).lower().strip().replace(" ", "")


def find_box_for_value(box_rows, value):
    if not value:
        return None

    value_norm = normalize(value)

    for row in box_rows:
        if normalize(row["text"]) == value_norm:
            return row["box"]

    for row in box_rows:
        text_norm = normalize(row["text"])
        if value_norm in text_norm or text_norm in value_norm:
            return row["box"]

    return None


def draw_white_box(draw, box):
    x0, y0, x1, y1 = box
    padding = 3
    draw.rectangle(
        [x0 - padding, y0 - padding, x1 + padding, y1 + padding],
        fill="white"
    )


def get_font(box):
    x0, y0, x1, y1 = box
    font_size = max(10, min(24, y1 - y0 + 2))

    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def write_value_on_image(base_image_path, box, new_value, out_path):
    img = Image.open(base_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if box is not None:
        draw_white_box(draw, box)
        draw.text((box[0], box[1]), str(new_value), fill="black", font=get_font(box))

    img.save(out_path)


def generate_value_history(field_name, old_value):
    field_name = field_name.lower()
    old_value = str(old_value).strip()

    if field_name == "total":
        clean = old_value.replace(",", "").replace("$", "").strip()
        try:
            base = float(clean)
        except Exception:
            base = 10.00

        return [
            f"{base:.2f}",
            f"{base + 2.00:.2f}",
            f"{base + 5.00:.2f}",
            f"{base + 10.00:.2f}",
        ]

    if field_name == "date":
        return [
            old_value,
            "04/01/2026",
            "04/15/2026",
            "04/27/2026",
        ]

    if field_name == "company":
        return [
            old_value,
            "Updated Mart",
            "Revised Shop",
            "Final Store",
        ]

    if field_name == "address":
        return [
            old_value,
            "123 Updated Street",
            "456 Revised Road",
            "789 Final Ave",
        ]

    return [
        old_value,
        "Value 1",
        "Value 2",
        "Value 3",
    ]


def build_multiversion_questions(doc_id, field_name, value_history, version_paths, box):
    v0, v1, v2, v3 = value_history

    base = {
        "dataset": "sroie",
        "doc_id": doc_id,
        "field_name": field_name,
        "version_paths": version_paths,
        "value_history": value_history,
        "box": box,
        "num_versions": 4,
    }

    qa_pairs = [
        (
            "final_value",
            f"What is the final {field_name} value?",
            str(v3),
        ),
        (
            "original_value",
            f"What was the original {field_name} value?",
            str(v0),
        ),
        (
            "before_final_value",
            f"What was the {field_name} value before the final revision?",
            str(v2),
        ),
        (
            "first_revision_value",
            f"What was the first revised {field_name} value?",
            str(v1),
        ),
        (
            "last_change_description",
            f"What was the last change made to the {field_name}?",
            f"The {field_name} changed from {v2} to {v3}.",
        ),
        (
            "first_change_description",
            f"What was the first change made to the {field_name}?",
            f"The {field_name} changed from {v0} to {v1}.",
        ),
        (
            "modification_count",
            f"How many times was the {field_name} modified?",
            "3",
        ),
        (
            "full_history",
            f"What is the full value history of the {field_name}?",
            f"{v0} -> {v1} -> {v2} -> {v3}",
        ),
    ]

    rows = []

    for idx, (question_type, question, answer) in enumerate(qa_pairs):
        row = dict(base)
        row["qa_id"] = f"{doc_id}_{field_name}_multiversion_{idx}"
        row["question_type"] = question_type
        row["question"] = question
        row["answer"] = answer
        rows.append(row)

    return rows


def main(max_docs=200):
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing SROIE image directory: {img_dir}")

    if not box_dir.exists():
        raise FileNotFoundError(f"Missing SROIE box directory: {box_dir}")

    if not ent_dir.exists():
        raise FileNotFoundError(f"Missing SROIE entities directory: {ent_dir}")

    random.seed(42)

    if out_qa.exists():
        out_qa.unlink()

    entity_files = sorted(ent_dir.glob("*.txt"))

    qa_rows = []
    skipped = 0

    for ent_path in tqdm(entity_files[:max_docs], desc="Creating SROIE multi-version chains"):
        doc_id = ent_path.stem

        image_path = find_image_for_doc(doc_id)
        box_path = box_dir / f"{doc_id}.txt"

        if image_path is None or not box_path.exists():
            skipped += 1
            continue

        try:
            entities = load_json(ent_path)
        except Exception:
            skipped += 1
            continue

        available_fields = []
        for field in ["total", "date", "company", "address"]:
            if str(entities.get(field, "")).strip():
                available_fields.append(field)

        if not available_fields:
            skipped += 1
            continue

        field_name = random.choice(available_fields)
        old_value = str(entities.get(field_name, "")).strip()

        box_rows = parse_box_file(box_path)
        box = find_box_for_value(box_rows, old_value)

        if box is None:
            skipped += 1
            continue

        value_history = generate_value_history(field_name, old_value)

        version_paths = []

        for version_idx, value in enumerate(value_history):
            out_path = out_img / f"{doc_id}_v{version_idx}.jpg"

            if version_idx == 0:
                shutil.copy(image_path, out_path)
            else:
                write_value_on_image(
                    base_image_path=image_path,
                    box=box,
                    new_value=value,
                    out_path=out_path
                )

            version_paths.append(str(out_path.relative_to(project_root)))

        metadata = {
            "dataset": "sroie",
            "doc_id": doc_id,
            "field_name": field_name,
            "value_history": value_history,
            "version_paths": version_paths,
            "box": box,
        }

        meta_path = out_meta / f"{doc_id}_multiversion.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        doc_qa_rows = build_multiversion_questions(
            doc_id=doc_id,
            field_name=field_name,
            value_history=value_history,
            version_paths=version_paths,
            box=box
        )

        qa_rows.extend(doc_qa_rows)

    with open(out_qa, "w", encoding="utf-8") as f:
        for row in qa_rows:
            f.write(json.dumps(row) + "\n")

    field_counts = {}
    question_counts = {}

    for row in qa_rows:
        field_counts[row["field_name"]] = field_counts.get(row["field_name"], 0) + 1
        question_counts[row["question_type"]] = question_counts.get(row["question_type"], 0) + 1

    print(f"Saved multi-version QA file to: {out_qa}", flush=True)
    print(f"Created {len(qa_rows)} multi-version QA examples.", flush=True)
    print(f"Skipped {skipped} documents.", flush=True)
    print("QA count by field:", field_counts, flush=True)
    print("QA count by question type:", question_counts, flush=True)


if __name__ == "__main__":
    main(max_docs=200)
