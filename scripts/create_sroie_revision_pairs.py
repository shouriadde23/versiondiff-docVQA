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


PROJECT_ROOT = Path.home() / "VersionDiff-DocVQA"

SROIE_ROOT = PROJECT_ROOT / "data/raw/SROIE/SROIE2019"
IMG_DIR = SROIE_ROOT / "train/img"
BOX_DIR = SROIE_ROOT / "train/box"
ENT_DIR = SROIE_ROOT / "train/entities"

OUT_ORIG = PROJECT_ROOT / "data/processed/sroie/original"
OUT_REV = PROJECT_ROOT / "data/processed/sroie/revised"
OUT_META = PROJECT_ROOT / "data/processed/sroie/metadata"
OUT_QA = PROJECT_ROOT / "data/processed/qa_jsonl/sroie_revision_qa.jsonl"

OUT_ORIG.mkdir(parents=True, exist_ok=True)
OUT_REV.mkdir(parents=True, exist_ok=True)
OUT_META.mkdir(parents=True, exist_ok=True)
OUT_QA.parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_image_for_doc(doc_id):
    for ext in [".jpg", ".png", ".jpeg"]:
        p = IMG_DIR / f"{doc_id}{ext}"
        if p.exists():
            return p
    return None


def parse_box_file(path):
    """
    SROIE box files usually contain lines like:
    x1,y1,x2,y2,x3,y3,x4,y4,text
    """
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
    """
    Try to find the OCR box that contains the entity value.
    If exact matching fails, try substring matching.
    """
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


def make_new_value(field_name, old_value):
    field_name = field_name.lower()

    if field_name == "total":
        clean = str(old_value).replace(",", "").replace("$", "").strip()
        try:
            val = float(clean)
            return f"{val + random.choice([1.00, 2.50, 5.00, 10.00]):.2f}"
        except Exception:
            return "99.99"

    if field_name == "date":
        return random.choice(["04/27/2026", "2026-04-27", "27/04/2026"])

    if field_name == "company":
        return random.choice(["NEW STORE", "UPDATED MART", "REVISED SHOP"])

    if field_name == "address":
        return random.choice(["123 UPDATED STREET", "456 NEW ROAD", "789 REVISED AVE"])

    return random.choice(["UPDATED", "REVISED", "NEWVALUE"])


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


def edit_image(image_path, box, change_type, old_value, new_value, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if box is None:
        # If no box is found, save original as revised.
        img.save(output_path)
        return False

    if change_type == "replacement":
        draw_white_box(draw, box)
        draw.text((box[0], box[1]), str(new_value), fill="black", font=get_font(box))

    elif change_type == "deletion":
        draw_white_box(draw, box)

    elif change_type == "insertion":
        x0, y0, x1, y1 = box
        insert_x = x0
        insert_y = y1 + 8
        draw.text((insert_x, insert_y), str(new_value), fill="black", font=get_font(box))

    else:
        raise ValueError(f"Unknown change type: {change_type}")

    img.save(output_path)
    return True


def build_questions(doc_id, original_out, revised_out, field_name, change_type, old_value, new_value, box):
    base = {
        "dataset": "sroie",
        "doc_id": doc_id,
        "original_image": str(original_out.relative_to(PROJECT_ROOT)),
        "revised_image": str(revised_out.relative_to(PROJECT_ROOT)),
        "field_name": field_name,
        "change_type": change_type,
        "old_value": old_value,
        "new_value": new_value,
        "box": box
    }

    rows = []

    if change_type == "replacement":
        qa_pairs = [
            (
                f"What was the {field_name} changed to?",
                str(new_value)
            ),
            (
                f"What was the original {field_name} before the revision?",
                str(old_value)
            ),
            (
                f"How did the {field_name} change?",
                f"The {field_name} changed from {old_value} to {new_value}."
            ),
            (
                "What type of revision occurred?",
                "replacement"
            )
        ]

    elif change_type == "deletion":
        qa_pairs = [
            (
                f"What {field_name} value was removed?",
                str(old_value)
            ),
            (
                "What type of revision occurred?",
                "deletion"
            ),
            (
                f"Was the {field_name} deleted?",
                "Yes"
            )
        ]

    elif change_type == "insertion":
        qa_pairs = [
            (
                f"What {field_name} value was added?",
                str(new_value)
            ),
            (
                "What type of revision occurred?",
                "insertion"
            ),
            (
                f"Was a new {field_name} inserted?",
                "Yes"
            )
        ]

    else:
        qa_pairs = []

    for i, (question, answer) in enumerate(qa_pairs):
        row = dict(base)
        row["qa_id"] = f"{doc_id}_{field_name}_{change_type}_{i}"
        row["question"] = question
        row["answer"] = answer
        rows.append(row)

    return rows


def main(max_docs=200):
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Missing SROIE image directory: {IMG_DIR}")

    if not BOX_DIR.exists():
        raise FileNotFoundError(f"Missing SROIE box directory: {BOX_DIR}")

    if not ENT_DIR.exists():
        raise FileNotFoundError(f"Missing SROIE entities directory: {ENT_DIR}")

    entity_files = sorted(ENT_DIR.glob("*.txt"))

    random.seed(42)
    qa_rows = []
    skipped = 0

    if OUT_QA.exists():
        OUT_QA.unlink()

    for ent_path in tqdm(entity_files[:max_docs], desc="Creating SROIE revision pairs"):
        doc_id = ent_path.stem

        image_path = find_image_for_doc(doc_id)
        box_path = BOX_DIR / f"{doc_id}.txt"

        if image_path is None or not box_path.exists():
            skipped += 1
            continue

        try:
            entities = load_json(ent_path)
        except Exception:
            skipped += 1
            continue

        box_rows = parse_box_file(box_path)

        available_fields = []
        for field in ["total", "date", "company", "address"]:
            val = entities.get(field, "")
            if val:
                available_fields.append(field)

        if not available_fields:
            skipped += 1
            continue

        field_name = random.choice(available_fields)
        old_value = str(entities.get(field_name, "")).strip()

        box = find_box_for_value(box_rows, old_value)

        if box is None:
            skipped += 1
            continue

        change_type = random.choice(["replacement", "deletion", "insertion"])

        if change_type == "replacement":
            new_value = make_new_value(field_name, old_value)
        elif change_type == "deletion":
            new_value = ""
        elif change_type == "insertion":
            new_value = make_new_value(field_name, old_value)
        else:
            new_value = "UPDATED"

        original_out = OUT_ORIG / f"{doc_id}.jpg"
        revised_out = OUT_REV / f"{doc_id}_v1.jpg"
        metadata_out = OUT_META / f"{doc_id}_v1.json"

        shutil.copy(image_path, original_out)

        edited = edit_image(
            image_path=image_path,
            box=box,
            change_type=change_type,
            old_value=old_value,
            new_value=new_value,
            output_path=revised_out
        )

        if not edited:
            skipped += 1
            continue

        doc_qa_rows = build_questions(
            doc_id=doc_id,
            original_out=original_out,
            revised_out=revised_out,
            field_name=field_name,
            change_type=change_type,
            old_value=old_value,
            new_value=new_value,
            box=box
        )

        metadata = {
            "dataset": "sroie",
            "doc_id": doc_id,
            "original_image": str(original_out.relative_to(PROJECT_ROOT)),
            "revised_image": str(revised_out.relative_to(PROJECT_ROOT)),
            "field_name": field_name,
            "change_type": change_type,
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

    with open(OUT_QA, "w", encoding="utf-8") as f:
        for row in qa_rows:
            f.write(json.dumps(row) + "\n")

    counts = {}
    for row in qa_rows:
        counts[row["change_type"]] = counts.get(row["change_type"], 0) + 1

    field_counts = {}
    for row in qa_rows:
        field_counts[row["field_name"]] = field_counts.get(row["field_name"], 0) + 1

    print(f"Saved QA file to: {OUT_QA}", flush=True)
    print(f"Created {len(qa_rows)} SROIE revision QA examples.", flush=True)
    print(f"Skipped {skipped} documents.", flush=True)
    print("QA count by change type:", counts, flush=True)
    print("QA count by field:", field_counts, flush=True)


if __name__ == "__main__":
    main(max_docs=200)
