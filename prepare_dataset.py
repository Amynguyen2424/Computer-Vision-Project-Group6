"""Build the YOLO dataset for the gesture-controlled photo booth (Group 6).

Reads the Google Drive export (images/ + labels/, files named <class>_<NNN>.<ext>),
checks every image/label pair, fixes Joseph's class-ID order, and copies the files
into dataset/images/{train,val,test} and dataset/labels/{train,val,test}.

Split (per class, from the group's split doc, 80/10/10):
    HaGRID 001-368 -> 320 train / 40 val / 8 test (random, seed 42)
    Team   369-400 -> test (our own photos are the test data, as the brief says)

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --images <dir> [<dir> ...] --labels <dir>
"""
import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

DOWNLOADS = Path.home() / "Downloads"
DEFAULT_IMAGES = [
    DOWNLOADS / "images-20261105T072605Z-1-001" / "images",
    DOWNLOADS / "images-20261105T072605Z-1-002" / "images",
]
DEFAULT_LABELS = DOWNLOADS / "labels-20261105T072606Z-1-001" / "labels"

CLASSES = {"palm": 0, "fist": 1, "like": 2, "dislike": 3}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
NAME_RE = re.compile(r"^(palm|fist|like|dislike)_(\d{3})$")

HAGRID_MAX = 368  # 001-368 are HaGRID, 369-400 are team photos
TEAM = [(369, 376, "Amy"), (377, 384, "Ben"), (385, 392, "Daff"), (393, 400, "Joseph")]
N_VAL, N_TEST_HAGRID = 40, 8
SEED = 42

# Joseph's labels don't use our class order. His first export (MakeSense list like,
# dislike, palm, fist) and the redone Drive labels of 2026-11-05 (palm, like, dislike,
# fist) differ, so each file is mapped with whichever order matches its file name.
JOSEPH_REMAPS = [
    {0: 0, 1: 2, 2: 3, 3: 1},   # 2026-11-05 Drive labels: palm=0, like=1, dislike=2, fist=3
    {0: 2, 1: 3, 2: 0, 3: 1},   # first export: like=0, dislike=1, palm=2, fist=3
]

OUT = Path(__file__).parent / "dataset"


def person(num):
    if num <= HAGRID_MAX:
        return ""
    return next(name for lo, hi, name in TEAM if lo <= num <= hi)


def read_label(path):
    """Return the label's rows as (class_id, x, y, w, h); raise ValueError if malformed."""
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"line {line_no}: expected 5 values, got {len(parts)}")
        cls, coords = int(parts[0]), [float(v) for v in parts[1:]]
        if not all(0.0 <= v <= 1.0 for v in coords) or coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"line {line_no}: coordinates out of range {coords}")
        rows.append((cls, *coords))
    if not rows:
        raise ValueError("empty label file")
    return rows


def remap_joseph(rows, expected):
    """Map Joseph's class IDs to ours with the order that matches the file name."""
    for remap in JOSEPH_REMAPS:
        mapped = [(remap.get(c, c), *xywh) for c, *xywh in rows]
        if all(c == expected for c, *_ in mapped):
            return mapped
    return rows   # no order fits; the class-ID check reports it


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = ap.parse_args()

    # Files not named <class>_<NNN> (e.g. Drive's "Copy of palm_369.txt") are skipped.
    images, problems, ignored = {}, [], []
    for d in args.images:
        for p in d.iterdir():
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            if not NAME_RE.match(p.stem):
                ignored.append(p.name)
            elif p.stem in images:
                problems.append(f"duplicate image: {p.stem}")
            else:
                images[p.stem] = p
    labels = {}
    for p in args.labels.glob("*.txt"):
        if NAME_RE.match(p.stem):
            labels[p.stem] = p
        else:
            ignored.append(p.name)

    for stem in sorted(images.keys() - labels.keys()):
        problems.append(f"image without label: {stem}")
    for stem in sorted(labels.keys() - images.keys()):
        problems.append(f"label without image: {stem}")

    records = []
    for stem in sorted(images.keys() & labels.keys()):
        cls_name, num = NAME_RE.match(stem).groups()
        num = int(num)
        try:
            rows = read_label(labels[stem])
        except ValueError as e:
            problems.append(f"bad label {stem}: {e}")
            continue
        if person(num) == "Joseph":
            rows = remap_joseph(rows, CLASSES[cls_name])
        wrong = [r[0] for r in rows if r[0] != CLASSES[cls_name]]
        if wrong:
            problems.append(f"class-ID mismatch {stem}: expected {CLASSES[cls_name]}, got {wrong}")
            continue
        records.append({"stem": stem, "image": images[stem], "rows": rows, "class": cls_name,
                        "num": num, "source": "hagrid" if num <= HAGRID_MAX else "team",
                        "person": person(num)})

    print(f"images found: {len(images)}   labels found: {len(labels)}   usable pairs: {len(records)}")
    if ignored:
        print(f"ignored {len(ignored)} file(s) not named <class>_<NNN>: {', '.join(sorted(ignored))}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        sys.exit("\nFix the problems above before building the dataset.")

    # Seeded per-class split: team photos -> test; HaGRID -> val / test top-up / train.
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["class"]].append(r)
    for cls_name in CLASSES:
        hagrid = sorted((r for r in by_class[cls_name] if r["source"] == "hagrid"), key=lambda r: r["num"])
        rng.shuffle(hagrid)
        for i, r in enumerate(hagrid):
            r["split"] = "val" if i < N_VAL else "test" if i < N_VAL + N_TEST_HAGRID else "train"
        for r in by_class[cls_name]:
            if r["source"] == "team":
                r["split"] = "test"

    for sub in ("images", "labels"):
        if (OUT / sub).exists():
            shutil.rmtree(OUT / sub)
        for split in ("train", "val", "test"):
            (OUT / sub / split).mkdir(parents=True)

    for r in records:
        shutil.copy2(r["image"], OUT / "images" / r["split"] / r["image"].name)
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in r["rows"]]
        (OUT / "labels" / r["split"] / f"{r['stem']}.txt").write_text("\n".join(lines) + "\n")

    with open(OUT / "split_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "source", "person", "split", "boxes"])
        for r in sorted(records, key=lambda r: (r["split"], r["stem"])):
            w.writerow([r["image"].name, r["class"], r["source"], r["person"], r["split"], len(r["rows"])])

    # Image list for the team-only test evaluation (data_team_test.yaml).
    team_test = sorted(r["image"].name for r in records if r["source"] == "team")
    (OUT / "test_team.txt").write_text("".join(f"./images/test/{n}\n" for n in team_test))

    counts = Counter((r["split"], r["class"]) for r in records)
    print(f"\n{'split':<6}" + "".join(f"{c:>9}" for c in CLASSES) + f"{'total':>9}")
    for split in ("train", "val", "test"):
        row = [counts[(split, c)] for c in CLASSES]
        print(f"{split:<6}" + "".join(f"{n:>9}" for n in row) + f"{sum(row):>9}")
    team = Counter(r["person"] for r in records if r["source"] == "team")
    print("\nteam photos in test:", dict(team), "=", sum(team.values()))
    print(f"Joseph's labels remapped: {sum(1 for r in records if r['person'] == 'Joseph')} files")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
