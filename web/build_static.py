#!/usr/bin/env python3
"""Build the static GitHub Pages site into ../docs.

Run locally (needs TensorFlow + the raw data/model on disk):

    python3 web/build_static.py

Outputs:
    docs/model_weights.json   - Dense-layer weights for the in-browser MLP
    docs/problems.json        - trimmed problem records with predictions baked in
    docs/<static tree>        - copy of web/static (html/css/js)
    docs/assets/mbsetup-2016.jpg

The deployed site has no backend; inference and lookup run client-side.
"""

import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

# --- Paths (mirror web/api.py) ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "moonboard_model" / "mlp" / "custom_mlp_model"
DATA_PATH = PROJECT_ROOT / "data" / "problems MoonBoard 2016 .json"
IMAGE_PATH = PROJECT_ROOT / "image_resources" / "mbsetup-2016.jpg"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DOCS_DIR = PROJECT_ROOT / "docs"

GRADE_LABELS = {
    1: "6B", 2: "6B+", 3: "6C", 4: "6C+", 5: "7A", 6: "7A+",
    7: "7B", 8: "7B+", 9: "7C", 10: "7C+", 11: "8A", 12: "8A+",
    13: "8B", 14: "8B+",
}
NUMERIC_GRADES = {v: k for k, v in GRADE_LABELS.items()}


# --- Hold conversion (ported from web/api.py) ---

def hold_to_coord(hold_string: str) -> tuple[int, int]:
    row = int(re.search(r"\d+", hold_string).group())
    col = ord(re.search(r"[a-zA-Z]", hold_string).group().upper()) - ord("A") + 1
    return (row, col)


def holds_to_array(hold_positions: list[str]) -> np.ndarray:
    arr = np.zeros((18, 11), dtype=np.float32)
    for hold in hold_positions:
        row, col = hold_to_coord(hold)
        arr[row - 1, col - 1] = 1.0
    return arr


def numeric_to_grade(val: float) -> str:
    rounded = max(1, min(14, round(val)))
    return GRADE_LABELS.get(rounded, str(rounded))


# --- Weight export ---

def export_weights(model: tf.keras.Model, out_path: Path) -> None:
    """Export Dense layers (weight, bias, activation) for the JS forward pass.

    The output transform (final Lambda) is `1 + softplus(x)` and is reimplemented
    directly in model.js, so it is not exported here.
    """
    layers = []
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) != 2:  # only Dense layers carry [kernel, bias]
            continue
        W, b = weights
        activation = layer.get_config().get("activation", "linear")
        layers.append({
            "shape": [int(W.shape[0]), int(W.shape[1])],
            "activation": activation,
            "W": W.astype(np.float32).flatten().round(7).tolist(),
            "b": b.astype(np.float32).round(7).tolist(),
        })
    out_path.write_text(json.dumps({"layers": layers}, separators=(",", ":")))
    print(f"  wrote {out_path.name}: {len(layers)} dense layers, "
          f"{out_path.stat().st_size / 1e6:.2f} MB")


# --- Problem data export (ported from web/api.py startup) ---

def export_problems(model: tf.keras.Model, out_path: Path) -> None:
    with open(DATA_PATH) as f:
        all_problems = json.load(f)["data"]
    print(f"  loaded {len(all_problems)} raw problems")

    arrays, valid = [], []
    for p in all_problems:
        moves = p.get("moves") or []
        if not moves:
            continue
        try:
            arrays.append(holds_to_array([m["description"] for m in moves]))
        except Exception:
            continue
        valid.append(p)

    x_all = np.stack(arrays)
    print(f"  predicting on {len(x_all)} problems...")
    preds = model.predict(x_all, batch_size=4096, verbose=0).flatten()

    records = []
    for i, p in enumerate(valid):
        predicted = float(preds[i])
        records.append({
            "apiId": p["apiId"],
            "name": p.get("name", ""),
            "grade": p.get("grade", ""),
            "gradeNumeric": NUMERIC_GRADES.get(p.get("grade")),
            "userGrade": p.get("userGrade", ""),
            "setby": p.get("setby", ""),
            "repeats": p.get("repeats", 0),
            "userRating": p.get("userRating", 0),
            "isBenchmark": p.get("isBenchmark", False),
            "predictedGradeNumeric": round(predicted, 2),
            "predictedGrade": numeric_to_grade(predicted),
            "moves": [
                {"d": m["description"],
                 "s": m.get("isStart", False),
                 "e": m.get("isEnd", False)}
                for m in p["moves"]
            ],
        })

    out_path.write_text(json.dumps(records, separators=(",", ":")))
    print(f"  wrote {out_path.name}: {len(records)} problems, "
          f"{out_path.stat().st_size / 1e6:.2f} MB")


def copy_static() -> None:
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    shutil.copytree(STATIC_DIR, DOCS_DIR)
    assets = DOCS_DIR / "assets"
    assets.mkdir(exist_ok=True)
    shutil.copy(IMAGE_PATH, assets / IMAGE_PATH.name)
    # calibrate.html is a local dev tool (needs the API); not part of the static site
    (DOCS_DIR / "calibrate.html").unlink(missing_ok=True)
    # Disable Jekyll processing on GitHub Pages
    (DOCS_DIR / ".nojekyll").touch()
    print(f"  copied static tree + {IMAGE_PATH.name} -> docs/")


def main() -> None:
    print("Copying static site...")
    copy_static()

    print("Loading model...")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    print("Exporting weights...")
    export_weights(model, DOCS_DIR / "model_weights.json")

    print("Exporting problems...")
    export_problems(model, DOCS_DIR / "problems.json")

    print("Done. Deploy by enabling GitHub Pages on /docs.")


if __name__ == "__main__":
    main()
