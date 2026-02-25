#!/usr/bin/env python3
"""MoonBoard Web API - FastAPI backend for grade prediction and problem lookup."""

import json
import re
import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --- Constants ---

GRADE_LABELS = {
    1: "6B", 2: "6B+", 3: "6C", 4: "6C+", 5: "7A", 6: "7A+",
    7: "7B", 8: "7B+", 9: "7C", 10: "7C+", 11: "8A", 12: "8A+",
    13: "8B", 14: "8B+",
}

NUMERIC_GRADES = {v: k for k, v in GRADE_LABELS.items()}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "moonboard_model" / "mlp" / "custom_mlp_model"
DATA_PATH = PROJECT_ROOT / "data" / "problems MoonBoard 2016 .json"
IMAGE_PATH = PROJECT_ROOT / "image_resources" / "mbsetup-2016.jpg"


# --- Hold conversion utilities ---

def hold_to_coord(hold_string: str) -> tuple[int, int]:
    """Convert hold string like 'E6' to (row, col) 1-indexed."""
    row = int(re.search(r'\d+', hold_string).group())
    col = ord(re.search(r'[a-zA-Z]', hold_string).group().upper()) - ord('A') + 1
    return (row, col)


def holds_to_array(hold_positions: list[str]) -> np.ndarray:
    """Convert list of hold strings to 18x11 binary array."""
    arr = np.zeros((18, 11), dtype=np.float32)
    for hold in hold_positions:
        row, col = hold_to_coord(hold)
        arr[row - 1, col - 1] = 1.0
    return arr


def numeric_to_grade(val: float) -> str:
    """Convert numeric grade to French grade string (nearest integer key)."""
    rounded = max(1, min(14, round(val)))
    return GRADE_LABELS.get(rounded, str(rounded))


# --- Pydantic models ---

class HoldInput(BaseModel):
    position: str
    type: str = "middle"  # start, middle, end


class PredictRequest(BaseModel):
    holds: list[HoldInput]


# --- App state (populated at startup) ---

app = FastAPI(title="MoonBoard Grade Predictor")

problems_by_id: dict[int, dict] = {}
problems_list: list[dict] = []  # summary records for listing
hold_index: dict[frozenset, list[int]] = {}  # frozenset of holds -> [apiIds]
model: tf.keras.Model = None


@app.on_event("startup")
async def startup():
    global model
    print("Loading MLP model...")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    print("Loading problem data...")
    with open(DATA_PATH, "r") as f:
        raw = json.load(f)

    all_problems = raw["data"]
    print(f"Loaded {len(all_problems)} problems")

    # Build arrays for batch prediction
    arrays = []
    valid_problems = []
    for p in all_problems:
        moves = p.get("moves", [])
        if not moves:
            continue
        hold_positions = [m["description"] for m in moves]
        try:
            arr = holds_to_array(hold_positions)
        except Exception:
            continue
        arrays.append(arr)
        valid_problems.append(p)

    x_all = np.stack(arrays)  # (N, 18, 11)
    print(f"Running batch prediction on {len(x_all)} problems...")
    predictions = model.predict(x_all, batch_size=4096, verbose=0).flatten()

    # Build lookups
    for i, p in enumerate(valid_problems):
        api_id = p["apiId"]
        moves = p["moves"]
        hold_positions = sorted(m["description"] for m in moves)
        predicted = float(predictions[i])

        grade_numeric = NUMERIC_GRADES.get(p.get("grade"), None)
        user_grade_numeric = NUMERIC_GRADES.get(p.get("userGrade"), None)

        record = {
            "apiId": api_id,
            "name": p.get("name", ""),
            "grade": p.get("grade", ""),
            "gradeNumeric": grade_numeric,
            "userGrade": p.get("userGrade", ""),
            "userGradeNumeric": user_grade_numeric,
            "predictedGradeNumeric": round(predicted, 2),
            "predictedGrade": numeric_to_grade(predicted),
            "setby": p.get("setby", ""),
            "repeats": p.get("repeats", 0),
            "userRating": p.get("userRating", 0),
            "isBenchmark": p.get("isBenchmark", False),
            "dateInserted": p.get("dateInserted", ""),
            "moves": [
                {"description": m["description"],
                 "isStart": m.get("isStart", False),
                 "isEnd": m.get("isEnd", False)}
                for m in moves
            ],
        }
        problems_by_id[api_id] = record
        problems_list.append({
            "apiId": api_id,
            "name": record["name"],
            "grade": record["grade"],
            "gradeNumeric": grade_numeric,
            "predictedGrade": record["predictedGrade"],
            "predictedGradeNumeric": record["predictedGradeNumeric"],
            "repeats": record["repeats"],
            "userRating": record["userRating"],
            "isBenchmark": record["isBenchmark"],
            "setby": record["setby"],
        })

        key = frozenset(hold_positions)
        hold_index.setdefault(key, []).append(api_id)

    print(f"Ready: {len(problems_by_id)} problems indexed")


# --- API endpoints ---

@app.get("/api/problems")
async def list_problems(
    q: str = "",
    grade: Optional[str] = None,
    benchmark: Optional[bool] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    filtered = problems_list
    if q:
        q_lower = q.lower()
        filtered = [p for p in filtered if q_lower in p["name"].lower()]
    if grade:
        filtered = [p for p in filtered if p["grade"] == grade]
    if benchmark is not None:
        filtered = [p for p in filtered if p["isBenchmark"] == benchmark]

    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    return {
        "problems": filtered[start:end],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@app.get("/api/problems/{api_id}")
async def get_problem(api_id: int):
    problem = problems_by_id.get(api_id)
    if not problem:
        return {"error": "Problem not found"}, 404
    return problem


@app.post("/api/predict")
async def predict_grade(req: PredictRequest):
    hold_positions = [h.position for h in req.holds]
    arr = holds_to_array(hold_positions)
    x = arr[np.newaxis, :, :]  # (1, 18, 11)
    pred = model.predict(x, verbose=0).flatten()[0]
    predicted_numeric = float(pred)

    # Check for matching existing problems
    key = frozenset(sorted(hold_positions))
    matching_ids = hold_index.get(key, [])
    matching_problems = []
    for aid in matching_ids:
        p = problems_by_id[aid]
        matching_problems.append({
            "apiId": p["apiId"],
            "name": p["name"],
            "grade": p["grade"],
            "userGrade": p["userGrade"],
            "repeats": p["repeats"],
            "isBenchmark": p["isBenchmark"],
        })

    return {
        "predictedGradeNumeric": round(predicted_numeric, 2),
        "predictedGrade": numeric_to_grade(predicted_numeric),
        "holdCount": len(hold_positions),
        "matchingProblems": matching_problems,
    }


@app.get("/api/grades")
async def get_grades():
    return {"grades": GRADE_LABELS}


# --- Static files & image ---

@app.get("/api/board-image")
async def board_image():
    return FileResponse(str(IMAGE_PATH), media_type="image/jpeg")


# Mount static files last so API routes take priority
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")
