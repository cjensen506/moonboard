# MoonBoard Grade Predictor

A neural network that predicts the difficulty grade of a [MoonBoard](https://www.moonboard.com/)
climbing problem from its hold layout, with an interactive web app for exploring the problem
database and grading your own problems.

**Live demo:** https://cjensen506.github.io/moonboard/

The deployed app is fully static — the model runs in your browser and the problem data is
bundled as a static file, so there is no backend to run or pay for.

## What it does

The MoonBoard is a standardised training wall (11 columns A–K × 18 rows) used by climbers
worldwide, so problems are directly comparable. This project trains a model on **63,388**
public 2016-setup problems to predict a French climbing grade (6B–8B+) from the binary
18×11 grid of holds a problem uses.

The web app has three views:

- **About** — what the MoonBoard is and how the model works.
- **Problem Lookup** — search/filter the problem database, view hold layouts on the board,
  and compare the consensus grade with the model's prediction.
- **Grade Predictor** — click any combination of holds and get an instant predicted grade,
  plus any existing problems that use the exact same holds.

## Model

A small MLP trained in TensorFlow/Keras:

```
Input(18×11) → Flatten(198) → Dense(512,relu) → Dropout → Dense(256,relu)
            → Dropout → Dense(64,relu) → Dense(1) → 1 + softplus(x)
```

The final `1 + softplus(x)` transform is a soft lower bound at grade 1. Input is a raw
binary hold grid — no scaling or normalisation. The model sees only *which* holds are used,
not move order or start/finish designation.

Training and data-prep code lives in `moonboard_model_trainer.py` and
`moonboard_data_prep.py`; the trained Keras model and raw data are kept out of git (see
`.gitignore`).

## How the static site is built

The deployed site lives in [`docs/`](docs/) and is **generated** — don't edit it by hand.
It is produced from the editable source in `web/static/` plus the trained model and raw data:

```bash
python3 web/build_static.py
```

This:

1. Copies `web/static/` (HTML/CSS/JS) into `docs/`.
2. Exports the Keras Dense-layer weights to `docs/model_weights.json`.
3. Runs the model over every problem and writes the trimmed records, with predictions
   baked in, to `docs/problems.json`.

At runtime, [`web/static/js/model.js`](web/static/js/model.js) reimplements the MLP forward
pass in dependency-free JavaScript (verified to match the Keras model to ~5e-3 end-to-end),
and [`web/static/js/app.js`](web/static/js/app.js) does all search, filtering, and hold-set
matching client-side. Because the raw data and model are gitignored, the generated
`docs/*.json` artifacts are committed so GitHub Pages can serve them.

## Running locally

Static site (matches production):

```bash
python3 -m http.server -d docs 8000
# open http://localhost:8000/
```

Original FastAPI backend (development only; not used by the deployed site):

```bash
uvicorn web.api:app --reload
```

## Deployment

GitHub Pages → **Settings → Pages → Deploy from a branch → `main` / `docs`**.

## License

[MIT](LICENSE)
