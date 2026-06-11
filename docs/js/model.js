/**
 * In-browser MoonBoard grade predictor.
 *
 * Pure-JS port of the trained Keras MLP:
 *   Flatten(18x11=198) -> Dense(512,relu) -> Dense(256,relu)
 *   -> Dense(64,relu) -> Dense(1,linear) -> output = 1 + softplus(x)
 *
 * Dropout layers are identity at inference and are omitted. The final output
 * transform (a soft lower bound at grade 1) is applied here, matching the
 * model's Lambda layer to ~1e-6. Weights are loaded from model_weights.json.
 */

const GRADE_LABELS = {
  1: "6B", 2: "6B+", 3: "6C", 4: "6C+", 5: "7A", 6: "7A+",
  7: "7B", 8: "7B+", 9: "7C", 10: "7C+", 11: "8A", 12: "8A+",
  13: "8B", 14: "8B+",
};

let LAYERS = null; // [{shape:[in,out], activation, W:Float32Array, b:Float32Array}]

/** Load and cache the model weights. Call once before predicting. */
export async function loadModel(url = "model_weights.json") {
  if (LAYERS) return;
  const data = await fetch(url).then((r) => r.json());
  LAYERS = data.layers.map((l) => ({
    shape: l.shape,
    activation: l.activation,
    W: Float32Array.from(l.W),
    b: Float32Array.from(l.b),
  }));
}

/** Convert a hold string like "E6" to (row, col), both 1-indexed. */
function holdToCoord(hold) {
  const row = parseInt(hold.match(/\d+/)[0], 10);
  const col = hold.match(/[a-zA-Z]/)[0].toUpperCase().charCodeAt(0) - 64;
  return [row, col];
}

/** Convert hold strings to a flat length-198 (18x11, row-major) binary vector. */
export function holdsToVector(positions) {
  const v = new Float32Array(198);
  for (const h of positions) {
    const [row, col] = holdToCoord(h);
    v[(row - 1) * 11 + (col - 1)] = 1.0;
  }
  return v;
}

/** Dense layer: y = act(x @ W + b). W is flat row-major [in, out]. */
function dense(x, layer) {
  const [nIn, nOut] = layer.shape;
  const { W, b, activation } = layer;
  const y = new Float32Array(nOut);
  for (let j = 0; j < nOut; j++) {
    let sum = b[j];
    for (let i = 0; i < nIn; i++) sum += x[i] * W[i * nOut + j];
    y[j] = activation === "relu" ? Math.max(0, sum) : sum;
  }
  return y;
}

/** Overflow-safe softplus: ln(1 + e^z). */
function softplus(z) {
  return z > 20 ? z : Math.log1p(Math.exp(z));
}

function numericToGrade(val) {
  const rounded = Math.max(1, Math.min(14, Math.round(val)));
  return GRADE_LABELS[rounded] ?? String(rounded);
}

/**
 * Predict a grade from hold strings.
 * @returns {{predictedGradeNumeric:number, predictedGrade:string}}
 */
export function predict(positions) {
  let x = holdsToVector(positions);
  for (const layer of LAYERS) x = dense(x, layer);
  const numeric = 1 + softplus(x[0]); // final Lambda
  return {
    predictedGradeNumeric: Math.round(numeric * 100) / 100,
    predictedGrade: numericToGrade(numeric),
  };
}

export { GRADE_LABELS };
