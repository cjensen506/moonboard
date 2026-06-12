import { MoonBoard } from "./board.js";
import { loadModel, predict, GRADE_LABELS } from "./model.js";

// --- Client-side data store (loaded once from problems.json) ---
let ALL_PROBLEMS = []; // normalized records
const byId = new Map(); // apiId -> record
const holdIndex = new Map(); // sorted-holds key -> [records]

function holdKey(descriptions) {
  return [...descriptions].sort().join(",");
}

async function loadData() {
  const raw = await fetch("problems.json").then((r) => r.json());
  ALL_PROBLEMS = raw.map((p) => {
    const moves = p.moves.map((m) => ({
      description: m.d,
      isStart: m.s,
      isEnd: m.e,
    }));
    return { ...p, moves };
  });
  for (const rec of ALL_PROBLEMS) {
    byId.set(rec.apiId, rec);
    const key = holdKey(rec.moves.map((m) => m.description));
    if (!holdIndex.has(key)) holdIndex.set(key, []);
    holdIndex.get(key).push(rec);
  }
}

// --- View routing ---
const navBtns = document.querySelectorAll(".nav-btn");
const views = document.querySelectorAll(".view");

navBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.view;
    navBtns.forEach((b) => b.classList.remove("active"));
    views.forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`view-${target}`).classList.add("active");
  });
});

// --- Lookup View ---
const lookupBoard = new MoonBoard(
  document.getElementById("lookup-board-img"),
  document.getElementById("lookup-board-canvas")
);

const searchInput = document.getElementById("search-input");
const gradeFilter = document.getElementById("grade-filter");
const benchmarkFilter = document.getElementById("benchmark-filter");
const resultsList = document.getElementById("results-list");
const pagination = document.getElementById("pagination");
const problemDetails = document.getElementById("problem-details");

let currentPage = 1;
const perPage = 50;
let debounceTimer = null;

// Populate grade filter from local grade labels
for (const label of Object.values(GRADE_LABELS)) {
  const opt = document.createElement("option");
  opt.value = label;
  opt.textContent = label;
  gradeFilter.appendChild(opt);
}

function fetchProblems() {
  const q = searchInput.value.trim().toLowerCase();
  const grade = gradeFilter.value;
  const benchmark = benchmarkFilter.checked;

  let filtered = ALL_PROBLEMS;
  if (q) filtered = filtered.filter((p) => p.name.toLowerCase().includes(q));
  if (grade) filtered = filtered.filter((p) => p.grade === grade);
  if (benchmark) filtered = filtered.filter((p) => p.isBenchmark);

  const total = filtered.length;
  const start = (currentPage - 1) * perPage;
  const problems = filtered.slice(start, start + perPage);
  renderResults({ problems, total });
}

function renderResults(data) {
  resultsList.innerHTML = "";
  if (data.problems.length === 0) {
    resultsList.innerHTML = '<div class="result-item"><span class="result-name">No results found</span></div>';
    pagination.innerHTML = "";
    return;
  }

  for (const p of data.problems) {
    const div = document.createElement("div");
    div.className = "result-item";
    div.dataset.apiId = p.apiId;
    div.innerHTML = `
      <div class="result-name">${escapeHtml(p.name)}</div>
      <div class="result-meta">${p.grade} | Predicted: ${p.predictedGrade} | ${p.repeats.toLocaleString()} repeats${p.isBenchmark ? " | BM" : ""}</div>
    `;
    div.addEventListener("click", () => selectProblem(p.apiId, div));
    resultsList.appendChild(div);
  }

  // Pagination
  const totalPages = Math.ceil(data.total / perPage);
  pagination.innerHTML = "";
  if (totalPages > 1) {
    const prevBtn = document.createElement("button");
    prevBtn.textContent = "Prev";
    prevBtn.disabled = currentPage <= 1;
    prevBtn.addEventListener("click", () => { currentPage--; fetchProblems(); });
    pagination.appendChild(prevBtn);

    const info = document.createElement("span");
    info.textContent = `Page ${currentPage} of ${totalPages} (${data.total.toLocaleString()} problems)`;
    pagination.appendChild(info);

    const nextBtn = document.createElement("button");
    nextBtn.textContent = "Next";
    nextBtn.disabled = currentPage >= totalPages;
    nextBtn.addEventListener("click", () => { currentPage++; fetchProblems(); });
    pagination.appendChild(nextBtn);
  }
}

function selectProblem(apiId, element) {
  document.querySelectorAll(".result-item.selected").forEach((el) => el.classList.remove("selected"));
  if (element) element.classList.add("selected");

  const p = byId.get(apiId);
  if (!p) return;
  lookupBoard.drawHolds(p.moves);
  problemDetails.innerHTML = `
    <div class="detail-grid">
      <div class="detail-name detail-item">
        <label>Problem</label>
        <div class="value">${escapeHtml(p.name)}</div>
      </div>
      <div class="detail-item">
        <label>Grade</label>
        <div class="value grade">${p.grade}</div>
      </div>
      <div class="detail-item">
        <label>Predicted Grade</label>
        <div class="value grade">${p.predictedGrade}</div>
      </div>
      <div class="detail-item">
        <label>User Grade</label>
        <div class="value">${p.userGrade || "N/A"}</div>
      </div>
      <div class="detail-item">
        <label>Set By</label>
        <div class="value">${escapeHtml(p.setby)}</div>
      </div>
      <div class="detail-item">
        <label>Repeats</label>
        <div class="value">${p.repeats.toLocaleString()}</div>
      </div>
      <div class="detail-item">
        <label>Rating</label>
        <div class="value">${p.userRating}/5</div>
      </div>
      <div class="detail-item">
        <label>Benchmark</label>
        <div class="value">${p.isBenchmark ? "Yes" : "No"}</div>
      </div>
      <div class="detail-item">
        <label>Holds</label>
        <div class="value">${p.moves.length} holds</div>
      </div>
      <div class="detail-item detail-holds-list">
        <label>Hold Positions</label>
        <div class="value">${p.moves.map(m => {
          const type = m.isStart ? "start" : m.isEnd ? "end" : "middle";
          return `<span class="hold-tag ${type}">${m.description} (${type})</span>`;
        }).join(" ")}</div>
      </div>
    </div>
  `;
}

// Search with debounce
searchInput.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => { currentPage = 1; fetchProblems(); }, 300);
});
gradeFilter.addEventListener("change", () => { currentPage = 1; fetchProblems(); });
benchmarkFilter.addEventListener("change", () => { currentPage = 1; fetchProblems(); });

// --- Create View ---
const createBoard = new MoonBoard(
  document.getElementById("create-board-img"),
  document.getElementById("create-board-canvas")
);

const selectedHoldsList = document.getElementById("selected-holds-list");
const predictBtn = document.getElementById("predict-btn");
const clearBtn = document.getElementById("clear-btn");
const predictionResult = document.getElementById("prediction-result");

// State: array of {position: "E6", type: "start"|"middle"|"end"}
let selectedHolds = [];

function getHoldType() {
  return document.querySelector('input[name="hold-type"]:checked').value;
}

function redrawCreateBoard() {
  const moves = selectedHolds.map((h) => ({
    description: h.position,
    isStart: h.type === "start",
    isEnd: h.type === "end",
  }));
  createBoard.drawHolds(moves);
}

function renderSelectedHolds() {
  selectedHoldsList.innerHTML = "";
  if (selectedHolds.length === 0) {
    selectedHoldsList.innerHTML = '<div style="color:#666;font-size:0.85rem;padding:0.5rem">Click holds on the board to add them.</div>';
  }
  for (let i = 0; i < selectedHolds.length; i++) {
    const h = selectedHolds[i];
    const div = document.createElement("div");
    div.className = `hold-chip ${h.type}`;
    div.innerHTML = `
      <span>${h.position} (${h.type})</span>
      <button data-index="${i}">&times;</button>
    `;
    div.querySelector("button").addEventListener("click", () => {
      selectedHolds.splice(i, 1);
      renderSelectedHolds();
      redrawCreateBoard();
      updatePredictBtn();
    });
    selectedHoldsList.appendChild(div);
  }
}

function updatePredictBtn() {
  predictBtn.disabled = selectedHolds.length === 0;
}

createBoard.enableInteractive((grid) => {
  // Toggle: if hold already selected, remove it; otherwise add it
  const existing = selectedHolds.findIndex((h) => h.position === grid.label);
  if (existing >= 0) {
    selectedHolds.splice(existing, 1);
  } else {
    selectedHolds.push({ position: grid.label, type: getHoldType() });
  }
  renderSelectedHolds();
  redrawCreateBoard();
  updatePredictBtn();
});

clearBtn.addEventListener("click", () => {
  selectedHolds = [];
  renderSelectedHolds();
  redrawCreateBoard();
  updatePredictBtn();
  predictionResult.innerHTML = "";
});

predictBtn.addEventListener("click", () => {
  const positions = selectedHolds.map((h) => h.position);
  const result = predict(positions);

  // Find existing problems with the exact same hold set
  const matches = holdIndex.get(holdKey(positions)) || [];

  let html = `
    <div class="predicted-grade">${result.predictedGrade}</div>
    <div class="predicted-numeric">${positions.length} holds</div>
  `;

  if (matches.length > 0) {
    html += '<div class="match-info"><h4>Matching Problem(s) Found</h4>';
    for (const m of matches) {
      html += `
        <div class="match-item">
          <strong>${escapeHtml(m.name)}</strong><br>
          Grade: ${m.grade} | User Grade: ${m.userGrade || "N/A"} | Repeats: ${m.repeats.toLocaleString()}
          ${m.isBenchmark ? " | Benchmark" : ""}
        </div>
      `;
    }
    html += "</div>";
  }

  predictionResult.innerHTML = html;
});

// --- Utility ---
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str || "";
  return div.innerHTML;
}

// --- Init ---
renderSelectedHolds();
updatePredictBtn();

(async function init() {
  resultsList.innerHTML = '<div class="result-item"><span class="result-name">Loading…</span></div>';
  await Promise.all([loadModel(), loadData()]);

  // Update the About-page total to the actual loaded count
  const statTotal = document.getElementById("stat-total");
  if (statTotal) statTotal.textContent = ALL_PROBLEMS.length.toLocaleString();

  fetchProblems();
})();
