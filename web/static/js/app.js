import { MoonBoard, parseHold } from "./board.js";

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

// Populate grade filter
fetch("/api/grades")
  .then((r) => r.json())
  .then((data) => {
    for (const [num, label] of Object.entries(data.grades)) {
      const opt = document.createElement("option");
      opt.value = label;
      opt.textContent = label;
      gradeFilter.appendChild(opt);
    }
  });

function fetchProblems() {
  const params = new URLSearchParams({
    page: currentPage,
    per_page: perPage,
  });
  const q = searchInput.value.trim();
  if (q) params.set("q", q);
  const grade = gradeFilter.value;
  if (grade) params.set("grade", grade);
  if (benchmarkFilter.checked) params.set("benchmark", "true");

  fetch(`/api/problems?${params}`)
    .then((r) => r.json())
    .then((data) => renderResults(data));
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

  fetch(`/api/problems/${apiId}`)
    .then((r) => r.json())
    .then((p) => {
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
    });
}

// Search with debounce
searchInput.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => { currentPage = 1; fetchProblems(); }, 300);
});
gradeFilter.addEventListener("change", () => { currentPage = 1; fetchProblems(); });
benchmarkFilter.addEventListener("change", () => { currentPage = 1; fetchProblems(); });

// Initial load
fetchProblems();

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
  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";

  fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ holds: selectedHolds }),
  })
    .then((r) => r.json())
    .then((data) => {
      let html = `
        <div class="predicted-grade">${data.predictedGrade}</div>
        <div class="predicted-numeric">${data.holdCount} holds</div>
      `;

      if (data.matchingProblems.length > 0) {
        html += '<div class="match-info"><h4>Matching Problem(s) Found</h4>';
        for (const m of data.matchingProblems) {
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
    })
    .catch((err) => {
      predictionResult.innerHTML = `<div style="color:#f44336">Error: ${err.message}</div>`;
    })
    .finally(() => {
      predictBtn.disabled = false;
      predictBtn.textContent = "Predict Grade";
    });
});

// Init
renderSelectedHolds();
updatePredictBtn();

// --- Utility ---
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str || "";
  return div.innerHTML;
}
