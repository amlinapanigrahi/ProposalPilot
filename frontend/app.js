/* ===================================================================
   ProposalPilot — app.js
   Plain vanilla JS, no build step. Connects to the FastAPI backend.
   =================================================================== */

const API_BASE_URL = "http://localhost:8000";

// ─── DOM refs ────────────────────────────────────────────────────────
const intakeView   = document.getElementById("intake-view");
const casefileView = document.getElementById("casefile-view");

const uploadForm   = document.getElementById("upload-form");
const dropZone     = document.getElementById("drop-zone");
const dropPrompt   = document.getElementById("drop-zone-prompt");
const fileInput    = document.getElementById("file-input");
const budgetInput  = document.getElementById("budget-input");
const submitBtn    = document.getElementById("submit-btn");
const formError    = document.getElementById("form-error");
const intakePanel  = document.querySelector(".intake-panel");

const proposalTitle = document.getElementById("proposal-title");
const proposalMeta  = document.getElementById("proposal-meta");
const verdictBar    = document.getElementById("verdict-bar");
const scoresGrid    = document.getElementById("scores-grid");
const riskFlagsList = document.getElementById("risk-flags-list");
const shapMethodology = document.getElementById("shap-methodology");
const shapBars      = document.getElementById("shap-bars");
const similarProjects = document.getElementById("similar-projects");
const narrativePanel  = document.getElementById("narrative-panel");
const newEvalBtn    = document.getElementById("new-eval-btn");

// ─── State ───────────────────────────────────────────────────────────
let selectedFile = null;

// ─── Drop zone interactions ──────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length === 0) return; // user cancelled dialog
  selectedFile = fileInput.files[0];
  showSelectedFile(selectedFile.name);
});

// Drag-and-drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drop-zone--dragover");
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drop-zone--dragover");
});
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drop-zone--dragover");
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    selectedFile = files[0];
    fileInput.files = e.dataTransfer.files; // sync the input
    showSelectedFile(selectedFile.name);
  }
});

function showSelectedFile(name) {
  dropPrompt.innerHTML = "";
  const span = document.createElement("span");
  span.className = "drop-zone__filename";
  span.textContent = name;
  dropPrompt.appendChild(span);
  dropZone.classList.add("drop-zone--has-file");
}

// ─── Form submission ─────────────────────────────────────────────────
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();

  // Client-side validation
  if (!selectedFile) {
    showError("Please select a proposal PDF to upload.");
    return;
  }
  const budgetValue = parseFloat(budgetInput.value);
  if (!budgetInput.value || isNaN(budgetValue) || budgetValue <= 0) {
    showError("Please enter a valid positive budget amount.");
    return;
  }

  // Enter loading state
  setLoading(true);

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("budget", budgetValue);

  try {
    const response = await fetch(`${API_BASE_URL}/api/proposals/upload`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      // Try to extract the server's detail message
      let errMsg = `Server returned ${response.status}.`;
      try {
        const errBody = await response.json();
        if (errBody.detail) {
          errMsg = typeof errBody.detail === "string"
            ? errBody.detail
            : JSON.stringify(errBody.detail);
        }
      } catch { /* ignore parse failure */ }
      showError(errMsg);
      setLoading(false);
      return;
    }

    const data = await response.json();

    try {
      renderCaseFile(data);
      switchToView("casefile");
    } catch (renderErr) {
      console.error("Render error:", renderErr);
      showError("Failed to render results: " + renderErr.message);
    }
  } catch (networkErr) {
    console.error("Network error:", networkErr);
    showError("Could not reach the evaluation server. Is the backend running?");
  } finally {
    setLoading(false);
  }
});

// ─── View toggling ───────────────────────────────────────────────────
function switchToView(view) {
  if (view === "casefile") {
    intakeView.hidden = true;
    casefileView.hidden = false;
    // Re-trigger the entrance animation
    casefileView.style.animation = "none";
    casefileView.offsetHeight; // force reflow
    casefileView.style.animation = "";
  } else {
    casefileView.hidden = true;
    intakeView.hidden = false;
    intakeView.style.animation = "none";
    intakeView.offsetHeight;
    intakeView.style.animation = "";
    resetForm();
  }
}

newEvalBtn.addEventListener("click", () => switchToView("intake"));

// ─── Helpers ─────────────────────────────────────────────────────────
function showError(msg) {
  formError.textContent = msg;
}
function clearError() {
  formError.textContent = "";
}

function setLoading(on) {
  submitBtn.disabled = on;
  submitBtn.textContent = on ? "Reviewing…" : "Submit for Review";
  intakePanel.classList.toggle("intake-panel--loading", on);
}

function resetForm() {
  selectedFile = null;
  fileInput.value = "";
  budgetInput.value = "";
  clearError();
  dropPrompt.innerHTML =
    'Drop proposal PDF here or <span class="drop-zone__browse">click to browse</span>';
  dropZone.classList.remove("drop-zone--has-file");
}

// ─── Humanize helper ─────────────────────────────────────────────────
function humanize(key) {
  const map = {
    novelty_score: "Novelty Score",
    financial_score: "Financial Score",
    budget: "Budget",
    tech_alignment: "Technical Alignment"
  };
  return map[key] || key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

// ===================================================================
//   RENDER CASE FILE
// ===================================================================
function renderCaseFile(data) {
  const a = data.analysis;
  const m = a.metrics;
  const v = a.evaluation_verdict;

  // ── Header ──
  proposalTitle.textContent = data.title || "Untitled Proposal";
  proposalMeta.textContent = `Case #${data.proposal_id} · Reviewed just now`;

  // ── Verdict bar ──
  renderVerdictBar(v);

  // ── Exhibit I — Scores ──
  scoresGrid.innerHTML = "";
  const scores = [
    { label: "Novelty",            value: m.novelty_score,       rating: m.novelty_rating },
    { label: "Financial",          value: m.financial_score,     rating: m.financial_rating },
    { label: "Tech Alignment",     value: m.technical_alignment, rating: null },
    { label: "Final Score",        value: m.final_score,         rating: null }
  ];
  scores.forEach(s => {
    const card = document.createElement("div");
    card.className = "score-card";
    card.innerHTML = `
      <div class="score-card__value">${formatNum(s.value)}</div>
      <div class="score-card__label">${s.label}</div>
      ${s.rating ? `<div class="score-card__rating">${s.rating}</div>` : ""}
    `;
    scoresGrid.appendChild(card);
  });

  // ── Exhibit II — Risk flags ──
  riskFlagsList.innerHTML = "";
  const flags = m.financial_risk_flags || [];
  if (flags.length === 0) {
    const li = document.createElement("li");
    li.className = "risk-flag--none";
    li.textContent = "No risk flags identified.";
    riskFlagsList.appendChild(li);
  } else {
    flags.forEach(flag => {
      const li = document.createElement("li");
      li.className = "risk-flag";
      li.innerHTML = `<span class="risk-flag__marker" aria-hidden="true"></span><span>${escapeHTML(flag)}</span>`;
      riskFlagsList.appendChild(li);
    });
  }

  // ── Exhibit III — SHAP ──
  const xai = a.explainable_ai_attribution;
  shapMethodology.textContent = xai.methodology_type || "";
  shapBars.innerHTML = "";

  const contributions = xai.feature_contributions || {};
  const entries = Object.entries(contributions);
  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v.shap_value)), 0.001);

  entries.forEach(([key, val]) => {
    const isPositive = val.shap_value >= 0;
    const pct = (Math.abs(val.shap_value) / maxAbs) * 100;

    const row = document.createElement("div");
    row.className = "shap-row";
    row.innerHTML = `
      <div class="shap-row__label">${humanize(key)}</div>
      <div class="shap-row__bar-track">
        <div class="shap-row__bar-fill ${isPositive ? "shap-row__bar-fill--approve" : "shap-row__bar-fill--review"}"
             style="width:${pct.toFixed(1)}%"></div>
      </div>
      <div class="shap-row__value ${isPositive ? "shap-row__value--positive" : "shap-row__value--negative"}">
        ${isPositive ? "+" : ""}${val.shap_value.toFixed(4)}
      </div>
    `;
    shapBars.appendChild(row);
  });

  // ── Exhibit IV — Similar projects ──
  similarProjects.innerHTML = "";
  (a.similar_past_projects || []).forEach(p => {
    const row = document.createElement("div");
    row.className = "similar-project-row";
    row.innerHTML = `
      <span class="similar-project-row__title">${escapeHTML(p.title)}</span>
      <span class="similar-project-row__score">${formatNum(p.similarity_score)}%</span>
    `;
    similarProjects.appendChild(row);
  });

  // ── Exhibit V — Narrative ──
  narrativePanel.innerHTML = `<p>${escapeHTML(a.ai_generated_narrative || "")}</p>`;
}

// ===================================================================
//   VERDICT BAR — horizontal status strip
// ===================================================================
function renderVerdictBar(verdict) {
  const isApproved = verdict.decision.toUpperCase().includes("APPROVAL");
  const confidence = `${verdict.confidence_percentage}%`;
  const decisionText = verdict.decision.toUpperCase();

  // Reset classes
  verdictBar.className = "verdict-bar";
  verdictBar.classList.add(isApproved ? "verdict-bar--approved" : "verdict-bar--review");

  verdictBar.innerHTML = `
    <span class="verdict-bar__decision">${escapeHTML(decisionText)}</span>
    <span class="verdict-bar__divider" aria-hidden="true"></span>
    <span class="verdict-bar__confidence">${confidence}</span>
  `;

  verdictBar.setAttribute("aria-label", `${decisionText} — Confidence ${confidence}`);
}

// ─── Utility ─────────────────────────────────────────────────────────
function formatNum(n) {
  if (n == null) return "—";
  return Number(n).toFixed(2);
}

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
