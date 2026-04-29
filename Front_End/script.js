const API_URL = "https://gwc-ai-image-detection-1.onrender.com/predict";

const dropZone     = document.getElementById("dropZone");
const fileInput    = document.getElementById("fileInput");
const placeholder  = document.getElementById("placeholder");
const previewImg   = document.getElementById("previewImg");
const imgOverlay   = document.getElementById("imgOverlay");
const scanOverlay  = document.getElementById("scanOverlay");
const fileInfo     = document.getElementById("fileInfo");
const fileName     = document.getElementById("fileName");
const fileSize     = document.getElementById("fileSize");
const analyzeBtn   = document.getElementById("analyzeBtn");

const emptyState   = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultState  = document.getElementById("resultState");

let currentFile = null;

// Label → CSS variable color
const LABEL_COLORS = {
  "AI_GENERATED":        "var(--label-ai)",
  "LIKELY_AI_GENERATED": "var(--label-likely-ai)",
  "LIKELY_REAL":         "var(--label-likely-real)",
  "UNCERTAIN":           "var(--label-uncertain)"
};

// Label → human-readable display text
const LABEL_DISPLAY = {
  "AI_GENERATED":        "AI Generated",
  "LIKELY_AI_GENERATED": "Likely AI",
  "LIKELY_REAL":         "Likely Real",
  "UNCERTAIN":           "Uncertain"
};

/* ── File selection ── */
dropZone.addEventListener("click", () => fileInput.click());

imgOverlay.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) loadFile(f);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

/* ── Load a selected file ── */
function loadFile(file) {
  currentFile = file;
  const url = URL.createObjectURL(file);

  previewImg.src = url;
  previewImg.style.display = "block";
  placeholder.style.display = "none";
  dropZone.classList.add("has-image");

  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileInfo.classList.add("visible");

  analyzeBtn.disabled = false;

  // Reset right panel to empty state
  showEmpty();
}

/* ── Helpers ── */
function formatBytes(b) {
  if (b < 1024)    return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  return (b / 1048576).toFixed(2) + " MB";
}

function showEmpty() {
  emptyState.style.display = "flex";
  emptyState.querySelector(".empty-text").textContent = "Awaiting image input";
  emptyState.querySelector(".empty-hint").textContent =
    "Upload an image on the left to begin AI authenticity analysis";
  loadingState.classList.remove("visible");
  resultState.classList.remove("visible");
}

function showLoading() {
  emptyState.style.display = "none";
  loadingState.classList.add("visible");
  resultState.classList.remove("visible");
  scanOverlay.classList.add("active");

  // Re-trigger step animations by forcing reflow
  const steps = loadingState.querySelectorAll(".step");
  steps.forEach((s) => {
    s.style.animation = "none";
    void s.offsetHeight;
    s.style.animation = "";
  });
}

function showResult(data, filename, elapsed) {
  emptyState.style.display = "none";
  loadingState.classList.remove("visible");
  scanOverlay.classList.remove("active");
  resultState.classList.add("visible");

  const label   = data.label || "UNCERTAIN";
  const color   = LABEL_COLORS[label] || "var(--accent)";
  const display = LABEL_DISPLAY[label] || label;
  const conf    = parseFloat(data.confidence) || 0;

  // Apply verdict color to the card via CSS variable
  const card = document.getElementById("verdictCard");
  card.style.setProperty("--verdict-color", color);

  document.getElementById("verdictLabel").textContent = display;
  document.getElementById("verdictBadge").textContent = label.replace(/_/g, " ");
  document.getElementById("confValue").textContent    = (conf * 100).toFixed(1) + "%";

  // Animate confidence bar
  const bar = document.getElementById("confBar");
  bar.style.width = "0%";
  setTimeout(() => { bar.style.width = (conf * 100) + "%"; }, 80);

  document.getElementById("reasonText").textContent  = data.reason || "No reason provided.";
  document.getElementById("metaFilename").textContent = filename;
  document.getElementById("metaModel").textContent   = (data.model_used || "—").replace("models/", "");
  document.getElementById("metaSize").textContent    = fileSize.textContent;
  document.getElementById("metaTime").textContent    = elapsed + " ms";
}

/* ── Main analysis call ── */
async function analyzeImage() {
  if (!currentFile) return;

  showLoading();
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", currentFile);

  const t0 = Date.now();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const data    = await response.json();
    const elapsed = Date.now() - t0;

    showResult(data.result, data.filename, elapsed);

  } catch (err) {
    loadingState.classList.remove("visible");
    scanOverlay.classList.remove("active");

    emptyState.style.display = "flex";
    emptyState.querySelector(".empty-text").textContent = "Connection failed";
    emptyState.querySelector(".empty-hint").textContent =
      "Could not reach the API at " + API_URL + ". Make sure your FastAPI server is running.";

    console.error("API error:", err);
  } finally {
    analyzeBtn.disabled = false;
  }
}
