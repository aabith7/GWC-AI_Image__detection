const API_URL = "https://gwc-ai-image-detection-1.onrender.com/predict";

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const placeholder = document.getElementById("placeholder");
const previewImg = document.getElementById("previewImg");
const imgOverlay = document.getElementById("imgOverlay");
const scanOverlay = document.getElementById("scanOverlay");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const analyzeBtn = document.getElementById("analyzeBtn");

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultState = document.getElementById("resultState");

let currentFile = null;

const LABEL_COLORS = {
  AI_GENERATED: "var(--label-ai)",
  REAL_IMAGE: "var(--label-likely-real)",
  LIKELY_AI_GENERATED: "var(--label-likely-ai)",
  LIKELY_REAL: "var(--label-likely-real)",
  UNCERTAIN: "var(--label-uncertain)"
};

const LABEL_DISPLAY = {
  AI_GENERATED: "AI Generated",
  REAL_IMAGE: "Real Image",
  LIKELY_AI_GENERATED: "Likely AI",
  LIKELY_REAL: "Likely Real",
  UNCERTAIN: "Uncertain"
};

dropZone.addEventListener("click", () => fileInput.click());

imgOverlay.addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.click();
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");

  const file = event.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) loadFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

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
  showEmpty();
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(2) + " MB";
}

function showEmpty() {
  emptyState.style.display = "flex";
  emptyState.querySelector(".empty-text").textContent = "Awaiting image input";
  emptyState.querySelector(".empty-hint").textContent =
    "Upload an image on the left to begin AI authenticity analysis";

  loadingState.classList.remove("visible");
  resultState.classList.remove("visible");
  scanOverlay.classList.remove("active");
}

function showLoading() {
  emptyState.style.display = "none";
  loadingState.classList.add("visible");
  resultState.classList.remove("visible");
  scanOverlay.classList.add("active");

  const steps = loadingState.querySelectorAll(".step");
  steps.forEach((step) => {
    step.style.animation = "none";
    void step.offsetHeight;
    step.style.animation = "";
  });
}

function showResult(data, filename, elapsed) {
  emptyState.style.display = "none";
  loadingState.classList.remove("visible");
  scanOverlay.classList.remove("active");
  resultState.classList.add("visible");

  const label = data.label || "UNCERTAIN";
  const color = LABEL_COLORS[label] || "var(--accent)";
  const display = LABEL_DISPLAY[label] || label;
  const confidence = parseFloat(data.confidence) || 0;

  const card = document.getElementById("verdictCard");
  card.style.setProperty("--verdict-color", color);

  setText("verdictLabel", display);
  setText("verdictBadge", label.replace(/_/g, " "));
  setText("confValue", (confidence * 100).toFixed(1) + "%");
  setText("reasonText", data.reason || "No reason provided.");
  setText("metaFilename", filename || "-");
  setText("metaModel", (data.model_used || "-").replace("models/", ""));
  setText("metaSize", fileSize.textContent);
  setText("metaTime", elapsed + " ms");

  const bar = document.getElementById("confBar");
  bar.style.width = "0%";
  setTimeout(() => {
    bar.style.width = confidence * 100 + "%";
  }, 80);
}

function showError(title, message) {
  loadingState.classList.remove("visible");
  resultState.classList.remove("visible");
  scanOverlay.classList.remove("active");

  emptyState.style.display = "flex";
  emptyState.querySelector(".empty-text").textContent = title;
  emptyState.querySelector(".empty-hint").textContent = message;
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

async function analyzeImage() {
  if (!currentFile) return;

  showLoading();
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", currentFile);

  const startedAt = Date.now();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      let message = "Server responded with " + response.status;
      try {
        const errorData = await response.json();
        message = errorData.detail || message;
      } catch (parseError) {
        console.warn("Could not parse error response:", parseError);
      }
      throw new Error(message);
    }

    const data = await response.json();
    const elapsed = Date.now() - startedAt;

    showResult(data.result, data.filename, elapsed);
  } catch (error) {
    showError(
      "Gemini error",
      error.message || "Gemini did not return a usable response. Please try again."
    );
    console.error("API error:", error);
  } finally {
    analyzeBtn.disabled = false;
  }
}
