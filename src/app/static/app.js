// src/app/static/app.js

const fileInput = document.getElementById("file");
const runBtn = document.getElementById("run");
const clearBtn = document.getElementById("clear");

const statusText = document.getElementById("statusText");
const statusPill = document.getElementById("statusPill");

const detCountEl = document.getElementById("detCount");
const latencyEl = document.getElementById("latency");
const classesEl = document.getElementById("classes");

const enhancedImg = document.getElementById("enhanced");
const annotatedImg = document.getElementById("annotated");
const enhEmpty = document.getElementById("enhEmpty");
const annEmpty = document.getElementById("annEmpty");

// overlay
const overlay = document.getElementById("overlay");
const overlayImg = document.getElementById("overlayImg");
const overlayTitle = document.getElementById("overlayTitle");
const overlayClose = document.getElementById("overlayClose");

let currentFile = null;

function setStatus(mode, text){
  statusText.textContent = text;

  statusPill.classList.remove("idle", "run", "ok");
  if (mode === "idle") { statusPill.classList.add("idle"); statusPill.textContent = "Idle"; }
  if (mode === "run")  { statusPill.classList.add("run");  statusPill.textContent = "Running"; }
  if (mode === "ok")   { statusPill.classList.add("ok");   statusPill.textContent = "Done"; }
}

function setDisabled(run, clear){
  runBtn.disabled = run;
  clearBtn.disabled = clear;
}

function resetImages(){
  enhancedImg.src = "";
  annotatedImg.src = "";
  enhancedImg.style.display = "none";
  annotatedImg.style.display = "none";
  enhEmpty.style.display = "block";
  annEmpty.style.display = "block";
}

function resetUI(){
  currentFile = null;
  detCountEl.textContent = "—";
  latencyEl.textContent = "—";
  classesEl.innerHTML = "";
  setStatus("idle", "Idle");
  resetImages();
  setDisabled(true, true);
}

function addTag(name, conf){
  const el = document.createElement("div");
  el.className = "tag";
  el.innerHTML = `<b>${name}</b> • ${(conf*100).toFixed(1)}%`;
  classesEl.appendChild(el);
}

fileInput.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  currentFile = f;

  classesEl.innerHTML = "";
  detCountEl.textContent = "—";
  latencyEl.textContent = "—";
  setStatus("idle", "Ready");

  resetImages();
  setDisabled(false, false);
});

runBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  setStatus("run", "Running inference...");
  setDisabled(true, true);

  const fd = new FormData();
  fd.append("file", currentFile);

  try{
    const resp = await fetch("/api/predict", { method:"POST", body: fd });
    if (!resp.ok){
      setStatus("idle", "Server error");
      setDisabled(false, false);
      return;
    }

    const data = await resp.json();

    detCountEl.textContent = data.detections;
    latencyEl.textContent = `${data.latency_ms} ms`;

    // show images (contain, no crop)
    enhancedImg.src = data.enhanced_png;
    annotatedImg.src = data.annotated_png;

    enhancedImg.style.display = "block";
    annotatedImg.style.display = "block";
    enhEmpty.style.display = "none";
    annEmpty.style.display = "none";

    // tags: best conf per class
    classesEl.innerHTML = "";
    const best = {};
    for (const it of (data.items || [])){
      const k = it.class_name;
      best[k] = Math.max(best[k] ?? 0, it.confidence ?? 0);
    }
    const keys = Object.keys(best).sort((a,b) => best[b]-best[a]);
    if (keys.length === 0){
      const el = document.createElement("div");
      el.className = "tag";
      el.innerHTML = `<b>No detections</b>`;
      classesEl.appendChild(el);
    } else {
      keys.forEach(k => addTag(k, best[k]));
    }

    setStatus("ok", "Done");
  } catch(err){
    console.error(err);
    setStatus("idle", "Network / runtime error");
  } finally {
    setDisabled(false, false);
  }
});

clearBtn.addEventListener("click", () => {
  fileInput.value = "";
  resetUI();
});

/* ======= overlay zoom ======= */
function openOverlay(title, src){
  overlayTitle.textContent = title;
  overlayImg.src = src;
  overlay.classList.add("open");
}

function closeOverlay(){
  overlay.classList.remove("open");
  overlayImg.src = "";
}

overlayClose.addEventListener("click", closeOverlay);
overlay.addEventListener("click", (e) => {
  if (e.target === overlay) closeOverlay();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeOverlay();
});

document.querySelectorAll(".viewer").forEach(v => {
  v.addEventListener("click", () => {
    const which = v.getAttribute("data-view");
    if (which === "enhanced" && enhancedImg.src) openOverlay("Enhanced (zoom)", enhancedImg.src);
    if (which === "annotated" && annotatedImg.src) openOverlay("Annotated (zoom)", annotatedImg.src);
  });
});

resetUI();
