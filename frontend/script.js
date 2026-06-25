/* ============================================================
   VIRTUAL CLOSET — script.js
   ============================================================ */

/* ── Constants ── */
const BACKEND = "http://127.0.0.1:8000";

const TOPS    = ["t-shirt","shirt","top","blouse","sweater","sweatshirt","hoodie","long-sleeve","tank","camisole","crop top","polo"];
const BOTTOMS = ["pants","jeans","shorts","skirt","leggings","trousers"];
const LAYERS  = ["jacket","coat","cardigan","vest","blazer","windbreaker"];
const FULL    = ["dress","jumpsuit","romper","overalls"];

/* ── State ── */
let closetItems   = [];
let currentFilter = "all";
let currentView   = "grid";
let editIndex     = -1;
let openModalIdx  = -1;
let outfitHistory = [];
let plannerData   = {};     // { "Mon": [{itemIdx}], ... }

/* ── Boot ── */
document.addEventListener("DOMContentLoaded", () => {
  loadFromStorage();
  initUploadZone();
  initSidebar();
  buildPlanner();
  refreshAll();
  registerServiceWorker();
});

/* ============================================================
   STORAGE
   ============================================================ */
function loadFromStorage() {
  try {
    const raw = localStorage.getItem("vc_closet");
    closetItems = raw ? JSON.parse(raw) : [];
  } catch { closetItems = []; }

  try {
    const raw = localStorage.getItem("vc_planner");
    plannerData = raw ? JSON.parse(raw) : {};
  } catch { plannerData = {}; }

  try {
    const raw = localStorage.getItem("vc_history");
    outfitHistory = raw ? JSON.parse(raw) : [];
  } catch { outfitHistory = []; }
}

function saveCloset() {
  try {
    localStorage.setItem("vc_closet", JSON.stringify(closetItems));
  } catch (e) {
    if (e.name === "QuotaExceededError" || e.code === 22) {
      showToast("⚠️ Storage full — try removing some items or clearing old data.");
      console.error("localStorage quota exceeded:", e);
    } else {
      throw e;
    }
  }
}
function savePlanner() { localStorage.setItem("vc_planner", JSON.stringify(plannerData)); }
function saveHistory() { localStorage.setItem("vc_history", JSON.stringify(outfitHistory)); }

/* ============================================================
   NAVIGATION
   ============================================================ */
function showTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));

  const tab = document.getElementById(`tab-${name}`);
  if (tab) tab.classList.add("active");

  const btn = document.querySelector(`.nav-btn[data-tab="${name}"]`);
  if (btn) btn.classList.add("active");

  closeSidebar();
  if (name === "outfits") renderOutfitHistory();
  if (name === "planner") renderPlanner();
  if (name === "closet")  refreshClosetView();
}

/* wire nav buttons */
document.querySelectorAll(".nav-btn[data-tab]").forEach(btn => {
  btn.addEventListener("click", () => showTab(btn.dataset.tab));
});

/* ============================================================
   SIDEBAR / MOBILE
   ============================================================ */
function initSidebar() {
  document.getElementById("hamburger")?.addEventListener("click", openSidebar);
  document.getElementById("sidebarClose")?.addEventListener("click", closeSidebar);
}
function openSidebar() {
  document.getElementById("sidebar").classList.add("open");
  document.getElementById("overlayBg").classList.add("show");
}
function closeSidebar() {
  document.getElementById("sidebar").classList.remove("open");
  document.getElementById("overlayBg").classList.remove("show");
}

/* ============================================================
   UPLOAD ZONE — drag & drop + file picker (multi-file)
   ============================================================ */

// Pending file queue for batch upload
let pendingFiles = [];

function initUploadZone() {
  const zone  = document.getElementById("uploadZone");
  const input = document.getElementById("upload");
  if (!zone || !input) return;

  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const files = [...(e.dataTransfer?.files || [])].filter(f => f.type.startsWith("image/"));
    if (files.length) setQueueFiles(files);
  });

  input.addEventListener("change", () => {
    const files = [...(input.files || [])].filter(f => f.type.startsWith("image/"));
    if (files.length) setQueueFiles(files);
  });
}

function setQueueFiles(files) {
  if (editIndex >= 0 || files.length === 1) {
    // Single-file mode: show old preview
    pendingFiles = files;
    setPreviewFile(files[0]);
    document.getElementById("uploadQueue").style.display = "none";
  } else {
    // Multi-file mode: show queue
    pendingFiles = files;
    document.getElementById("uploadPreview").style.display = "none";
    document.getElementById("uploadPlaceholder").style.display = "none";
    renderQueue();
    document.getElementById("addBtnText").textContent = `➕ Add ${files.length} Items`;
  }
}

function renderQueue() {
  const wrap = document.getElementById("uploadQueue");
  if (!wrap) return;
  wrap.style.display = "";
  wrap.innerHTML = pendingFiles.map((f, i) => {
    const url = URL.createObjectURL(f);
    return `
      <div class="queue-item" id="qitem-${i}">
        <img class="queue-thumb" src="${url}" alt="${escHtml(f.name)}" />
        <div class="queue-info">
          <div class="queue-name">${escHtml(f.name)}</div>
          <div class="queue-status" id="qstatus-${i}">Ready</div>
        </div>
        <span class="queue-badge pending" id="qbadge-${i}">Pending</span>
      </div>`;
  }).join("");
}

function setQueueItemState(i, state, statusText) {
  const item  = document.getElementById(`qitem-${i}`);
  const badge = document.getElementById(`qbadge-${i}`);
  const status = document.getElementById(`qstatus-${i}`);
  if (!item) return;
  item.className  = `queue-item ${state}`;
  badge.className = `queue-badge ${state}`;
  badge.textContent  = state === "active" ? "Analyzing…" : state === "done" ? "✓ Added" : state === "error" ? "Failed" : "Pending";
  if (status) status.textContent = statusText || "";
}

function setPreviewFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById("previewImg").src = e.target.result;
    document.getElementById("uploadPreview").style.display = "block";
    document.getElementById("uploadPlaceholder").style.display = "none";
  };
  reader.readAsDataURL(file);
}

function clearUpload() {
  document.getElementById("upload").value = "";
  document.getElementById("previewImg").src = "";
  document.getElementById("uploadPreview").style.display = "none";
  document.getElementById("uploadPlaceholder").style.display = "";
  document.getElementById("uploadQueue").style.display = "none";
  document.getElementById("aiStatus").style.display = "none";
  document.getElementById("addBtnText").textContent = "➕ Add to Closet";
  pendingFiles = [];
}

/* ============================================================
   ADD / EDIT ITEM  (single or batch)
   ============================================================ */
async function addItem() {
  // ── Edit mode: always single item ──
  if (editIndex >= 0) {
    await _addSingleItem(pendingFiles[0] || null);
    return;
  }

  // ── Single file: old flow ──
  if (pendingFiles.length <= 1) {
    await _addSingleItem(pendingFiles[0] || null);
    return;
  }

  // ── Batch flow ──
  await _addBatchItems();
}

/* ── Single-item path ─────────────────────────────────────── */
async function _addSingleItem(file) {
  const typeEl  = document.getElementById("type");
  const colorEl = document.getElementById("color");
  const eventEl = document.getElementById("event");
  const patEl   = document.getElementById("pattern");
  const tagsEl  = document.getElementById("tags");
  const addBtn  = document.getElementById("addBtn");
  const btnText = document.getElementById("addBtnText");

  if (!file && editIndex < 0) { showToast("Please select an image first."); return; }

  addBtn.disabled = true;
  btnText.textContent = "Analyzing…";

  const aiStatus = document.getElementById("aiStatus");
  const aiText   = document.getElementById("aiStatusText");
  aiStatus.style.display = "flex";
  aiText.textContent = "Analyzing with AI…";

  let imageData = editIndex >= 0 ? closetItems[editIndex].image : null;
  if (file) {
    const raw = await readFileAsDataURL(file);
    imageData  = await compressImage(raw);
  }

  let type = typeEl.value.trim(), color = colorEl.value.trim(), color_hex = "#cccccc";
  let colorsMeta = [], colorsHexMeta = [];

  if (file) {
    try {
      aiText.textContent = "Running clothing detection…";
      const result = await detectClothingAndColor(file);
      const items  = Array.isArray(result.detected_items) ? result.detected_items : [];
      if (items.length > 0) {
        type      = type  || items[0].label || "unknown";
        color     = color || items[0].color || "unknown";
        color_hex = items[0].color_hex || color_hex;
        // Auto-fill pattern if the vision model returned one and user hasn't set it
        const patEl2 = document.getElementById("pattern");
        if (patEl2 && !patEl2.value && items[0].pattern && items[0].pattern !== "solid") {
          patEl2.value = items[0].pattern.charAt(0).toUpperCase() + items[0].pattern.slice(1);
        }
        // Store multi-color data
        if (items[0].colors)     colorsMeta     = items[0].colors;
        if (items[0].colors_hex) colorsHexMeta  = items[0].colors_hex;
      } else {
        aiText.textContent = "Using colour estimate…";
        const fb = await estimateFromImage(imageData);
        type = type || fb.type; color = color || fb.color; color_hex = fb.color_hex;
      }
      if (!typeEl.value.trim())  typeEl.value  = type;
      if (!colorEl.value.trim()) colorEl.value = color;
      updateColorDot(color_hex);
    } catch (err) {
      console.warn("AI detection failed:", err);
      if (!type || !color) {
        const fb = await estimateFromImage(imageData);
        type = type || fb.type; color = color || fb.color; color_hex = fb.color_hex;
      }
      aiText.textContent = "AI unavailable — using estimate";
    }
  } else {
    color_hex = closetItems[editIndex]?.color_hex || color_hex;
  }

  type  = typeEl.value.trim()  || type;
  color = colorEl.value.trim() || color;

  const item = {
    image: imageData, type, color, color_hex,
    colors:     colorsMeta.length  ? colorsMeta     : [color],
    colors_hex: colorsHexMeta.length ? colorsHexMeta : [color_hex],
    event: eventEl.value, pattern: patEl.value, tags: tagsEl.value.trim(),
    addedAt: editIndex >= 0 ? (closetItems[editIndex].addedAt || Date.now()) : Date.now(),
  };

  if (editIndex >= 0) {
    closetItems[editIndex] = item;
    editIndex = -1;
    showToast("Item updated ✓");
  } else {
    closetItems.push(item);
    showToast("Added to your closet ✓");
  }

  saveCloset(); refreshAll(); resetForm();
  aiStatus.style.display = "none";
  addBtn.disabled = false;
  btnText.textContent = "➕ Add to Closet";
  document.getElementById("cancelEditBtn").style.display = "none";
}

/* ── Batch path ───────────────────────────────────────────── */
async function _addBatchItems() {
  const addBtn    = document.getElementById("addBtn");
  const btnText   = document.getElementById("addBtnText");
  const progWrap  = document.getElementById("batchProgress");
  const progFill  = document.getElementById("batchProgressFill");
  const progLabel = document.getElementById("batchProgressLabel");

  const files  = pendingFiles;
  const total  = files.length;
  let done = 0, errors = 0;

  addBtn.disabled = true;
  progWrap.style.display = "flex";

  const updateProgress = () => {
    const pct = Math.round((done / total) * 100);
    progFill.style.width  = pct + "%";
    progLabel.textContent = `${done} / ${total} added`;
    btnText.textContent   = `Adding… ${done}/${total}`;
  };
  updateProgress();

  // Build a single FormData with all files under the "files" field
  // but process sequentially so the queue UI updates correctly
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    // Small gap between items to avoid rate-limiting (OpenAI RPM limits)
    if (i > 0) await new Promise(r => setTimeout(r, 600));
    setQueueItemState(i, "active", "Analyzing…");

    let type = "", color = "", color_hex = "#cccccc";
    let colors = [], colors_hex = [];
    let imageData;

    try {
      imageData = await readFileAsDataURL(file);
      imageData = await compressImage(imageData);

      const result = await detectClothingAndColor(file);
      const items  = Array.isArray(result.detected_items) ? result.detected_items : [];

      if (items.length > 0) {
        type       = items[0].label || "unknown";
        color      = items[0].color || "unknown";
        color_hex  = items[0].color_hex || color_hex;
        colors     = items[0].colors     || [color];
        colors_hex = items[0].colors_hex || [color_hex];
        const pat  = items[0].pattern || "";
        closetItems.push({
          image: imageData, type, color, color_hex, colors, colors_hex,
          event: "", pattern: (pat && pat !== "solid") ? pat : "", tags: "",
          addedAt: Date.now(),
        });

        setQueueItemState(i, "done", `${type} · ${color}`);
      } else {
        const fb = await estimateFromImage(imageData);
        closetItems.push({
          image: imageData, type: fb.type, color: fb.color, color_hex: fb.color_hex,
          event: "", pattern: "", tags: "", addedAt: Date.now(),
        });
        setQueueItemState(i, "done", `${fb.type} · ${fb.color} (estimated)`);
      }
    } catch (err) {
      console.warn(`Item ${i} failed:`, err);
      errors++;
      // Check if it's a rate limit — pause and retry once
      if (err.message && err.message.includes("rate_limit")) {
        setQueueItemState(i, "active", "Rate limited — waiting 10s…");
        await new Promise(r => setTimeout(r, 10000));
        try {
          const result2 = await detectClothingAndColor(file);
          const items2  = Array.isArray(result2.detected_items) ? result2.detected_items : [];
          if (items2.length > 0) {
            type      = items2[0].label || "unknown";
            color     = items2[0].color || "unknown";
            color_hex = items2[0].color_hex || color_hex;
          }
          closetItems.push({ image: imageData, type, color, color_hex, event: "", pattern: "", tags: "", addedAt: Date.now() });
          setQueueItemState(i, "done", `${type} · ${color}`);
          errors--;
        } catch { setQueueItemState(i, "error", "Rate limited — try again later"); }
      } else
      // Still save with fallback if we at least got the image
      if (imageData) {
        try {
          const fb = await estimateFromImage(imageData);
          closetItems.push({
            image: imageData, type: fb.type, color: fb.color, color_hex: fb.color_hex,
            event: "", pattern: "", tags: "", addedAt: Date.now(),
          });
          setQueueItemState(i, "done", `${fb.type} · ${fb.color} (estimated)`);
          errors--; // recovered
        } catch { setQueueItemState(i, "error", "Could not read image"); }
      } else {
        setQueueItemState(i, "error", err.message || "Failed");
      }
    }

    done++;
    updateProgress();
    saveCloset();
    refreshAll();
  }

  const msg = errors > 0
    ? `Added ${done - errors} items (${errors} failed)`
    : `Added ${done} item${done !== 1 ? "s" : ""} ✓`;
  showToast(msg);

  addBtn.disabled = false;
  btnText.textContent = "➕ Add to Closet";
  progWrap.style.display = "none";
  pendingFiles = [];

  // Leave queue visible so user can see results, clear after delay
  setTimeout(() => {
    document.getElementById("uploadQueue").style.display = "none";
    document.getElementById("uploadPlaceholder").style.display = "";
  }, 3000);
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload  = e => resolve(e.target.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

/* ── Compress an image dataURL to a smaller JPEG dataURL ── */
function compressImage(dataURL, maxDim = 800, quality = 0.75) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      let { naturalWidth: w, naturalHeight: h } = img;

      // Scale down if larger than maxDim on either side
      if (w > maxDim || h > maxDim) {
        if (w >= h) { h = Math.round(h * maxDim / w); w = maxDim; }
        else        { w = Math.round(w * maxDim / h); h = maxDim; }
      }

      const canvas = document.createElement("canvas");
      canvas.width  = w;
      canvas.height = h;
      canvas.getContext("2d").drawImage(img, 0, 0, w, h);
      resolve(canvas.toDataURL("image/jpeg", quality));
    };
    img.onerror = () => resolve(dataURL); // fall back to original on error
    img.src = dataURL;
  });
}

function resetForm() {
  document.getElementById("upload").value   = "";
  document.getElementById("type").value     = "";
  document.getElementById("color").value    = "";
  document.getElementById("event").value    = "";
  document.getElementById("pattern").value  = "";
  document.getElementById("tags").value     = "";
  pendingFiles = [];
  clearUpload();
  updateColorDot("#cccccc");
}

function cancelEdit() {
  editIndex = -1;
  resetForm();
  document.getElementById("cancelEditBtn").style.display = "none";
  document.getElementById("addBtnText").textContent = "➕ Add to Closet";
}

/* colour dot in form updates live */
document.getElementById("color")?.addEventListener("input", e => {
  const hex = namedColorToHex(e.target.value.trim());
  updateColorDot(hex);
});

function updateColorDot(hex) {
  const dot = document.getElementById("colorDot");
  if (dot) dot.style.background = hex || "#ccc";
}

/* ============================================================
   BACKEND AI DETECTION
   ============================================================ */
async function detectClothingAndColor(file) {
  const tryPost = async (fieldName) => {
    const fd = new FormData();
    fd.append(fieldName, file);
    const res = await fetch(`${BACKEND}/detect`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  };
  try { return await tryPost("file"); }
  catch { return await tryPost("image"); }
}

/* ============================================================
   CANVAS COLOUR / TYPE FALLBACK
   ============================================================ */
async function averageColorFromImage(imageData) {
  const img = new Image();
  img.src = imageData;
  await new Promise(r => { img.onload = r; img.onerror = r; });

  const W = Math.min(200, img.naturalWidth  || img.width);
  const H = Math.min(200, img.naturalHeight || img.height);
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, W, H);
  const { data } = ctx.getImageData(0, 0, W, H);

  let r = 0, g = 0, b = 0, n = 0;
  for (let i = 0; i < data.length; i += 4 * 16) { r += data[i]; g += data[i+1]; b += data[i+2]; n++; }
  r = Math.round(r / Math.max(1,n));
  g = Math.round(g / Math.max(1,n));
  b = Math.round(b / Math.max(1,n));

  const hex = "#" + [r,g,b].map(v => v.toString(16).padStart(2,"0")).join("");
  const V = Math.max(r,g,b), S = V === 0 ? 0 : (V - Math.min(r,g,b)) / V;
  let name = "gray";
  if (V < 60)       name = "black";
  else if (S < 0.1) name = V > 200 ? "white" : "gray";
  else if (r>=g && r>=b) name = "red";
  else if (g>=r && g>=b) name = "green";
  else                   name = "blue";

  return { hex, name };
}

async function estimateFromImage(imageData) {
  const img = new Image();
  img.src = imageData;
  await new Promise(r => { img.onload = r; img.onerror = r; });
  const tall = (img.naturalHeight||img.height) / Math.max(1, img.naturalWidth||img.width) > 1.4;
  const avg = await averageColorFromImage(imageData);
  return { type: tall ? "pants" : "t-shirt", color: avg.name, color_hex: avg.hex };
}

/* Map common CSS color names → hex for the live colour dot */
function namedColorToHex(name) {
  if (!name) return "#ccc";
  const el = document.createElement("div");
  el.style.color = name;
  document.body.appendChild(el);
  const c = getComputedStyle(el).color;
  document.body.removeChild(el);
  const m = c.match(/\d+/g);
  if (!m || m.length < 3) return "#ccc";
  return "#" + m.slice(0,3).map(v => (+v).toString(16).padStart(2,"0")).join("");
}

/* ============================================================
   CLOSET — REFRESH / RENDER
   ============================================================ */
function refreshAll() {
  refreshClosetView();
  updateStats();
  updateBadge();
}

function refreshClosetView() {
  filterCloset();
}

function itemMatchesFilter(item) {
  const t = (item.type || "").toLowerCase();
  switch (currentFilter) {
    case "tops":     return TOPS.some(k => t.includes(k));
    case "bottoms":  return BOTTOMS.some(k => t.includes(k));
    case "jackets":  return LAYERS.some(k => t.includes(k));
    case "dresses":  return FULL.some(k => t.includes(k));
    case "other":    return ![...TOPS,...BOTTOMS,...LAYERS,...FULL].some(k => t.includes(k));
    default:         return true;
  }
}

function filterCloset() {
  const query = (document.getElementById("search")?.value || "").toLowerCase();
  const clearBtn = document.getElementById("clearSearch");
  if (clearBtn) clearBtn.style.display = query ? "" : "none";

  const filtered = closetItems.map((item, idx) => ({ item, idx })).filter(({ item }) => {
    if (!itemMatchesFilter(item)) return false;
    if (!query) return true;
    return (item.type||"").toLowerCase().includes(query)
        || (item.color||"").toLowerCase().includes(query)
        || (item.tags||"").toLowerCase().includes(query)
        || (item.event||"").toLowerCase().includes(query)
        || (item.pattern||"").toLowerCase().includes(query);
  });

  const grid  = document.getElementById("closet");
  const empty = document.getElementById("emptyState");
  const sub   = document.getElementById("closetSubtitle");

  grid.className = `closet-grid${currentView === "list" ? " list-view" : ""}`;
  grid.innerHTML = "";

  if (filtered.length === 0) {
    grid.style.display = "none";
    if (empty) empty.style.display = "";
    if (sub)   sub.textContent = closetItems.length === 0
      ? "Your closet is empty — add your first item!"
      : `No items match your search.`;
    return;
  }

  grid.style.display = "";
  if (empty) empty.style.display = "none";
  if (sub)   sub.textContent = `${filtered.length} item${filtered.length !== 1 ? "s" : ""}`;

  filtered.forEach(({ item, idx }) => grid.appendChild(buildItemCard(item, idx)));
}

function buildItemCard(item, idx) {
  const div = document.createElement("div");
  div.className = "item-card";
  div.setAttribute("role", "button");
  div.setAttribute("tabindex", "0");
  div.setAttribute("aria-label", `${item.type}, ${item.color}`);

  const tags = (item.tags || "").split(",").map(t => t.trim()).filter(Boolean);
  const tagHtml = tags.slice(0,3).map(t => `<span class="tag-pill">${escHtml(t)}</span>`).join("");

  div.innerHTML = `
    <div class="item-img-wrap">
      <img src="${item.image}" alt="${escHtml(item.type)}" loading="lazy" />
    </div>
    <div class="item-body">
      <div class="item-type">${escHtml(item.type)}</div>
      <div class="item-color-row">
        ${(item.colors_hex||[item.color_hex||'#ccc']).map(h=>`<span class="color-swatch" style="background:${h}"></span>`).join("")}
        <span class="item-color-name">${escHtml((item.colors||[item.color]).join(" · "))}</span>
      </div>
      ${(item.pattern && item.pattern.toLowerCase() !== "solid") ? `<div class="item-pattern">${escHtml(item.pattern)}</div>` : ""}
      ${tagHtml ? `<div class="item-tags">${tagHtml}</div>` : ""}
    </div>
    <div class="item-footer">
      <button class="item-action-btn" onclick="startEdit(${idx});event.stopPropagation()">✏️ Edit</button>
      <button class="item-action-btn del" onclick="deleteItem(${idx});event.stopPropagation()">🗑</button>
    </div>`;

  div.addEventListener("click", () => openItemModal(idx));
  div.addEventListener("keydown", e => { if (e.key === "Enter") openItemModal(idx); });
  return div;
}

/* ============================================================
   ITEM MODAL
   ============================================================ */
function openItemModal(idx) {
  const item = closetItems[idx];
  if (!item) return;
  openModalIdx = idx;

  document.getElementById("modalImg").src          = item.image;
  document.getElementById("modalImg").alt          = item.type;
  document.getElementById("modalTitle").textContent = item.type;
  document.getElementById("modalColorDot").style.background = item.color_hex || "#ccc";

  const colorLabel = (item.colors && item.colors.length > 1)
    ? item.colors.join(" & ")
    : item.color;
  const details = [
    colorLabel   ? `Color: ${colorLabel}` : null,
    item.event   ? `Occasion: ${item.event}` : null,
    item.pattern ? `Pattern: ${item.pattern}` : null,
  ].filter(Boolean).join(" · ");
  document.getElementById("modalColor").textContent = details;

  const tagsWrap = document.getElementById("modalTags");
  const tags = (item.tags || "").split(",").map(t => t.trim()).filter(Boolean);
  tagsWrap.innerHTML = tags.map(t => `<span class="tag-pill">${escHtml(t)}</span>`).join("");

  document.getElementById("modalEditBtn").onclick   = () => { closeItemModal(); startEdit(idx); };
  document.getElementById("modalDeleteBtn").onclick = () => { closeItemModal(); deleteItem(idx); };

  document.getElementById("itemModal").classList.add("open");
}

function closeItemModal() {
  document.getElementById("itemModal").classList.remove("open");
  openModalIdx = -1;
}

function closeModal(e) {
  if (e.target === document.getElementById("itemModal")) closeItemModal();
}

/* ============================================================
   EDIT / DELETE
   ============================================================ */
function startEdit(idx) {
  const item = closetItems[idx];
  if (!item) return;
  editIndex = idx;

  document.getElementById("type").value    = item.type    || "";
  document.getElementById("color").value   = item.color   || "";
  document.getElementById("event").value   = item.event   || "";
  document.getElementById("pattern").value = item.pattern || "";
  document.getElementById("tags").value    = item.tags    || "";
  updateColorDot(item.color_hex || "#ccc");

  /* Show existing image as preview */
  document.getElementById("previewImg").src = item.image;
  document.getElementById("uploadPreview").style.display  = "block";
  document.getElementById("uploadPlaceholder").style.display = "none";

  document.getElementById("addBtnText").textContent = "💾 Save Changes";
  document.getElementById("cancelEditBtn").style.display = "";

  showTab("add");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function deleteItem(idx) {
  if (!confirm(`Delete this ${closetItems[idx]?.type || "item"}?`)) return;
  closetItems.splice(idx, 1);
  saveCloset();
  refreshAll();
  showToast("Item deleted");
}

function resetCloset() {
  if (!confirm("Clear your entire closet? This cannot be undone.")) return;
  closetItems  = [];
  plannerData  = {};
  outfitHistory = [];
  localStorage.removeItem("vc_closet");
  localStorage.removeItem("vc_planner");
  localStorage.removeItem("vc_history");
  refreshAll();
  renderPlanner();
  renderOutfitHistory();
  showToast("Closet cleared");
}

/* ============================================================
   SEARCH / FILTER / VIEW
   ============================================================ */
function setFilter(name, btn) {
  currentFilter = name;
  document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
  btn.classList.add("active");
  filterCloset();
}

function setView(v) {
  currentView = v;
  document.getElementById("gridViewBtn")?.classList.toggle("active", v === "grid");
  document.getElementById("listViewBtn")?.classList.toggle("active", v === "list");
  filterCloset();
}

function clearSearch() {
  const s = document.getElementById("search");
  if (s) s.value = "";
  filterCloset();
}

/* ============================================================
   STATS & BADGE
   ============================================================ */
function updateStats() {
  const tops    = closetItems.filter(i => TOPS.some(k   => (i.type||"").toLowerCase().includes(k))).length;
  const bottoms = closetItems.filter(i => BOTTOMS.some(k => (i.type||"").toLowerCase().includes(k))).length;
  const other   = closetItems.length - tops - bottoms;

  setText("statTotal",   closetItems.length);
  setText("statTops",    tops);
  setText("statBottoms", bottoms);
  setText("statOther",   Math.max(0, other));
}

function updateBadge() {
  const badge = document.getElementById("closet-count");
  if (badge) {
    badge.textContent = closetItems.length || "";
  }
}

/* ============================================================
   OUTFIT SUGGESTION
   ============================================================ */
function suggestOutfit() {
  const occasion = document.getElementById("outfitOccasion")?.value || "";
  const result   = document.getElementById("outfitResult");

  /* Filter pool by occasion if selected */
  const pool = occasion
    ? closetItems.filter(i => !i.event || i.event === occasion)
    : closetItems;

  const tops    = pool.filter(i => TOPS.some(k    => (i.type||"").toLowerCase().includes(k)));
  const bottoms = pool.filter(i => BOTTOMS.some(k  => (i.type||"").toLowerCase().includes(k)));
  const full    = pool.filter(i => FULL.some(k     => (i.type||"").toLowerCase().includes(k)));
  const layers  = pool.filter(i => LAYERS.some(k   => (i.type||"").toLowerCase().includes(k)));

  /* Build outfit combinations */
  let pieces = [];
  if (full.length > 0 && Math.random() > 0.4) {
    /* one-piece option */
    pieces.push(randFrom(full));
    if (layers.length > 0 && Math.random() > 0.5) pieces.push(randFrom(layers));
  } else if (tops.length > 0 && bottoms.length > 0) {
    pieces.push(randFrom(tops));
    pieces.push(randFrom(bottoms));
    if (layers.length > 0 && Math.random() > 0.6) pieces.push(randFrom(layers));
  } else if (tops.length > 0) {
    pieces.push(randFrom(tops));
  } else if (bottoms.length > 0) {
    pieces.push(randFrom(bottoms));
  }

  if (pieces.length === 0) {
    result.innerHTML = `
      <div class="outfit-placeholder">
        <span>🤷</span>
        <p>Not enough items in your closet yet. Add some tops and bottoms to get started!</p>
      </div>`;
    return;
  }

  /* Render outfit */
  const cardsHtml = pieces.map((item, i) => `
    <div class="outfit-piece">
      <img class="outfit-img" src="${item.image}" alt="${escHtml(item.type)}" />
      <div class="outfit-piece-label">${escHtml(item.type)}</div>
    </div>
    ${i < pieces.length - 1 ? '<div class="outfit-connector">+</div>' : ""}
  `).join("");

  const desc = buildOutfitDesc(pieces);
  result.innerHTML = `
    <div style="width:100%">
      <div class="outfit-cards">${cardsHtml}</div>
      <p class="outfit-desc">${desc}</p>
    </div>`;

  /* Save to history */
  outfitHistory.unshift({ pieces, ts: Date.now() });
  if (outfitHistory.length > 12) outfitHistory.pop();
  saveHistory();
  renderOutfitHistory();
}

function buildOutfitDesc(pieces) {
  if (pieces.length === 1) {
    return `Rock your ${pieces[0].color} ${pieces[0].type}.`;
  }
  if (pieces.length === 2) {
    return `Pair your ${pieces[0].color} ${pieces[0].type} with your ${pieces[1].color} ${pieces[1].type}.`;
  }
  const [a, b, c] = pieces;
  return `Wear your ${a.color} ${a.type} with your ${b.color} ${b.type}, and layer on the ${c.color} ${c.type}.`;
}

function renderOutfitHistory() {
  const wrap = document.getElementById("outfitHistory");
  if (!wrap) return;
  if (outfitHistory.length === 0) { wrap.innerHTML = ""; return; }

  const scrollHtml = outfitHistory.map(entry => {
    const thumbs = entry.pieces.map(p =>
      `<img src="${p.image}" alt="${escHtml(p.type)}" />`
    ).join("");
    return `<div class="history-card">${thumbs}</div>`;
  }).join("");

  wrap.innerHTML = `<h3>Recent Outfits</h3><div class="history-scroll">${scrollHtml}</div>`;
}

/* ============================================================
   PLANNER
   ============================================================ */
const DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];

function buildPlanner() {
  const grid = document.getElementById("plannerGrid");
  if (!grid) return;

  const todayIdx = (new Date().getDay() + 6) % 7; // Mon=0

  grid.innerHTML = DAYS.map((day, i) => `
    <div class="planner-day${i === todayIdx ? " today" : ""}" id="planner-${day}">
      <div class="planner-day-label">${day}${i === todayIdx ? " · Today" : ""}</div>
      <div class="planner-slot${plannerData[day] ? " filled" : ""}"
           id="slot-${day}"
           onclick="openPlannerPicker('${day}')"
           title="Click to assign an outfit">
        ${plannerData[day]
          ? `<img src="${plannerData[day].image}" alt="${escHtml(plannerData[day].type)}" />`
          : "＋ Add outfit"}
      </div>
    </div>
  `).join("");
}

function renderPlanner() { buildPlanner(); }

function openPlannerPicker(day) {
  if (closetItems.length === 0) {
    showToast("Add some items to your closet first.");
    return;
  }

  /* Simple prompt-free picker: show a mini modal with item thumbnails */
  const existing = document.getElementById("plannerPickerModal");
  if (existing) existing.remove();

  const modal = document.createElement("div");
  modal.id = "plannerPickerModal";
  modal.style.cssText = `
    position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:600;
    display:flex;align-items:center;justify-content:center;padding:16px;
  `;

  const thumbs = closetItems.map((item, idx) => `
    <div onclick="assignToPlanner('${day}', ${idx})"
         style="cursor:pointer;border-radius:10px;overflow:hidden;border:2px solid transparent;
                transition:.15s;flex-shrink:0;width:90px;"
         onmouseover="this.style.borderColor='#a855f7'"
         onmouseout="this.style.borderColor='transparent'">
      <img src="${item.image}" style="width:90px;height:110px;object-fit:cover;display:block;" alt="${escHtml(item.type)}" />
      <div style="font-size:.68rem;padding:3px 4px;text-align:center;color:#3f3f46;font-weight:600;">
        ${escHtml(item.type)}
      </div>
    </div>`
  ).join("");

  modal.innerHTML = `
    <div style="background:#fff;border-radius:18px;max-width:520px;width:100%;max-height:80vh;
                overflow:hidden;display:flex;flex-direction:column;box-shadow:0 12px 40px rgba(0,0,0,.15);">
      <div style="padding:18px 20px;border-bottom:1px solid #e4e4e7;display:flex;
                  justify-content:space-between;align-items:center;">
        <strong style="font-size:.95rem;">Pick outfit for ${day}</strong>
        <button onclick="document.getElementById('plannerPickerModal').remove()"
                style="background:none;border:none;font-size:1.1rem;cursor:pointer;color:#71717a;">✕</button>
      </div>
      <div style="padding:16px;overflow-y:auto;display:flex;flex-wrap:wrap;gap:10px;justify-content:center;">
        ${thumbs}
      </div>
      ${plannerData[day] ? `<div style="padding:12px 20px;border-top:1px solid #e4e4e7;text-align:center;">
        <button onclick="clearPlannerDay('${day}')"
          style="background:#fff1f2;color:#f43f5e;border:1.5px solid #fecdd3;border-radius:8px;
                 padding:7px 16px;font-size:.8rem;font-weight:600;cursor:pointer;">
          🗑 Clear ${day}
        </button>
      </div>` : ""}
    </div>`;

  modal.addEventListener("click", e => { if (e.target === modal) modal.remove(); });
  document.body.appendChild(modal);
}

function assignToPlanner(day, idx) {
  plannerData[day] = closetItems[idx];
  savePlanner();
  document.getElementById("plannerPickerModal")?.remove();
  buildPlanner();
  showToast(`${closetItems[idx]?.type} planned for ${day} ✓`);
}

function clearPlannerDay(day) {
  delete plannerData[day];
  savePlanner();
  document.getElementById("plannerPickerModal")?.remove();
  buildPlanner();
}

/* ============================================================
   TOAST
   ============================================================ */
let _toastTimer = null;
function showToast(msg) {
  const t = document.getElementById("toast");
  if (!t) return;
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.remove("show"), 2800);
}

/* ============================================================
   UTILITIES
   ============================================================ */
function randFrom(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function escHtml(str) {
  return String(str ?? "")
    .replace(/&/g,"&amp;")
    .replace(/</g,"&lt;")
    .replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;");
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

/* ============================================================
   SERVICE WORKER
   ============================================================ */
function registerServiceWorker() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("service-worker.js")
      .then(() => console.log("[SW] registered"))
      .catch(err => console.warn("[SW] registration failed:", err));
  }
}

/* ============================================================
   KEYBOARD SHORTCUTS
   ============================================================ */
document.addEventListener("keydown", e => {
  if (e.key === "Escape") {
    closeItemModal();
    document.getElementById("plannerPickerModal")?.remove();
  }
});

/* ============================================================
   STYLE DNA
   ============================================================ */
function updateStyleDna() {
  const dna = document.getElementById("styleDna");
  if (!dna) return;

  if (closetItems.length < 5) {
    dna.style.display = "none";
    return;
  }
  dna.style.display = "";

  // ── Color frequency map ──
  const colorCounts = {};
  const colorHexMap = {};
  closetItems.forEach(item => {
    const c = (item.color || "unknown").toLowerCase().trim();
    colorCounts[c] = (colorCounts[c] || 0) + 1;
    if (!colorHexMap[c]) colorHexMap[c] = item.color_hex || "#ccc";
  });

  const topColors = Object.entries(colorCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);

  const swatchWrap = document.getElementById("dnaSwatches");
  if (swatchWrap) {
    swatchWrap.innerHTML = topColors.map(([name]) => `
      <div class="dna-swatch-wrap">
        <div class="dna-swatch" style="background:${colorHexMap[name]}"></div>
        <span class="dna-swatch-name">${escHtml(name)}</span>
      </div>`).join("");
  }

  // ── Wardrobe breakdown ──
  const total   = closetItems.length;
  const cats = [
    { label: "Tops",    items: closetItems.filter(i => TOPS.some(k    => (i.type||"").toLowerCase().includes(k))) },
    { label: "Bottoms", items: closetItems.filter(i => BOTTOMS.some(k => (i.type||"").toLowerCase().includes(k))) },
    { label: "Layers",  items: closetItems.filter(i => LAYERS.some(k  => (i.type||"").toLowerCase().includes(k))) },
    { label: "Dresses", items: closetItems.filter(i => FULL.some(k    => (i.type||"").toLowerCase().includes(k))) },
  ];

  const breakdown = document.getElementById("dnaBreakdown");
  if (breakdown) {
    breakdown.innerHTML = cats.map(({ label, items }) => {
      const pct = total > 0 ? Math.round((items.length / total) * 100) : 0;
      return `
        <div class="dna-bar-row">
          <span class="dna-bar-label">${label}</span>
          <div class="dna-bar-track">
            <div class="dna-bar-fill" style="width:${pct}%"></div>
          </div>
          <span class="dna-bar-pct">${pct}%</span>
        </div>`;
    }).join("");
  }

  // ── Personality label ──
  const personality = derivePersonality(colorCounts, cats);
  const label = document.getElementById("dnaPersonality");
  if (label) label.textContent = personality;
}

function derivePersonality(colorCounts, cats) {
  const neutrals  = ["black","white","gray","grey","beige","cream","navy","tan","brown"];
  const bolds     = ["red","orange","yellow","pink","purple","green","cobalt","lime","magenta"];
  const total     = closetItems.length;

  const neutralPct = Object.entries(colorCounts)
    .filter(([c]) => neutrals.some(n => c.includes(n)))
    .reduce((s, [, n]) => s + n, 0) / Math.max(1, total);

  const boldPct = Object.entries(colorCounts)
    .filter(([c]) => bolds.some(b => c.includes(b)))
    .reduce((s, [, n]) => s + n, 0) / Math.max(1, total);

  const layerCount  = cats.find(c => c.label === "Layers")?.items.length  || 0;
  const dressCount  = cats.find(c => c.label === "Dresses")?.items.length || 0;
  const bottomCount = cats.find(c => c.label === "Bottoms")?.items.length || 0;

  if (neutralPct > 0.65)                          return "Minimalist";
  if (boldPct > 0.5)                              return "Maximalist";
  if (layerCount / Math.max(1, total) > 0.3)      return "Streetwear";
  if (dressCount / Math.max(1, total) > 0.3)      return "Feminine";
  if (bottomCount / Math.max(1, total) > 0.5)     return "Classic";
  if (boldPct > 0.25 && neutralPct > 0.3)         return "Eclectic";
  return "Versatile";
}

/* ============================================================
   LIVE WEATHER  (Open-Meteo, free, no key needed)
   ============================================================ */
let liveWeather = null;

const WMO_CONDITION = c =>
  c === 0 ? "Clear sky" : c <= 3 ? "Partly cloudy" : c <= 48 ? "Foggy" :
  c <= 67 ? "Rainy" : c <= 77 ? "Snowy" : c <= 82 ? "Showers" : "Thunderstorm";

const WMO_ICON = c =>
  c === 0 ? "☀️" : c <= 3 ? "⛅" : c <= 48 ? "🌫️" :
  c <= 67 ? "🌧️" : c <= 77 ? "❄️" : c <= 82 ? "🌦️" : "⛈️";

function getTimeOfDay() {
  const h = new Date().getHours();
  if (h >= 5  && h < 12) return "morning";
  if (h >= 12 && h < 17) return "afternoon";
  if (h >= 17 && h < 21) return "evening";
  return "night";
}

function updateWeatherBar() {
  const bar  = document.getElementById("weatherBar");
  const text = document.getElementById("weatherText");
  if (!bar || !text) return;
  const tod = getTimeOfDay();
  if (liveWeather) {
    text.textContent = `${liveWeather.icon} ${liveWeather.temp_f}°F · ${liveWeather.condition} · ${tod.charAt(0).toUpperCase() + tod.slice(1)}`;
    bar.style.display = "flex";
  } else {
    text.textContent = `🕐 ${tod.charAt(0).toUpperCase() + tod.slice(1)} · weather unavailable`;
    bar.style.display = "flex";
  }
}

async function fetchWeather(forceRefresh = false) {
  if (liveWeather && !forceRefresh) { updateWeatherBar(); return; }
  if (!navigator.geolocation) { updateWeatherBar(); return; }

  const text = document.getElementById("weatherText");
  if (text) text.textContent = "📍 Getting location…";
  const bar = document.getElementById("weatherBar");
  if (bar) bar.style.display = "flex";

  navigator.geolocation.getCurrentPosition(async pos => {
    try {
      const { latitude: lat, longitude: lon } = pos.coords;
      const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
        `&current=temperature_2m,weather_code&temperature_unit=fahrenheit&forecast_days=1`;
      const res  = await fetch(url);
      const data = await res.json();
      const cur  = data.current;
      liveWeather = {
        temp_f:    Math.round(cur.temperature_2m),
        condition: WMO_CONDITION(cur.weather_code),
        icon:      WMO_ICON(cur.weather_code),
        code:      cur.weather_code,
      };
    } catch { liveWeather = null; }
    updateWeatherBar();
  }, () => { liveWeather = null; updateWeatherBar(); });
}

/* ============================================================
   AI OUTFIT SUGGESTION  (calls POST /suggest)
   ============================================================ */
async function askAiOutfit() {
  if (closetItems.length === 0) {
    showToast("Add some items to your closet first.");
    return;
  }

  const btn       = document.getElementById("askAiBtn");
  const thinking  = document.getElementById("aiThinking");
  const result    = document.getElementById("outfitResult");
  const occasion  = document.getElementById("outfitOccasion")?.value || "";

  btn.disabled = true;
  thinking.style.display = "flex";
  result.innerHTML = "";

  // Build a lightweight payload — no images, just attributes
  const itemPayload = closetItems.map((item, idx) => ({
    id:      idx,
    type:    item.type    || "unknown",
    color:   item.color   || "unknown",
    pattern: item.pattern || "",
    event:   item.event   || "",
    tags:    item.tags    || "",
  }));

  try {
    const res = await fetch(`${BACKEND}/suggest`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        items:       itemPayload,
        occasion,
        time_of_day: getTimeOfDay(),
        weather:     liveWeather ? `${liveWeather.temp_f}°F, ${liveWeather.condition}` : "",
        temp_f:      liveWeather?.temp_f ?? null,
      }),
    });

    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();

    // data.outfit = [{ id, reason_for_pick }, ...]  +  data.reason (overall)
    renderAiOutfit(data);

    // Save to history
    const pieces = (data.outfit || [])
      .map(o => closetItems[o.id])
      .filter(Boolean);
    if (pieces.length > 0) {
      outfitHistory.unshift({ pieces, ts: Date.now(), ai: true });
      if (outfitHistory.length > 12) outfitHistory.pop();
      saveHistory();
      renderOutfitHistory();
    }

  } catch (err) {
    console.error("AI suggest failed:", err);
    result.innerHTML = `
      <div class="ai-error">
        <span class="ai-error-icon">⚠️</span>
        <div>
          <strong>AI suggestion unavailable</strong><br/>
          ${err.message.includes("401") ? "Check your OpenAI API key in the backend." :
            err.message.includes("fetch") ? "Make sure the backend is running at " + BACKEND + "." :
            "Something went wrong — try again in a moment."}
        </div>
      </div>`;
  } finally {
    btn.disabled = false;
    thinking.style.display = "none";
  }
}

function renderAiOutfit(data) {
  const result  = document.getElementById("outfitResult");
  const outfit  = data.outfit || [];
  const reason  = data.reason || "";

  if (outfit.length === 0) {
    result.innerHTML = `<div class="outfit-placeholder"><span>🤷</span><p>AI couldn't build an outfit from your current closet.</p></div>`;
    return;
  }

  const cardsHtml = outfit.map((o, i) => {
    const item = closetItems[o.id];
    if (!item) return "";
    return `
      <div class="outfit-piece">
        <img class="outfit-img" src="${item.image}" alt="${escHtml(item.type)}" />
        <div class="outfit-piece-label">${escHtml(item.type)}</div>
      </div>
      ${i < outfit.length - 1 ? '<div class="outfit-connector">+</div>' : ""}`;
  }).join("");

  result.innerHTML = `
    <div class="ai-outfit-card">
      <div class="ai-badge">🤖 AI Pick</div>
      ${reason ? `<div class="ai-outfit-reason">${escHtml(reason)}</div>` : ""}
      <div class="outfit-cards">${cardsHtml}</div>
    </div>`;
}

/* ============================================================
   MATCH CHECK  (calls POST /match)
   ============================================================ */
let matchSelection = { A: null, B: null };

function openMatchPicker(slot) {
  if (closetItems.length === 0) {
    showToast("Add some items to your closet first.");
    return;
  }

  const existing = document.getElementById("matchPickerModal");
  if (existing) existing.remove();

  const modal = document.createElement("div");
  modal.id = "matchPickerModal";
  modal.style.cssText = `
    position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:600;
    display:flex;align-items:center;justify-content:center;padding:16px;`;

  const thumbs = closetItems.map((item, idx) => {
    const selected = matchSelection[slot]?.idx === idx;
    return `
      <div onclick="selectMatchItem('${slot}',${idx})"
           style="cursor:pointer;border-radius:10px;overflow:hidden;
                  border:2px solid ${selected ? "#a855f7" : "transparent"};
                  transition:.15s;flex-shrink:0;width:90px;"
           onmouseover="this.style.borderColor='#a855f7'"
           onmouseout="this.style.borderColor='${selected ? "#a855f7" : "transparent"}'">
        <img src="${item.image}" style="width:90px;height:110px;object-fit:cover;display:block;" alt="${escHtml(item.type)}" />
        <div style="font-size:.68rem;padding:3px 4px;text-align:center;color:#3f3f46;font-weight:600;">
          ${escHtml(item.type)}
        </div>
      </div>`;
  }).join("");

  modal.innerHTML = `
    <div style="background:#fff;border-radius:18px;max-width:520px;width:100%;
                max-height:80vh;overflow:hidden;display:flex;flex-direction:column;
                box-shadow:0 12px 40px rgba(0,0,0,.15);">
      <div style="padding:18px 20px;border-bottom:1px solid #e4e4e7;
                  display:flex;justify-content:space-between;align-items:center;">
        <strong style="font-size:.95rem;">Pick item ${slot}</strong>
        <button onclick="document.getElementById('matchPickerModal').remove()"
                style="background:none;border:none;font-size:1.1rem;cursor:pointer;color:#71717a;">✕</button>
      </div>
      <div style="padding:16px;overflow-y:auto;display:flex;flex-wrap:wrap;
                  gap:10px;justify-content:center;">
        ${thumbs}
      </div>
    </div>`;

  modal.addEventListener("click", e => { if (e.target === modal) modal.remove(); });
  document.body.appendChild(modal);
}

function selectMatchItem(slot, idx) {
  matchSelection[slot] = { idx, item: closetItems[idx] };
  document.getElementById("matchPickerModal")?.remove();

  // Update picker UI
  const img   = document.getElementById(`matchImg${slot}`);
  const label = document.getElementById(`matchLabel${slot}`);
  const empty  = document.getElementById(`matchEmpty${slot}`);
  const filled = document.getElementById(`matchFilled${slot}`);
  const picker = document.getElementById(`matchPicker${slot}`);

  img.src         = closetItems[idx].image;
  label.textContent = closetItems[idx].type;
  empty.style.display  = "none";
  filled.style.display = "";
  picker.classList.add("selected");

  // Enable check button when both slots filled
  const bothFilled = matchSelection.A !== null && matchSelection.B !== null;
  const checkBtn = document.getElementById("checkMatchBtn");
  if (checkBtn) checkBtn.disabled = !bothFilled;

  // Reset any previous result
  document.getElementById("matchResult").style.display = "none";
}

function clearMatchItem(slot) {
  matchSelection[slot] = null;
  document.getElementById(`matchEmpty${slot}`).style.display  = "";
  document.getElementById(`matchFilled${slot}`).style.display = "none";
  document.getElementById(`matchPicker${slot}`).classList.remove("selected");
  document.getElementById("checkMatchBtn").disabled = true;
  document.getElementById("matchResult").style.display = "none";
}

async function checkMatch() {
  const a = matchSelection.A?.item;
  const b = matchSelection.B?.item;
  if (!a || !b) return;

  const btn      = document.getElementById("checkMatchBtn");
  const thinking = document.getElementById("matchThinking");
  const resultWrap = document.getElementById("matchResult");

  btn.disabled = true;
  thinking.style.display = "flex";
  resultWrap.style.display = "none";

  try {
    const res = await fetch(`${BACKEND}/match`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        item_a: { type: a.type, color: a.color, pattern: a.pattern, event: a.event },
        item_b: { type: b.type, color: b.color, pattern: b.pattern, event: b.event },
      }),
    });

    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();
    renderMatchResult(data);

  } catch (err) {
    console.error("Match check failed:", err);
    resultWrap.style.display = "";
    document.getElementById("matchResultCard").innerHTML = `
      <div class="ai-error">
        <span class="ai-error-icon">⚠️</span>
        <div>
          <strong>Match check unavailable</strong><br/>
          ${err.message.includes("fetch") ? "Make sure the backend is running." : err.message}
        </div>
      </div>`;
  } finally {
    btn.disabled = false;
    thinking.style.display = "none";
  }
}

function renderMatchResult(data) {
  // data = { rating: "Great Match"|"Works"|"Clash", explanation: "...", tips: ["..."] }
  const ratingMap = {
    "great match": { cls: "great", emoji: "✅ Great Match" },
    "works":       { cls: "works", emoji: "🟡 Works"       },
    "clash":       { cls: "clash", emoji: "❌ Clash"        },
  };

  const raw   = (data.rating || "works").toLowerCase().trim();
  const info  = ratingMap[raw] || ratingMap["works"];
  const tips  = Array.isArray(data.tips) ? data.tips : [];

  document.getElementById("matchRating").className      = `match-rating ${info.cls}`;
  document.getElementById("matchRating").textContent    = info.emoji;
  document.getElementById("matchExplanation").textContent = data.explanation || "";
  document.getElementById("matchTips").innerHTML = tips.map(tip => `
    <div class="match-tip">
      <span class="match-tip-icon">💡</span>
      <span>${escHtml(tip)}</span>
    </div>`).join("");

  document.getElementById("matchResult").style.display = "";
}

/* ============================================================
   HOOK NEW FEATURES INTO refreshAll + showTab
   ============================================================ */

// Patch refreshAll to also update Style DNA
const _origRefreshAll = refreshAll;
refreshAll = function () {
  _origRefreshAll();
  updateStyleDna();
};

// Patch showTab to handle match tab + weather
const _origShowTab = showTab;
showTab = function (name) {
  _origShowTab(name);
  if (name === "outfits") fetchWeather();
};
