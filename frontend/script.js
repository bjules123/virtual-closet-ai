/* ============================================================
   VIRTUAL CLOSET — script.js
   ============================================================ */

/* ── Constants ── */
const BACKEND = "https://virtual-closet-gkz7.onrender.com";

const TOPS    = ["t-shirt","shirt","top","blouse","sweater","sweatshirt","hoodie","long-sleeve","tank","camisole","crop top","polo"];
const BOTTOMS = ["pants","jeans","shorts","skirt","leggings","trousers"];
const LAYERS  = ["jacket","coat","cardigan","vest","blazer","windbreaker"];
const FULL    = ["dress","jumpsuit","romper","overalls"];
const SHOES   = ["sneaker","flip flop","sandal","ankle boot","boot","heel","loafer","flat","slide","mule","oxford","pump","clog","platform shoe","shoe","footwear"];
const ACCESSORIES = ["belt","sun hat","bucket hat","cowboy hat","fedora","beret","flat cap","beanie","baseball cap","hat","cap","scarf","clutch","crossbody bag","shoulder bag","fanny pack","duffle bag","mini bag","tote bag","backpack","handbag","bag","purse","sunglasses","glasses","watch","necklace","earring","bracelet","ring","tie","glove","wallet","accessory"];

/* ── State ── */
let closetItems   = [];
let inspoItems    = [];
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
  initInspoUpload();
  initShopOutfits();
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
    // Migrate old format: single item object → array
    Object.keys(plannerData).forEach(day => {
      if (plannerData[day] && !Array.isArray(plannerData[day])) {
        plannerData[day] = [plannerData[day]];
      }
    });
  } catch { plannerData = {}; }

  try {
    const raw = localStorage.getItem("vc_history");
    outfitHistory = raw ? JSON.parse(raw) : [];
  } catch { outfitHistory = []; }

  try {
    const raw = localStorage.getItem("vc_inspo");
    inspoItems = raw ? JSON.parse(raw) : [];
  } catch { inspoItems = []; }
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
    case "tops":        return TOPS.some(k => t.includes(k));
    case "bottoms":     return BOTTOMS.some(k => t.includes(k));
    case "jackets":     return LAYERS.some(k => t.includes(k));
    case "dresses":     return FULL.some(k => t.includes(k));
    case "shoes":       return SHOES.some(k => t.includes(k));
    case "accessories": return ACCESSORIES.some(k => t.includes(k));
    case "other":       return ![...TOPS,...BOTTOMS,...LAYERS,...FULL,...SHOES,...ACCESSORIES].some(k => t.includes(k));
    default:            return true;
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
  inspoItems   = [];
  localStorage.removeItem("vc_closet");
  localStorage.removeItem("vc_planner");
  localStorage.removeItem("vc_history");
  localStorage.removeItem("vc_inspo");
  refreshAll();
  renderPlanner();
  renderOutfitHistory();
  renderInspoBoard();
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

  grid.innerHTML = DAYS.map((day, i) => {
    const items = plannerData[day] || [];
    const hasItems = items.length > 0;
    const thumbsHtml = items.map((item, ii) => `
      <div class="pd-thumb" title="${escHtml(item.type)} · ${escHtml(item.color)}">
        <img src="${item.image}" alt="${escHtml(item.type)}" />
        <button class="pd-remove-btn"
                onclick="removePlannerItem('${day}',${ii});event.stopPropagation()"
                title="Remove">✕</button>
      </div>`).join("");

    return `
      <div class="planner-day${i === todayIdx ? " today" : ""}" id="planner-${day}">
        <div class="planner-day-label">${day}${i === todayIdx ? " · Today" : ""}</div>
        ${hasItems ? `<div class="pd-thumbs">${thumbsHtml}</div>` : ""}
        ${hasItems
          ? `<div class="pd-foot">
               <button class="pd-add-btn-sm" onclick="openPlannerPicker('${day}')">＋ Add</button>
               <button class="pd-clear-btn" onclick="clearPlannerDay('${day}')">Clear</button>
             </div>`
          : `<button class="pd-add-btn" onclick="openPlannerPicker('${day}')">＋ Add outfit</button>`}
      </div>`;
  }).join("");
}

function renderPlanner() { buildPlanner(); }

function _ppThumbs(day) {
  const current = plannerData[day] || [];
  const currentImages = new Set(current.map(p => p.image));
  return closetItems.map((item, idx) => {
    const added = currentImages.has(item.image);
    return `
      <div class="pp-item${added ? " pp-added" : ""}"
           onclick="assignToPlanner('${day}',${idx})"
           title="${added ? "Remove from outfit" : "Add to outfit"}">
        <img src="${item.image}" alt="${escHtml(item.type)}" />
        ${added ? `<div class="pp-check">✓</div>` : ""}
        <div class="pp-label">${escHtml(item.type)}</div>
      </div>`;
  }).join("");
}

function _ppFooter(day) {
  const count = (plannerData[day] || []).length;
  return count > 0
    ? `<div style="padding:12px 20px;border-top:1px solid #e4e4e7;display:flex;align-items:center;justify-content:space-between;gap:12px;">
         <span style="font-size:.8rem;color:#71717a;">${count} item${count !== 1 ? "s" : ""} in outfit</span>
         <button onclick="clearPlannerDay('${day}')"
           style="background:#fff1f2;color:#f43f5e;border:1.5px solid #fecdd3;border-radius:8px;
                  padding:6px 14px;font-size:.78rem;font-weight:600;cursor:pointer;">
           Clear day
         </button>
       </div>`
    : "";
}

function openPlannerPicker(day) {
  if (closetItems.length === 0) { showToast("Add some items to your closet first."); return; }
  const existing = document.getElementById("plannerPickerModal");
  if (existing) existing.remove();

  const modal = document.createElement("div");
  modal.id = "plannerPickerModal";
  modal.style.cssText = `position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:600;
    display:flex;align-items:center;justify-content:center;padding:16px;`;

  modal.innerHTML = `
    <div style="background:#fff;border-radius:18px;max-width:560px;width:100%;max-height:82vh;
                overflow:hidden;display:flex;flex-direction:column;box-shadow:0 12px 40px rgba(0,0,0,.15);">
      <div style="padding:18px 20px;border-bottom:1px solid #e4e4e7;display:flex;
                  justify-content:space-between;align-items:center;">
        <div>
          <strong style="font-size:.95rem;">Build outfit for ${day}</strong>
          <div style="font-size:.75rem;color:#71717a;margin-top:2px;">Tap to add or remove each piece</div>
        </div>
        <button onclick="document.getElementById('plannerPickerModal').remove()"
                style="background:none;border:none;font-size:1.1rem;cursor:pointer;color:#71717a;padding:4px;">✕</button>
      </div>
      <div id="ppContent" style="padding:16px;overflow-y:auto;display:flex;flex-wrap:wrap;gap:10px;justify-content:center;">
        ${_ppThumbs(day)}
      </div>
      <div id="ppFooter">${_ppFooter(day)}</div>
    </div>`;

  modal.addEventListener("click", e => { if (e.target === modal) modal.remove(); });
  document.body.appendChild(modal);
}

function assignToPlanner(day, closetIdx) {
  if (!plannerData[day]) plannerData[day] = [];
  const item = closetItems[closetIdx];
  const existingIdx = plannerData[day].findIndex(p => p.image === item.image);
  if (existingIdx !== -1) {
    plannerData[day].splice(existingIdx, 1);
  } else {
    plannerData[day].push({ ...item });
  }
  if (plannerData[day].length === 0) delete plannerData[day];
  savePlanner();
  buildPlanner();
  // Update modal in place
  const ppContent = document.getElementById("ppContent");
  if (ppContent) ppContent.innerHTML = _ppThumbs(day);
  const ppFooter  = document.getElementById("ppFooter");
  if (ppFooter)  ppFooter.innerHTML  = _ppFooter(day);
}

function removePlannerItem(day, thumbIdx) {
  if (!plannerData[day]) return;
  plannerData[day].splice(thumbIdx, 1);
  if (plannerData[day].length === 0) delete plannerData[day];
  savePlanner();
  buildPlanner();
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
   STYLE PROFILE
   ============================================================ */
function updateStyleDna() {
  const dna = document.getElementById("styleDna");
  if (!dna) return;
  if (closetItems.length < 5) { dna.style.display = "none"; return; }
  dna.style.display = "";

  const total = closetItems.length;

  // ── Color data ──────────────────────────────────────────────
  const colorCounts = {};
  const colorHexMap = {};
  closetItems.forEach(item => {
    const c = (item.color || "unknown").toLowerCase().trim();
    colorCounts[c] = (colorCounts[c] || 0) + 1;
    if (!colorHexMap[c]) colorHexMap[c] = item.color_hex || "#ccc";
  });
  const topColors = Object.entries(colorCounts).sort((a, b) => b[1] - a[1]).slice(0, 7);

  // ── Category data ────────────────────────────────────────────
  const cats = [
    { label: "tops",        items: closetItems.filter(i => TOPS.some(k        => (i.type||"").toLowerCase().includes(k))) },
    { label: "bottoms",     items: closetItems.filter(i => BOTTOMS.some(k     => (i.type||"").toLowerCase().includes(k))) },
    { label: "layers",      items: closetItems.filter(i => LAYERS.some(k      => (i.type||"").toLowerCase().includes(k))) },
    { label: "dresses",     items: closetItems.filter(i => FULL.some(k        => (i.type||"").toLowerCase().includes(k))) },
    { label: "shoes",       items: closetItems.filter(i => SHOES.some(k       => (i.type||"").toLowerCase().includes(k))) },
    { label: "accessories", items: closetItems.filter(i => ACCESSORIES.some(k => (i.type||"").toLowerCase().includes(k))) },
  ];
  const biggestCat = [...cats].sort((a, b) => b.items.length - a.items.length)[0];

  // ── Personality signal ───────────────────────────────────────
  const NEUTRALS = ["black","white","gray","grey","beige","cream","navy","tan","brown","ivory","stone","sand","camel","charcoal","off-white"];
  const BOLDS    = ["red","orange","yellow","pink","purple","cobalt","lime","magenta","teal","coral","turquoise","fuchsia","electric","neon"];
  const neutralPct = Object.entries(colorCounts)
    .filter(([c]) => NEUTRALS.some(n => c.includes(n)))
    .reduce((s, [, n]) => s + n, 0) / Math.max(1, total);
  const boldPct = Object.entries(colorCounts)
    .filter(([c]) => BOLDS.some(b => c.includes(b)))
    .reduce((s, [, n]) => s + n, 0) / Math.max(1, total);

  let personality = "versatile";
  if (neutralPct > 0.65) personality = "minimalist";
  else if (boldPct > 0.5) personality = "maximalist";
  else if ((cats.find(c => c.label === "layers")?.items.length || 0) / total > 0.3) personality = "streetwear";
  else if ((cats.find(c => c.label === "dresses")?.items.length || 0) / total > 0.3) personality = "romantic";
  else if (boldPct > 0.25 && neutralPct > 0.3) personality = "eclectic";

  // ── Inspo data ───────────────────────────────────────────────
  let inspoAesthetic = "";
  let sigColorNames  = [];
  let wishColorNames = [];
  const inspoColorHexMap = {};

  if (inspoItems.length > 0) {
    const vibeCounts = {};
    inspoItems.forEach(i => { if (i.vibe) vibeCounts[i.vibe] = (vibeCounts[i.vibe] || 0) + 1; });
    inspoAesthetic = Object.entries(vibeCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "";

    const inspoColorSet = new Set();
    inspoItems.forEach(item => {
      (item.colors || []).forEach((c, ci) => {
        const key = c.toLowerCase().trim();
        inspoColorSet.add(key);
        if (!inspoColorHexMap[key] && item.colors_hex?.[ci]) inspoColorHexMap[key] = item.colors_hex[ci];
      });
    });
    const closetColorSet = new Set(closetItems.map(i => (i.color || "").toLowerCase().trim()));
    sigColorNames  = [...inspoColorSet].filter(c => closetColorSet.has(c)).slice(0, 3);
    wishColorNames = [...inspoColorSet].filter(c => !closetColorSet.has(c)).slice(0, 3);
  }

  // ── Deterministic phrase index (stable for a given closet size) ──
  const phraseIdx = total % 3;

  // ── Render headline ──────────────────────────────────────────
  const headlineEl = document.getElementById("sdnaHeadline");
  if (headlineEl) headlineEl.innerHTML = _sdnaHeadline(personality, inspoAesthetic, phraseIdx);

  // ── Render paragraph ─────────────────────────────────────────
  const paraEl = document.getElementById("sdnaPara");
  if (paraEl) paraEl.textContent = _sdnaPara(personality, topColors, cats, total, neutralPct, phraseIdx);

  // ── Render color strip ───────────────────────────────────────
  const colorsEl = document.getElementById("sdnaColors");
  if (colorsEl) {
    colorsEl.innerHTML = topColors.map(([name]) =>
      `<span class="sdna-swatch" style="background:${colorHexMap[name] || '#ccc'}" title="${escHtml(name)}"></span>`
    ).join("");
  }

  // ── Render inspo line ────────────────────────────────────────
  const inspoLineEl = document.getElementById("sdnaInspoLine");
  if (inspoLineEl) {
    if (inspoItems.length > 0 && inspoAesthetic) {
      inspoLineEl.textContent = _sdnaInspoLine(inspoAesthetic, sigColorNames, wishColorNames);
      inspoLineEl.style.display = "";
    } else {
      inspoLineEl.style.display = "none";
    }
  }
}

function _sdnaHeadline(personality, inspoAesthetic, idx) {
  const byAesthetic = {
    "Clean Girl":      ["Clean lines, effortless execution.", "You make getting dressed look easy.", "Clean Girl through and through."],
    "Quiet Luxury":    ["Quiet luxury isn't a trend for you — it's the baseline.", "You spend where it counts and it shows.", "Understated. Intentional. Always right."],
    "Streetwear":      ["Your style has an edge and knows how to use it.", "Casual but deliberate. Effortless but considered.", "You dress for yourself. It works."],
    "Euro Summer":     ["Your wardrobe has a passport.", "Sun-washed, easy, and always put together.", "Effortless in the way that actually takes thought."],
    "Boho":            ["Free-spirited with a serious eye for texture.", "Your style is layered — literally and figuratively.", "You wear what feels right. It always does."],
    "Dark Academia":   ["You dress like you have a library card and actually use it.", "Thoughtful, intentional, deeply stylish.", "Dark Academia as a lifestyle, not just an aesthetic."],
    "Preppy":          ["Classic with a clear point of view.", "Polished, put-together, and never overdone.", "You dress like you mean it."],
  };
  const byPersonality = {
    minimalist:  ["You dress in whispers, not shouts.", "Less, but better — that's your whole thing.", "Your wardrobe is edited to perfection."],
    maximalist:  ["You dress to be noticed. And you are.", "Bold palettes, confident choices. Your closet has a personality.", "Your wardrobe doesn't do quiet — and that's the point."],
    streetwear:  ["Casual but considered. Your style walks the line perfectly.", "You dress for yourself, not for the occasion.", "Comfort and cool, in equal measure."],
    romantic:    ["Soft silhouettes, intentional choices.", "You lean into your aesthetic and own it completely.", "Feminine and considered — your wardrobe says it before you do."],
    eclectic:    ["You're hard to pin down, which is exactly the point.", "A little bold, a little quiet — always interesting.", "Your style doesn't pick a lane. It doesn't need to."],
    versatile:   ["Your wardrobe keeps its options open.", "Adaptable and considered. You dress for the moment.", "Your closet can do anything you ask of it."],
  };

  const pool = (inspoAesthetic && byAesthetic[inspoAesthetic]) || byPersonality[personality] || byPersonality.versatile;
  return pool[idx % pool.length];
}

function _sdnaPara(personality, topColors, cats, total, neutralPct, idx) {
  const colorNames = topColors.slice(0, 3).map(([name]) => name);
  const colorStr = colorNames.length >= 3
    ? `${colorNames[0]}, ${colorNames[1]}, and ${colorNames[2]}`
    : colorNames.join(" and ");

  const topsCount    = cats.find(c => c.label === "tops")?.items.length    || 0;
  const bottomsCount = cats.find(c => c.label === "bottoms")?.items.length || 0;
  const layerCount   = cats.find(c => c.label === "layers")?.items.length  || 0;
  const dressCount   = cats.find(c => c.label === "dresses")?.items.length || 0;

  // Color opening
  let colorLine = "";
  if (neutralPct > 0.65) {
    colorLine = `You gravitate toward neutrals — ${colorStr} form the foundation of almost everything you own.`;
  } else if (neutralPct > 0.4) {
    colorLine = `Your palette is anchored in neutrals — ${colorStr} — with a little room to experiment.`;
  } else {
    colorLine = `Your color palette is confident: ${colorStr} show up most in your closet.`;
  }

  // Category observation
  let catLine = "";
  if (topsCount > 0 && bottomsCount > 0 && topsCount > bottomsCount * 1.7) {
    catLine = `Your closet skews heavily toward tops, which means you probably have a few go-to bottoms you rely on more than you'd like. Adding more variety down below would multiply your outfit options instantly.`;
  } else if (bottomsCount > 0 && topsCount > 0 && bottomsCount > topsCount * 1.7) {
    catLine = `You have a strong bottom selection — now imagine what a few more interesting tops could unlock for you.`;
  } else if (layerCount / total > 0.3) {
    catLine = `You love a layer — your outerwear game is clearly a priority. That kind of intentional layering is what makes an outfit feel finished.`;
  } else if (dressCount / total > 0.3) {
    catLine = `One-and-done dressing is clearly your move. It's efficient, it's easy, and it always looks intentional.`;
  } else {
    const balancedLines = [
      `Your wardrobe is well-balanced — you've got a solid mix across categories, which means more flexibility when you actually need to get dressed.`,
      `You've built a wardrobe that can do most things. That kind of range is harder to achieve than it looks.`,
      `There's a good spread here. The kind of closet where you can actually get dressed in the morning without a crisis.`,
    ];
    catLine = balancedLines[idx % balancedLines.length];
  }

  return `${colorLine} ${catLine}`;
}

function _sdnaInspoLine(aesthetic, sigColors, wishColors) {
  let line = `Your inspo board leans ${aesthetic}.`;
  if (sigColors.length > 0) {
    const sig = sigColors.slice(0, 2).join(" and ");
    line += ` You already own the ${sig} — those colors are working for you.`;
  }
  if (wishColors.length > 0) {
    const wish = wishColors.slice(0, 2).join(" and ");
    line += ` You're still reaching for ${wish}.`;
  } else if (sigColors.length === 0) {
    line += ` Keep building toward it.`;
  }
  return line;
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

/* ============================================================
   INSPO BOARD
   ============================================================ */

const INSPO_STORES = ["", "Shein", "Zara", "H&M", "Aritzia", "Garage"];
let _buildMatches = [];   // current set of matches for store switching

function saveInspo() {
  try {
    localStorage.setItem("vc_inspo", JSON.stringify(inspoItems));
  } catch (e) {
    if (e.name === "QuotaExceededError") showToast("⚠️ Storage full — remove some inspo items.");
  }
}

/* ── Upload zone ── */
function initInspoUpload() {
  const zone  = document.getElementById("inspoUploadZone");
  const input = document.getElementById("inspoUpload");
  if (!zone || !input) return;

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("dragover",  e => { e.preventDefault(); e.stopPropagation(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault(); e.stopPropagation();
    zone.classList.remove("drag-over");
    const file = [...(e.dataTransfer?.files || [])].find(f => f.type.startsWith("image/"));
    if (file) _handleInspoFile(file);
  });
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) _handleInspoFile(file);
    input.value = "";
  });
}

async function _handleInspoFile(file) {
  const raw        = await readFileAsDataURL(file);
  const compressed = await compressImage(raw);
  await _processInspoImage(compressed, "upload");
}

/* ── Pinterest import ── */
async function importPinterestUrl() {
  const input  = document.getElementById("inspoUrlInput");
  const status = document.getElementById("inspoUrlStatus");
  const url    = (input.value || "").trim();
  if (!url) { showToast("Paste a Pinterest URL first."); return; }

  status.style.display = "";
  status.className     = "inspo-url-status loading";
  status.textContent   = "Fetching images from Pinterest…";

  try {
    const res = await fetch(`${BACKEND}/fetch-image?url=${encodeURIComponent(url)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }
    const data   = await res.json();
    const images = data.images || [];
    if (!images.length) throw new Error("No images found");

    const total = images.length;
    for (let i = 0; i < total; i++) {
      status.textContent = total > 1
        ? `Analyzing ${i + 1} of ${total}…`
        : "Analyzing…";
      const compressed = await compressImage(images[i].image_b64);
      await _processInspoImage(compressed, "pinterest", url, { quiet: true });
    }

    input.value          = "";
    status.style.display = "none";
    renderInspoBoard();
    updateStyleDna();
    showToast(`Added ${total} inspo image${total !== 1 ? "s" : ""} ✓`);
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    status.className   = "inspo-url-status error";
  }
}

/* ── AI analysis ── */
// quiet:true suppresses spinner/toast so the caller can batch-manage UI
async function _processInspoImage(base64, source, url = "", { quiet = false } = {}) {
  const spinner = document.getElementById("inspoAnalyzing");
  if (!quiet && spinner) spinner.style.display = "flex";

  try {
    const res = await fetch(`${BACKEND}/analyze-inspo`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ image: base64 }),
    });
    if (!res.ok) throw new Error(`Analysis failed (${res.status})`);
    const analysis = await res.json();

    inspoItems.unshift({
      id:          Date.now(),
      image:       base64,
      source,
      url,
      vibe:        analysis.vibe        || "style inspo",
      pieces:      analysis.pieces      || [],
      colors:      analysis.colors      || [],
      colors_hex:  analysis.colors_hex  || [],
      style_notes: analysis.style_notes || "",
      addedAt:     Date.now(),
    });
    saveInspo();
    if (!quiet) { renderInspoBoard(); updateStyleDna(); showToast("Inspo added ✓"); }
  } catch (err) {
    if (!quiet) showToast(`Could not analyze image: ${err.message}`);
    else console.warn("[inspo] analyze failed:", err.message);
  } finally {
    if (!quiet && spinner) spinner.style.display = "none";
  }
}

/* ── Render board ── */
function renderInspoBoard() {
  const grid  = document.getElementById("inspoGrid");
  const empty = document.getElementById("inspoEmpty");
  if (!grid) return;

  if (inspoItems.length === 0) {
    grid.innerHTML = "";
    if (empty) empty.style.display = "";
    return;
  }
  if (empty) empty.style.display = "none";

  grid.innerHTML = inspoItems.map((item, idx) => {
    const swatches = (item.colors_hex || []).slice(0, 5).map(h =>
      `<span class="inspo-color-swatch" style="background:${h}"></span>`
    ).join("");
    return `
      <div class="inspo-card">
        <div class="inspo-img-wrap">
          <img src="${item.image}" alt="${escHtml(item.vibe)}" loading="lazy" />
          <span class="inspo-vibe-badge">${escHtml(item.vibe)}</span>
        </div>
        <div class="inspo-card-body">
          ${swatches ? `<div class="inspo-palette">${swatches}</div>` : ""}
          ${item.style_notes ? `<p class="inspo-notes">${escHtml(item.style_notes)}</p>` : ""}
          <div class="inspo-actions">
            <button class="btn-primary inspo-build-btn" onclick="buildThisOutfit(${idx})">✨ Build This Outfit</button>
            <button class="inspo-remove-btn" onclick="removeInspo(${idx})" title="Remove">🗑</button>
          </div>
        </div>
      </div>`;
  }).join("");
}

function removeInspo(idx) {
  inspoItems.splice(idx, 1);
  saveInspo();
  renderInspoBoard();
  updateStyleDna();
  showToast("Removed from inspo board");
}

/* ── Build This Outfit ── */
async function buildThisOutfit(idx) {
  const inspo = inspoItems[idx];
  if (!inspo) return;

  // Create and show modal
  const modal = document.createElement("div");
  modal.id        = "buildOutfitModal";
  modal.className = "build-modal-overlay";
  modal.innerHTML = `
    <div class="build-modal">
      <div class="build-modal-top">
        <button class="build-modal-close" onclick="document.getElementById('buildOutfitModal').remove()">✕</button>
      </div>
      <div class="build-modal-header">
        <img src="${inspo.image}" class="build-inspo-thumb" alt="inspo" />
        <div class="build-header-info">
          <span class="inspo-vibe-badge" style="position:static;margin-bottom:8px;display:inline-block">${escHtml(inspo.vibe)}</span>
          <p class="build-notes">${escHtml(inspo.style_notes || "")}</p>
        </div>
      </div>
      <div class="build-modal-content">
        <div class="ai-thinking" id="buildThinking" style="display:flex;margin:8px 0 24px">
          <div class="ai-spinner"></div>
          <span>Finding matches in your closet…</span>
        </div>
      </div>
    </div>`;
  modal.addEventListener("click", e => { if (e.target === modal) modal.remove(); });
  document.body.appendChild(modal);

  if (closetItems.length === 0) {
    document.querySelector("#buildOutfitModal .build-modal-content").innerHTML = `
      <div class="empty-state" style="padding:40px 20px">
        <p>Add items to your closet first, then come back to build this outfit.</p>
      </div>`;
    return;
  }

  try {
    const res = await fetch(`${BACKEND}/match-inspo`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        inspo:       { vibe: inspo.vibe, pieces: inspo.pieces, colors: inspo.colors },
        inspo_image: inspo.image,   // send the actual photo so Claude can visually compare
        closet:      closetItems.map((item, i) => ({ index: i, type: item.type, color: item.color, pattern: item.pattern || "" })),
      }),
    });
    if (!res.ok) throw new Error(`Match failed (${res.status})`);
    const data = await res.json();
    _buildMatches = data.matches || [];
    await _renderBuildResults(modal);
  } catch (err) {
    const content = document.querySelector("#buildOutfitModal .build-modal-content");
    if (content) content.innerHTML = `<div class="ai-error"><span class="ai-error-icon">⚠️</span><div>${escHtml(err.message)}</div></div>`;
  }
}

async function _renderBuildResults(modal) {
  const content = modal.querySelector(".build-modal-content");
  const matches = _buildMatches;

  const storeBtns = INSPO_STORES.map((s, i) =>
    `<button class="chip${i === 0 ? " active" : ""}" onclick="switchBuildStore('${escHtml(s)}',this)">${escHtml(s || "All")}</button>`
  ).join("");

  const matchesHtml = matches.map((m, mIdx) => {
    if (m.closet_item !== null && m.closet_item !== undefined) {
      const item = closetItems[m.closet_item.index];
      if (!item) return "";
      return `
        <div class="build-match-row">
          <div class="build-piece-label">${escHtml(m.piece)}</div>
          <div class="build-closet-match">
            <img src="${item.image}" class="build-closet-thumb" alt="${escHtml(item.type)}" />
            <div>
              <div class="build-match-badge">✓ You have this</div>
              <div class="build-match-type">${escHtml(item.type)} · ${escHtml(item.color)}</div>
            </div>
          </div>
        </div>`;
    } else {
      return `
        <div class="build-match-row" data-gap-idx="${mIdx}">
          <div class="build-piece-label">${escHtml(m.piece)}</div>
          <div class="build-gap-results" id="gap-${mIdx}">
            <div class="ai-thinking" style="display:flex"><div class="ai-spinner"></div><span>Searching…</span></div>
          </div>
        </div>`;
    }
  }).join("");

  content.innerHTML = `
    <div class="build-section-title">Outfit Pieces</div>
    <div class="build-store-filter">
      <span class="build-store-label">Shop at:</span>
      <div class="filter-chips">${storeBtns}</div>
    </div>
    <div class="build-matches">${matchesHtml}</div>`;

  // Fire all gap searches in parallel
  const gaps = matches.map((m, i) => ({...m, idx: i})).filter(m => !m.closet_item && m.closet_item !== 0);
  await Promise.all(gaps.map(m => _fetchGapProduct(m.idx, m.gap_query || m.piece, "")));
}

async function switchBuildStore(store, btn) {
  document.querySelectorAll(".build-store-filter .chip").forEach(c => c.classList.remove("active"));
  btn.classList.add("active");

  const gaps = _buildMatches.map((m, i) => ({...m, idx: i})).filter(m => !m.closet_item && m.closet_item !== 0);
  gaps.forEach(m => {
    const el = document.getElementById(`gap-${m.idx}`);
    if (el) el.innerHTML = `<div class="ai-thinking" style="display:flex"><div class="ai-spinner"></div><span>Searching…</span></div>`;
  });
  await Promise.all(gaps.map(m => _fetchGapProduct(m.idx, m.gap_query || m.piece, store)));
}

// Strip vibe/style/aesthetic words — query must be [shade] [silhouette] [type] only.
function _cleanGapQuery(raw) {
  const vibeWords = /\b(clean|minimalist|effortless|aesthetic|chic|casual|elevated|luxe|classic|modern|simple|soft|quiet|cozy|boho|grunge|edgy|street|cool|girl|style|inspired|look|vibe|outfit|core|cottagecore|fitted|relaxed|oversized|streetwear|feminine|romantic|preppy|sporty|trendy)\b/gi;
  return raw.replace(vibeWords, "").replace(/\s{2,}/g, " ").trim();
}

// Retry fallback: drop to last 2-3 words (silhouette + garment type).
function _simplerQuery(q) {
  const words = q.trim().split(/\s+/);
  return words.length > 2 ? words.slice(-2).join(" ") : q;
}

function _manualSearchBox(gapIdx, placeholderQ) {
  return `
    <div class="manual-search-box">
      <span class="inspo-url-status">No results found.</span>
      <div class="manual-search-row">
        <input type="text" class="manual-search-input" id="manual-q-${gapIdx}"
               placeholder="${escHtml(placeholderQ)}" value="${escHtml(placeholderQ)}" />
        <button class="manual-search-btn" onclick="manualGapSearch(${gapIdx})">Search</button>
      </div>
    </div>`;
}

// ─── SerpAPI cache ───────────────────────────────────────────────────────────
const _SERP_TTL  = 24 * 60 * 60 * 1000;  // 24 h in ms
const _SERP_DEV  = location.hostname === "127.0.0.1" || location.hostname === "localhost";

async function _serpFetch(q) {
  const key     = "serp_cache_" + q;
  const cached  = (() => { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } })();

  if (cached && (Date.now() - cached.cachedAt) < _SERP_TTL) {
    return { shopping_results: cached.results, fromCache: true };
  }

  const res = await fetch(`${BACKEND}/shop?q=${encodeURIComponent(q)}`);
  if (!res.ok) throw new Error("Search error");
  const data = await res.json();

  try {
    localStorage.setItem(key, JSON.stringify({ results: data.shopping_results || [], cachedAt: Date.now() }));
  } catch { /* storage full — ignore */ }

  return { shopping_results: data.shopping_results || [], fromCache: false };
}

async function manualGapSearch(gapIdx) {
  const input = document.getElementById(`manual-q-${gapIdx}`);
  if (!input) return;
  const q = input.value.trim();
  if (!q) return;
  const target = document.getElementById(`gap-${gapIdx}`);
  if (target) target.innerHTML = `<span class="inspo-url-status">Searching…</span>`;
  await _fetchGapProduct(gapIdx, q, "", true);
}

async function _fetchGapProduct(gapIdx, query, store, skipClean = false) {
  const cleaned = skipClean ? query : _cleanGapQuery(query);
  const q       = store ? `${cleaned} ${store}` : cleaned;
  const target  = document.getElementById(`gap-${gapIdx}`);

  async function _doSearch(searchQ) {
    const data = await _serpFetch(searchQ);
    if (_SERP_DEV && data.fromCache) console.debug(`[serp cache hit] ${searchQ}`);
    return (data.shopping_results || []).slice(0, 3);
  }

  try {
    let products = await _doSearch(q);

    // Retry 1: simpler query (last 2 words)
    if (!products.length && cleaned !== _simplerQuery(cleaned)) {
      const simpler = store ? `${_simplerQuery(cleaned)} ${store}` : _simplerQuery(cleaned);
      products = await _doSearch(simpler);
    }

    if (!target) return;
    if (!products.length) {
      target.innerHTML = _manualSearchBox(gapIdx, cleaned);
      return;
    }

    target.innerHTML = products.map(p => `
      <div class="shop-card">
        ${p.thumbnail
          ? `<img src="${escHtml(p.thumbnail)}" class="shop-thumb" alt="" loading="lazy" />`
          : `<div class="shop-thumb-placeholder">🛍️</div>`}
        <div class="shop-info">
          <div class="shop-title">${escHtml((p.title || "").substring(0, 70))}</div>
          <div class="shop-meta">
            ${p.price  ? `<span class="shop-price">${escHtml(p.price)}</span>` : ""}
            ${p.source ? `<span class="shop-store-badge">${escHtml(p.source)}</span>` : ""}
          </div>
          ${p.link ? `<a href="${escHtml(p.link)}" target="_blank" rel="noopener noreferrer" class="shop-now-btn">Shop Now →</a>` : ""}
        </div>
      </div>`).join("");
  } catch {
    if (target) target.innerHTML = `<span class="inspo-url-status error">Search unavailable</span>`;
  }
}

// ─── Shop Outfits ────────────────────────────────────────────────────────────
let _shopOutfits  = [];   // outfit concepts from Claude
let _shopProducts = {};   // key: "outfitIdx-pieceIdx" → product object
let _shopFilters  = { occasion: "", vibe: "", store: "" };

function initShopOutfits() {
  [["sofOccasion", "occasion"], ["sofVibe", "vibe"], ["sofStore", "store"]].forEach(([id, key]) => {
    document.getElementById(id)?.addEventListener("click", e => {
      const chip = e.target.closest(".sof-chip");
      if (!chip) return;
      e.currentTarget.querySelectorAll(".sof-chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      _shopFilters[key] = chip.dataset.val;
    });
  });
}

function _shopSkeletonHtml() {
  return `
    <div class="shop-outfit-card soc-skeleton">
      <div class="soc-skel-header"></div>
      <div class="soc-skel-pieces">
        <div class="soc-skel-piece"></div><div class="soc-skel-piece"></div>
        <div class="soc-skel-piece"></div><div class="soc-skel-piece"></div>
      </div>
      <div class="soc-skel-btn-row"><div class="soc-skel-btn"></div><div class="soc-skel-btn"></div></div>
    </div>`;
}

function _pieceProductHtml(role, product, fromCache = false) {
  if (!product) return `
    <div class="soc-role-label">${escHtml(role)}</div>
    <div class="soc-no-result">No results<br>
      <span class="soc-retry-note">Try another store</span></div>`;
  const cacheTag = (_SERP_DEV && fromCache) ? `<span class="serp-cache-badge">cached</span>` : "";
  return `
    <div class="soc-role-label">${escHtml(role)}${cacheTag}</div>
    ${product.thumbnail
      ? `<img src="${escHtml(product.thumbnail)}" class="soc-piece-img" alt="" loading="lazy" />`
      : `<div class="soc-piece-img-ph">🛍️</div>`}
    <div class="soc-piece-title">${escHtml((product.title || "").substring(0, 55))}</div>
    <div class="soc-piece-meta">
      ${product.price  ? `<span class="shop-price">${escHtml(product.price)}</span>` : ""}
      ${product.source ? `<span class="shop-store-badge">${escHtml(product.source)}</span>` : ""}
    </div>
    ${product.link ? `<a href="${escHtml(product.link)}" target="_blank" rel="noopener noreferrer" class="soc-shop-btn">Shop →</a>` : ""}`;
}

function _shopOutfitCardHtml(outfit, idx) {
  const piecesHtml = (outfit.pieces || []).map((p, pi) => `
    <div class="soc-piece" id="soc-${idx}-p-${pi}">
      <div class="soc-role-label">${escHtml(p.role)}</div>
      <div class="soc-piece-loading"><span class="spinner" style="width:20px;height:20px"></span></div>
    </div>`).join("");

  return `
    <div class="shop-outfit-card" id="soc-${idx}">
      <div class="soc-header">
        <div class="soc-name">${escHtml(outfit.name)}</div>
        <span class="soc-vibe-badge">${escHtml(outfit.vibe)}</span>
      </div>
      <div class="soc-pieces">${piecesHtml}</div>
      <div class="soc-actions">
        <div class="soc-closet-summary" id="soc-summary-${idx}" style="display:none"></div>
        <div class="soc-btn-row">
          <button class="btn-secondary soc-btn" onclick="saveShopOutfitToInspo(${idx})">💾 Save to Inspo</button>
          <button class="btn-secondary soc-btn" onclick="findShopOutfitInCloset(${idx})">🔍 Find in My Closet</button>
        </div>
      </div>
    </div>`;
}

async function generateShopOutfits() {
  const grid = document.getElementById("shopOutfitsGrid");
  grid.innerHTML = Array.from({length: 4}, _shopSkeletonHtml).join("");

  try {
    const res = await fetch(`${BACKEND}/generate-outfits`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        ..._shopFilters,
        closet:       closetItems.map(i => ({ type: i.type, color: i.color, pattern: i.pattern || "" })),
        inspo_vibes:  [...new Set(inspoItems.map(i => i.vibe).filter(Boolean))].slice(0, 8),
        inspo_colors: [...new Set(inspoItems.flatMap(i => i.colors || []).filter(Boolean))].slice(0, 10),
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    _shopOutfits  = data.outfits || [];
    _shopProducts = {};

    if (!_shopOutfits.length) {
      grid.innerHTML = `<div class="sof-empty" style="grid-column:1/-1"><span class="sof-empty-icon">😕</span><p>Couldn't generate outfits — try again.</p><button class="btn-primary" onclick="generateShopOutfits()">Retry</button></div>`;
      return;
    }

    grid.innerHTML = _shopOutfits.map((o, i) => _shopOutfitCardHtml(o, i)).join("");

    // Load all outfit products in parallel (outfits × pieces all fire at once)
    await Promise.all(_shopOutfits.map((outfit, idx) => _loadShopOutfitProducts(outfit, idx)));

  } catch (err) {
    grid.innerHTML = `<div class="sof-empty" style="grid-column:1/-1"><span class="sof-empty-icon">⚠️</span><p>${escHtml(err.message || "Something went wrong")}</p><button class="btn-primary" onclick="generateShopOutfits()">Retry</button></div>`;
  }
}

async function _loadShopOutfitProducts(outfit, idx) {
  const store = _shopFilters.store;
  await Promise.all((outfit.pieces || []).map(async (piece, pi) => {
    const cellEl = document.getElementById(`soc-${idx}-p-${pi}`);
    if (!cellEl) return;

    const q = store ? `${piece.search_query} ${store}` : piece.search_query;
    let product = null;
    let fromCache = false;
    try {
      const d = await _serpFetch(q);
      fromCache = d.fromCache;
      product = (d.shopping_results || [])[0] || null;

      // Retry with simpler query if nothing found
      if (!product) {
        const words = piece.search_query.trim().split(/\s+/);
        if (words.length > 2) {
          const simpler = store
            ? `${words.slice(-2).join(" ")} ${store}`
            : words.slice(-2).join(" ");
          const d2 = await _serpFetch(simpler);
          fromCache = d2.fromCache;
          product = (d2.shopping_results || [])[0] || null;
        }
      }
    } catch { /* leave product null */ }

    _shopProducts[`${idx}-${pi}`] = product;
    if (cellEl) cellEl.innerHTML = _pieceProductHtml(piece.role, product, fromCache);
  }));
}

async function retryShopPiece(idx, pi) {
  const outfit = _shopOutfits[idx];
  const piece  = outfit?.pieces?.[pi];
  const cellEl = document.getElementById(`soc-${idx}-p-${pi}`);
  if (!piece || !cellEl) return;

  cellEl.innerHTML = `<div class="soc-role-label">${escHtml(piece.role)}</div><div class="soc-piece-loading"><span class="spinner" style="width:20px;height:20px"></span></div>`;

  const altStores = ["Shein", "Zara", "H&M", "Aritzia", "Garage"].filter(s => s !== _shopFilters.store);
  const nextStore = altStores[Math.floor(Math.random() * altStores.length)];
  const q = `${piece.search_query} ${nextStore}`;

  let product = null;
  let fromCache = false;
  try {
    const d = await _serpFetch(q);
    fromCache = d.fromCache;
    product = (d.shopping_results || [])[0] || null;
  } catch { /* leave null */ }

  _shopProducts[`${idx}-${pi}`] = product;
  cellEl.innerHTML = _pieceProductHtml(piece.role, product, fromCache);
}

function saveShopOutfitToInspo(idx) {
  const outfit = _shopOutfits[idx];
  if (!outfit) return;
  inspoItems.unshift({
    id:          Date.now(),
    image:       null,
    source:      "shop-outfits",
    vibe:        outfit.vibe,
    pieces:      (outfit.pieces || []).map(p => p.search_query),
    colors:      [],
    colors_hex:  [],
    style_notes: outfit.name,
  });
  saveInspo();
  showToast(`"${outfit.name}" saved to Inspo Board`);
}

function findShopOutfitInCloset(idx) {
  const outfit = _shopOutfits[idx];
  if (!outfit) return;
  if (!closetItems.length) { showToast("Add items to your closet first"); return; }

  // Remove any previous "owned" state on this card
  document.querySelectorAll(`#soc-${idx} .soc-owned`).forEach(el => {
    el.classList.remove("soc-owned");
    el.querySelector(".soc-owned-badge")?.remove();
  });

  let ownedCount = 0;
  let shopTotal  = 0;

  (outfit.pieces || []).forEach((piece, pi) => {
    const cellEl = document.getElementById(`soc-${idx}-p-${pi}`);
    if (!cellEl) return;

    const q = piece.search_query.toLowerCase();
    const match = closetItems.find(item => {
      const type  = (item.type  || "").toLowerCase().replace(/-/g, " ");
      const color = (item.color || "").toLowerCase();
      const typeWords  = type.split(/\s+/).filter(w => w.length > 2);
      const colorWords = color.split(/\s+/).filter(w => w.length > 2);
      const typeMatch  = typeWords.length > 0 && typeWords.every(w => q.includes(w));
      const colorMatch = colorWords.some(w => q.includes(w));
      return typeMatch && colorMatch;
    });

    if (match) {
      ownedCount++;
      cellEl.classList.add("soc-owned");
      const badge = document.createElement("div");
      badge.className = "soc-owned-badge";
      badge.textContent = "✓ You own this";
      cellEl.prepend(badge);
    } else {
      const product = _shopProducts[`${idx}-${pi}`];
      if (product?.price) {
        const n = parseFloat(product.price.replace(/[^0-9.]/g, ""));
        if (!isNaN(n)) shopTotal += n;
      }
    }
  });

  const total     = (outfit.pieces || []).length;
  const summary   = document.getElementById(`soc-summary-${idx}`);
  if (summary) {
    summary.style.display = "block";
    const priceStr = shopTotal > 0 ? ` — shop the rest for ~$${shopTotal.toFixed(0)}` : "";
    summary.textContent = `You own ${ownedCount} of ${total} pieces${priceStr}`;
  }
}

// Patch showTab to handle match tab + weather + inspo
const _origShowTab = showTab;
showTab = function (name) {
  _origShowTab(name);
  if (name === "outfits") fetchWeather();
  if (name === "inspo")   renderInspoBoard();
};
