/* ========== CALL BACKEND AI DETECT CLOTHING + COLOR ========== */
async function detectClothingAndColor(imageFile) {
  const tryPost = async (fieldName) => {
    const fd = new FormData();
    fd.append(fieldName, imageFile);

    const res = await fetch("http://127.0.0.1:8000/detect", {
      method: "POST",
      body: fd,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status} from /detect (${fieldName}): ${text}`);
    }
    return res.json();
  };

  try {
    // First try: field name "file"
    return await tryPost("file");
  } catch (e1) {
    console.warn("detect using 'file' failed:", e1.message);
    try {
      // Second try: field name "image"
      return await tryPost("image");
    } catch (e2) {
      console.error(
        "❌ detect failed with both 'file' and 'image':",
        e1.message,
        e2.message
      );
      return { detected_items: [] }; // graceful fallback
    }
  }
}

/* ========== GLOBAL CLOSET STATE ========== */
let closetItems = [];
let currentEditIndex = -1;

if (localStorage.getItem("closet")) {
  try {
    closetItems = JSON.parse(localStorage.getItem("closet"));
  } catch (e) {
    console.error("Error parsing closet from localStorage:", e);
    closetItems = [];
  }
  refreshClosetView();
}

/* ========== ADD ITEM (AI) ========== */
async function addItem() {
  const fileInput = document.getElementById("upload");
  const file = fileInput.files[0];
  const event = document.getElementById("event").value.trim();
  const pattern = document.getElementById("pattern").value.trim();
  const tags = document.getElementById("tags").value.trim();

  if (!file) {
    alert("Please select an image first.");
    return;
  }

  const addBtn = document.querySelector(".upload-btn");
  const suggestion = document.getElementById("suggestion");

  // 🔒 show loading + disable button
  const oldBtnText = addBtn.innerText;
  addBtn.innerText = "Analyzing...";
  addBtn.disabled = true;
  suggestion.innerText = "Analyzing your item, this can take a few seconds...";

  console.log("[addItem] starting...");

  const reader = new FileReader();
  reader.onload = async function (e) {
    const imageData = e.target.result;

    try {
      console.log("[addItem] calling detectClothingAndColor...");
      const result = await detectClothingAndColor(file);

      console.log("[addItem] /detect result:", result);

      const detected_items = Array.isArray(result.detected_items)
        ? result.detected_items
        : result.detected_items
        ? [result.detected_items]
        : [];

      let type = "Unknown";
      let color = "Unknown";
      let color_hex = "#cccccc";

      if (detected_items.length > 0) {
        const d = detected_items[0];
        type = d.label || "Unknown";
        color = d.color || "Unknown";
        color_hex = d.color_hex || "#cccccc";
      } else {
        // fallback to simple image-based guess
        const fb = await estimateFromImage(imageData);
        type = fb.type;
        color = fb.color;
        color_hex = fb.color_hex;
      }

      // fill manual fields
      const typeInput = document.getElementById("type");
      if (typeInput) typeInput.value = type;
      const colorInput = document.getElementById("color");
      if (colorInput) colorInput.value = color;

      const item = {
        image: imageData,
        type,
        color,
        color_hex,
        event,
        pattern,
        tags,
      };

      if (currentEditIndex >= 0) {
        closetItems[currentEditIndex] = item;
        currentEditIndex = -1;
      } else {
        closetItems.push(item);
      }

      localStorage.setItem("closet", JSON.stringify(closetItems));
      refreshClosetView();
      resetForm(fileInput);

      suggestion.innerText = "Item added to your closet!";
    } catch (err) {
      console.error("Error in addItem:", err);
      alert("Something went wrong while adding this item. Check console for details.");
      suggestion.innerText = "Error analyzing item.";
    } finally {
      // 🔓 restore button
      addBtn.innerText = oldBtnText;
      addBtn.disabled = false;
    }
  };

  reader.readAsDataURL(file);
}


function resetForm(fileInput) {
  fileInput.value = "";
  document.getElementById("event").value = "";
  document.getElementById("pattern").value = "";
  document.getElementById("tags").value = "";
}

/* ========== SIMPLE COLOR AVERAGE FALLBACK ========== */
async function averageColorFromImage(imageData) {
  const img = new Image();
  img.src = imageData;
  await new Promise((r) => {
    img.onload = r;
    img.onerror = r;
  });

  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const canvas = document.createElement("canvas");
  canvas.width = Math.min(200, w);
  canvas.height = Math.min(200, h);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);

  let r = 0,
    g = 0,
    b = 0,
    count = 0;
  const step = 4 * 16; // sample every 16th pixel

  for (let i = 0; i < data.length; i += step) {
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
    count++;
  }

  r = Math.round(r / Math.max(1, count));
  g = Math.round(g / Math.max(1, count));
  b = Math.round(b / Math.max(1, count));

  const hex =
    "#" +
    [r, g, b]
      .map((v) => v.toString(16).padStart(2, "0"))
      .join("");

  // crude color name
  let name = "gray";
  const maxc = Math.max(r, g, b);
  const minc = Math.min(r, g, b);
  const V = maxc;
  const S = V === 0 ? 0 : (maxc - minc) / V;

  if (V < 60) {
    name = "black";
  } else if (S < 0.1) {
    name = "gray";
  } else if (r >= g && r >= b) {
    name = "red";
  } else if (g >= r && g >= b) {
    name = "green";
  } else {
    name = "blue";
  }

  return { r, g, b, hex, name };
}

/* ========== IMAGE-ONLY TYPE FALLBACK ========== */
async function estimateFromImage(imageData) {
  const img = new Image();
  img.src = imageData;
  await new Promise((r) => {
    img.onload = r;
    img.onerror = r;
  });

  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const tall = h / Math.max(1, w) > 1.4;

  const avg = await averageColorFromImage(imageData);
  const color_hex = avg.hex;
  let colorName = avg.name;

  let type;
  if (tall) {
    type = "pants";
  } else {
    type = "t-shirt";
  }

  return { type, color: colorName, color_hex };
}

/* ========== CLOSET VIEW / EDIT / DELETE / SEARCH / SUGGEST ========== */
function refreshClosetView() {
  const closet = document.getElementById("closet");
  closet.innerHTML = "";

  closetItems.forEach((item, index) => {
    const div = document.createElement("div");
    div.className = "item";
    div.innerHTML = `
      <img src="${item.image}" alt="${item.type}">
      <p>
        <strong>Color:</strong> ${item.color}
        <span class="color-swatch" style="background:${item.color_hex || "#ccc"}"></span>
      </p>
      <p><strong>Type:</strong> ${item.type}</p>
      <p><strong>Event:</strong> ${item.event || "N/A"}</p>
      <p><strong>Pattern:</strong> ${item.pattern || "N/A"}</p>
      <p><strong>Tags:</strong> ${item.tags || "N/A"}</p>
      <button onclick="editItem(${index})">Edit</button>
      <button onclick="deleteItem(${index})">Delete</button>
    `;
    closet.appendChild(div);
  });
}

function editItem(index) {
  const item = closetItems[index];
  currentEditIndex = index;

  const typeInput = document.getElementById("type");
  if (typeInput) typeInput.value = item.type;
  const colorInput = document.getElementById("color");
  if (colorInput) colorInput.value = item.color;

  document.getElementById("event").value = item.event || "";
  document.getElementById("pattern").value = item.pattern || "";
  document.getElementById("tags").value = item.tags || "";

  showTab("add");
}

function deleteItem(index) {
  closetItems.splice(index, 1);
  localStorage.setItem("closet", JSON.stringify(closetItems));
  refreshClosetView();
}

function resetCloset() {
  if (confirm("Are you sure you want to clear your closet?")) {
    localStorage.clear();
    closetItems = [];
    refreshClosetView();
    alert("Closet has been reset!");
  }
}

function searchCloset() {
  const query = document.getElementById("search").value.toLowerCase();
  const closet = document.getElementById("closet");
  closet.innerHTML = "";

  closetItems.forEach((item, index) => {
    const tags = (item.tags || "").toLowerCase();
    if (
      (item.type || "").toLowerCase().includes(query) ||
      (item.color || "").toLowerCase().includes(query) ||
      tags.includes(query)
    ) {
      const div = document.createElement("div");
      div.className = "item";
      div.innerHTML = `
        <img src="${item.image}" alt="${item.type}">
        <p>
          <strong>Color:</strong> ${item.color}
          <span class="color-swatch" style="background:${item.color_hex || "#ccc"}"></span>
        </p>
        <p><strong>Type:</strong> ${item.type}</p>
        <p><strong>Event:</strong> ${item.event || "N/A"}</p>
        <p><strong>Pattern:</strong> ${item.pattern || "N/A"}</p>
        <p><strong>Tags:</strong> ${item.tags || "N/A"}</p>
        <button onclick="editItem(${index})">Edit</button>
        <button onclick="deleteItem(${index})">Delete</button>
      `;
      closet.appendChild(div);
    }
  });
}

function suggestOutfit() {
  const shirts = closetItems.filter((i) =>
    /(t-shirt|long-sleeve|blouse|sweatshirt|sweater|hoodie|shirt)/i.test(i.type)
  );
  const pants = closetItems.filter((i) =>
    /(pants|jeans|shorts)/i.test(i.type)
  );
  if (shirts.length === 0 || pants.length === 0) {
    document.getElementById("suggestion").innerText =
      "Add at least one shirt and one pants/jeans/shorts to get a suggestion.";
    return;
  }
  const top = shirts[Math.floor(Math.random() * shirts.length)];
  const bottom = pants[Math.floor(Math.random() * pants.length)];
  document.getElementById("suggestion").innerText =
    `Try your ${top.color} ${top.type} with your ${bottom.color} ${bottom.type}.`;
}

/* ========== TABS ========== */
function showTab(tabName) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.remove("active");
  });
  document.getElementById(`tab-${tabName}`).classList.add("active");
}
