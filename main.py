import os
import io
from typing import List, Optional

import numpy as np
import torch
import webcolors
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import YolosForObjectDetection, YolosImageProcessor
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------- INIT APP -----------------
app = FastAPI()

# 🚨 For dev, allow any origin (so it works from file:// or http://localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- LOAD MODEL -----------------
model_name = "valentinafeve/yolos-fashionpedia"
model = YolosForObjectDetection.from_pretrained(model_name, use_safetensors=True)
processor = YolosImageProcessor.from_pretrained(model_name)

# ----------------- COLOR UTILS -----------------
def closest_css_color(rgb):
    """Return nearest CSS3 color name for an RGB triplet."""
    try:
        return webcolors.rgb_to_name(rgb, spec="css3")
    except ValueError:
        min_dist, closest = float("inf"), None
        for hex_code, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r, g, b = webcolors.hex_to_rgb(hex_code)
            dist = (r - rgb[0])**2 + (g - rgb[1])**2 + (b - rgb[2])**2
            if dist < min_dist:
                min_dist, closest = dist, name
        return closest or "gray"


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


def sleeve_likely_by_edges(pil_image: Image.Image, filename: str = None) -> bool:
    """
    Compare left/right vertical bands (excluding background) to a center torso patch.
    Scans inward from edges until it finds bands whose pixels mostly match torso color.
    """
    import numpy as np

    w, h = pil_image.size
    small = pil_image.resize((min(360, w), min(360, h)))
    arr = np.array(small).astype(np.float32)  # H x W x 3
    H, W, _ = arr.shape

    # torso reference (center box)
    cx0, cx1 = int(W * 0.35), int(W * 0.65)
    cy0, cy1 = int(H * 0.30), int(H * 0.55)
    torso = arr[cy0:cy1, cx0:cx1]
    torso_mean = torso.reshape(-1, 3).mean(0)

    # background from corners
    corners = np.vstack([
        arr[0:int(H * 0.08), 0:int(W * 0.08)].reshape(-1, 3),
        arr[0:int(H * 0.08), W-int(W * 0.08):W].reshape(-1, 3),
        arr[H-int(H * 0.08):H, 0:int(W * 0.08)].reshape(-1, 3),
        arr[H-int(H * 0.08):H, W-int(W * 0.08):W].reshape(-1, 3),
    ])
    bg_mean = corners.mean(0)
    bg_dist = np.linalg.norm(arr - bg_mean, axis=2)
    bg_mask = bg_dist < 28.0

    # vertical band region
    y0, y1 = int(H * 0.50), int(H * 0.88)

    def band_matches(x0, x1):
        band = arr[y0:y1, x0:x1]
        band_bg = bg_mask[y0:y1, x0:x1]
        if band.size == 0:
            return 0.0, 999.0
        pix = band[~band_bg]
        if pix.size == 0:
            return 0.0, 999.0
        d = np.linalg.norm(pix - torso_mean, axis=1)
        frac = float((d < 32.0).mean())
        return frac, float(d.mean()) if d.size else 999.0

    band_w = max(2, int(W * 0.08))
    step = max(2, int(W * 0.02))
    left_best, right_best = 0.0, 0.0
    left_debug, right_debug = (0.0, 999.0), (0.0, 999.0)

    # scan from left
    for x in range(int(W * 0.02), int(W * 0.30), step):
        frac, md = band_matches(x, min(x + band_w, W))
        if frac > left_best:
            left_best = frac
            left_debug = (frac, md)

    # scan from right
    for x in range(W - int(W * 0.02) - band_w, W - int(W * 0.30), -step):
        frac, md = band_matches(max(0, x), min(x + band_w, W))
        if frac > right_best:
            right_best = frac
            right_debug = (frac, md)

    print(
        f"[Sleeve Debug] {filename or ''} "
        f"left_frac={left_best:.2f} right_frac={right_best:.2f} "
        f"left_md={left_debug[1]:.1f} right_md={right_debug[1]:.1f}"
    )

    return (left_best >= 0.35) and (right_best >= 0.35)


def hsvish_bucket_name(r: int, g: int, b: int) -> str:
    V = max(r, g, b)
    chroma = V - min(r, g, b)
    S = 0 if V == 0 else chroma / V

    if V >= 240:
        return "white"
    if V < 80 or (V < 120 and S < 0.12):
        return "blue" if (b > r + 8 and b > g + 8) else "black"
    if S < 0.12:
        return "gray"
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    return "blue"


def np_kmeans(pixels: np.ndarray, k: int = 3, iters: int = 7, seed: int = 42):
    """
    Simple NumPy k-means for (N,3) pixels.
    Returns (centers[k,3], labels[N]).
    """
    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError("pixels must be (N,3)")

    rng = np.random.default_rng(seed)
    idx = rng.choice(pixels.shape[0], size=min(k, pixels.shape[0]), replace=False)
    centers = pixels[idx].astype(np.float32)

    for _ in range(iters):
        d2 = ((pixels[:, None, :].astype(np.float32) - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)

        new_centers = np.empty_like(centers)
        for c in range(centers.shape[0]):
            mask = labels == c
            if mask.any():
                new_centers[c] = pixels[mask].mean(axis=0)
            else:
                new_centers[c] = pixels[rng.integers(0, pixels.shape[0])]
        if np.allclose(new_centers, centers, atol=1e-1):
            centers = new_centers
            break
        centers = new_centers
    return centers, labels


def get_dominant_color(pil_image: Image.Image) -> str:
    image = pil_image.resize((64, 64))
    pixels = np.asarray(image, dtype=np.uint16).reshape(-1, 3)

    sums = pixels.sum(axis=1)
    mask = (sums > 50) & (sums < 700)
    pixels_f = pixels[mask] if mask.any() else pixels

    centers, labels = np_kmeans(pixels_f, k=3, iters=7)
    vals, counts = np.unique(labels, return_counts=True)
    dominant = centers[vals[counts.argmax()]]
    r, g, b = map(int, dominant)

    bucket = hsvish_bucket_name(r, g, b)
    return bucket if bucket in {"black", "gray", "white", "blue", "red", "green"} \
                  else closest_css_color((r, g, b))


def get_avg_rgb_from_bbox(pil_image, bbox, img_w, img_h):
    cx, cy, bw, bh = bbox
    x_min = int((cx - bw / 2) * img_w)
    y_min = int((cy - bh / 2) * img_h)
    x_max = int((cx + bw / 2) * img_w)
    y_max = int((cy + bh / 2) * img_h)
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_w, x_max), min(img_h, y_max)
    if x_max <= x_min or y_max <= y_min:
        return [128, 128, 128]
    cropped = pil_image.crop((x_min, y_min, x_max, y_max))
    return np.array(cropped).mean(axis=(0, 1)).astype(int).tolist()


def get_color_from_bbox(pil_image, bbox, img_w, img_h):
    cx, cy, bw, bh = bbox
    x_min = int((cx - bw / 2) * img_w)
    y_min = int((cy - bh / 2) * img_h)
    x_max = int((cx + bw / 2) * img_w)
    y_max = int((cy + bh / 2) * img_h)
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_w, x_max), min(img_h, y_max)
    if x_max <= x_min or y_max <= y_min:
        return "gray"

    cropped = pil_image.crop((x_min, y_min, x_max, y_max)).resize((64, 64))
    pixels = np.asarray(cropped, dtype=np.uint16).reshape(-1, 3)

    sums = pixels.sum(axis=1)
    mask = (sums > 50) & (sums < 700)
    pixels_f = pixels[mask] if mask.any() else pixels

    centers, labels = np_kmeans(pixels_f, k=3, iters=7)
    vals, counts = np.unique(labels, return_counts=True)
    dominant = centers[vals[counts.argmax()]]
    r, g, b = map(int, dominant)

    bucket = hsvish_bucket_name(r, g, b)
    return bucket if bucket in {"black", "gray", "white", "blue", "red", "green"} \
                  else closest_css_color((r, g, b))


def is_shorts_bbox(box, threshold=0.55):
    # YOLOS normalized height (bh). Shorter boxes → shorts.
    return float(box[3]) < threshold


# ----------------- LABEL RULES -----------------
PART_LABELS = {
    "collar", "lapel", "epaulette", "sleeve", "pocket", "neckline", "buckle", "zipper",
    "applique", "bead", "bow", "flower", "fringe", "ribbon", "rivet", "ruffle", "sequin",
    "tassel", "button", "hem", "seam", "logo"
}
GARMENT_LABELS = {
    "t-shirt", "shirt", "top", "sweatshirt", "hoodie", "blouse",
    "pants", "jeans", "shorts", "skirt", "dress",
    "jacket", "coat", "cardigan", "vest", "sweater"
}


def is_denim_color_name(name: Optional[str]) -> bool:
    return bool(name) and any(k in name.lower() for k in ["denim", "indigo", "blue", "navy", "slate"])


def is_blue_dominant(avg_rgb):
    if len(avg_rgb) != 3:
        return False
    r, g, b = avg_rgb
    return (b > r + 15) and (b > g + 15)


def choose_final_label(labels: List[str], sleeve_type: str) -> str:
    labels = [l.lower() for l in labels]
    if "hoodie" in labels:
        return "hoodie"
    if any(l in labels for l in ["jacket", "coat", "cardigan", "vest"]):
        return "jacket"
    if "sweatshirt" in labels:
        return "long-sleeve"
    if any(l in labels for l in ["t-shirt", "shirt", "top", "blouse"]):
        return "long-sleeve" if sleeve_type == "long" else "t-shirt"
    if "jeans" in labels:
        return "jeans"
    if "pants" in labels:
        return "pants"
    if "shorts" in labels:
        return "shorts"
    if "skirt" in labels:
        return "skirt"
    if "dress" in labels:
        return "dress"
    return labels[0] if labels else "unknown"


# ----------------- CORE INFERENCE PER IMAGE -----------------
def detect_items_in_image(pil_image: Image.Image):
    w, h = pil_image.size
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    bboxes = outputs.pred_boxes
    probabilities = logits.softmax(-1)[0, :, :-1]
    scores, labels = probabilities.max(-1)

    # sleeve detection from part labels
    sleeve_height_ratio = 0.0
    rivet_count = 0
    garments = []

    for score, label, box in zip(scores, labels, bboxes[0]):
        raw_label = model.config.id2label[label.item()].lower()
        if "sleeve" in raw_label and score > 0.5:
            sleeve_height_ratio = max(sleeve_height_ratio, float(box[3].item()))
        if raw_label in GARMENT_LABELS and score > 0.4:
            garments.append({
                "raw_label": raw_label,
                "score": float(score.item()),
                "box": box.tolist()
            })
        elif raw_label in PART_LABELS and score > 0.5 and raw_label == "rivet":
            rivet_count += 1

    sleeve_type = "short" if sleeve_height_ratio < 0.2 else "long" if sleeve_height_ratio > 0.5 else "medium"

    detected_items = []
    for g in garments:
        final_label = choose_final_label([g["raw_label"]], sleeve_type)

        color_name = get_color_from_bbox(pil_image, g["box"], w, h)
        avg_rgb = get_avg_rgb_from_bbox(pil_image, g["box"], w, h)

        if final_label == "pants":
            if is_shorts_bbox(g["box"]):
                final_label = "shorts"
            else:
                blue_by_name = is_denim_color_name(color_name)
                blue_by_rgb = is_blue_dominant(avg_rgb)
                if (rivet_count >= 2 and (blue_by_name or blue_by_rgb)) or rivet_count >= 4:
                    final_label = "jeans"

        detected_items.append({
            "label": final_label,
            "color": color_name,
            "color_hex": rgb_to_hex(tuple(avg_rgb)),
            "bbox": g["box"],
            "avg_rgb": avg_rgb
        })

    # fallback: nothing detected
    if not detected_items:
        fallback_name = get_dominant_color(pil_image)
        avg_rgb = np.array(pil_image).mean(axis=(0, 1)).astype(int).tolist()
        if rivet_count >= 3:
            label = "jeans" if is_denim_color_name(fallback_name) or is_blue_dominant(avg_rgb) else "pants"
        else:
            label = "t-shirt" if (h / max(1, w)) < 1.5 else "pants"
        detected_items.append({
            "label": label,
            "color": fallback_name,
            "color_hex": rgb_to_hex(tuple(avg_rgb)),
            "bbox": [0.5, 0.5, 1.0, 1.0],
            "avg_rgb": avg_rgb
        })

    return detected_items


# ----------------- ROUTES -----------------
@app.get("/")
def home():
    return {"message": "Clothing detection API running with YOLOS-Fashionpedia!"}


@app.post("/detect")
async def detect(
    file: UploadFile = File(None),
    image: UploadFile = File(None),
    files: Optional[List[UploadFile]] = File(None)
):
    """
    Accepts: 'file' or 'image' (single) OR 'files' (list).
    - If exactly one file was uploaded, return {"detected_items":[...]}.
    - If multiple files were uploaded, return {"results":[...]}.
    """
    file_list: List[UploadFile] = []
    if files and len(files) > 0:
        file_list = files
    else:
        upl = file or image
        if upl is not None:
            file_list = [upl]

    if not file_list:
        raise HTTPException(
            status_code=422,
            detail="No file provided (expected 'file', 'image', or 'files')."
        )

    results_all = []
    for f in file_list:
        content = await f.read()
        if not content:
            results_all.append({"filename": getattr(f, "filename", None), "detected_items": []})
            continue

        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        detected_items = detect_items_in_image(pil_image)

        w, h = pil_image.size
        tall = (h / max(1, w)) > 1.4

        # refine full-image t-shirt fallback -> long-sleeve if sleeves detected
        if (
            len(detected_items) == 1
            and detected_items[0].get("label") in {"t-shirt", "unknown"}
            and detected_items[0].get("bbox") == [0.5, 0.5, 1.0, 1.0]
        ):
            if not tall and sleeve_likely_by_edges(pil_image, getattr(f, "filename", None)):
                detected_items[0]["label"] = "long-sleeve"

        results_all.append({
            "filename": getattr(f, "filename", None),
            "detected_items": detected_items
        })

    if len(results_all) == 1:
        return {"detected_items": results_all[0]["detected_items"]}
    else:
        return {"results": results_all}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
