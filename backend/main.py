"""
Virtual Closet — FastAPI backend  v4.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detection pipeline (priority order):
  1. Claude Haiku vision  — most accurate (set ANTHROPIC_API_KEY)
  2. GPT-4o-mini vision   — accurate alternative (set OPENAI_API_KEY)
  3. YOLOS-Fashionpedia   — local fallback, no API key needed
  4. Canvas heuristic     — last resort

AI outfit/match features work with either Anthropic or OpenAI key.
"""

import asyncio
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import webcolors
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from transformers import YolosForObjectDetection, YolosImageProcessor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ── Anthropic (primary vision) ─────────────────────────────────
try:
    import anthropic as _anthropic_module
    _ANT_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    if _ANT_KEY:
        _ant = _anthropic_module.AsyncAnthropic(api_key=_ANT_KEY)
        ANTHROPIC_AVAILABLE = True
    else:
        _ant = None
        ANTHROPIC_AVAILABLE = False
except (ImportError, Exception):
    _ant = None
    ANTHROPIC_AVAILABLE = False

# ── OpenAI (secondary vision) ──────────────────────────────────
try:
    from openai import AsyncOpenAI
    _OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
    if _OPENAI_KEY:
        _oai = AsyncOpenAI(api_key=_OPENAI_KEY)
        OPENAI_AVAILABLE = True
    else:
        _oai = None
        OPENAI_AVAILABLE = False
except (ImportError, Exception):
    _oai = None
    OPENAI_AVAILABLE = False

AI_AVAILABLE = ANTHROPIC_AVAILABLE or OPENAI_AVAILABLE


# ── App ────────────────────────────────────────────────────────
app = FastAPI(title="Virtual Closet API", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND = Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/app", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")


# ── YOLOS (local fallback, loaded lazily) ─────────────────────
_yolos_model     = None
_yolos_processor = None
YOLOS_MODEL_NAME = "valentinafeve/yolos-fashionpedia"

def _get_yolos():
    global _yolos_model, _yolos_processor
    if _yolos_model is None:
        print(f"[yolos] Loading {YOLOS_MODEL_NAME} …")
        _yolos_model     = YolosForObjectDetection.from_pretrained(YOLOS_MODEL_NAME, use_safetensors=True)
        _yolos_processor = YolosImageProcessor.from_pretrained(YOLOS_MODEL_NAME)
        _yolos_model.eval()
        print("[yolos] Ready.")
    return _yolos_model, _yolos_processor


# ══════════════════════════════════════════════════════════════
#  COLOUR UTILITIES
# ══════════════════════════════════════════════════════════════

def rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb[:3])


def closest_css_color(rgb: tuple) -> str:
    try:
        return webcolors.rgb_to_name(rgb, spec="css3")
    except ValueError:
        min_dist, closest = float("inf"), "gray"
        for hex_code, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r, g, b = webcolors.hex_to_rgb(hex_code)
            dist = (r - rgb[0]) ** 2 + (g - rgb[1]) ** 2 + (b - rgb[2]) ** 2
            if dist < min_dist:
                min_dist, closest = dist, name
        return closest


def hsvish_bucket(r: int, g: int, b: int) -> str:
    V      = max(r, g, b)
    chroma = V - min(r, g, b)
    S      = 0 if V == 0 else chroma / V

    if V >= 230 and S < 0.12:  return "white"

    # Dark teal: G and B both above R, and they're CLOSE to each other (abs < 25).
    # Navy has B >> G (abs ~40-50), so tightening this prevents navy → teal.
    if V < 120 and g > r + 12 and b > r + 12 and abs(g - b) < 25:
        return "teal"

    if V < 50:                 return "black"
    if V < 110 and S < 0.15:  return "black"
    if S < 0.12:               return "gray"

    # Bright teal: G and B both clearly dominate R, close to each other
    if g > r + 20 and b > r + 20 and abs(g - b) < 30:  return "teal"

    if r >= g and r >= b:
        # Tan/khaki/beige: R dominant, G clearly above B, saturation under 0.55.
        # Runs before pink/red — true red has S > 0.60 and G ≈ B; tan does not.
        if g > b + 15 and S < 0.55 and V > 85:  return "tan"
        if S < 0.32:                              return "pink"
        if b > 90 and S < 0.50:                  return "pink"
        return "red"

    if g >= r and g >= b:
        return "green"

    if b >= r and b >= g:
        if V < 80:  return "navy"
        # Purple: R close to G (lavender/violet) — G > R by 20+ means it's just blue
        if r > 140 and S > 0.25 and (g - r) < 20:  return "purple"
        return "blue"

    return "gray"


def np_kmeans(pixels: np.ndarray, k: int = 5, iters: int = 12, seed: int = 42):
    rng     = np.random.default_rng(seed)
    idx     = rng.choice(pixels.shape[0], size=min(k, pixels.shape[0]), replace=False)
    centers = pixels[idx].astype(np.float32)
    for _ in range(iters):
        d2     = ((pixels[:, None].astype(np.float32) - centers[None]) ** 2).sum(2)
        labels = d2.argmin(1)
        new_c  = np.array([
            pixels[labels == c].mean(0) if (labels == c).any()
            else pixels[rng.integers(0, pixels.shape[0])]
            for c in range(k)
        ], dtype=np.float32)
        if np.allclose(new_c, centers, atol=0.5):
            centers = new_c; break
        centers = new_c
    return centers, labels


def _is_skin_tone(r: float, g: float, b: float) -> bool:
    """Detect skin-tone pixels so they don't pollute garment color detection."""
    return (r > 90 and g > 50 and b > 20 and
            r > g > b and
            (r - b) > 30 and
            (r - g) < 90 and
            r < 240)


def dominant_color_from_image(pil_image: Image.Image) -> tuple:
    """Returns (color_name, hex_str) for the dominant non-background garment colour."""
    arr = np.asarray(pil_image.resize((100, 100)), dtype=np.float32).reshape(-1, 3)
    r_ch, g_ch, b_ch = arr[:, 0], arr[:, 1], arr[:, 2]
    sums   = arr.sum(1)
    vmax   = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    vmin   = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    chroma = vmax - vmin

    # Skin tones: R > G > B, strong R-B gap (>55), G not too far below R.
    # Threshold R-B > 55 to avoid catching tan/khaki clothing pixels (which have smaller gap).
    skin_mask = (
        (r_ch > 100) & (g_ch > 60) & (b_ch > 30) &
        (r_ch > g_ch) & (g_ch > b_ch) &
        ((r_ch - b_ch) > 55) &
        ((r_ch - g_ch) < 65) &
        (r_ch < 240)
    )
    bg_mask   = (sums > 660) | (sums < 45) | ((chroma < 20) & (sums > 320))
    keep      = ~skin_mask & ~bg_mask
    px        = arr[keep].astype(np.uint16)
    if len(px) < 50:
        # Fallback: just drop white/near-white
        px = arr[(sums > 80) & (sums < 700)].astype(np.uint16)
    if len(px) < 10:
        px = arr.astype(np.uint16)

    centers, labels = np_kmeans(px, k=5)
    vals, counts    = np.unique(labels, return_counts=True)

    # Pick most common cluster that isn't white or a product-photo gray
    sorted_idx = vals[np.argsort(-counts)]
    best = None
    for ci in sorted_idx:
        cr, cg, cb = map(int, centers[ci])
        cs = cr + cg + cb
        cc = max(cr, cg, cb) - min(cr, cg, cb)
        if cs < 650 and not (cc < 20 and cs > 320):  # skip white and mid-gray bg
            best = (cr, cg, cb)
            break
    if best is None:
        best = tuple(map(int, centers[sorted_idx[0]]))
    r, g, b = best

    name = hsvish_bucket(r, g, b)
    if name not in {"black", "gray", "white", "blue", "navy", "red", "green",
                    "pink", "teal", "purple", "tan", "cream"}:
        name = closest_css_color((r, g, b))
    return name, rgb_to_hex((r, g, b))


def detect_pattern(pil_image: Image.Image) -> str:
    """Detect solid / striped / polka dot using row+column variance analysis."""
    small = np.array(pil_image.resize((80, 80)), dtype=np.float32)

    bg = small[:8, :8].mean(axis=(0, 1))
    flat = small.reshape(-1, 3)
    bg_dist = np.linalg.norm(flat - bg, axis=1)

    masked = small.copy()
    masked[bg_dist.reshape(80, 80) <= 25] = np.nan

    row_means = np.nanmean(masked, axis=1)   # (80, 3)
    col_means = np.nanmean(masked, axis=0)   # (80, 3)
    row_std   = np.nanstd(row_means, axis=0).mean()
    col_std   = np.nanstd(col_means, axis=0).mean()

    STRIPE_T = 18.0

    # Both axes vary → polka dots (circular spots create variance in both directions)
    if row_std > STRIPE_T and col_std > STRIPE_T:
        return "polka dot"

    # One axis varies strongly → stripes
    if row_std > STRIPE_T or col_std > STRIPE_T:
        return "striped"

    return "solid"


# ══════════════════════════════════════════════════════════════
#  VISION PROMPT  (shared by Anthropic + OpenAI)
# ══════════════════════════════════════════════════════════════

_VISION_PROMPT = """You are an expert clothing analyst. Examine this clothing image carefully and return a JSON object.

Be VERY precise about these details:

GARMENT TYPE — look at the exact item:
Tops:
• Short sleeves ending above/at elbow → "t-shirt"
• Long sleeves reaching the wrist → "long-sleeve"
• No sleeves → "tank top"
• Collar + short sleeves → "polo"
• With hood → "hoodie"
• Zip-up/button jacket → "jacket"
• Longer outerwear → "coat"

Bottoms / full:
• One-piece dress → "dress"
• Separate bottom flared/flowing → "skirt"
• Full-length leg coverage → "pants" or "jeans" (if denim)
• Knee-length or above leg coverage → "shorts"

Shoes — pick the most specific match:
• Running/athletic lace-up with rubber sole → "sneakers"
• Y-strap between toes, open sole → "flip flops"
• Open strappy warm-weather shoe (not flip flop) → "sandals"
• Ankle-height boot → "ankle boots"
• Knee-high or over-knee boot → "boots"
• Chunky block heel or stiletto → "heels"
• Pointed-toe elegant pump → "heels"
• Slip-on with no heel, casual → "loafers"
• Simple slim flat slip-on dress shoe → "flats"
• Open backless slider/pool slide → "slides"
• Formal lace-up Oxford/Derby → "oxford shoes"
• Platform sole shoe → "platform shoes"
• Any other shoe → "shoes"

Accessories — pick the most specific match:
• Waist/trouser strap with buckle → "belt"
• Wide-brim floppy fabric hat (beach/sun) → "sun hat"
• Rounded-top bucket-style hat → "bucket hat"
• Cowboy/western hat with pinched crown → "cowboy hat"
• Structured brim + flat top (fedora shape) → "fedora"
• Soft beret / flat cap / newsboy cap → "beret" or "flat cap"
• Knit stretchy winter hat → "beanie"
• Baseball-style cap with curved brim → "baseball cap"
• Any other hat with brim → "hat"
• Lightweight neck wrap → "scarf"
• Small flat hand-held bag with no strap → "clutch"
• Small bag worn across the body on a long thin strap → "crossbody bag"
• Bag hanging from one shoulder with a short strap → "shoulder bag"
• Pouch worn around the waist / hips → "fanny pack"
• Large cylindrical soft bag for travel/gym → "duffle bag"
• Very small decorative bag → "mini bag"
• Large open-top shopping/carry bag → "tote bag"
• Bag with two straps worn on the back → "backpack"
• Medium structured hand/arm bag → "handbag"
• Generic carried bag → "bag"
• Dark tinted lens eyewear → "sunglasses"
• Clear lens eyewear → "glasses"
• Wrist timepiece → "watch"
• Chain/pendant around neck → "necklace"
• Ear jewelry → "earrings"
• Wrist chain/cuff → "bracelet"
• Finger band → "ring"
• General accessory → "accessory"

COLOR — be specific:
• Pure white = "white", off-white/cream = "cream"
• Light blue = "light blue", dark blue = "navy", medium blue = "blue"
• Blue-green = "teal", yellow-green = "olive"
• Tan/khaki/beige = "tan", light brown = "beige"
• List ALL significant colors in the "colors" array (up to 3). Put the dominant one first and also in "color".

PATTERN — be specific:
• One uniform color throughout = "solid"
• Alternating parallel bands/lines of color = "striped"
• Intersecting horizontal and vertical lines forming a grid = "plaid"
• Small repeated circular dots = "polka dot"
• Flower or botanical prints = "floral"
• Logo, text, or image screen-printed on = "graphic"
• Irregular swirled or marbled colors = "abstract"
• Rainbow swirl dye pattern = "tie-dye"
• Spotted animal skin pattern (leopard, zebra, etc.) = "animal print"
• Military-style irregular color patches = "camouflage"

Return ONLY this JSON, no other text:
{
  "type": <one of: t-shirt, long-sleeve, shirt, blouse, tank top, crop top, polo, hoodie, sweatshirt, sweater, cardigan, jacket, coat, vest, blazer, dress, skirt, pants, jeans, shorts, leggings, jumpsuit, overalls, sneakers, flip flops, sandals, ankle boots, boots, heels, loafers, flats, slides, oxford shoes, platform shoes, shoes, belt, sun hat, bucket hat, cowboy hat, fedora, beret, flat cap, beanie, baseball cap, hat, scarf, clutch, crossbody bag, shoulder bag, fanny pack, duffle bag, mini bag, tote bag, backpack, handbag, bag, sunglasses, glasses, watch, necklace, earrings, bracelet, ring, accessory, other>,
  "color": <dominant color in plain English>,
  "color_hex": <hex code for dominant color, e.g. "#2e5fa3">,
  "colors": [<all significant colors as plain English strings, dominant first, up to 3>],
  "colors_hex": [<hex code for each color in the colors array>],
  "pattern": <one of: solid, striped, plaid, polka dot, floral, graphic, animal print, abstract, tie-dye, camouflage, other>,
  "notes": <one sentence describing the item>
}"""


def _extract_json(text: str) -> dict:
    """Parse JSON from model response, with regex fallback."""
    text = text.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group())
    return json.loads(text)


# ══════════════════════════════════════════════════════════════
#  VISION DETECTION — ANTHROPIC (primary)
# ══════════════════════════════════════════════════════════════

async def _vision_detect_anthropic(image_bytes: bytes, mime: str = "image/jpeg") -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    for attempt in range(3):
        try:
            msg = await _ant.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": mime, "data": b64},
                        },
                        {"type": "text", "text": _VISION_PROMPT},
                    ],
                }],
            )
            return _extract_json(msg.content[0].text)
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "529" in err or "overload" in err:
                await asyncio.sleep(2 ** (attempt + 1))
            else:
                raise
    raise RuntimeError("Anthropic vision failed after 3 attempts")


# ══════════════════════════════════════════════════════════════
#  VISION DETECTION — OPENAI (secondary)
# ══════════════════════════════════════════════════════════════

async def _vision_detect_openai(image_bytes: bytes, mime: str = "image/jpeg") -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    for attempt in range(3):
        try:
            response = await _oai.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": _VISION_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url":    f"data:{mime};base64,{b64}",
                            "detail": "high",
                        }},
                    ],
                }],
                max_tokens=400,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err or "overload" in err:
                await asyncio.sleep(2 ** (attempt + 1))
            else:
                raise
    raise RuntimeError("OpenAI vision failed after 3 attempts")


# ══════════════════════════════════════════════════════════════
#  YOLOS FALLBACK DETECTION
# ══════════════════════════════════════════════════════════════

YOLOS_GARMENT_MAP = {
    "top, t-shirt, sweatshirt": "t-shirt",
    "shirt, blouse":            "shirt",
    "pants":                    "pants",
    "jeans":                    "jeans",
    "shorts":                   "shorts",
    "skirt":                    "skirt",
    "dress":                    "dress",
    "jacket":                   "jacket",
    "coat":                     "coat",
    "cardigan":                 "cardigan",
    "vest":                     "vest",
    "sweater":                  "sweater",
    "jumpsuit":                 "jumpsuit",
    "cape":                     "jacket",
    # Shoes
    "shoe":                     "shoes",
    "boot":                     "boots",
    "sandal":                   "sandals",
    "sneaker":                  "sneakers",
    "pump":                     "heels",
    "heel":                     "heels",
    "loafer":                   "loafers",
    "slipper":                  "slides",
    # Accessories
    "belt":                     "belt",
    "hat":                      "hat",
    "cap":                      "cap",
    "scarf":                    "scarf",
    "bag, wallet":              "bag",
    "backpack":                 "backpack",
    "glasses":                  "sunglasses",
    "glove":                    "accessory",
    "watch":                    "watch",
    "tie":                      "accessory",
    "headband, head covering, hair accessory": "accessory",
}


def _yolos_detect(pil_image: Image.Image) -> dict:
    model, processor = _get_yolos()
    img_w, img_h     = pil_image.size

    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probs             = outputs.logits.softmax(-1)[0, :, :-1]
    scores, label_ids = probs.max(-1)
    bboxes            = outputs.pred_boxes[0]

    best_garment = None
    rivet_count  = 0

    for sc, lid, box in zip(scores, label_ids, bboxes):
        raw  = model.config.id2label[lid.item()]
        sc_f = float(sc.item())
        if raw in YOLOS_GARMENT_MAP and sc_f > 0.35:
            if best_garment is None or sc_f > best_garment["score"]:
                best_garment = {"raw": raw, "score": sc_f, "box": box.tolist()}
        elif raw == "rivet" and sc_f > 0.5:
            rivet_count += 1

    if best_garment is None:
        color_name, hex_str = dominant_color_from_image(pil_image)
        ratio = img_h / max(img_w, 1)
        garment_type = "pants" if ratio > 1.5 else "t-shirt"
        pattern = detect_pattern(pil_image)
        return {"type": garment_type, "color": color_name, "color_hex": hex_str,
                "pattern": pattern, "notes": ""}

    mapped = YOLOS_GARMENT_MAP[best_garment["raw"]]
    box    = best_garment["box"]

    color_name, hex_str = _color_from_bbox(pil_image, box, img_w, img_h)
    pattern = _pattern_from_bbox(pil_image, box, img_w, img_h)

    # Sleeve length cannot be reliably detected from pixels alone.
    # Set ANTHROPIC_API_KEY or OPENAI_API_KEY for accurate sleeve detection.

    if mapped == "pants":
        bh = float(box[3])
        if bh < 0.50:
            mapped = "shorts"
        elif rivet_count >= 2 and color_name in {"blue", "navy", "light blue", "indigo"}:
            mapped = "jeans"

    # If YOLOS said "jeans" but color isn't blue, it's probably just dark pants
    if mapped == "jeans" and color_name not in {"blue", "navy", "light blue", "indigo",
                                                  "darkblue", "midnightblue", "steelblue"}:
        mapped = "pants"

    return {"type": mapped, "color": color_name, "color_hex": hex_str,
            "pattern": pattern, "notes": ""}


def _color_from_bbox(pil_image: Image.Image, bbox: list, img_w: int, img_h: int) -> tuple:
    cx, cy, bw, bh = bbox
    # Crop aggressively inward (30%) to avoid model skin/background bleed at edges
    shrink = 0.30
    x0 = max(0, int((cx - bw * (0.5 - shrink)) * img_w))
    y0 = max(0, int((cy - bh * (0.5 - shrink)) * img_h))
    x1 = min(img_w, int((cx + bw * (0.5 - shrink)) * img_w))
    y1 = min(img_h, int((cy + bh * (0.5 - shrink)) * img_h))
    if x1 <= x0 or y1 <= y0:
        return dominant_color_from_image(pil_image)
    return dominant_color_from_image(pil_image.crop((x0, y0, x1, y1)))


def _pattern_from_bbox(pil_image: Image.Image, bbox: list, img_w: int, img_h: int) -> str:
    cx, cy, bw, bh = bbox
    shrink = 0.15
    x0 = max(0, int((cx - bw * (0.5 - shrink)) * img_w))
    y0 = max(0, int((cy - bh * (0.5 - shrink)) * img_h))
    x1 = min(img_w, int((cx + bw * (0.5 - shrink)) * img_w))
    y1 = min(img_h, int((cy + bh * (0.5 - shrink)) * img_h))
    crop = pil_image.crop((x0, y0, x1, y1)) if x1 > x0 and y1 > y0 else pil_image
    return detect_pattern(crop)


def _sleeve_type(pil_image: Image.Image, bbox: list) -> str:
    """
    Compare garment coverage at shoulder level vs mid-arm level.
    Long-sleeve shirts stay wide throughout; t-shirts narrow to just the torso below the sleeve stub.
    """
    cx, cy, bw, bh = bbox
    w, h = pil_image.size
    x0 = max(0, int((cx - bw / 2) * w)); x1 = min(w, int((cx + bw / 2) * w))
    y0 = max(0, int((cy - bh / 2) * h)); y1 = min(h, int((cy + bh / 2) * h))
    if x1 - x0 < 60 or y1 - y0 < 60:
        return "short"

    crop    = np.array(pil_image.crop((x0, y0, x1, y1)).resize((100, 100)))
    corners = np.vstack([crop[:8, :8], crop[:8, 92:],
                         crop[92:, :8], crop[92:, 92:]]).reshape(-1, 3).astype(float)
    bg      = corners.mean(0)

    def coverage(strip):
        diffs = np.linalg.norm(strip.reshape(-1, 3).astype(float) - bg, axis=1)
        return (diffs > 30).mean()

    shoulder = coverage(crop[10:25])   # top 10-25 %: shoulder + top-of-sleeve
    mid_arm  = coverage(crop[40:60])   # mid 40-60 %: where long sleeve hangs, t-shirt torso only

    if shoulder < 0.30:
        return "short"   # too little garment to judge

    # Long-sleeve: mid-arm nearly as wide as shoulder (ratio > 0.72)
    # T-shirt: mid-arm is just the narrow torso (ratio < 0.72)
    return "long" if (mid_arm / shoulder) > 0.72 else "short"


# ══════════════════════════════════════════════════════════════
#  MAIN DETECTION ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

def _mime_from_bytes(image_bytes: bytes) -> str:
    if image_bytes[:4] == b"\x89PNG":    return "image/png"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"): return "image/gif"
    if image_bytes[:4] == b"RIFF":      return "image/webp"
    return "image/jpeg"


def _normalise(result: dict, source: str) -> dict:
    primary_color = result.get("color", "unknown").lower().strip()
    primary_hex   = result.get("color_hex", "#cccccc")
    raw_colors    = result.get("colors", [])
    raw_hexes     = result.get("colors_hex", [])

    colors     = [c.lower().strip() for c in raw_colors] if raw_colors else [primary_color]
    colors_hex = list(raw_hexes) if raw_hexes else [primary_hex]

    # Keep arrays same length
    while len(colors_hex) < len(colors):
        colors_hex.append("#cccccc")

    return {
        "label":      result.get("type", "unknown").lower().strip(),
        "color":      colors[0] if colors else primary_color,
        "color_hex":  colors_hex[0] if colors_hex else primary_hex,
        "colors":     colors,
        "colors_hex": colors_hex,
        "pattern":    result.get("pattern", "solid").lower().strip(),
        "notes":      result.get("notes", ""),
        "source":     source,
    }


async def detect_clothing(image_bytes: bytes) -> dict:
    mime = _mime_from_bytes(image_bytes)

    # 1. Claude Haiku vision
    if ANTHROPIC_AVAILABLE and _ant is not None:
        try:
            result = await _vision_detect_anthropic(image_bytes, mime)
            return _normalise(result, "claude-vision")
        except Exception as e:
            print(f"[claude-vision] failed ({e}), trying next")

    # 2. GPT-4o-mini vision
    if OPENAI_AVAILABLE and _oai is not None:
        try:
            result = await _vision_detect_openai(image_bytes, mime)
            return _normalise(result, "gpt-vision")
        except Exception as e:
            print(f"[gpt-vision] failed ({e}), trying YOLOS")

    # 3. YOLOS local model
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        r   = _yolos_detect(pil)
        return _normalise(r, "yolos")
    except Exception as e:
        print(f"[yolos] failed ({e}), using heuristic")

    # 4. Heuristic last resort
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        color_name, hex_str = dominant_color_from_image(pil)
        w, h = pil.size
        label   = "pants" if h / max(w, 1) > 1.5 else "t-shirt"
        pattern = detect_pattern(pil)
        return {"label": label, "color": color_name, "color_hex": hex_str,
                "pattern": pattern, "notes": "", "source": "heuristic"}
    except Exception:
        return {"label": "unknown", "color": "unknown", "color_hex": "#cccccc",
                "pattern": "solid", "notes": "", "source": "error"}


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["health"])
def health():
    return {
        "status":             "ok",
        "anthropic_vision":   ANTHROPIC_AVAILABLE,
        "openai_vision":      OPENAI_AVAILABLE,
        "yolos_loaded":       _yolos_model is not None,
    }


@app.post("/detect", tags=["inference"])
async def detect(
    file:  UploadFile = File(None),
    image: UploadFile = File(None),
    files: Optional[List[UploadFile]] = File(None),
):
    file_list: List[UploadFile] = []
    if files:
        file_list = list(files)
    elif file or image:
        file_list = [f for f in (file, image) if f is not None]

    if not file_list:
        raise HTTPException(422, "No file provided. Use 'file', 'image', or 'files'.")

    all_results = []
    for i, upload in enumerate(file_list):
        if i > 0 and AI_AVAILABLE:
            await asyncio.sleep(0.3)

        raw_bytes = await upload.read()
        if not raw_bytes:
            all_results.append({"filename": upload.filename, "detected_items": []})
            continue
        try:
            item = await detect_clothing(raw_bytes)
            all_results.append({"filename": upload.filename, "detected_items": [item]})
        except Exception as exc:
            err_msg = str(exc)
            if "rate" in err_msg.lower() or "429" in err_msg:
                err_msg = "rate_limit"
            all_results.append({
                "filename": upload.filename,
                "error":    err_msg,
                "detected_items": [],
            })

    if len(all_results) == 1:
        return all_results[0]
    return {"results": all_results}


# ══════════════════════════════════════════════════════════════
#  AI OUTFIT + MATCH  (works with Anthropic or OpenAI)
# ══════════════════════════════════════════════════════════════

class ItemPayload(BaseModel):
    id:      int
    type:    str = "unknown"
    color:   str = "unknown"
    pattern: str = ""
    event:   str = ""
    tags:    str = ""

class SuggestRequest(BaseModel):
    items:       List[ItemPayload]
    occasion:    str   = ""
    time_of_day: str   = ""
    weather:     str   = ""
    temp_f:      Optional[float] = None

class ItemAttributes(BaseModel):
    type:    str = "unknown"
    color:   str = "unknown"
    pattern: str = ""
    event:   str = ""

class MatchRequest(BaseModel):
    item_a: ItemAttributes
    item_b: ItemAttributes


def _require_ai():
    if not AI_AVAILABLE:
        raise HTTPException(
            503,
            "No AI key set. Set ANTHROPIC_API_KEY or OPENAI_API_KEY and restart."
        )


async def _chat(system: str, user: str, temperature: float = 0.7) -> str:
    """Run a text chat using whichever AI backend is available."""
    if ANTHROPIC_AVAILABLE and _ant is not None:
        msg = await _ant.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    if OPENAI_AVAILABLE and _oai is not None:
        r = await _oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return r.choices[0].message.content

    raise RuntimeError("No AI backend available")


@app.post("/suggest", tags=["ai"])
async def suggest(req: SuggestRequest) -> Dict[str, Any]:
    _require_ai()
    if not req.items:
        raise HTTPException(422, "items list is empty")

    # Build context string from occasion / time / weather
    ctx_parts = []
    if req.occasion:    ctx_parts.append(f"Occasion: {req.occasion}")
    if req.time_of_day: ctx_parts.append(f"Time of day: {req.time_of_day}")
    if req.weather:     ctx_parts.append(f"Current weather: {req.weather}")
    if req.temp_f is not None:
        feel = ("very cold" if req.temp_f < 32 else
                "cold"      if req.temp_f < 50 else
                "cool"      if req.temp_f < 65 else
                "warm"      if req.temp_f < 80 else "hot")
        ctx_parts.append(f"Feel: {feel}")
    context = " | ".join(ctx_parts) if ctx_parts else "No specific context"

    sys_prompt = (
        "You are a professional fashion stylist. Pick 2-5 items from the wardrobe that work "
        "well together given the context below. Build a complete outfit:\n"
        "• Include a top + bottom (or full-body piece like a dress)\n"
        "• Add a layer/jacket if weather is cold or cool\n"
        "• Include shoes if any are available in the wardrobe\n"
        "• Include one accessory (belt, bag, hat, etc.) if it completes the look\n"
        "• Weather & temperature: layers for cold/cool, light pieces for warm/hot, waterproof for rain\n"
        "• Time of day: relaxed for morning, put-together for afternoon, elevated for evening/night\n"
        "• Occasion: match formality — casual for errands, polished for work, dressed-up for formal\n"
        "• Colour harmony: complementary or matching tones\n\n"
        "Return ONLY JSON with keys:\n"
        '  "outfit": [{id, reason_for_pick}, ...]\n'
        '  "reason": 1-2 sentences explaining why this outfit fits the context.\n'
        "Only use item IDs from the list provided."
    )
    user_msg = f"Context: {context}\n\nWardrobe:\n{json.dumps([i.model_dump() for i in req.items])}"

    try:
        raw  = await _chat(sys_prompt, user_msg)
        data = _extract_json(raw)
        valid = {i.id for i in req.items}
        data["outfit"] = [o for o in data.get("outfit", []) if isinstance(o.get("id"), int) and o["id"] in valid]
        return data
    except Exception as e:
        raise HTTPException(502, str(e))


@app.post("/match", tags=["ai"])
async def match(req: MatchRequest) -> Dict[str, Any]:
    _require_ai()
    sys = (
        "You are a fashion stylist. Rate how well two items go together. Return JSON:\n"
        '  "rating": "Great Match" | "Works" | "Clash"\n'
        '  "explanation": 1-2 sentences\n'
        '  "tips": [] or up to 2 short tips\n'
        "Return ONLY JSON."
    )
    try:
        raw  = await _chat(sys, f"A: {req.item_a.model_dump()}\nB: {req.item_b.model_dump()}", 0.4)
        data = _extract_json(raw)
        r    = data.get("rating", "Works").lower()
        data["rating"] = "Great Match" if "great" in r else "Clash" if "clash" in r else "Works"
        data.setdefault("explanation", "")
        data.setdefault("tips", [])
        return data
    except Exception as e:
        raise HTTPException(502, str(e))


# ══════════════════════════════════════════════════════════════
#  INSPO BOARD  (Pinterest proxy · AI analysis · outfit matching · shopping)
# ══════════════════════════════════════════════════════════════

def _extract_og_image(html: str) -> Optional[str]:
    for pat in [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'"og:image"\s*:\s*"([^"]+)"',
    ]:
        m = re.search(pat, html)
        if m:
            return m.group(1).replace("&amp;", "&").replace("\\/", "/")
    return None


def _extract_board_images(html: str, max_images: int = 20) -> List[str]:
    """Pull all 736x pin image URLs from Pinterest board page HTML."""
    seen: set = set()
    urls: List[str] = []
    # Pinterest embeds image URLs both as-is and JSON-escaped (\/), so match both
    for pat in [
        r'https?:\\?/\\?/i\.pinimg\.com\\?/736x\\?/[^"\'>\s\\]+\.(?:jpg|jpeg|png|webp)',
        r'https?:\\?/\\?/i\.pinimg\.com\\?/originals\\?/[^"\'>\s\\]+\.(?:jpg|jpeg|png|webp)',
        r'https?:\\?/\\?/i\.pinimg\.com\\?/[^"\'>\s\\]+\.(?:jpg|jpeg|png|webp)',
    ]:
        for m in re.finditer(pat, html, re.IGNORECASE):
            cleaned = m.group(0).replace("\\/", "/")
            # Skip tiny thumbnails (60x, 30x)
            if re.search(r'/(?:30x|60x|75x)/', cleaned):
                continue
            if cleaned not in seen:
                seen.add(cleaned)
                urls.append(cleaned)
        if len(urls) >= max_images:
            break
    return urls[:max_images]


async def _fetch_as_b64(client, image_url: str, ua: str) -> Optional[str]:
    try:
        r    = await client.get(image_url, headers={"User-Agent": ua})
        ct   = r.headers.get("content-type", "")
        mime = "image/png" if "png" in ct else "image/webp" if "webp" in ct else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(r.content).decode()}"
    except Exception:
        return None


@app.get("/fetch-image", tags=["utility"])
async def fetch_image(url: str):
    """
    Fetch images from a Pinterest URL.
    - Single pin  → returns one image in the `images` list.
    - Board / any other Pinterest page → returns all pin images found (up to 20).
    Response: {"images": [{"image_b64": "data:image/jpeg;base64,..."}, ...]}
    """
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    req_headers = {
        "User-Agent":      ua,
        "Accept":          "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(follow_redirects=True, timeout=20,
                                      headers=req_headers) as client:
            page = await client.get(url)
        html = page.text

        from urllib.parse import urlparse as _urlparse
        is_single_pin = bool(re.search(r'/pin/\d+', _urlparse(url).path))

        if is_single_pin:
            image_url = _extract_og_image(html)
            if not image_url:
                raise HTTPException(404, "No image found for this pin")
            async with _httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
                b64 = await _fetch_as_b64(client, image_url, ua)
            if not b64:
                raise HTTPException(502, "Could not download pin image")
            return {"images": [{"image_b64": b64}]}

        # Board / search / profile → extract all pin images
        board_urls = _extract_board_images(html)

        if not board_urls:
            # Fallback: at least grab the og:image
            og = _extract_og_image(html)
            if og:
                board_urls = [og]
            else:
                raise HTTPException(404, "No images found. Try a direct pin URL instead.")

        results = []
        async with _httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            for img_url in board_urls:
                b64 = await _fetch_as_b64(client, img_url, ua)
                if b64:
                    results.append({"image_b64": b64})

        if not results:
            raise HTTPException(502, "Found image URLs but could not download any")
        return {"images": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, str(e))


_INSPO_PROMPT = (
    "Analyze this outfit inspiration image. Return JSON only:\n"
    '{\n'
    '  "vibe": "one short aesthetic label e.g. euro summer, quiet luxury, clean girl, streetwear, coastal grandmother, dark academia, boho, preppy",\n'
    '  "pieces": ["piece 1", "piece 2", "piece 3"],\n'
    '  "colors": ["color1", "color2", "color3"],\n'
    '  "colors_hex": ["#hex1", "#hex2", "#hex3"],\n'
    '  "style_notes": "one sentence describing the overall look"\n'
    "}"
)


class InspoRequest(BaseModel):
    image: str  # base64 data URL


@app.post("/analyze-inspo", tags=["ai"])
async def analyze_inspo(req: InspoRequest) -> Dict[str, Any]:
    """Run Claude Sonnet vision on an inspo image to extract vibe, pieces, colors, and style notes."""
    _require_ai()
    b64, mime = req.image, "image/jpeg"
    if "," in b64:
        header, b64 = b64.split(",", 1)
        if "png"  in header: mime = "image/png"
        elif "webp" in header: mime = "image/webp"

    if ANTHROPIC_AVAILABLE and _ant is not None:
        try:
            msg = await _ant.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text",  "text": _INSPO_PROMPT},
                ]}],
            )
            return _extract_json(msg.content[0].text)
        except Exception as e:
            if not (OPENAI_AVAILABLE and _oai):
                raise HTTPException(502, str(e))

    if OPENAI_AVAILABLE and _oai is not None:
        try:
            r = await _oai.chat.completions.create(
                model="gpt-4o-mini", temperature=0.1,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": [
                    {"type": "text",      "text": _INSPO_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                ]}],
                max_tokens=400,
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            raise HTTPException(502, str(e))

    raise HTTPException(503, "No AI key set")


class InspoAnalysis(BaseModel):
    vibe:   str       = ""
    pieces: List[str] = []
    colors: List[str] = []


class MatchInspoRequest(BaseModel):
    inspo:       InspoAnalysis
    closet:      List[Dict[str, Any]] = []
    inspo_image: Optional[str] = None   # base64 data URL of the inspo photo


_MATCH_INSPO_PROMPT = """\
You are a brutally honest personal stylist reviewing someone's wardrobe against an outfit photo.
Your job is to save them money — only tell them they own something if it is virtually identical \
to what is in the photo. When in doubt, always recommend shopping.

Study the photo carefully, then check the closet list below.

━━━ THE ONE TEST THAT MATTERS ━━━
Would a real stylist look at the closet item and the inspo piece and say
"yes, wear what you have — it's the same thing"?
  → YES → closet_item match
  → NO, they'd say "you need to buy this specific thing" → gap + shop

━━━ WHEN TO MATCH (all three must be true) ━━━
  ✓ Same garment category (see aliases below)
  ✓ Same shade — not just same base color (see shade rules below)
  ✓ Same silhouette / sleeve length (see silhouette rules below)
If ANY of the three differ → GAP → shop. No exceptions.

━━━ GARMENT CATEGORY ALIASES ━━━
These count as the same type:
  • tank top = cami = spaghetti strap = sleeveless top
  • t-shirt = tee = short-sleeve top (but NOT long-sleeve or sleeveless)
  • sneakers = trainers = athletic shoes
  • jeans = denim pants (but NOT leggings, trousers, or dress pants)
  • boots = ankle boots (same height only)

━━━ SHADE RULES (must be same shade, not just same color) ━━━
  • Light blue ≠ dark blue ≠ bright blue ≠ navy
  • Light wash denim ≠ medium wash ≠ dark wash denim
  • White ≠ cream ≠ off-white
  • Black ≠ charcoal ≠ dark gray
  • Bright red ≠ burgundy ≠ wine
  • Tan ≠ camel ≠ brown
  Same shade = the two items could swap and no one would notice the color difference.

━━━ SILHOUETTE RULES ━━━
BOTTOMS — cut must match exactly:
  • Wide leg ≠ straight ≠ skinny ≠ slim ≠ flare ≠ bootcut ≠ cargo ≠ baggy ≠ boyfriend
  • Midi skirt ≠ mini skirt ≠ maxi skirt ≠ A-line ≠ pencil skirt

TOPS — sleeve length must match exactly:
  • Sleeveless / tank ≠ short sleeve (above elbow) ≠ long sleeve (wrist) ≠ crop (midriff)

━━━ ALWAYS SHOP FOR THESE — NO MATCH POSSIBLE ━━━
  • Statement pieces with distinctive details: fringe, corset boning, snake/animal print, \
lace-up, cargo pockets, cutouts, ruching, tie-dye, sequins, sheer panels, structured boning, \
embroidery, patchwork, distressing. A plain version of that garment is NOT a match.
  • Matching sets: if the photo shows a co-ord / matching set (same fabric+print top+bottom), \
the user must own the matching piece from that exact set. Having just one piece = gap for the other.

━━━ BAGS & ACCESSORIES (looser rules) ━━━
  • Match on color family + general type only.
  • A tan shoulder bag matches any tan bag (crossbody, tote, etc.).
  • Color shade flexibility allowed for accessories.

Closet:
{closet_lines}

Return ONLY valid JSON — no commentary before or after:
{{
  "matches": [
    {{
      "piece": "<exact description of what you see in the photo>",
      "closet_item": {{"type": "...", "color": "...", "index": <number>}} or null,
      "gap_query": "<3-6 words: [shade] [silhouette] [garment type] ONLY. \
Zero vibe/style/aesthetic words. Examples: 'dark wash baggy cargo jeans', \
'grey midi skirt', 'white short sleeve fitted tee'. NOT 'grey skirt minimalist clean'>" or null
    }}
  ]
}}
Rules: closet_item=non-null means match found. gap_query=non-null means send to shopping. \
Never set both non-null for the same piece. Never set both null.\
"""


@app.post("/match-inspo", tags=["ai"])
async def match_inspo(req: MatchInspoRequest) -> Dict[str, Any]:
    """Match each inspo outfit piece to a closet item or flag as a shopping gap using vision."""
    _require_ai()

    closet_lines = "\n".join(
        f"  [{i['index']}] {i['color']} {i['type']}"
        + (f" ({i['pattern']})" if i.get("pattern") else "")
        for i in req.closet
    )
    prompt_text = _MATCH_INSPO_PROMPT.format(closet_lines=closet_lines)

    try:
        # ── Vision path: send the actual inspo photo so Claude sees exact colors/styles ──
        if req.inspo_image and ANTHROPIC_AVAILABLE and _ant is not None:
            b64, mime = req.inspo_image, "image/jpeg"
            if "," in b64:
                header, b64 = b64.split(",", 1)
                if "png"  in header: mime = "image/png"
                elif "webp" in header: mime = "image/webp"
            msg = await _ant.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1200,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text",  "text": prompt_text},
                ]}],
            )
            return _extract_json(msg.content[0].text)

        # ── Text-only fallback (no image or no Anthropic) ──
        text_prompt = (
            f"Outfit vibe: {req.inspo.vibe}\n"
            f"Pieces: {', '.join(req.inspo.pieces)}\n"
            f"Colors: {', '.join(req.inspo.colors)}\n\n"
            + prompt_text
        )

        if ANTHROPIC_AVAILABLE and _ant is not None:
            msg = await _ant.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1200,
                messages=[{"role": "user", "content": text_prompt}],
            )
            return _extract_json(msg.content[0].text)

        if OPENAI_AVAILABLE and _oai is not None:
            r = await _oai.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a fashion stylist. Return only valid JSON."},
                    {"role": "user",   "content": text_prompt},
                ],
                max_tokens=1200,
            )
            return json.loads(r.choices[0].message.content)

    except Exception as e:
        raise HTTPException(502, str(e))

    raise HTTPException(503, "No AI key set")


class GenerateOutfitsRequest(BaseModel):
    occasion: str = ""
    vibe: str = ""
    store: str = ""
    closet: List[Dict] = []       # [{type, color, pattern}]
    inspo_vibes: List[str] = []   # aesthetic labels from inspo board
    inspo_colors: List[str] = []  # color names from inspo board


_GENERATE_OUTFITS_PROMPT = """\
You are a personal stylist generating shoppable outfit ideas for a specific person.

THEIR WARDROBE (what they already own):
{closet_section}

THEIR STYLE AESTHETIC (from their inspo board):
{inspo_section}

FILTERS:
Occasion: {occasion}
Vibe filter: {vibe}
{store_line}

Generate 4 complete shoppable outfit ideas that feel authentic to THEIR personal style and \
complement THEIR existing wardrobe colors and aesthetic. If they have a strong color palette \
in their closet, build outfits around those colors. If their inspo board shows a clear aesthetic, \
lean into it. Make the outfits feel like they were curated for this specific person, not generic.

Return ONLY this JSON — no text before or after:
{{
  "outfits": [
    {{
      "name": "Short outfit name e.g. Sunday Market Look",
      "vibe": "Aesthetic label e.g. Clean Girl",
      "pieces": [
        {{ "role": "Top",    "search_query": "white oversized linen button shirt" }},
        {{ "role": "Bottom", "search_query": "wide leg beige linen trouser" }},
        {{ "role": "Shoes",  "search_query": "brown leather loafer women" }},
        {{ "role": "Bag",    "search_query": "tan leather tote bag" }}
      ]
    }}
  ]
}}

Rules for search_query:
- Format: [color/shade] [style adjective] [garment type]
- Be specific enough to find real products on Google Shopping
- NO brand names, NO aesthetic/vibe words ("clean girl", "minimalist", etc.)
- Good examples: "dark wash straight leg jeans", "strappy black heeled sandal", "ribbed cream tank top"
- Include 3-4 pieces per outfit (top, bottom, shoes, and optionally bag or accessory)
- Make all 4 outfits distinct from each other in color palette and silhouette\
"""


@app.post("/generate-outfits", tags=["ai"])
async def generate_outfits(req: GenerateOutfitsRequest) -> Dict[str, Any]:
    """Use Claude to generate outfit concepts personalised to the user's closet + inspo board."""
    _require_ai()

    # ── Build closet section ──────────────────────────────────────────────────
    if req.closet:
        from collections import Counter
        color_counts = Counter(i.get("color", "") for i in req.closet if i.get("color"))
        top_colors   = [c for c, _ in color_counts.most_common(8)]
        type_counts  = Counter(i.get("type",  "") for i in req.closet if i.get("type"))
        top_types    = [t for t, _ in type_counts.most_common(10)]
        closet_section = (
            f"Color palette: {', '.join(top_colors)}\n"
            f"Garment types owned: {', '.join(top_types)}"
        )
    else:
        closet_section = "No closet data provided — generate outfits for any style."

    # ── Build inspo section ───────────────────────────────────────────────────
    inspo_parts = []
    if req.inspo_vibes:
        inspo_parts.append(f"Aesthetics they love: {', '.join(req.inspo_vibes[:8])}")
    if req.inspo_colors:
        inspo_parts.append(f"Colors they're drawn to: {', '.join(req.inspo_colors[:10])}")
    inspo_section = "\n".join(inspo_parts) if inspo_parts else "No inspo board yet — use general fashion trends."

    store_line = f"Preferred store: {req.store}" if req.store else "No store preference — suggest anything"
    prompt = _GENERATE_OUTFITS_PROMPT.format(
        closet_section=closet_section,
        inspo_section=inspo_section,
        occasion=req.occasion or "any occasion",
        vibe=req.vibe or "any vibe (match their aesthetic)",
        store_line=store_line,
    )

    try:
        if ANTHROPIC_AVAILABLE and _ant is not None:
            msg = await _ant.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1400,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_json(msg.content[0].text)

        if OPENAI_AVAILABLE and _oai is not None:
            r = await _oai.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.8,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a fashion stylist. Return only valid JSON."},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=1400,
            )
            return json.loads(r.choices[0].message.content)

    except Exception as e:
        raise HTTPException(502, str(e))

    raise HTTPException(503, "No AI key set")


@app.get("/shop", tags=["utility"])
async def shop_proxy(q: str, key: str = ""):
    """Proxy SerpAPI Google Shopping to avoid browser CORS restrictions."""
    serp_key = key or os.environ.get("SERP_KEY", "")
    if not serp_key:
        return {"shopping_results": [], "error": "No SERP key"}
    try:
        import httpx as _httpx
        search_url = f"https://serpapi.com/search.json?engine=google_shopping&q={q}&api_key={serp_key}&num=5"
        async with _httpx.AsyncClient(timeout=10) as client:
            r = await client.get(search_url)
            return r.json()
    except Exception as e:
        return {"shopping_results": [], "error": str(e)}
