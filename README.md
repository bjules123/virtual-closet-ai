# Virtual Closet

An AI-powered wardrobe app that helps you build outfits from what you own, save style inspiration, and shop what's missing — all in one place.

**Live:** [virtual-closet-ai.vercel.app](https://virtual-closet-ai.vercel.app)

---

## What It Does

Upload photos of your clothes. Claude vision automatically identifies each item's type, color, and pattern and saves it to your digital closet. From there:

- **Outfit Ideas** — AI stylist builds complete outfits from your actual wardrobe, considering occasion, time of day, and live weather
- **Inspo Board** — save Pinterest pins or photos. Claude reads the vibe, color palette, and key pieces from each image. Your inspo data personalizes everything else in the app
- **Build This Outfit** — pick an inspo photo and the app matches it piece-by-piece against your closet using strict rules (exact color shade, sleeve length, silhouette). Matches get checked off; missing pieces get real shopping links
- **Shop Outfits** — generates 4 complete shoppable outfit concepts tailored to your closet colors and inspo aesthetic, with live product results from Google Shopping
- **Weekly Planner** — plan full outfits (top, bottom, shoes, bag, accessories) for each day of the week
- **Personal Style Profile** — once you have 5+ items, the home page generates a personal paragraph describing your aesthetic, a color swatch strip, and an inspiration line — all derived from what's actually in your closet
- **Match Check** — pick any two items and AI rates how well they work together with styling tips

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla JS, HTML5, CSS3 |
| Backend | Python, FastAPI, Uvicorn |
| Primary AI | Claude Sonnet 4.6 vision (Anthropic) |
| Fallback AI | GPT-4o-mini vision (OpenAI) |
| Local fallback | YOLOS-Fashionpedia (optional, runs offline) |
| Shopping | SerpAPI Google Shopping |
| Weather | Open-Meteo API (free, no key needed) |
| Storage | Browser localStorage |
| Frontend hosting | Vercel |
| Backend hosting | Render |

---

## Architecture

```
Browser (Vercel)
    │
    ├── /detect         →  Claude Haiku vision → GPT-4o-mini → YOLOS → pixel heuristic
    ├── /suggest        →  Claude Sonnet — outfit picks from wardrobe items
    ├── /match-inspo    →  Claude Sonnet vision — matches inspo photo to closet + shopping gaps
    ├── /analyze-inspo  →  Claude Sonnet — extracts vibe, colors, pieces from inspo photo
    ├── /generate-outfits → Claude Sonnet — personalized shoppable outfit concepts
    ├── /shop           →  SerpAPI proxy (server-side + client-side 7-day cache)
    └── /match          →  Claude Haiku — compatibility rating for two items

Backend (Render)
```

**Detection pipeline priority:**
```
Claude Haiku vision  →  GPT-4o-mini vision  →  YOLOS (local, optional)  →  pixel heuristic
```

YOLOS is optional — the app deploys and runs fine on Render without torch/transformers installed. Claude vision handles detection in production.

**SerpAPI caching:**
All shopping searches are cached in localStorage for 7 days and in server memory for the Render session. Store-switching in Build This Outfit is cache-first — no live API calls unless the user explicitly requests them.

---

## Local Setup

**Requirements:** Python 3.11+, modern browser

```bash
# 1. Clone the repo
git clone https://github.com/bjules123/virtual-closet-ai
cd virtual-closet-ai

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies (local — includes YOLOS)
pip install -r requirements-local.txt

# 4. Add API keys to .env
cp .env.example .env
# Edit .env and add your keys

# 5. Start the backend
./start.sh
```

Open **http://127.0.0.1:8000/app/** in your browser.

---

## API Keys

| Key | Purpose | Required? |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude vision + all AI features | Yes (primary) |
| `SERP_KEY` | SerpAPI Google Shopping — Build This Outfit + Shop Outfits | Yes (shopping features) |
| `OPENAI_API_KEY` | GPT-4o-mini vision fallback | Optional |

---

## Project Structure

```
virtual-closet/
├── backend/
│   └── main.py              # FastAPI — all endpoints, detection pipeline, AI prompts
├── frontend/
│   ├── index.html           # 8-tab layout
│   ├── script.js            # All app logic, state, SerpAPI cache
│   ├── style.css            # Design system
│   └── manifest.json        # PWA manifest
├── requirements.txt         # Production (no torch — for Render)
├── requirements-local.txt   # Local dev (includes YOLOS/torch)
├── render.yaml              # Render deployment config
└── start.sh                 # Local startup script (sources .env)
```

---

## Author

Brianna Jules
