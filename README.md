# AI Virtual Closet

A personal wardrobe management PWA with AI-powered clothing detection, multi-color recognition, pattern detection, and outfit suggestions based on occasion, time of day, and live weather.

---

## Features

- **AI clothing detection** — upload any photo and the app automatically identifies type, color(s), and pattern
- **Multi-color support** — items with multiple colors (striped, polka dot, plaid) show all swatches
- **Pattern detection** — identifies solid, striped, polka dot, floral, graphic, plaid, tie-dye, camouflage, and more
- **Sleeve length detection** — distinguishes t-shirts from long-sleeve shirts
- **Smart outfit suggestions** — AI stylist picks 2-4 items from your wardrobe based on:
  - Occasion (casual, work, date night, formal, gym, beach, travel)
  - Time of day (morning / afternoon / evening / night)
  - Live weather and temperature via geolocation
- **Outfit match check** — rate how well any two items go together
- **Weekly planner** — assign outfits to days of the week
- **Style DNA** — auto-generated breakdown of your wardrobe by type and color
- **Search & filter** — search by type, color, pattern, tag, or occasion
- **PWA** — installable on mobile, works offline for browsing your closet

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla JS, HTML5, CSS3 (PWA + service worker) |
| Backend | Python, FastAPI, Uvicorn |
| Primary AI | Claude Haiku vision (Anthropic API) |
| Fallback AI | GPT-4o-mini vision (OpenAI API) |
| Local fallback | YOLOS-Fashionpedia (HuggingFace, runs offline) |
| Weather | Open-Meteo API (free, no key needed) |
| Storage | Browser localStorage |

---

## Setup

**Requirements:** Python 3.10+, a modern browser

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd virtual-closet

# 2. Create and activate a virtual environment
python3 -m virtualenv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Anthropic API key (recommended for best accuracy)
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.zshrc
source ~/.zshrc

# 5. Start the app
./start.sh
```

Open **http://127.0.0.1:8000/app/** in your browser.

---

## API Keys

| Key | Purpose | Required? |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude Haiku vision — best accuracy for type, color, pattern, sleeve length | Recommended |
| `OPENAI_API_KEY` | GPT-4o-mini vision — alternative if no Anthropic key | Optional |

Without any key the app falls back to **YOLOS** (local model, no internet needed) for clothing detection. Outfit suggestions and match check require at least one AI key.

---

## Detection Pipeline

Every uploaded image goes through this priority order:

```
Claude Haiku vision  →  GPT-4o-mini vision  →  YOLOS (local)  →  pixel heuristic
```

Claude vision returns: garment type, all significant colors + hex codes, pattern, sleeve length, and a style note.

---

## Project Structure

```
virtual-closet/
├── backend/
│   └── main.py          # FastAPI app — detection, suggest, match endpoints
├── frontend/
│   ├── index.html       # 6-tab PWA layout
│   ├── script.js        # All app logic and state
│   ├── style.css        # Design system (purple palette, responsive)
│   ├── manifest.json    # PWA manifest
│   └── service-worker.js
├── requirements.txt
└── start.sh             # One-command startup
```

---

## Author

Brianna Jules
