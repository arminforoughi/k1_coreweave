# OpenClawdIRL — Updated Architecture Plan

**Project**: Self-Improving Vision Agent for WeaveHacks 3
**Deadline**: Sunday Feb 1, 1:15pm
**Repo**: OpenClawdIRL (must be public on GitHub before submission)
**Last updated**: Commit `ddd4f94` — thin-device refactor complete

---

## 1. What This Is

A camera system that **learns to recognize objects in real time**. Objects start as "unknown," the system can either research them on the web (Browserbase) or a human teaches it. It remembers — getting faster and more confident with every example. The dashboard shows the system visibly improving over time.

**Demo narrative (3 min):**
1. System sees objects on a table, labels some as "unknown"
2. User clicks "Research" — Browserbase searches the web, auto-identifies the object
3. If research isn't confident, user clicks "Teach" → types a label manually
4. Object moves/rotates → system now recognizes it with high confidence
5. Dashboard metrics: unknown rate drops, memory grows, recognition rate climbs
6. Open a Weave trace showing the full detect→embed→lookup→learn loop

---

## 2. Architecture — Thin Device, Fat Laptop

The Jetson does **as little as possible** (camera + YOLO only). Everything compute-heavy runs on the laptop.

```
JETSON (Booster K1)                    LAPTOP
┌──────────────────────┐               ┌──────────────────────────────────┐
│                      │               │                                  │
│  Camera              │               │  FastAPI Backend (port 8000)     │
│    ↓                 │               │  ┌────────────────────────────┐  │
│  YOLOv8n Detection   │               │  │ POST /ingest               │  │
│    ↓                 │  HTTP POST    │  │   → decode crop            │  │
│  Crop + base64       │──────────────▶│  │   → MobileNetV2 embed     │  │
│  encode              │  (crops,      │  │   → IoU tracker match     │  │
│                      │   bbox,       │  │   → EMA stabilize embed   │  │
│  3 dependencies:     │   class,      │  │   → Redis KNN lookup      │  │
│  - ultralytics       │   confidence) │  │   → known/unknown gate    │  │
│  - opencv            │               │  │   → publish to Streams    │  │
│  - requests          │  ◀────────────│  │   → return state per obj  │  │
│                      │  (state,      │  └────────────────────────────┘  │
└──────────────────────┘   label,      │                                  │
                           similarity) │  POST /label     → store embed  │
                                       │  POST /research  → Browserbase  │
                                       │  POST /rename    → update label │
                                       │  GET /events/*   → stream data  │
                                       │  GET /memory     → label list   │
                                       │  GET /metrics    → improvement  │
                                       │                                  │
                                       ├──────────────────────────────────┤
                                       │  Redis Stack (port 6379)         │
                                       │  ┌────────────────────────────┐  │
                                       │  │ RediSearch Vector Index    │  │
                                       │  │   emb:{id} → 1280-d vec   │  │
                                       │  │   cosine KNN (top-5)      │  │
                                       │  ├────────────────────────────┤  │
                                       │  │ Redis Streams (event bus)  │  │
                                       │  │   stream:vision:objects    │  │
                                       │  │   stream:vision:unknown    │  │
                                       │  │   stream:vision:labels     │  │
                                       │  │   stream:vision:researched │  │
                                       │  ├────────────────────────────┤  │
                                       │  │ Hashes + JSON (metadata)   │  │
                                       │  │   label:{id} → name, etc  │  │
                                       │  │   metrics:* → counters    │  │
                                       │  └────────────────────────────┘  │
                                       │                                  │
                                       ├──────────────────────────────────┤
                                       │  Browserbase/Stagehand Worker   │
                                       │   → web search unknown objects  │
                                       │   → extract: label, desc, facts │
                                       │   → auto-label if confident     │
                                       │                                  │
                                       ├──────────────────────────────────┤
                                       │  Next.js Dashboard (port 3000)  │
                                       │   → Live View                   │
                                       │   → Unknown Queue               │
                                       │   → Object Memory               │
                                       │   → Metrics                     │
                                       └──────────────────────────────────┘
```

### Why this split?
- **Latency**: Jetson→laptop over local WiFi is <10ms. Embedding on laptop CPU ~30ms. Total round trip ~50ms.
- **Device safety**: Only 2 files + a venv on the Jetson. No Redis, no torch, no system changes.
- **Simplicity**: One machine to debug (laptop). Jetson is a dumb sensor.

---

## 3. Component-by-Component Breakdown

### 3a. Jetson Client (`perception/jetson_client.py`)

**What it does**: Captures camera frames, runs YOLO, sends crops to the laptop.

**Data out** (HTTP POST to `/ingest`):
```json
{
  "timestamp": 1706745600.0,
  "frame_id": 42,
  "detections": [
    {
      "bbox": [120, 80, 340, 290],
      "yolo_class": "bottle",
      "yolo_confidence": 0.87,
      "crop_b64": "/9j/4AAQ..."
    }
  ]
}
```

**How YOLO fits in**: YOLOv8n knows 80 COCO classes (person, car, bottle, cup, etc). For each detected object it returns a class guess + confidence 0-1. This gives us two signals:
- High confidence (>0.75) + real class → YOLO probably knows what this is
- Low confidence (<0.4) or wrong class → object is outside YOLO's training data

We set YOLO's confidence floor to 0.3 (low) so we catch uncertain detections rather than discarding them.

**Config**: `--backend http://<laptop-ip>:8000 --camera 0 --fps 3 --confidence 0.3`

---

### 3b. Backend API (`backend/api.py`)

**What it does**: The brain. Receives crops, embeds them, looks them up, decides known/unknown, triggers research.

**Runs on startup**:
- Loads MobileNetV2 embedder (~14MB model, CPU or GPU)
- Connects to Redis
- Initializes Weave tracing

**Core endpoint — `POST /ingest`** (called by Jetson every frame):
1. Receives list of detections with crops
2. Matches each detection to an existing track via IoU (stable identity across frames)
3. Decodes base64 crop → numpy array
4. Runs MobileNetV2 embedding → 1280-dim normalized vector
5. EMA-stabilizes the track's embedding: `e = 0.8 * e_old + 0.2 * e_new`
6. KNN lookup against Redis vector index (top-5 nearest)
7. Gating decision (combines KNN similarity + YOLO confidence):
   - **Known**: KNN top match ≥ 0.7 similarity AND margin ≥ 0.15 over 2nd match
   - **Uncertain**: KNN 0.4–0.7 similarity, OR KNN says unknown but YOLO ≥ 0.75 confidence
   - **Unknown**: KNN < 0.4 AND YOLO not confident
8. Publishes `ObjectEvent` to `stream:vision:objects`
9. If unknown + seen for ≥10 frames + cooldown expired → publishes `UnknownEvent` to `stream:vision:unknown`
10. Returns state per object back to Jetson client (for logging)

**Teaching endpoint — `POST /label`**:
1. Finds the object's embedding from recent stream events
2. Stores embedding in Redis under the new label name
3. Publishes `LabelEvent` to `stream:vision:labels`
4. Next KNN lookup will now match this object → "known"

**Research endpoint — `POST /research`**:
1. Finds the unknown object's thumbnail and YOLO hint
2. Queues a background task that calls the Browserbase worker
3. If research returns confidence > 0.7 → auto-labels the object
4. Publishes result to `stream:vision:researched`

---

### 3c. Embedder (`perception/embedder.py`)

**What it does**: Converts an object crop into a 1280-dimensional vector.

- Uses **MobileNetV2** pretrained on ImageNet (frozen, inference only)
- Classifier head removed → outputs the penultimate feature layer
- Input: 224x224 RGB image (auto-resized from crop)
- Output: L2-normalized 1280-dim float32 vector
- Runs on laptop CPU (~30ms) or GPU (<5ms)

---

### 3d. Vector Memory + Gating (`perception/memory.py`)

**VectorMemory class** — wraps Redis vector operations:
- `store_embedding(vector, label_id, label_name)` → stores as JSON doc at `emb:{uuid}`
- `knn_lookup(vector, k=5)` → RediSearch KNN query, returns label + similarity per match
- `learn_label(label_name, vectors)` → creates `label:{uuid}` hash + stores exemplar embeddings
- `get_all_labels()` → scans all `label:*` keys
- `get_memory_count()` → count of docs in the vector index

**GatingLogic class** — decides object state from KNN results:
- Takes list of KNN matches (sorted by similarity)
- Returns `(state, label_name, top_similarity)`
- Three thresholds: `known_threshold=0.7`, `unknown_threshold=0.4`, `margin_threshold=0.15`

---

### 3e. Tracker (`perception/tracker.py`)

**What it does**: Assigns stable track IDs across frames using IoU matching.

- Runs server-side (on the laptop) since the backend receives bboxes each frame
- Matches incoming detections to existing tracks by bounding box overlap
- Creates new tracks for unmatched detections
- Drops tracks not seen for >5 seconds
- Each track holds: track_id, bbox, embedding, frames_seen, state, cooldown timer

---

### 3f. Research Worker (`workers/research/researcher.py`)

**What it does**: Uses Browserbase + Stagehand to identify unknown objects via web search.

**Flow**:
1. Receives thumbnail + YOLO hint (e.g., "bottle" at 0.35 confidence)
2. Opens a Browserbase browser session via Stagehand
3. Navigates to Google, searches "what is a {yolo_hint} object identify"
4. Uses Stagehand's `extract()` to pull structured data from results:
   - `label`: 1-3 word object name
   - `description`: one sentence
   - `facts[]`: 2-3 interesting facts
   - `confidence`: 0.0 to 1.0
5. Returns result to backend

**Fallback**: If Browserbase is unavailable, returns the YOLO hint with 0.3 confidence.

**Auto-labeling**: If research confidence > 0.7, the backend automatically stores the embedding under the researched label — no human needed.

---

### 3g. Redis (`infra/setup_redis.py` + `shared/redis_keys.py`)

**Three roles in one instance**:

**1. Vector Database (RediSearch)**
- Index `idx:embeddings` over `emb:*` JSON documents
- Schema: vector (1280-dim float32, FLAT, cosine), label_id, label_name, created_at
- KNN query: `*=>[KNN 5 @vector $query_vec AS score]`
- Cosine distance 0=identical → converted to similarity 1.0

**2. Event Bus (Redis Streams)**
- `stream:vision:objects` — every tracked object, ~3Hz per track (maxlen 1000)
- `stream:vision:unknown` — gated unknown events, sparse (maxlen 500)
- `stream:vision:labels` — label/teach events (maxlen 500)
- `stream:vision:researched` — research results (maxlen 500)
- Consumer groups: `dashboard`, `backend`, `research_worker`, `voice_worker`

**3. Metadata Store (Hashes + JSON + Counters)**
- `label:{id}` → hash: name, created_at, n_examples
- `emb:{id}` → JSON: vector, label_id, label_name, created_at
- `metrics:unknown_count`, `metrics:known_count`, `metrics:total_queries` → counters

---

### 3h. Dashboard (`dashboard/`)

**Tech**: Next.js 14, React 18, TypeScript, vanilla CSS (dark theme)

**4 tabs**:

| Tab | Data Source | What It Shows | User Actions |
|-----|-----------|---------------|-------------|
| **Live View** | `GET /events/objects` | Tracked objects with thumbnails, state badges (known/uncertain/unknown), similarity bars | Teach, Research |
| **Unknown Queue** | `GET /events/unknown` | Objects that passed the unknown gate | Teach, Research |
| **Object Memory** | `GET /memory` | Learned labels, exemplar count, time learned | — |
| **Metrics** | `GET /metrics` | Recognition rate %, unknown count, embeddings in memory, labels learned, total queries, unknown rate % | — |

**Polling**: Every 1.5 seconds on the active tab.

**Teach Modal**: Click "Teach" → modal with thumbnail + text input → POST /label.

**Research Button**: Click "Research" → POST /research → shows "Researching..." for 5s → refreshes.

---

### 3i. Weave Instrumentation

Every core function in the backend is decorated with `@weave.op()`:

| Op | What It Traces |
|----|---------------|
| `embed_crop_op` | MobileNetV2 inference (input crop → output vector) |
| `knn_lookup_op` | Redis vector search (input vector → output matches) |
| `gate_decision_op` | State decision (matches + YOLO signals → known/uncertain/unknown) |
| `learn_label_op` | Memory write (label name + vectors → label_id) |
| `ingest_frame` | Full pipeline per frame (all of the above) |
| `label_object` | Teach action (track_id + name → stored) |
| `trigger_research` | Browserbase research trigger |
| `research_object` | Actual web research execution |

This produces nested traces in W&B Weave. Each `/ingest` call creates a parent trace containing embed → knn → gate child spans.

---

## 4. Sponsor Tool Usage

| Sponsor | Tool | Role in System | How Deeply Used |
|---------|------|---------------|----------------|
| **W&B** | Weave | Traces every step of the detect→embed→lookup→gate→learn loop. Metrics logged. Deep link for judges. | **Core** (required) |
| **Redis** | Redis Stack | Vector DB (KNN memory), event bus (Streams), metadata store (JSON/Hashes), counters | **Core** (3 features in 1) |
| **Browserbase** | Stagehand | Web research to auto-identify unknown objects. Searches Google, extracts structured Object Cards. | **Core** (primary learning path) |
| **Daily** | Pipecat | Voice interface to speak object descriptions, accept corrections | Stretch |
| **Vercel** | v0 / hosting | Could generate or host dashboard | Optional |

---

## 5. Data Flow — End to End

```
1. Camera captures frame (Jetson)
2. YOLOv8n detects objects → bboxes + class guesses + confidence (Jetson)
3. Crops encoded as base64 JPEG, POSTed to laptop (Jetson → Laptop)
4. Backend matches detections to tracks via IoU (Laptop)
5. MobileNetV2 embeds each crop → 1280-dim vector (Laptop)
6. EMA stabilizes track embedding (Laptop)
7. Redis KNN lookup → top-5 nearest stored embeddings (Laptop → Redis)
8. Gate decision: known / uncertain / unknown (Laptop)
9. Object event published to stream:vision:objects (Laptop → Redis)
10. Dashboard polls and displays live state (Browser → Laptop)

UNKNOWN PATH:
11. If unknown + persistent + cooldown → unknown event published (Laptop → Redis)
12. User clicks "Research" → Browserbase searches the web (Laptop → Browserbase)
13. If confident → auto-label stored in Redis memory (Laptop → Redis)
14. OR user clicks "Teach" → manual label stored (Browser → Laptop → Redis)
15. Next detection of same object → KNN returns high similarity → "known" ✓

IMPROVEMENT MEASURABLES:
- Unknown rate decreases over time
- Recognition rate increases
- Confidence (similarity scores) increases per object
- Memory grows (more labels, more exemplars)
- All visible in dashboard Metrics tab and Weave traces
```

---

## 6. Repo Structure (Current)

```
OpenClawdIRL/
├── perception/
│   ├── jetson_client.py        # [JETSON] Camera + YOLO + POST crops
│   ├── requirements-jetson.txt # [JETSON] 3 deps only
│   ├── detector.py             # [LAPTOP] YOLOv8n wrapper (used by old pipeline)
│   ├── embedder.py             # [LAPTOP] MobileNetV2 feature extraction
│   ├── memory.py               # [LAPTOP] VectorMemory (Redis KNN) + GatingLogic
│   ├── tracker.py              # [LAPTOP] IoU-based tracker
│   └── pipeline.py             # [OLD] On-device pipeline (kept for reference)
├── backend/
│   └── api.py                  # [LAPTOP] FastAPI: /ingest, /label, /research, /memory, /metrics
├── workers/
│   └── research/
│       └── researcher.py       # [LAPTOP] Browserbase/Stagehand web research
├── dashboard/
│   ├── app/
│   │   ├── page.tsx            # [LAPTOP] 4-tab dashboard with Teach + Research
│   │   ├── layout.tsx          # Root layout
│   │   └── globals.css         # Dark theme
│   ├── lib/api.ts              # API client
│   ├── package.json            # Next.js 14 + React 18
│   └── tsconfig.json
├── shared/
│   ├── config.py               # Env-based config for all services
│   ├── events.py               # Event dataclasses (ObjectEvent, UnknownEvent, LabelEvent)
│   └── redis_keys.py           # All Redis key patterns, stream names, index name
├── infra/
│   ├── setup_redis.py          # Create vector index + consumer groups
│   ├── deploy_jetson.sh        # Copy thin client to device (2 files + venv)
│   ├── recon_device.sh         # Read-only device resource check
│   └── install_redis_stack_jetson.sh  # (Legacy, Docker instructions)
├── .env.example
├── requirements.txt            # Laptop deps (torch, fastapi, redis, stagehand, weave)
└── Makefile                    # make backend, make dashboard, make jetson-client, etc.
```

---

## 7. Deployment Plan

### Jetson (2 files, ~50MB footprint)

The deploy script `infra/deploy_jetson.sh` does exactly this:
1. Creates `~/OpenClawdIRL/perception/` on device
2. Copies `jetson_client.py` and `requirements-jetson.txt` (via scp)
3. Creates a Python venv at `~/OpenClawdIRL/venv/`
4. Installs 3 packages: ultralytics, opencv-python-headless, requests
5. Creates `~/OpenClawdIRL/run.sh` convenience script

**Isolation guarantees**:
- Only touches `~/OpenClawdIRL/` — nothing else
- No sudo, no global pip, no system config changes
- Cleanup: `rm -rf ~/OpenClawdIRL`

### Laptop (everything else)

```bash
# One-time setup:
pip install -r requirements.txt
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest
python infra/setup_redis.py

# Run (3 terminals):
make backend                    # Terminal 1: API + embedder + KNN
make dashboard                  # Terminal 2: Next.js dashboard
# (Jetson sends to this laptop's IP automatically)
```

### Startup order
1. Redis (Docker container)
2. `setup_redis.py` (create index + consumer groups, once)
3. Backend API (loads embedder, connects to Redis)
4. Dashboard (connects to backend)
5. Jetson client (sends frames to backend)

---

## 8. Environment Variables

```bash
# Required
WANDB_API_KEY=<your W&B key>
WANDB_PROJECT=openclawdirl

# Required for Browserbase research
BROWSERBASE_API_KEY=<from stagehand.dev>
BROWSERBASE_PROJECT_ID=<from Browserbase dashboard>

# Optional (all have defaults)
REDIS_URL=redis://localhost:6379
CAMERA_INDEX=0
DETECTION_CONFIDENCE=0.5
KNOWN_THRESHOLD=0.7
UNKNOWN_THRESHOLD=0.4
MARGIN_THRESHOLD=0.15
COOLDOWN_SECONDS=15
PERSISTENCE_FRAMES=10
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

---

## 9. What's Left To Do

### Must-have
- [ ] Get API keys: WANDB_API_KEY, BROWSERBASE_API_KEY, BROWSERBASE_PROJECT_ID
- [ ] Run device recon script (read-only check)
- [ ] Deploy thin client to Jetson
- [ ] Test end-to-end: Jetson → laptop → dashboard
- [ ] Verify Weave traces in W&B dashboard
- [ ] Make GitHub repo public
- [ ] Record 1-2 min demo video
- [ ] Write submission: summary, sponsor usage, architecture description

### Nice-to-have
- [ ] Voice worker (Pipecat) — speak object cards
- [ ] Weave "Last Learning Loop" deep link in dashboard
- [ ] Time-series chart in Metrics tab
- [ ] Auto-research unknowns (trigger Browserbase automatically, not just on button click)

### Known Risks
- **Browserbase latency**: Web search + extraction can take 5-10 seconds. Background task handles this.
- **YOLO on Jetson without GPU**: If CUDA isn't available, YOLO falls back to CPU (~200ms/frame). Still workable at 3 FPS.
- **WiFi latency**: If Jetson and laptop are on congested WiFi, crop transfer could be slow. Crops are ~10-20KB each.
- **Redis Stack Docker on ARM**: If the Jetson needed Redis (old plan), ARM Docker images can be flaky. But now Redis is on the laptop — no issue.

---

*Updated after thin-device refactor, commit `ddd4f94`*
