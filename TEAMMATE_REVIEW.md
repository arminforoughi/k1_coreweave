# OpenClawdIRL — Teammate Review Document

**Project**: Self-Improving Vision Agent for WeaveHacks 3
**Deadline**: Sunday Feb 1, 1:15pm
**Repo**: OpenClawdIRL (must be public on GitHub before submission)

---

## 1. What This Is

A camera system that **learns to recognize objects in real time**. Objects start as "unknown," a human teaches the system what they are, and it remembers — getting faster and more confident with every example. The dashboard shows the system visibly improving over time.

**Demo narrative (3 min):**
1. System sees objects on a table, labels them all "unknown"
2. User clicks "Teach" on one → types "stapler"
3. Object moves/rotates → system now recognizes it as "stapler" with high confidence
4. Dashboard metrics: unknown rate drops, memory grows
5. Open a Weave trace showing the full detect→embed→lookup→learn loop
6. (Stretch) System researches an unknown object on the web and explains it via voice

---

## 2. Architecture

Everything runs on the **Booster K1 (Jetson Orin NX)**, except the dashboard which can run on a laptop pointed at the device's IP.

```
┌─────────────────────────────────────────────────────────┐
│  Jetson Orin NX                                         │
│                                                         │
│  ┌──────────────┐    ┌─────────────────────────────┐    │
│  │   Camera      │───▶│   Perception Pipeline       │    │
│  └──────────────┘    │  YOLOv8n → IoU Tracker →    │    │
│                      │  MobileNetV2 Embedder →     │    │
│                      │  Redis KNN Lookup →          │    │
│                      │  Known/Unknown Gate          │    │
│                      └──────────┬──────────────────┘    │
│                                 │ Redis Streams         │
│                      ┌──────────▼──────────────────┐    │
│                      │   Redis Stack               │    │
│                      │  • Vector Index (KNN)       │    │
│                      │  • Streams (event bus)      │    │
│                      │  • JSON (metadata)          │    │
│                      └──────────┬──────────────────┘    │
│                                 │                       │
│                      ┌──────────▼──────────────────┐    │
│                      │   FastAPI Backend           │    │
│                      │  POST /label (teach)        │    │
│                      │  GET  /memory               │    │
│                      │  GET  /metrics              │    │
│                      │  GET  /events/*             │    │
│                      └─────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
         │ HTTP (port 8000)
         ▼
┌─────────────────────┐
│  Laptop / Browser   │
│  Next.js Dashboard  │
│  (port 3000)        │
└─────────────────────┘
```

---

## 3. How the "Learning" Works

This is NOT fine-tuning. The learning is **instance memory** — storing embedding vectors and doing KNN lookup.

### Embedding
- Each detected object is cropped and run through **MobileNetV2** (pretrained, inference only)
- Produces a **1280-dimensional** normalized vector
- Per-track **EMA stabilization**: `e_track = 0.8 * e_track + 0.2 * e_frame`

### KNN Lookup
- Every track's stabilized embedding is queried against **Redis vector index** (cosine similarity)
- Top-K results come back with labels and similarity scores

### Known/Unknown Gating
- **s1** = similarity to nearest match, **s2** = second nearest, **margin** = s1 - s2
- `Known`: s1 >= 0.7 AND margin >= 0.15 → system is confident
- `Uncertain`: 0.4 <= s1 < 0.7 → show to user, might be right
- `Unknown`: s1 < 0.4 → new object, never seen before
- **Spam prevention**: unknown events only fire after 10 consecutive frames, then cooldown for 15 seconds

### Teaching
- User clicks "Teach" on an unknown → types a label name
- System stores the track's embedding in Redis under that label
- Next KNN lookup against the same object → now matches the stored exemplar → becomes "Known"
- More exemplars stored = more robust recognition from different angles

### What Improves Over Time
1. **Unknown rate decreases** — fewer objects trigger the unknown gate
2. **Recognition accuracy increases** — more exemplars per label = better KNN matches
3. **Confidence (similarity) increases** — multiple angles/lighting stored
4. All of this is measurable and visible in the dashboard metrics

---

## 4. Sponsor Tool Usage

| Sponsor | Tool | How We Use It | Required? |
|---------|------|---------------|-----------|
| W&B | Weave | Trace every step: detect, track, embed, knn_lookup, gate_decision, learn_label. Dashboard links to latest trace. | **Yes (required)** |
| Redis | Redis Stack (Vector + Streams + JSON) | Vector index for KNN memory, Streams as event bus between services, JSON for metadata | **Yes** |
| Browserbase | Stagehand | Research unknown objects on the web, produce structured Object Cards | Stretch |
| Daily | Pipecat | Voice interface to speak object descriptions, accept corrections | Stretch |
| Vercel | v0 / hosting | Generate and/or host dashboard | Optional |

---

## 5. Repo Structure

```
OpenClawdIRL/
├── perception/
│   ├── detector.py       # YOLOv8n detection wrapper
│   ├── tracker.py        # IoU-based tracker, assigns stable track_id
│   ├── embedder.py       # MobileNetV2 feature extraction (1280-dim)
│   ├── memory.py         # VectorMemory (Redis KNN) + GatingLogic
│   └── pipeline.py       # Main loop: camera → detect → track → embed → classify → publish
├── backend/
│   └── api.py            # FastAPI: /label, /rename, /memory, /metrics, /events/*
├── dashboard/
│   ├── app/page.tsx      # Single-page app with 4 tabs (Live, Unknown Queue, Memory, Metrics)
│   ├── app/globals.css   # Dark theme styling
│   ├── lib/api.ts        # API client
│   ├── package.json
│   └── next.config.js
├── shared/
│   ├── config.py         # Env-based config (Redis URL, thresholds, etc.)
│   ├── events.py         # Event dataclasses with Redis serialization
│   └── redis_keys.py     # All Redis key patterns, stream names, index name
├── workers/
│   ├── research/         # (Stub) Browserbase/Stagehand research worker
│   └── voice/            # (Stub) Pipecat voice worker
├── infra/
│   ├── setup_redis.py    # Creates vector index + consumer groups
│   ├── deploy_jetson.sh  # rsync deploy (sandboxed to project dir only)
│   ├── install_redis_stack_jetson.sh
│   └── recon_device.sh   # Check device resources before deploying
├── .env.example          # Template for environment variables
├── requirements.txt      # Python dependencies
└── Makefile              # make setup, make redis-setup, make perception, make backend, etc.
```

---

## 6. Deployment Plan — Jetson Isolation Strategy

**Critical rule: We NEVER touch existing files on the device.**

All our work is sandboxed inside `~/OpenClawdIRL/` on the Jetson. Here's exactly what happens:

### What we DO on the device
- `mkdir -p ~/OpenClawdIRL` — create our project directory
- `rsync` our code into `~/OpenClawdIRL/` only
- `pip install --user` or use a venv inside `~/OpenClawdIRL/venv/`
- Run a Docker container for Redis Stack (isolated, no host filesystem changes)
- Run our Python services from within `~/OpenClawdIRL/`

### What we NEVER do
- Modify any file outside `~/OpenClawdIRL/`
- Install system packages with `sudo apt install` (unless absolutely required and reviewed)
- Change system configs, env vars globally, or modify `.bashrc`
- Run anything as root
- Touch any existing Docker containers, services, or data

### Pre-deployment: Device Recon
Before deploying anything, we run a **read-only recon script** that checks:
- Available disk space
- Available RAM and GPU memory
- Whether Docker is installed
- Whether CUDA/JetPack is available
- What Python version is present
- Whether a camera device exists at `/dev/video*`
- What ports are already in use (6379, 8000, 3000)

This tells us if deployment is safe before we copy anything.

### Deployment Steps (in order)
1. **Recon** — SSH in, run `recon_device.sh` (read-only, zero changes)
2. **Review recon output** — confirm resources are sufficient
3. **Deploy code** — `rsync` to `~/OpenClawdIRL/`
4. **Create venv** — `python3 -m venv ~/OpenClawdIRL/venv` (isolated)
5. **Install deps** — `~/OpenClawdIRL/venv/bin/pip install -r requirements.txt`
6. **Start Redis** — Docker container on port 6379 (check not already in use)
7. **Setup Redis indexes** — `make redis-setup`
8. **Start backend** — `make backend` (port 8000)
9. **Start perception** — `make perception`
10. **Start dashboard** — On laptop, pointing to Jetson IP

### Rollback / Cleanup
To completely remove everything we did:
```bash
# Stop services (Ctrl+C on running terminals)
docker stop redis-stack && docker rm redis-stack   # remove Redis container
rm -rf ~/OpenClawdIRL                               # remove all project files
# That's it. Device is back to its original state.
```

---

## 7. Environment Variables

Copy `.env.example` to `.env` and fill in:

```
REDIS_URL=redis://localhost:6379
WANDB_API_KEY=<your W&B API key>
WANDB_PROJECT=openclawdirl
CAMERA_INDEX=0
```

All other values have sensible defaults. Thresholds can be tuned live by editing `.env` and restarting the perception pipeline.

---

## 8. Dashboard Features

| Tab | What It Shows | Actions |
|-----|---------------|---------|
| **Live View** | All currently tracked objects with thumbnails, state badge (known/uncertain/unknown), similarity bar | "Teach" button on non-known objects |
| **Unknown Queue** | Objects that triggered the unknown gate, with track ID and top similarity | "Teach" button |
| **Object Memory** | All learned labels, number of exemplars, when learned | (Rename planned) |
| **Metrics** | Recognition rate %, unknown count, embeddings in memory, labels learned, total KNN queries, unknown rate % | Auto-refreshes every 1.5s |

The dashboard polls the backend every 1.5 seconds. There's a teach modal that lets you type a label name and submit.

---

## 9. Weave Integration

Every core function is decorated with `@weave.op()`:
- `detect_objects` — YOLO inference
- `track_objects` — IoU tracker update
- `embed_crop` — MobileNetV2 embedding
- `knn_lookup` — Redis vector search
- `gate_decision` — known/unknown classification
- `label_object` — the teach/learn endpoint

This produces full traces in W&B Weave showing the entire learning loop. The plan is to include a "Last Learning Loop" deep link in the dashboard for judges.

---

## 10. What's Left To Do

### Must-have (before submission)
- [ ] Deploy to Jetson and get end-to-end working
- [ ] Verify Weave traces appear in W&B dashboard
- [ ] Make GitHub repo public
- [ ] Record 1-2 min demo video
- [ ] Write submission description with sponsor usage list

### Nice-to-have
- [ ] Research worker (Browserbase + Stagehand) — auto-research unknown objects
- [ ] Voice worker (Pipecat) — speak object descriptions
- [ ] Weave trace link embedded in dashboard
- [ ] Time-series metrics chart (not just current numbers)
- [ ] Merge labels action

### Known Risks
- **Redis Stack on ARM64**: Need Docker or compile from source. Docker is the safe path.
- **MobileNetV2 GPU performance on Jetson**: Should be fine with CUDA, but may need to throttle if memory is tight.
- **Camera access**: Need to verify the right `/dev/video*` device index.
- **Port conflicts**: Recon script checks for this.

---

## 11. Questions for Review

1. Does the architecture make sense for a 3-minute demo?
2. Any concerns about the isolation/sandboxing approach on the Jetson?
3. Should we prioritize research worker (Browserbase) or voice worker (Pipecat) for stretch goals?
4. Any other sponsor tools we should weave in?

---

*Generated from commit `12cc432` on Jan 31, 2025*
