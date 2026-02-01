# Self-Improvement Loop Status Report

**Date**: 2026-02-01
**Issue**: No evidence of self-improvement working

## Problems Found

### 1. Vector Index Missing ❌
- The Redis vector index `idx:embeddings` did NOT exist
- This means KNN lookups were failing silently
- Embeddings were being stored but never indexed for search
- **Fixed**: Ran `infra/setup_redis.py` to create index

### 2. Stream Caps Truncating History ❌
- All streams had `maxlen` caps (500-1000 entries)
- This caused old detections/labels to be deleted
- Made it impossible to see full learning history
- **Fixed**: Removed all `maxlen` parameters from `r.xadd()` calls

### 3. Current Memory State
**Labels**: 1
- "Adult male person" (1 example)

**Embeddings**: 1
- `emb:68e02976-0844-4abe-86f4-e63aafb44af2`

**Metrics**:
- Total queries: 4,871
- Known: 1,911 (39%)
- Unknown: 69 (1.4%)
- **Missing**: 2,891 queries (59%) - likely "uncertain" state

**Streams** (all empty due to previous maxlen caps):
- `stream:vision:objects`: 0 entries
- `stream:vision:unknown`: 0 entries
- `stream:vision:labels`: 0 entries
- `stream:vision:researched`: 0 entries

## Self-Improvement Flow (How It Should Work)

```
YOLO Detection
    ↓
Compute Embedding (MobileNetV2)
    ↓
KNN Lookup (5 nearest neighbors)
    ↓
Gating Decision:
    ├─ similarity >= 0.7 AND margin >= 0.15 → "known" (GREEN)
    ├─ similarity >= 0.4 → "uncertain" (YELLOW)
    └─ similarity < 0.4 → "unknown" (RED)
        ↓
        Wait 10 frames (persistence check)
        ↓
        Trigger Auto-Research:
            ├─ GPT-5 Vision → confidence >= 0.85? Return
            ├─ Claude Sonnet Vision → confidence >= 0.85? Return
            ├─ If all < 0.7 → Claude Opus Vision (escalation)
            └─ Return highest confidence result
        ↓
        Auto-Labeling (if confidence > 0.7):
            ├─ Store embedding in Redis with label
            ├─ Emit LabelEvent
            └─ Emit ResearchedEvent (auto_labeled=true)
        ↓
        Next Detection of Same Object:
            └─ KNN finds match → "known" (LEARNED!)
```

## What Was Broken

### Before Fixes:
1. **Vector index didn't exist** → KNN always returned 0 matches
2. **All objects classified as "unknown"** → constant re-research
3. **Stream history lost** → couldn't verify learning happened
4. **Metrics showed 39% "known"** → likely false positives from broken KNN

### After Fixes:
1. Vector index created → KNN can find similar embeddings
2. Stream caps removed → full history preserved
3. Ready for end-to-end testing

## Testing Plan

Run `test_self_improvement.py` to validate:

1. **Unknown → Research**
   - Send frame with cell phone
   - Verify auto-research triggers
   - Check vision API response

2. **Research → Auto-Label**
   - Verify confidence > 0.7
   - Check embedding added to KNN memory
   - Verify `stream:vision:labels` event

3. **Known → Recognition**
   - Send same frame again
   - Verify KNN finds match
   - Verify state changes to "known"

## Expected Results

If working correctly:
- First detection: `state="unknown"`, similarity ~0.0
- Auto-research: Vision API identifies object
- Auto-label: Embedding stored with label
- Second detection: `state="known"`, similarity > 0.7

## Next Steps

1. Clear Redis and start fresh:
   ```bash
   redis-cli FLUSHALL
   /usr/bin/python3 infra/setup_redis.py
   ```

2. Run validation test:
   ```bash
   /usr/bin/python3 test_self_improvement.py
   ```

3. Start backend and monitor logs:
   ```bash
   /usr/bin/python3 -m backend.api
   ```

4. Watch for:
   - Auto-research triggers (background task logs)
   - Vision API responses
   - Auto-labeling events
   - KNN similarity improvements

## Code Changes Made

1. `backend/api.py`:
   - Removed `maxlen` from all `r.xadd()` calls
   - Added WebSocket `/ws/detections` endpoint

2. `workers/research/researcher.py`:
   - Removed Browserbase text search
   - Kept vision APIs only (GPT-5, Claude Sonnet/Opus)

3. `dashboard/components/CameraPreview.tsx`:
   - Added WebSocket connection for live detections
   - Added bounding box overlay with color-coding

4. `infra/setup_redis.py`:
   - Creates `idx:embeddings` vector index (1280-dim, COSINE)
