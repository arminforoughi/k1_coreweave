#!/usr/bin/env python3
"""Test that the self-improvement loop works end-to-end.

This script validates:
1. Unknown object detection triggers auto-research
2. Vision API identifies the object
3. Auto-labeling adds embedding to KNN memory
4. Subsequent detections recognize the object via KNN
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import base64
import requests
import cv2
import numpy as np
from perception.embedder import Embedder
from perception.memory import VectorMemory, GatingLogic
import redis

API_BASE = "http://localhost:8003"

def main():
    print("="*60)
    print("SELF-IMPROVEMENT VALIDATION TEST")
    print("="*60)
    print()

    # Connect to Redis
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r_binary = redis.Redis(host='localhost', port=6379, decode_responses=False)

    # Initialize embedder and memory
    print("Initializing embedder and memory...")
    embedder = Embedder()
    memory = VectorMemory(r_binary, embedding_dim=embedder.dim)
    gating = GatingLogic()

    # Load test video
    print("Loading test video...")
    cap = cv2.VideoCapture("test_video.mp4")
    if not cap.isOpened():
        print("ERROR: Cannot open test_video.mp4")
        print("Please ensure test_video.mp4 exists in the project root")
        return

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame from test_video.mp4")
        return

    cap.release()
    print(f"✓ Loaded frame: {frame.shape}")
    print()

    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    # Compute embedding
    print("Computing embedding...")
    embedding = embedder.embed(frame)
    print(f"✓ Embedding: {embedding.shape} dims, norm={np.linalg.norm(embedding):.4f}")
    print()

    # Check if this embedding is known
    print("Checking KNN memory...")
    matches = memory.knn_lookup(embedding, k=5)
    if matches:
        print(f"Found {len(matches)} matches:")
        for m in matches:
            print(f"  - {m['label_name']}: similarity={m['similarity']:.3f}")
    else:
        print("No matches found (empty memory or no similar objects)")

    state, label, similarity = gating.decide(matches)
    print(f"✓ Gating decision: state={state}, label={label}, similarity={similarity:.3f}")
    print()

    # Send 12 frames to trigger persistence (needs 10+ frames)
    print("Sending 12 detections to trigger auto-research...")
    track_id = None
    for frame_num in range(1, 13):
        payload = {
            "timestamp": time.time(),
            "frame_id": frame_num,
            "detections": [
                {
                    "bbox": [100, 100, 300, 300],
                    "yolo_class": "cell phone",
                    "yolo_confidence": 0.65,
                    "crop_b64": frame_b64,
                }
            ]
        }

        response = requests.post(f"{API_BASE}/ingest", json=payload)
        if response.status_code != 200:
            print(f"ERROR: Backend returned {response.status_code} on frame {frame_num}")
            print(response.text)
            return

        result = response.json()
        if frame_num == 1:
            track_id = result['objects'][0]['track_id']
            print(f"✓ Frame {frame_num}: Created track {track_id}, state={result['objects'][0]['state']}")
        elif frame_num % 3 == 0:
            print(f"✓ Frame {frame_num}: state={result['objects'][0]['state']}")

        time.sleep(0.1)  # Small delay between frames

    print(f"✓ Sent 12 frames for track {track_id}")
    print()

    # Wait for auto-research to complete
    print("Waiting 15 seconds for auto-research...")
    for i in range(15, 0, -1):
        print(f"  {i}s...", end="\r")
        time.sleep(1)
    print()

    # Check if research happened
    print("Checking research stream...")
    researched = r.xrevrange("stream:vision:researched", count=5)
    if researched:
        print(f"Found {len(researched)} research results:")
        for entry_id, data in researched:
            print(f"  {entry_id}:")
            for k, v in data.items():
                print(f"    {k}: {v}")
    else:
        print("No research results found")
    print()

    # Check if auto-labeling happened
    print("Checking labels stream...")
    labels = r.xrevrange("stream:vision:labels", count=5)
    if labels:
        print(f"Found {len(labels)} label events:")
        for entry_id, data in labels:
            print(f"  {entry_id}:")
            for k, v in data.items():
                print(f"    {k}: {v}")
    else:
        print("No label events found")
    print()

    # Check memory again
    print("Checking KNN memory after auto-research...")
    all_labels = memory.get_all_labels()
    print(f"Total labels in memory: {len(all_labels)}")
    for lbl in all_labels:
        print(f"  - {lbl['name']}: {lbl['n_examples']} examples, created={time.ctime(lbl['created_at'])}")
    print()

    memory_count = memory.get_memory_count()
    print(f"Total embeddings in vector index: {memory_count}")
    print()

    # Test recognition on same frame again
    print("Testing recognition on same frame...")
    matches2 = memory.knn_lookup(embedding, k=5)
    if matches2:
        print(f"Found {len(matches2)} matches:")
        for m in matches2:
            print(f"  - {m['label_name']}: similarity={m['similarity']:.3f}")
    else:
        print("No matches found")

    state2, label2, similarity2 = gating.decide(matches2)
    print(f"✓ Gating decision: state={state2}, label={label2}, similarity={similarity2:.3f}")
    print()

    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Initial state: {state} (similarity={similarity:.3f})")
    print(f"Final state: {state2} (similarity={similarity2:.3f})")
    print(f"Memory size: {memory_count} embeddings, {len(all_labels)} labels")
    print()

    if state == "unknown" and state2 == "known":
        print("✅ SUCCESS: Object learned! Unknown → Known")
    elif state == "unknown" and state2 == "uncertain":
        print("⚠️  PARTIAL: Object researched but not confident enough to auto-label")
        print("   Check vision API keys and confidence thresholds")
    elif state == state2:
        if state == "known":
            print("✅ ALREADY KNOWN: Object was already in memory")
        else:
            print("❌ FAILED: State did not improve")
            print("   Check auto-research logs and vision API responses")
    else:
        print(f"⚠️  UNEXPECTED: State changed from {state} to {state2}")

if __name__ == "__main__":
    main()
