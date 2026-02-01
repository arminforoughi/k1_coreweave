#!/usr/bin/env python3
"""Test WebSocket detection stream."""
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8003/ws/detections"

    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected!")
            print("Listening for detections (Ctrl+C to stop)...")
            print("-" * 60)

            count = 0
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                count += 1

                print(f"[{count}] Track: {data['track_id']}, State: {data['state']}, "
                      f"Label: {data.get('label', 'None')}, Similarity: {data['similarity']:.2f}")

    except websockets.exceptions.ConnectionClosed:
        print("\nConnection closed")
    except KeyboardInterrupt:
        print(f"\n\nReceived {count} detections")

if __name__ == "__main__":
    asyncio.run(test_websocket())
