#!/usr/bin/env python3
"""Check Redis memory state to validate self-improvement loop."""
import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Check labels
print('=== LABELS ===')
cursor = 0
label_count = 0
while True:
    cursor, keys = r.scan(cursor, match='label:*', count=100)
    for key in keys:
        data = r.hgetall(key)
        if data:
            label_count += 1
            print(f'{key}: {data}')
    if cursor == 0:
        break
print(f'Total labels: {label_count}\n')

# Check embeddings
print('=== EMBEDDINGS ===')
cursor = 0
emb_count = 0
emb_samples = []
while True:
    cursor, keys = r.scan(cursor, match='emb:*', count=10)
    for key in keys:
        emb_count += 1
        if len(emb_samples) < 3:
            emb_samples.append(key)
    if cursor == 0:
        break
print(f'Total embeddings: {emb_count}')
if emb_samples:
    print('Sample embeddings:')
    for key in emb_samples:
        data = r.json().get(key)
        if data:
            print(f'  {key}: label={data.get("label_name")}, label_id={data.get("label_id")}')
print()

# Check vector index
print('=== VECTOR INDEX ===')
try:
    info = r.execute_command('FT.INFO', 'idx:embeddings')
    # Parse the flat list
    if isinstance(info, list):
        for i in range(0, len(info) - 1, 2):
            k = info[i]
            v = info[i + 1]
            if isinstance(k, bytes):
                k = k.decode()
            if k == 'num_docs':
                print(f'num_docs: {v}')
            elif k == 'num_records':
                print(f'num_records: {v}')
except Exception as e:
    print(f'Vector index error: {e}')

# Check streams
print('\n=== STREAM COUNTS ===')
for stream in ['vision:objects', 'vision:unknown', 'vision:labels', 'vision:researched']:
    try:
        length = r.xlen(stream)
        print(f'{stream}: {length} entries')

        # Show last few entries
        if length > 0 and stream in ['vision:labels', 'vision:researched']:
            entries = r.xrevrange(stream, count=3)
            print(f'  Last entries:')
            for entry_id, data in entries:
                print(f'    {entry_id}: {data}')
    except Exception as e:
        print(f'{stream}: error - {e}')

# Check metrics
print('\n=== METRICS ===')
for metric in ['metrics:unknown_count', 'metrics:known_count', 'metrics:total_queries']:
    val = r.get(metric)
    print(f'{metric}: {val or 0}')
