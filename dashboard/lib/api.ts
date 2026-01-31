const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchObjects(count = 50) {
  const res = await fetch(`${API_BASE}/events/objects?count=${count}`);
  if (!res.ok) throw new Error("Failed to fetch objects");
  return res.json();
}

export async function fetchUnknowns(count = 50) {
  const res = await fetch(`${API_BASE}/events/unknown?count=${count}`);
  if (!res.ok) throw new Error("Failed to fetch unknowns");
  return res.json();
}

export async function fetchMemory() {
  const res = await fetch(`${API_BASE}/memory`);
  if (!res.ok) throw new Error("Failed to fetch memory");
  return res.json();
}

export async function fetchMetrics() {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error("Failed to fetch metrics");
  return res.json();
}

export async function fetchMetricsHistory() {
  const res = await fetch(`${API_BASE}/metrics/history`);
  if (!res.ok) throw new Error("Failed to fetch history");
  return res.json();
}

export async function labelObject(trackId: string, labelName: string) {
  const res = await fetch(`${API_BASE}/label`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ track_id: trackId, label_name: labelName }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to label");
  }
  return res.json();
}

export async function renameLabel(labelId: string, newName: string) {
  const res = await fetch(`${API_BASE}/rename`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label_id: labelId, new_name: newName }),
  });
  if (!res.ok) throw new Error("Failed to rename");
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
