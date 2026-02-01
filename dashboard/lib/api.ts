const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8003";

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

export async function fetchResearched(count = 50) {
  const res = await fetch(`${API_BASE}/events/researched?count=${count}`);
  if (!res.ok) throw new Error("Failed to fetch researched");
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

export async function researchObject(trackId: string) {
  const res = await fetch(`${API_BASE}/research?track_id=${trackId}`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to trigger research");
  }
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function startCamera(cameraId = 0, fps = 2.0) {
  const res = await fetch(`${API_BASE}/camera/start?camera_id=${cameraId}&fps=${fps}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Failed to start camera");
  return res.json();
}

export async function stopCamera() {
  const res = await fetch(`${API_BASE}/camera/stop`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Failed to stop camera");
  return res.json();
}

export async function getCameraStatus() {
  const res = await fetch(`${API_BASE}/camera/status`);
  if (!res.ok) throw new Error("Failed to get camera status");
  return res.json();
}

// Voice API
export async function voiceQuery(audioBlob: Blob): Promise<{ transcription: string; response: string; detections: string }> {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");
  
  const res = await fetch(`${API_BASE}/voice/query`, {
    method: "POST",
    body: formData,
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Voice query failed");
  }
  return res.json();
}

export async function voiceTextQuery(text: string): Promise<{ transcription: string; response: string; detections: string }> {
  const res = await fetch(`${API_BASE}/voice/query?text=${encodeURIComponent(text)}`, {
    method: "POST",
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Voice query failed");
  }
  return res.json();
}

export function getVoiceSpeakUrl(text: string): string {
  return `${API_BASE}/voice/speak?text=${encodeURIComponent(text)}`;
}
