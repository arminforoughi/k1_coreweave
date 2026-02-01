"use client";

import { useEffect, useState } from "react";

interface CameraPreviewProps {
  onDeviceChange?: (deviceId: string) => void;
}

interface Detection {
  track_id: string;
  state: "known" | "uncertain" | "unknown";
  label: string;
  bbox: string; // JSON array [x1, y1, x2, y2]
  similarity: number;
  timestamp: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function CameraPreview({ onDeviceChange }: CameraPreviewProps) {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [streamError, setStreamError] = useState(false);

  // Poll for detections via HTTP
  useEffect(() => {
    const pollDetections = async () => {
      try {
        const response = await fetch(`${API_URL}/events/objects?count=20`);
        const data = await response.json();

        if (data.events && data.events.length > 0) {
          const now = Date.now() / 1000;
          const trackMap = new Map<string, Detection>();

          for (const event of data.events) {
            const timestamp = parseFloat(event.timestamp || "0");
            if (now - timestamp < 2) {
              const detection: Detection = {
                track_id: event.track_id,
                state: event.state as "known" | "uncertain" | "unknown",
                label: event.label || event.state,
                bbox: event.bbox,
                similarity: parseFloat(event.similarity || "0"),
                timestamp: timestamp,
              };

              if (
                !trackMap.has(detection.track_id) ||
                trackMap.get(detection.track_id)!.timestamp < timestamp
              ) {
                trackMap.set(detection.track_id, detection);
              }
            }
          }

          setDetections(Array.from(trackMap.values()));
        } else {
          setDetections([]);
        }
      } catch (err) {
        console.error("Failed to poll detections:", err);
      }
    };

    const interval = setInterval(pollDetections, 500);
    pollDetections();
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {/* Jetson camera MJPEG stream */}
      <div style={{ position: "relative" }}>
        {streamError ? (
          <div
            style={{
              width: "100%",
              height: "300px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "#1a1a2e",
              borderRadius: "8px",
              border: "2px solid var(--border)",
              color: "var(--text-dim)",
              fontSize: "0.9rem",
            }}
          >
            Waiting for Jetson camera feed...
          </div>
        ) : (
          <img
            src={`${API_URL}/stream/mjpeg`}
            alt="Jetson Camera Feed"
            onError={() => setStreamError(true)}
            onLoad={() => setStreamError(false)}
            style={{
              width: "100%",
              height: "auto",
              borderRadius: "8px",
              border: "2px solid var(--border)",
            }}
          />
        )}
      </div>

      {/* Live Detection List */}
      <div
        style={{
          padding: "1rem",
          background: "var(--card-bg)",
          borderRadius: "4px",
        }}
      >
        <h3 style={{ margin: "0 0 0.5rem 0", fontSize: "1rem" }}>
          Live Detections ({detections.length})
        </h3>
        {detections.length === 0 ? (
          <div style={{ color: "var(--text-dim)", fontSize: "0.85rem" }}>
            No recent detections. Point camera at objects (person, laptop, cup,
            phone, etc.)
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "0.5rem",
            }}
          >
            {detections.map((det) => {
              let bgColor = "#00ff0020";
              let borderColor = "#00ff00";
              if (det.state === "uncertain") {
                bgColor = "#ffff0020";
                borderColor = "#ffff00";
              } else if (det.state === "unknown") {
                bgColor = "#ff000020";
                borderColor = "#ff0000";
              }

              return (
                <div
                  key={det.track_id}
                  style={{
                    padding: "0.5rem",
                    background: bgColor,
                    border: `2px solid ${borderColor}`,
                    borderRadius: "4px",
                    fontSize: "0.85rem",
                  }}
                >
                  <div style={{ fontWeight: "bold" }}>
                    {det.label} ({(det.similarity * 100).toFixed(0)}%)
                  </div>
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--text-dim)",
                      marginTop: "0.25rem",
                    }}
                  >
                    {det.state.toUpperCase()} · Track {det.track_id}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
