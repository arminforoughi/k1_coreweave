"use client";

import { useEffect, useRef, useState } from "react";

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

export default function CameraPreview({ onDeviceChange }: CameraPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string>("");
  const [fps, setFps] = useState(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const fpsCounterRef = useRef({ frames: 0, lastTime: Date.now() });

  // Poll for detections via HTTP (simpler than WebSocket with Redis streams)
  useEffect(() => {
    const pollDetections = async () => {
      try {
        const response = await fetch("http://localhost:8003/events/objects?count=20");
        const data = await response.json();

        if (data.events && data.events.length > 0) {
          const now = Date.now() / 1000;
          const trackMap = new Map<string, Detection>();  // Deduplicate by track_id

          // Convert events to Detection format and filter recent ones
          for (const event of data.events) {
            const timestamp = parseFloat(event.timestamp || "0");
            if (now - timestamp < 2) {  // Only show last 2 seconds
              const detection: Detection = {
                track_id: event.track_id,
                state: event.state as "known" | "uncertain" | "unknown",
                label: event.label || event.state,
                bbox: event.bbox,
                similarity: parseFloat(event.similarity || "0"),
                timestamp: timestamp,
              };

              // Keep only newest per track_id
              if (!trackMap.has(detection.track_id) ||
                  trackMap.get(detection.track_id)!.timestamp < timestamp) {
                trackMap.set(detection.track_id, detection);
              }
            }
          }

          setDetections(Array.from(trackMap.values()));
        }
      } catch (err) {
        console.error("Failed to poll detections:", err);
      }
    };

    // Poll every 500ms
    const interval = setInterval(pollDetections, 500);
    pollDetections(); // Initial fetch

    return () => {
      clearInterval(interval);
    };
  }, []);

  // Enumerate cameras
  useEffect(() => {
    async function getCameras() {
      try {
        // Request permissions first
        await navigator.mediaDevices.getUserMedia({ video: true });
        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = allDevices.filter((d) => d.kind === "videoinput");
        setDevices(videoDevices);
        if (videoDevices.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevices[0].deviceId);
        }
      } catch (err) {
        setError("Failed to access cameras: " + (err as Error).message);
      }
    }
    getCameras();
  }, []);

  // Start/stop video stream
  useEffect(() => {
    if (!selectedDeviceId) return;

    let animationId: number;

    async function startStream() {
      try {
        // Stop existing stream
        if (stream) {
          stream.getTracks().forEach((track) => track.stop());
        }

        const newStream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: selectedDeviceId } },
        });

        setStream(newStream);
        setError("");

        if (videoRef.current) {
          videoRef.current.srcObject = newStream;
        }

        // FPS counter
        const updateFps = () => {
          const counter = fpsCounterRef.current;
          counter.frames++;
          const now = Date.now();
          if (now - counter.lastTime >= 1000) {
            setFps(counter.frames);
            counter.frames = 0;
            counter.lastTime = now;
          }

          // Draw to canvas with bounding boxes
          if (videoRef.current && canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            if (ctx && videoRef.current.readyState === videoRef.current.HAVE_ENOUGH_DATA) {
              const video = videoRef.current;
              canvasRef.current.width = video.videoWidth;
              canvasRef.current.height = video.videoHeight;
              ctx.drawImage(video, 0, 0);

              // Draw bounding boxes with color-coded confidence
              detections.forEach((det) => {
                try {
                  const bbox = JSON.parse(det.bbox);
                  const [x1, y1, x2, y2] = bbox;

                  // Color based on state
                  let color = "#00ff00"; // green = known
                  if (det.state === "uncertain") {
                    color = "#ffff00"; // yellow = uncertain
                  } else if (det.state === "unknown") {
                    color = "#ff0000"; // red = unknown
                  }

                  // Draw bounding box
                  ctx.strokeStyle = color;
                  ctx.lineWidth = 3;
                  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                  // Draw label background
                  const label = det.label || det.state;
                  const labelText = `${label} (${(det.similarity * 100).toFixed(0)}%)`;
                  ctx.font = "14px monospace";
                  const textMetrics = ctx.measureText(labelText);
                  const textWidth = textMetrics.width;
                  const textHeight = 16;

                  ctx.fillStyle = color;
                  ctx.fillRect(x1, y1 - textHeight - 4, textWidth + 8, textHeight + 4);

                  // Draw label text
                  ctx.fillStyle = "#000000";
                  ctx.fillText(labelText, x1 + 4, y1 - 6);
                } catch (err) {
                  console.error("Error drawing detection:", err);
                }
              });
            }
          }

          animationId = requestAnimationFrame(updateFps);
        };
        updateFps();

        onDeviceChange?.(selectedDeviceId);
      } catch (err) {
        setError("Failed to start camera: " + (err as Error).message);
      }
    }

    startStream();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [selectedDeviceId]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
        <select
          value={selectedDeviceId}
          onChange={(e) => setSelectedDeviceId(e.target.value)}
          style={{
            flex: 1,
            padding: "0.5rem",
            borderRadius: "4px",
            border: "1px solid var(--border)",
            background: "var(--bg)",
            color: "var(--text)",
            fontSize: "0.9rem",
          }}
        >
          {devices.map((device) => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label || `Camera ${devices.indexOf(device)}`}
            </option>
          ))}
        </select>
        <div
          style={{
            padding: "0.5rem 1rem",
            background: "var(--card-bg)",
            borderRadius: "4px",
            fontSize: "0.9rem",
          }}
        >
          {fps} FPS
        </div>
      </div>

      {error && (
        <div
          style={{
            padding: "1rem",
            background: "var(--red)",
            color: "white",
            borderRadius: "4px",
          }}
        >
          {error}
        </div>
      )}

      <div style={{ position: "relative" }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{
            display: "none",
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            width: "100%",
            height: "auto",
            borderRadius: "8px",
            border: "2px solid var(--border)",
          }}
        />
      </div>

      <div
        style={{
          padding: "1rem",
          background: "var(--card-bg)",
          borderRadius: "4px",
          fontSize: "0.85rem",
          color: "var(--text-dim)",
        }}
      >
        <strong>Note:</strong> This is a live browser preview of your selected camera.
        The backend is running YOLO on a separate camera feed.
      </div>

      {/* Live Detection List */}
      <div
        style={{
          marginTop: "1rem",
          padding: "1rem",
          background: "var(--card-bg)",
          borderRadius: "4px",
        }}
      >
        <h3 style={{ margin: "0 0 0.5rem 0", fontSize: "1rem" }}>
          Recent Detections ({detections.length})
        </h3>
        {detections.length === 0 ? (
          <div style={{ color: "var(--text-dim)", fontSize: "0.85rem" }}>
            No recent detections. Point camera at objects (person, laptop, cup, phone, etc.)
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {detections.map((det) => {
              let bgColor = "#00ff0020"; // green
              let borderColor = "#00ff00";
              if (det.state === "uncertain") {
                bgColor = "#ffff0020"; // yellow
                borderColor = "#ffff00";
              } else if (det.state === "unknown") {
                bgColor = "#ff000020"; // red
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
                  <div style={{ fontSize: "0.75rem", color: "var(--text-dim)", marginTop: "0.25rem" }}>
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
