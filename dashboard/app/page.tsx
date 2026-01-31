"use client";

import { useState, useEffect, useCallback } from "react";
import {
  fetchObjects,
  fetchUnknowns,
  fetchMemory,
  fetchMetrics,
  labelObject,
  checkHealth,
} from "@/lib/api";

type Tab = "live" | "unknown" | "memory" | "metrics";

interface ObjectEvent {
  track_id: string;
  state: string;
  label?: string;
  similarity: string;
  thumbnail_b64: string;
  bbox: string;
  timestamp: string;
  stream_id: string;
}

interface UnknownEvent {
  track_id: string;
  thumbnail_b64: string;
  top_similarity: string;
  timestamp: string;
  event_id: string;
  stream_id: string;
}

interface LabelInfo {
  label_id: string;
  name: string;
  n_examples: number;
  created_at: number;
}

interface Metrics {
  unknown_count: number;
  known_count: number;
  total_queries: number;
  recognition_rate: number;
  unknown_rate: number;
  memory_size: number;
  label_count: number;
}

export default function Dashboard() {
  const [tab, setTab] = useState<Tab>("live");
  const [objects, setObjects] = useState<ObjectEvent[]>([]);
  const [unknowns, setUnknowns] = useState<UnknownEvent[]>([]);
  const [labels, setLabels] = useState<LabelInfo[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [online, setOnline] = useState(false);
  const [teachModal, setTeachModal] = useState<{ trackId: string; thumb: string } | null>(null);
  const [teachName, setTeachName] = useState("");
  const [teachLoading, setTeachLoading] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const health = await checkHealth();
      setOnline(health.status === "ok");
    } catch {
      setOnline(false);
      return;
    }

    try {
      if (tab === "live") {
        const data = await fetchObjects();
        setObjects(data.events || []);
      } else if (tab === "unknown") {
        const data = await fetchUnknowns();
        setUnknowns(data.events || []);
      } else if (tab === "memory") {
        const data = await fetchMemory();
        setLabels(data.labels || []);
      } else if (tab === "metrics") {
        const data = await fetchMetrics();
        setMetrics(data);
      }
    } catch (e) {
      console.error("Refresh error:", e);
    }
  }, [tab]);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 1500);
    return () => clearInterval(interval);
  }, [refresh]);

  const handleTeach = async () => {
    if (!teachModal || !teachName.trim()) return;
    setTeachLoading(true);
    try {
      await labelObject(teachModal.trackId, teachName.trim());
      setTeachModal(null);
      setTeachName("");
      refresh();
    } catch (e) {
      alert("Failed to label: " + (e as Error).message);
    } finally {
      setTeachLoading(false);
    }
  };

  // Deduplicate objects by track_id (keep most recent)
  const uniqueObjects = objects.reduce<ObjectEvent[]>((acc, obj) => {
    if (!acc.find((o) => o.track_id === obj.track_id)) acc.push(obj);
    return acc;
  }, []);

  return (
    <div className="container">
      <div className="header">
        <h1>OpenClawdIRL</h1>
        <div className="status">
          <div className={`status-dot ${online ? "" : "offline"}`} />
          {online ? "System Online" : "Offline"}
        </div>
      </div>

      <div className="tabs">
        {(["live", "unknown", "memory", "metrics"] as Tab[]).map((t) => (
          <button
            key={t}
            className={`tab ${tab === t ? "active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t === "live"
              ? "Live View"
              : t === "unknown"
              ? `Unknown Queue (${unknowns.length})`
              : t === "memory"
              ? "Object Memory"
              : "Metrics"}
          </button>
        ))}
      </div>

      {/* Live View */}
      {tab === "live" && (
        <div>
          {uniqueObjects.length === 0 ? (
            <div className="empty-state">
              <p>No objects detected yet. Make sure the perception pipeline is running.</p>
            </div>
          ) : (
            <div className="grid grid-3">
              {uniqueObjects.map((obj) => (
                <div key={obj.track_id} className="object-card">
                  {obj.thumbnail_b64 ? (
                    <img
                      className="object-thumb"
                      src={`data:image/jpeg;base64,${obj.thumbnail_b64}`}
                      alt={obj.label || "object"}
                    />
                  ) : (
                    <div className="object-thumb" />
                  )}
                  <div className="object-info">
                    <h3>{obj.label || `Track ${obj.track_id}`}</h3>
                    <span className={`badge badge-${obj.state}`}>{obj.state}</span>
                    <div className="meta">
                      Similarity: {(parseFloat(obj.similarity) * 100).toFixed(1)}%
                    </div>
                    <div className="sim-bar">
                      <div
                        className="sim-bar-fill"
                        style={{
                          width: `${parseFloat(obj.similarity) * 100}%`,
                          background:
                            obj.state === "known"
                              ? "var(--green)"
                              : obj.state === "uncertain"
                              ? "var(--yellow)"
                              : "var(--red)",
                        }}
                      />
                    </div>
                    {obj.state !== "known" && (
                      <button
                        className="btn btn-primary"
                        style={{ marginTop: "0.5rem", fontSize: "0.8rem" }}
                        onClick={() =>
                          setTeachModal({
                            trackId: obj.track_id,
                            thumb: obj.thumbnail_b64,
                          })
                        }
                      >
                        Teach
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Unknown Queue */}
      {tab === "unknown" && (
        <div>
          {unknowns.length === 0 ? (
            <div className="empty-state">
              <p>No unknown objects in the queue.</p>
            </div>
          ) : (
            <div className="grid grid-3">
              {unknowns.map((u, i) => (
                <div key={u.stream_id || i} className="object-card">
                  {u.thumbnail_b64 ? (
                    <img
                      className="object-thumb"
                      src={`data:image/jpeg;base64,${u.thumbnail_b64}`}
                      alt="unknown"
                    />
                  ) : (
                    <div className="object-thumb" />
                  )}
                  <div className="object-info">
                    <h3>Unknown Object</h3>
                    <span className="badge badge-unknown">unknown</span>
                    <div className="meta">
                      Track: {u.track_id}
                      <br />
                      Top sim: {(parseFloat(u.top_similarity) * 100).toFixed(1)}%
                    </div>
                    <button
                      className="btn btn-primary"
                      style={{ marginTop: "0.5rem", fontSize: "0.8rem" }}
                      onClick={() =>
                        setTeachModal({
                          trackId: u.track_id,
                          thumb: u.thumbnail_b64,
                        })
                      }
                    >
                      Teach
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Object Memory */}
      {tab === "memory" && (
        <div>
          {labels.length === 0 ? (
            <div className="empty-state">
              <p>No objects learned yet. Teach the system by labeling unknown objects.</p>
            </div>
          ) : (
            <div className="grid grid-3">
              {labels.map((l) => (
                <div key={l.label_id} className="card">
                  <h3>{l.name}</h3>
                  <div className="meta">
                    {l.n_examples} exemplar{l.n_examples !== 1 ? "s" : ""} stored
                  </div>
                  <div className="meta">
                    Learned: {new Date(l.created_at * 1000).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Metrics */}
      {tab === "metrics" && (
        <div>
          {metrics ? (
            <div className="grid grid-4">
              <div className="card">
                <div className="metric-value" style={{ color: "var(--green)" }}>
                  {(metrics.recognition_rate * 100).toFixed(1)}%
                </div>
                <div className="metric-label">Recognition Rate</div>
              </div>
              <div className="card">
                <div className="metric-value" style={{ color: "var(--red)" }}>
                  {metrics.unknown_count}
                </div>
                <div className="metric-label">Unknown Events</div>
              </div>
              <div className="card">
                <div className="metric-value" style={{ color: "var(--accent)" }}>
                  {metrics.memory_size}
                </div>
                <div className="metric-label">Embeddings in Memory</div>
              </div>
              <div className="card">
                <div className="metric-value" style={{ color: "var(--yellow)" }}>
                  {metrics.label_count}
                </div>
                <div className="metric-label">Learned Labels</div>
              </div>
              <div className="card" style={{ gridColumn: "span 2" }}>
                <div className="card-header">
                  <span className="card-title">Total Queries</span>
                </div>
                <div className="metric-value">{metrics.total_queries}</div>
                <div className="metric-label">KNN lookups performed</div>
              </div>
              <div className="card" style={{ gridColumn: "span 2" }}>
                <div className="card-header">
                  <span className="card-title">Unknown Rate</span>
                </div>
                <div className="metric-value" style={{ color: "var(--orange)" }}>
                  {(metrics.unknown_rate * 100).toFixed(1)}%
                </div>
                <div className="metric-label">
                  This should decrease as you teach more objects
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <p>Loading metrics...</p>
            </div>
          )}
        </div>
      )}

      {/* Teach Modal */}
      {teachModal && (
        <div className="modal-overlay" onClick={() => setTeachModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Teach Object</h2>
            {teachModal.thumb && (
              <img
                src={`data:image/jpeg;base64,${teachModal.thumb}`}
                alt="object"
                style={{
                  width: "100%",
                  maxHeight: "200px",
                  objectFit: "contain",
                  borderRadius: "8px",
                  marginBottom: "1rem",
                }}
              />
            )}
            <p className="meta" style={{ marginBottom: "0.5rem" }}>
              Track: {teachModal.trackId}
            </p>
            <input
              className="input"
              style={{ width: "100%" }}
              placeholder="What is this object?"
              value={teachName}
              onChange={(e) => setTeachName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleTeach()}
              autoFocus
            />
            <div className="actions">
              <button className="btn" onClick={() => setTeachModal(null)}>
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={handleTeach}
                disabled={teachLoading || !teachName.trim()}
              >
                {teachLoading ? "Saving..." : "Teach"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
