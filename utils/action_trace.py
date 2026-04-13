from __future__ import annotations

import copy
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from utils.message.datatype import ExecutionMode, RobotAction
from utils.message.message_convert import robot_action_to_named_arrays

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - graceful runtime fallback
    go = None
    make_subplots = None
    _PLOTLY_IMPORT_ERROR = exc

try:
    import cv2 as cv
    _CV_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - graceful runtime fallback
    cv = None
    _CV_IMPORT_ERROR = exc


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _normalize_sequence(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D action sequence, got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def _normalize_vector(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    return array.reshape(-1).astype(np.float32, copy=False)


def _normalize_image(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected image with shape HxWx3, got {array.shape}")
    if (float(np.max(array)) if array.size else 0.0) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _component_labels(key: str, dim: int) -> list[str]:
    if key.endswith("_ee_pose") and dim >= 7:
        base = ["x", "y", "z", "qx", "qy", "qz", "qw"]
        return [f"{key}.{label}" for label in base[:dim]]
    if "arm" in key:
        return [f"{key}.j{i + 1}" for i in range(dim)]
    if "gripper" in key:
        return [key]
    if key == "torso":
        return [f"torso.{i + 1}" for i in range(dim)]
    if key == "chassis":
        return [f"chassis.{i + 1}" for i in range(dim)]
    return [f"{key}.{i + 1}" for i in range(dim)]


class ActionTraceRecorder:
    def __init__(
        self,
        output_dir: Path,
        execution_mode: ExecutionMode,
        control_frequency: float,
        action_steps: int,
        render_every_n_publishes: int = 5,
        keep_chunks: int = 12,
        keep_publish_events: int = 400,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.execution_mode = execution_mode
        self.control_frequency = float(control_frequency)
        self.action_steps = int(action_steps)
        self.dt = 1.0 / self.control_frequency
        self.chunk_duration_s = self.action_steps * self.dt
        self.last_step_horizon_s = max(self.action_steps - 1, 0) * self.dt

        self.render_every_n_publishes = max(int(render_every_n_publishes), 1)
        self.chunk_history: deque[dict[str, Any]] = deque(maxlen=max(int(keep_chunks), 1))
        self.publish_history: deque[dict[str, Any]] = deque(maxlen=max(int(keep_publish_events), 1))

        self.trace_path = self.output_dir / "trace.jsonl"
        self.latest_summary_path = self.output_dir / "latest_summary.json"
        self.latest_dashboard_path = self.output_dir / "latest_dashboard.html"
        self.live_dashboard_path = self.output_dir / "live_dashboard.html"
        self.object_view_path = self.output_dir / "latest_object_view.jpg"
        self.condition_view_path = self.output_dir / "latest_condition_view.jpg"

        self._lock = threading.Lock()
        self._next_chunk_id = 1
        self._next_publish_id = 1
        self._publish_events_since_render = 0
        self._plotly_warning_emitted = False

        logger.info(f"Action trace output: {self.output_dir}")
        self._write_live_dashboard_locked()

    def record_predicted_chunk(
        self,
        actions: dict[str, Any],
        obs_time: float,
        infer_start: float,
        infer_end: float,
        instruction: str | None = None,
        model_instruction: str | None = None,
        obs_state: dict[str, Any] | None = None,
        object_context: dict[str, Any] | None = None,
    ) -> int:
        named_actions = {
            key: _normalize_sequence(value)
            for key, value in actions.items()
            if value is not None
        }
        named_state = self._normalize_state_snapshot(obs_state)

        first_key = next(iter(named_actions), None)
        steps = named_actions[first_key].shape[0] if first_key is not None else self.action_steps
        last_step_time = float(obs_time + max(steps - 1, 0) * self.dt)

        with self._lock:
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
            object_meta = self._write_object_assets_locked(object_context)

            record = {
                "type": "chunk",
                "chunk_id": chunk_id,
                "instruction": instruction or "",
                "user_instruction": instruction or "",
                "model_instruction": model_instruction or instruction or "",
                "obs_time": float(obs_time),
                "infer_start": float(infer_start),
                "infer_end": float(infer_end),
                "infer_cost": float(infer_end - infer_start),
                "control_frequency": self.control_frequency,
                "action_steps": steps,
                "chunk_duration_s": float(steps * self.dt),
                "last_step_horizon_s": float(max(steps - 1, 0) * self.dt),
                "last_step_time": last_step_time,
                "ready_margin_s": float(last_step_time - infer_end),
                "is_ready_before_last_step": bool(infer_end <= last_step_time),
                "actions": {key: value.tolist() for key, value in named_actions.items()},
                "obs_state": {key: value.tolist() for key, value in named_state.items()},
                "object_context": object_meta,
                "created_at": float(time.time()),
            }
            self.chunk_history.append(record)
            self._append_jsonl_locked(record)
            self._write_latest_summary_locked()
            self._render_dashboard_locked(trigger="chunk")
            return chunk_id

    def record_publish_event(
        self,
        action: RobotAction,
        publish_time: float,
        manager_debug: dict[str, Any] | None,
        feedback_snapshot: dict[str, Any] | None = None,
    ) -> None:
        named_action = {
            key: _normalize_vector(value)
            for key, value in robot_action_to_named_arrays(action).items()
            if value is not None
        }
        feedback_state, feedback_timestamps = self._normalize_feedback_snapshot(feedback_snapshot)
        self._record_publish_base(
            published=True,
            publish_time=publish_time,
            manager_debug=manager_debug,
            action=named_action,
            feedback_state=feedback_state,
            feedback_timestamps=feedback_timestamps,
        )

    def record_publish_miss(
        self,
        publish_time: float,
        manager_debug: dict[str, Any] | None,
        feedback_snapshot: dict[str, Any] | None = None,
    ) -> None:
        feedback_state, feedback_timestamps = self._normalize_feedback_snapshot(feedback_snapshot)
        self._record_publish_base(
            published=False,
            publish_time=publish_time,
            manager_debug=manager_debug,
            action={},
            feedback_state=feedback_state,
            feedback_timestamps=feedback_timestamps,
        )

    def _record_publish_base(
        self,
        published: bool,
        publish_time: float,
        manager_debug: dict[str, Any] | None,
        action: dict[str, np.ndarray],
        feedback_state: dict[str, np.ndarray],
        feedback_timestamps: dict[str, float],
    ) -> None:
        safe_debug = _json_safe(copy.deepcopy(manager_debug or {}))
        record = {
            "type": "publish",
            "publish_id": None,
            "published": bool(published),
            "publish_time": float(publish_time),
            "primary_chunk_id": safe_debug.get("primary_chunk_id") or safe_debug.get("chunk_id"),
            "action": {key: value.tolist() for key, value in action.items()},
            "feedback": {key: value.tolist() for key, value in feedback_state.items()},
            "feedback_timestamps": feedback_timestamps,
            "manager_debug": safe_debug,
        }

        with self._lock:
            record["publish_id"] = self._next_publish_id
            self._next_publish_id += 1

            self.publish_history.append(record)
            self._append_jsonl_locked(record)
            self._publish_events_since_render += 1
            if self._publish_events_since_render >= self.render_every_n_publishes:
                self._write_latest_summary_locked()
                self._publish_events_since_render = 0

    def _normalize_state_snapshot(self, obs_state: dict[str, Any] | None) -> dict[str, np.ndarray]:
        if obs_state is None:
            return {}
        if isinstance(obs_state, dict):
            return {
                key: _normalize_vector(value)
                for key, value in obs_state.items()
                if value is not None
            }
        return {"state": _normalize_vector(obs_state)}

    def _normalize_feedback_snapshot(
        self,
        feedback_snapshot: dict[str, Any] | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        if feedback_snapshot is None:
            return {}, {}

        raw_state = feedback_snapshot.get("state", feedback_snapshot)
        raw_timestamps = feedback_snapshot.get("timestamps", {})
        state = {
            key: _normalize_vector(value)
            for key, value in raw_state.items()
            if value is not None
        }
        timestamps = {
            key: float(value)
            for key, value in raw_timestamps.items()
            if value is not None
        }
        return state, timestamps

    def _write_object_assets_locked(self, object_context: dict[str, Any] | None) -> dict[str, Any]:
        meta = {
            "bbox": None,
            "has_object_view": False,
            "has_condition_view": False,
            "object_view_filename": self.object_view_path.name,
            "condition_view_filename": self.condition_view_path.name,
            "note": "Object location is shown in camera image space only. This stack does not expose a trusted 3D object pose.",
        }

        if object_context is None:
            self._cleanup_object_assets_locked()
            return meta

        bbox = object_context.get("bbox")
        if bbox is not None and len(bbox) == 4:
            meta["bbox"] = [int(v) for v in bbox]

        head_rgb = object_context.get("head_rgb")
        if head_rgb is not None:
            try:
                image = _normalize_image(head_rgb)
                image = self._draw_bbox_overlay(image, meta["bbox"])
                self._write_rgb_image_locked(self.object_view_path, image)
                meta["has_object_view"] = True
            except Exception as exc:
                logger.warning(f"Failed to write object view image: {exc}")
                if self.object_view_path.exists():
                    self.object_view_path.unlink()
        elif self.object_view_path.exists():
            self.object_view_path.unlink()

        condition_image = object_context.get("condition_image")
        if condition_image is not None:
            try:
                image = _normalize_image(condition_image)
                self._write_rgb_image_locked(self.condition_view_path, image)
                meta["has_condition_view"] = True
            except Exception as exc:
                logger.warning(f"Failed to write condition image: {exc}")
                if self.condition_view_path.exists():
                    self.condition_view_path.unlink()
        elif self.condition_view_path.exists():
            self.condition_view_path.unlink()

        return meta

    def _cleanup_object_assets_locked(self) -> None:
        for path in (self.object_view_path, self.condition_view_path):
            if path.exists():
                path.unlink()

    def _draw_bbox_overlay(self, image: np.ndarray, bbox: list[int] | None) -> np.ndarray:
        vis = image.copy()
        if bbox is None or len(bbox) != 4:
            return vis
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, vis.shape[1] - 1))
        x2 = max(0, min(x2, vis.shape[1] - 1))
        y1 = max(0, min(y1, vis.shape[0] - 1))
        y2 = max(0, min(y2, vis.shape[0] - 1))
        if cv is not None:
            cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv.circle(vis, (cx, cy), 5, (255, 255, 0), -1)
        return vis

    def _write_rgb_image_locked(self, path: Path, image: np.ndarray) -> None:
        if cv is None:
            raise RuntimeError(f"OpenCV is unavailable: {_CV_IMPORT_ERROR}")
        bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if not cv.imwrite(str(path), bgr):
            raise RuntimeError(f"cv.imwrite returned False for {path}")

    def _append_jsonl_locked(self, record: dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_json_safe(record), ensure_ascii=True) + "\n")

    def _write_latest_summary_locked(self) -> None:
        latest_chunk = self.chunk_history[-1] if self.chunk_history else None
        recent_chunks = list(self.chunk_history)
        recent_publish_events = list(self.publish_history)[-120:]
        summary = {
            "latest_chunk": _json_safe(latest_chunk),
            "recent_chunks": _json_safe(recent_chunks),
            "recent_publish_events": _json_safe(recent_publish_events),
            "trace_path": str(self.trace_path),
            "dashboard_path": str(self.latest_dashboard_path),
            "live_dashboard_path": str(self.live_dashboard_path),
            "chunk_count_in_memory": len(self.chunk_history),
            "publish_count_in_memory": len(self.publish_history),
        }
        self.latest_summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def _render_dashboard_locked(self, trigger: str) -> None:
        if not self.chunk_history:
            return
        if go is None or make_subplots is None:
            if not self._plotly_warning_emitted:
                logger.warning(
                    f"Plotly is unavailable, skipping HTML dashboard rendering: {_PLOTLY_IMPORT_ERROR}"
                )
                self._plotly_warning_emitted = True
            return

        latest_chunk = self.chunk_history[-1]
        timing_fig = self._build_timing_figure()
        action_fig = self._build_action_figure(latest_chunk)
        infer_fig = self._build_infer_history_figure()
        path_fig = self._build_path_figure(latest_chunk)

        publish_events = self._events_for_chunk(latest_chunk["chunk_id"])
        publish_count = sum(1 for event in publish_events if event["published"])
        miss_count = sum(1 for event in publish_events if not event["published"])
        timing_warning = ""
        if latest_chunk["infer_cost"] > latest_chunk["last_step_horizon_s"]:
            timing_warning = (
                f"Inference ({latest_chunk['infer_cost']:.2f}s) is slower than the "
                f"chunk horizon ({latest_chunk['last_step_horizon_s']:.2f}s to the last step)."
            )

        html_parts = [
            "<html><head><meta charset='utf-8'>",
            "<title>EFMNode Action Trace</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 24px; background: #fafafa; color: #111; }",
            "h1, h2 { margin-bottom: 8px; }",
            "p, li { line-height: 1.45; }",
            "code { background: #eee; padding: 2px 4px; border-radius: 4px; }",
            ".warning { color: #8a1c1c; font-weight: 700; }",
            ".summary { display: grid; grid-template-columns: repeat(2, minmax(240px, 1fr)); gap: 8px 18px; margin-bottom: 20px; }",
            ".summary div { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; }",
            "</style></head><body>",
            "<h1>EFMNode Action Chunk Trace</h1>",
            f"<p>Render trigger: <code>{trigger}</code>. Files are being updated in <code>{self.output_dir}</code>.</p>",
            "<div class='summary'>",
            f"<div><strong>User Instruction</strong><br>{latest_chunk.get('user_instruction') or latest_chunk.get('instruction') or '(empty)'}</div>",
            f"<div><strong>Model Prompt</strong><br>{latest_chunk.get('model_instruction') or '(empty)'}</div>",
            f"<div><strong>Chunk ID</strong><br>{latest_chunk['chunk_id']}</div>",
            f"<div><strong>Inference Cost</strong><br>{latest_chunk['infer_cost']:.3f}s</div>",
            f"<div><strong>Chunk Horizon</strong><br>{latest_chunk['chunk_duration_s']:.3f}s total, {latest_chunk['last_step_horizon_s']:.3f}s to last step</div>",
            f"<div><strong>Ready Margin</strong><br>{latest_chunk['ready_margin_s']:.3f}s</div>",
            f"<div><strong>Publishes For Latest Chunk</strong><br>{publish_count} publishes, {miss_count} misses</div>",
            "</div>",
        ]
        if timing_warning:
            html_parts.append(f"<p class='warning'>{timing_warning}</p>")
        html_parts.extend(
            [
                "<h2>Timing</h2>",
                timing_fig.to_html(full_html=False, include_plotlyjs="cdn"),
                "<h2>Latest Chunk vs Executed Actions</h2>",
                action_fig.to_html(full_html=False, include_plotlyjs=False),
                "<h2>Inference History</h2>",
                infer_fig.to_html(full_html=False, include_plotlyjs=False),
            ]
        )
        if path_fig is not None:
            html_parts.extend(
                [
                    "<h2>3D End-Effector Path</h2>",
                    path_fig.to_html(full_html=False, include_plotlyjs=False),
                ]
            )
        html_parts.extend(
            [
                f"<p>Trace JSONL: <code>{self.trace_path}</code></p>",
                f"<p>Summary JSON: <code>{self.latest_summary_path}</code></p>",
                f"<p>Live Dashboard: <code>{self.live_dashboard_path}</code></p>",
                "</body></html>",
            ]
        )
        self.latest_dashboard_path.write_text("".join(html_parts), encoding="utf-8")

    def _write_live_dashboard_locked(self) -> None:
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>EFMNode Live Trace</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: sans-serif;
      margin: 24px;
      background: #fafafa;
      color: #111;
    }}
    h1, h2 {{
      margin-bottom: 8px;
    }}
    p, li {{
      line-height: 1.45;
    }}
    code {{
      background: #eee;
      padding: 2px 4px;
      border-radius: 4px;
    }}
    .warning {{
      color: #8a1c1c;
      font-weight: 700;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(2, minmax(240px, 1fr));
      gap: 8px 18px;
      margin-bottom: 20px;
    }}
    .summary div {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .explain {{
      margin: 8px 0 18px 0;
      color: #444;
    }}
    .muted {{
      color: #666;
    }}
    .image-row {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .image-card {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 10px 12px;
      min-width: 280px;
    }}
    .image-card img {{
      max-width: 420px;
      width: 100%;
      border-radius: 6px;
      border: 1px solid #ddd;
      display: block;
    }}
  </style>
</head>
<body>
  <h1>EFMNode Live Action Trace</h1>
  <p>
    This page updates in-place every second from <code>latest_summary.json</code>.
    Static snapshot: <code>{self.latest_dashboard_path.name}</code>.
  </p>
  <p class="muted" id="status">Waiting for trace data...</p>
  <div id="warning" class="warning"></div>
  <div class="summary" id="summary"></div>

  <h2>How To Read This</h2>
  <div class="explain">
    <p><strong>Object view</strong> shows where the object was found in the head camera image. This is 2D image space, not a true 3D object pose.</p>
    <p><strong>Robot snapshot</strong> compares the robot's current measured state with the latest command being sent. In <code>JOINT_STATE</code> mode, this is the clearest target-vs-current view.</p>
    <p><strong>3D robot view</strong> shows the current arm positions, the latest commanded positions, and recent measured motion. If the policy outputs EE poses, it also shows the full target chunk path.</p>
    <p><strong>Timing</strong> shows when the chunk started, how long inference took, and whether publish ticks had no valid action to send.</p>
  </div>

  <h2>Object View</h2>
  <div class="explain">If the model or bbox helper localized an object, you'll see the latest head-camera frame with the object box here. The optional crop is the image region used for image-conditioned prompts.</div>
  <div id="object_view"></div>

  <h2>Robot Snapshot</h2>
  <div class="explain">Measured state versus the latest command. This is the quickest way to see whether the robot is actually moving toward the current target.</div>
  <div id="snapshot_fig"></div>

  <h2>Latest Chunk vs Commands vs Feedback</h2>
  <div class="explain">For each action head, the line is the model's chunk, the diamonds are the commands actually sent, and the dotted line is the measured robot feedback when that signal exists.</div>
  <div id="action_fig"></div>

  <h2>3D Robot View</h2>
  <div class="explain">Large markers are the robot's current arm positions. Diamond markers are the latest commanded targets. If EE-pose actions are available, the target chunk path is shown as a line.</div>
  <div id="path_note" class="muted"></div>
  <div id="path_fig"></div>

  <h2>Timing</h2>
  <div class="explain">Blue is the chunk horizon, orange is inference time, green is when commands were actually published, and red x markers are publish ticks where no action was available.</div>
  <div id="timing_fig"></div>

  <h2>Inference History</h2>
  <div class="explain">This answers one simple question: is inference finishing before the chunk is already stale?</div>
  <div id="infer_fig"></div>

  <script>
    const SUMMARY_URL = "latest_summary.json";
    const REFRESH_MS = 1000;
    let lastFingerprint = "";

    function pretty(value) {{
      if (value === null || value === undefined || value === "") return "(empty)";
      return String(value);
    }}

    function componentLabels(key, dim) {{
      if (key.endsWith("_ee_pose") && dim >= 7) {{
        return ["x", "y", "z", "qx", "qy", "qz", "qw"].slice(0, dim).map(label => `${{key}}.${{label}}`);
      }}
      if (key.includes("arm")) return Array.from({{length: dim}}, (_, i) => `${{key}}.j${{i + 1}}`);
      if (key.includes("gripper")) return [key];
      if (key === "torso") return Array.from({{length: dim}}, (_, i) => `torso.${{i + 1}}`);
      if (key === "chassis") return Array.from({{length: dim}}, (_, i) => `chassis.${{i + 1}}`);
      return Array.from({{length: dim}}, (_, i) => `${{key}}.${{i + 1}}`);
    }}

    function eventUsesChunk(event, chunkId) {{
      if (!event || chunkId === null || chunkId === undefined) return false;
      if (event.primary_chunk_id === chunkId) return true;
      const debug = event.manager_debug || {{}};
      const selected = debug.selected_chunk_ids || [];
      const queued = debug.queue_chunk_ids || [];
      return selected.includes(chunkId) || queued.includes(chunkId);
    }}

    function normalize2D(seq) {{
      if (!Array.isArray(seq)) return [];
      if (seq.length === 0) return [];
      if (!Array.isArray(seq[0])) return seq.map(v => [v]);
      return seq;
    }}

    function buildSummary(summary) {{
      const latest = summary.latest_chunk;
      const container = document.getElementById("summary");
      if (!latest) {{
        container.innerHTML = "<div><strong>No chunks yet</strong><br>Waiting for first inference output.</div>";
        return;
      }}
      const events = (summary.recent_publish_events || []).filter(event => eventUsesChunk(event, latest.chunk_id));
      const publishCount = events.filter(event => event.published).length;
      const missCount = events.filter(event => !event.published).length;
      const latestEvent = [...(summary.recent_publish_events || [])].reverse().find(event => event.published || !event.published);
      const inferCost = Number(latest.infer_cost || 0);
      const lastStepHorizon = Number(latest.last_step_horizon_s || 0);
      const margin = Number(latest.ready_margin_s || 0);
      const timingVerdict = inferCost > lastStepHorizon
        ? `Inference finished ${{Math.abs(margin).toFixed(2)}}s after the chunk's last target step.`
        : `Inference finished ${{margin.toFixed(2)}}s before the chunk's last target step.`;
      const controlVerdict = latestEvent
        ? (latestEvent.published
            ? `Latest publish sent a command at ${{new Date(latestEvent.publish_time * 1000).toLocaleTimeString()}}.`
            : `Latest publish tick had no valid action to send.`)
        : "No publish ticks recorded yet.";
      const objectVerdict = latest.object_context && latest.object_context.bbox
        ? `Object bbox found in the head image at [${{latest.object_context.bbox.join(", ")}}].`
        : "No object bbox is available for this chunk.";
      container.innerHTML = [
        `<div><strong>Task</strong><br>${{pretty(latest.user_instruction || latest.instruction)}}<br><span class="muted">Model prompt: ${{pretty(latest.model_instruction)}}</span></div>`,
        `<div><strong>Timing Verdict</strong><br>${{timingVerdict}}<br><span class="muted">Infer: ${{inferCost.toFixed(2)}}s | chunk to last step: ${{lastStepHorizon.toFixed(2)}}s</span></div>`,
        `<div><strong>Control Verdict</strong><br>${{controlVerdict}}<br><span class="muted">Latest chunk: #${{pretty(latest.chunk_id)}} | publishes: ${{publishCount}} | misses: ${{missCount}}</span></div>`,
        `<div><strong>Object Signal</strong><br>${{objectVerdict}}<br><span class="muted">Camera-space only; no trusted 3D object pose is available in this live stack.</span></div>`
      ].join("");
    }}

    function buildObjectView(summary) {{
      const latest = summary.latest_chunk;
      const container = document.getElementById("object_view");
      if (!latest) {{
        container.innerHTML = "<p class='muted'>Waiting for first chunk.</p>";
        return;
      }}

      const objectContext = latest.object_context || {{}};
      const ts = Date.now();
      const cards = [];

      if (objectContext.has_object_view) {{
        cards.push(`
          <div class="image-card">
            <strong>Head Camera + Object Box</strong>
            <p class="muted">Green box = object location used for interpretation. This is 2D image space.</p>
            <img src="${{objectContext.object_view_filename}}?ts=${{ts}}" alt="Latest head camera with object box">
          </div>
        `);
      }}

      if (objectContext.has_condition_view) {{
        cards.push(`
          <div class="image-card">
            <strong>Condition Crop</strong>
            <p class="muted">This crop is what the image-conditioned prompt uses when that mode is enabled.</p>
            <img src="${{objectContext.condition_view_filename}}?ts=${{ts}}" alt="Latest condition crop">
          </div>
        `);
      }}

      if (!cards.length) {{
        container.innerHTML = "<p class='muted'>No object image or bbox is available for the latest chunk.</p>";
        return;
      }}

      container.innerHTML = `<div class="image-row">${{cards.join("")}}</div>`;
    }}

    function flattenSnapshot(latest, latestCommand) {{
      const rows = [];
      const pushVector = (labelPrefix, currentVec, commandVec, dims) => {{
        const safeCurrent = Array.isArray(currentVec) ? currentVec : [];
        const safeCommand = Array.isArray(commandVec) ? commandVec : [];
        const labels = componentLabels(labelPrefix, dims);
        for (let i = 0; i < dims; i += 1) {{
          rows.push({{
            label: labels[i],
            current: safeCurrent[i] ?? null,
            command: safeCommand[i] ?? null,
          }});
        }}
      }};

      const state = latest.obs_state || {{}};
      const command = latestCommand && latestCommand.action ? latestCommand.action : {{}};
      const orderedKeys = ["left_arm", "right_arm", "left_ee_pose", "right_ee_pose", "left_gripper", "right_gripper", "torso"];

      for (const key of orderedKeys) {{
        const currentVec = state[key];
        const commandVec = command[key];
        if (!currentVec && !commandVec) continue;
        const dims = Math.max(
          Array.isArray(currentVec) ? currentVec.length : 0,
          Array.isArray(commandVec) ? commandVec.length : 0
        );
        if (dims > 0) pushVector(key, currentVec, commandVec, dims);
      }}

      return rows;
    }}

    function buildSnapshotFigure(summary) {{
      const latest = summary.latest_chunk;
      const latestCommand = [...(summary.recent_publish_events || [])].reverse().find(event => event.published && eventUsesChunk(event, latest ? latest.chunk_id : null));
      if (!latest) {{
        Plotly.react("snapshot_fig", [], {{
          height: 280,
          template: "plotly_white",
          title: "Waiting for first chunk..."
        }}, {{responsive: true}});
        return;
      }}

      const rows = flattenSnapshot(latest, latestCommand);
      if (!rows.length) {{
        Plotly.react("snapshot_fig", [], {{
          height: 280,
          template: "plotly_white",
          title: "No current-vs-command snapshot available yet"
        }}, {{responsive: true}});
        return;
      }}

      const labels = rows.map(row => row.label);
      const current = rows.map(row => row.current);
      const command = rows.map(row => row.command);

      Plotly.react("snapshot_fig", [
        {{
          type: "bar",
          x: labels,
          y: current,
          name: "Current measured",
          marker: {{color: "#222"}}
        }},
        {{
          type: "bar",
          x: labels,
          y: command,
          name: "Latest command",
          marker: {{color: "#2ca02c"}}
        }}
      ], {{
        barmode: "group",
        height: 380,
        margin: {{l: 50, r: 20, t: 40, b: 120}},
        template: "plotly_white",
        xaxis: {{tickangle: -35}},
        yaxis: {{title: "Value"}},
        title: "Current measured state vs latest command"
      }}, {{responsive: true}});
    }}

    function buildTiming(summary) {{
      const chunks = summary.recent_chunks || [];
      const traces = [];
      const legendSeen = new Set();

      function addSegment(start, end, y, name, color, width) {{
        traces.push({{
          type: "scatter",
          x: [start, end],
          y: [y, y],
          mode: "lines",
          line: {{color, width}},
          name,
          showlegend: !legendSeen.has(name),
          hoverinfo: "skip"
        }});
        legendSeen.add(name);
      }}

      for (const chunk of chunks.slice(-6)) {{
        const label = `Chunk ${{chunk.chunk_id}}`;
        addSegment(chunk.obs_time, chunk.last_step_time, label, "Chunk horizon", "#1f77b4", 14);
        addSegment(chunk.infer_start, chunk.infer_end, label, "Inference", "#ff7f0e", 10);

        const chunkEvents = (summary.recent_publish_events || []).filter(event => eventUsesChunk(event, chunk.chunk_id));
        const publishTimes = chunkEvents.filter(event => event.published).map(event => event.publish_time);
        const missTimes = chunkEvents.filter(event => !event.published).map(event => event.publish_time);

        if (publishTimes.length) {{
          addSegment(Math.min(...publishTimes), Math.max(...publishTimes), label, "Publish window", "#2ca02c", 8);
          traces.push({{
            type: "scatter",
            x: publishTimes,
            y: publishTimes.map(() => label),
            mode: "markers",
            marker: {{color: "#2ca02c", size: 9}},
            name: "Published tick",
            showlegend: !legendSeen.has("Published tick"),
            hovertemplate: "publish=%{{x:.3f}}<extra></extra>"
          }});
          legendSeen.add("Published tick");
        }}
        if (missTimes.length) {{
          traces.push({{
            type: "scatter",
            x: missTimes,
            y: missTimes.map(() => label),
            mode: "markers",
            marker: {{color: "#d62728", size: 10, symbol: "x"}},
            name: "No action available",
            showlegend: !legendSeen.has("No action available"),
            hovertemplate: "miss=%{{x:.3f}}<extra></extra>"
          }});
          legendSeen.add("No action available");
        }}
      }}

      Plotly.react("timing_fig", traces, {{
        height: Math.max(340, 120 + 70 * Math.max(chunks.slice(-6).length, 1)),
        margin: {{l: 40, r: 20, t: 40, b: 40}},
        template: "plotly_white",
        xaxis: {{title: "Wall Time (s)"}},
        yaxis: {{title: "Chunk"}},
        title: "Chunk lifetime, inference delay, and publish activity"
      }}, {{responsive: true}});
    }}

    function buildActionFigure(summary) {{
      const latest = summary.latest_chunk;
      if (!latest || !latest.actions) {{
        Plotly.react("action_fig", [], {{
          height: 300,
          template: "plotly_white",
          title: "Waiting for first chunk..."
        }}, {{responsive: true}});
        return;
      }}

      const groups = Object.keys(latest.actions);
      const traces = [];
      const layout = {{
        grid: {{rows: Math.max(groups.length, 1), columns: 1, pattern: "independent"}},
        height: Math.max(360, 280 * Math.max(groups.length, 1)),
        margin: {{l: 60, r: 20, t: 60, b: 50}},
        template: "plotly_white",
        legend: {{orientation: "h"}}
      }};

      const chunkEvents = (summary.recent_publish_events || []).filter(event => eventUsesChunk(event, latest.chunk_id));
      const xUpperCandidates = [latest.chunk_duration_s || 0.5];

      groups.forEach((key, idx) => {{
        const row = idx + 1;
        const axisSuffix = row === 1 ? "" : String(row);
        const pred = normalize2D(latest.actions[key]);
        const labels = componentLabels(key, pred[0] ? pred[0].length : 0);
        const xPred = Array.from({{length: pred.length}}, (_, i) => i * (1.0 / (latest.control_frequency || 15.0)));

        labels.forEach((label, dimIdx) => {{
          traces.push({{
            type: "scatter",
            x: xPred,
            y: pred.map(step => step[dimIdx]),
            mode: "lines+markers",
            name: `pred ${{label}}`,
            legendgroup: `pred-${{key}}`,
            showlegend: row === 1,
            hovertemplate: `${{label}}<br>t=%{{x:.3f}}s<br>value=%{{y:.4f}}<extra></extra>`,
            xaxis: `x${{axisSuffix}}`,
            yaxis: `y${{axisSuffix}}`
          }});
        }});

        const publishedEvents = chunkEvents.filter(event => event.published && event.action && event.action[key]);
        if (publishedEvents.length) {{
          const xExec = publishedEvents.map(event => event.publish_time - latest.obs_time);
          xUpperCandidates.push(...xExec);
          const yExec = publishedEvents.map(event => event.action[key]);
          labels.slice(0, yExec[0].length).forEach((label, dimIdx) => {{
            traces.push({{
              type: "scatter",
              x: xExec,
              y: yExec.map(step => step[dimIdx]),
              mode: "markers",
              marker: {{size: 9, symbol: "diamond"}},
              name: `cmd ${{label}}`,
              legendgroup: `cmd-${{key}}`,
              showlegend: row === 1,
              hovertemplate: "publish dt=%{{x:.3f}}s<br>value=%{{y:.4f}}<extra></extra>",
              xaxis: `x${{axisSuffix}}`,
              yaxis: `y${{axisSuffix}}`
            }});
          }});
        }}

        const feedbackEvents = chunkEvents.filter(event => event.published && event.feedback && event.feedback[key]);
        if (feedbackEvents.length) {{
          const xFeedback = feedbackEvents.map(event => event.publish_time - latest.obs_time);
          xUpperCandidates.push(...xFeedback);
          const yFeedback = feedbackEvents.map(event => Array.isArray(event.feedback[key]) ? event.feedback[key] : [event.feedback[key]]);
          labels.slice(0, yFeedback[0].length).forEach((label, dimIdx) => {{
            traces.push({{
              type: "scatter",
              x: xFeedback,
              y: yFeedback.map(step => step[dimIdx]),
              mode: "lines+markers",
              line: {{dash: "dot"}},
              marker: {{size: 7}},
              name: `fb ${{label}}`,
              legendgroup: `fb-${{key}}`,
              showlegend: row === 1,
              hovertemplate: "feedback dt=%{{x:.3f}}s<br>value=%{{y:.4f}}<extra></extra>",
              xaxis: `x${{axisSuffix}}`,
              yaxis: `y${{axisSuffix}}`
            }});
          }});
        }}

        layout[`xaxis${{axisSuffix}}`] = {{
          title: row === groups.length ? "Seconds Since Observation Used For This Chunk" : "",
          range: [0, Math.max(0.5, Math.max(...xUpperCandidates) + 0.2)]
        }};
        layout[`yaxis${{axisSuffix}}`] = {{title: key}};
      }});

      Plotly.react("action_fig", traces, layout, {{responsive: true}});
    }}

    function buildInferFigure(summary) {{
      const chunks = summary.recent_chunks || [];
      const inferCosts = chunks.map(chunk => chunk.infer_cost || 0);
      const chunkIds = chunks.map(chunk => chunk.chunk_id);
      const lastHorizon = summary.latest_chunk ? (summary.latest_chunk.last_step_horizon_s || 0) : 0;
      const chunkDuration = summary.latest_chunk ? (summary.latest_chunk.chunk_duration_s || 0) : 0;

      const traces = [{{
        type: "scatter",
        x: chunkIds,
        y: inferCosts,
        mode: "lines+markers",
        name: "Inference cost"
      }}];

      Plotly.react("infer_fig", traces, {{
        height: 320,
        margin: {{l: 50, r: 20, t: 50, b: 40}},
        template: "plotly_white",
        xaxis: {{title: "Chunk ID"}},
        yaxis: {{title: "Seconds"}},
        title: "Inference time compared with chunk horizon",
        shapes: [
          {{
            type: "line",
            xref: "paper",
            x0: 0,
            x1: 1,
            y0: lastHorizon,
            y1: lastHorizon,
            line: {{dash: "dash", color: "#d62728"}}
          }},
          {{
            type: "line",
            xref: "paper",
            x0: 0,
            x1: 1,
            y0: chunkDuration,
            y1: chunkDuration,
            line: {{dash: "dot", color: "#1f77b4"}}
          }}
        ],
        annotations: [
          {{
            xref: "paper",
            x: 1,
            y: lastHorizon,
            yref: "y",
            text: `Last-step horizon (${{lastHorizon.toFixed(2)}}s)`,
            showarrow: false,
            xanchor: "right",
            bgcolor: "white"
          }},
          {{
            xref: "paper",
            x: 1,
            y: chunkDuration,
            yref: "y",
            text: `Nominal chunk duration (${{chunkDuration.toFixed(2)}}s)`,
            showarrow: false,
            xanchor: "right",
            bgcolor: "white"
          }}
        ]
      }}, {{responsive: true}});
    }}

    function buildPathFigure(summary) {{
      const latest = summary.latest_chunk;
      const note = document.getElementById("path_note");
      if (!latest) {{
        note.textContent = "Waiting for first chunk.";
        Plotly.react("path_fig", [], {{height: 420, template: "plotly_white"}}, {{responsive: true}});
        return;
      }}

      const chunkEvents = (summary.recent_publish_events || []).filter(event => eventUsesChunk(event, latest.chunk_id));
      const latestCommand = [...chunkEvents].reverse().find(event => event.published);
      const traces = [];
      const obsState = latest.obs_state || {{}};
      const armConfigs = [
        {{ key: "left_ee_pose", arm: "Left arm", predColor: "#4c78a8", cmdColor: "#2ca02c", fbColor: "#7f7f7f" }},
        {{ key: "right_ee_pose", arm: "Right arm", predColor: "#e45756", cmdColor: "#2ca02c", fbColor: "#ff9d9a" }},
      ];

      let hasTargetPath = false;
      let hasAny3D = false;

      for (const cfg of armConfigs) {{
        const pred = latest.actions[cfg.key] ? normalize2D(latest.actions[cfg.key]) : [];
        if (pred.length && pred[0].length >= 3) {{
          hasTargetPath = true;
          hasAny3D = true;
          traces.push({{
            type: "scatter3d",
            x: pred.map(step => step[0]),
            y: pred.map(step => step[1]),
            z: pred.map(step => step[2]),
            mode: "lines",
            name: `${{cfg.arm}} target chunk`,
            line: {{width: 5, color: cfg.predColor}}
          }});
        }}

        const currentPose = obsState[cfg.key];
        if (Array.isArray(currentPose) && currentPose.length >= 3) {{
          hasAny3D = true;
          const currentGripKey = cfg.key.startsWith("left") ? "left_gripper" : "right_gripper";
          const currentGrip = obsState[currentGripKey];
          traces.push({{
            type: "scatter3d",
            x: [currentPose[0]],
            y: [currentPose[1]],
            z: [currentPose[2]],
            mode: "markers",
            name: `${{cfg.arm}} current`,
            marker: {{size: 8, color: cfg.predColor, symbol: "circle"}},
            hovertemplate: `${{cfg.arm}} current<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<br>gripper=${{Array.isArray(currentGrip) ? Number(currentGrip[0] || 0).toFixed(3) : "n/a"}}<extra></extra>`
          }});
        }}

        const feedback = chunkEvents.filter(event => event.published && event.feedback && event.feedback[cfg.key]);
        if (feedback.length) {{
          hasAny3D = true;
          traces.push({{
            type: "scatter3d",
            x: feedback.map(event => event.feedback[cfg.key][0]),
            y: feedback.map(event => event.feedback[cfg.key][1]),
            z: feedback.map(event => event.feedback[cfg.key][2]),
            mode: "lines+markers",
            name: `${{cfg.arm}} measured path`,
            marker: {{size: 3, color: cfg.fbColor}},
            line: {{width: 3, color: cfg.fbColor}}
          }});
        }}

        const cmdPose = latestCommand && latestCommand.action ? latestCommand.action[cfg.key] : null;
        if (Array.isArray(cmdPose) && cmdPose.length >= 3) {{
          hasAny3D = true;
          const cmdGripKey = cfg.key.startsWith("left") ? "left_gripper" : "right_gripper";
          const cmdGrip = latestCommand.action[cmdGripKey];
          traces.push({{
            type: "scatter3d",
            x: [cmdPose[0]],
            y: [cmdPose[1]],
            z: [cmdPose[2]],
            mode: "markers",
            name: `${{cfg.arm}} latest command`,
            marker: {{size: 8, color: cfg.cmdColor, symbol: "diamond"}},
            hovertemplate: `${{cfg.arm}} latest command<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<br>gripper=${{Array.isArray(cmdGrip) ? Number(cmdGrip[0] || 0).toFixed(3) : "n/a"}}<extra></extra>`
          }});
        }}

        if (Array.isArray(currentPose) && currentPose.length >= 3 && Array.isArray(cmdPose) && cmdPose.length >= 3) {{
          traces.push({{
            type: "scatter3d",
            x: [currentPose[0], cmdPose[0]],
            y: [currentPose[1], cmdPose[1]],
            z: [currentPose[2], cmdPose[2]],
            mode: "lines",
            name: `${{cfg.arm}} current -> command`,
            line: {{width: 4, color: cfg.cmdColor}}
          }});
        }}
      }}

      if (!hasAny3D) {{
        note.textContent = "No end-effector pose data is available yet, so this 3D plot cannot be built.";
      }} else if (!hasTargetPath) {{
        note.textContent = "This policy is commanding joints rather than EE poses, so the full 3D target chunk is unavailable. Use the Robot Snapshot plot for the clearest current-vs-target view.";
      }} else {{
        note.textContent = "Current arm positions are circles, latest commanded targets are diamonds, recent measured motion is the thinner path, and the full model target chunk is the thicker arm-colored line.";
      }}

      Plotly.react("path_fig", traces, {{
        height: 540,
        margin: {{l: 0, r: 0, t: 40, b: 0}},
        template: "plotly_white",
        title: "3D robot view",
        scene: {{
          xaxis: {{title: "x"}},
          yaxis: {{title: "y"}},
          zaxis: {{title: "z"}},
          aspectmode: "data"
        }}
      }}, {{responsive: true}});
    }}

    function render(summary) {{
      const latest = summary.latest_chunk;
      const warning = document.getElementById("warning");
      const status = document.getElementById("status");
      const inferCost = latest ? Number(latest.infer_cost || 0) : 0;
      const horizon = latest ? Number(latest.last_step_horizon_s || 0) : 0;
      warning.textContent = (latest && inferCost > horizon)
        ? `Inference (${{
            inferCost.toFixed(2)
          }}s) is slower than the chunk horizon (${{
            horizon.toFixed(2)
          }}s to the last step).`
        : "";
      status.textContent = `Last update: ${{new Date().toLocaleTimeString()}} | chunks in memory: ${{
        summary.chunk_count_in_memory || 0
      }} | publish events in memory: ${{
        summary.publish_count_in_memory || 0
      }}`;
      buildSummary(summary);
      buildObjectView(summary);
      buildSnapshotFigure(summary);
      buildPathFigure(summary);
      buildTiming(summary);
      buildActionFigure(summary);
      buildInferFigure(summary);
    }}

    async function poll() {{
      try {{
        const response = await fetch(`${{SUMMARY_URL}}?ts=${{Date.now()}}`, {{cache: "no-store"}});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const summary = await response.json();
        const fingerprint = JSON.stringify([
          summary.chunk_count_in_memory,
          summary.publish_count_in_memory,
          summary.latest_chunk ? summary.latest_chunk.chunk_id : null,
          summary.latest_chunk ? summary.latest_chunk.infer_end : null
        ]);
        if (fingerprint !== lastFingerprint) {{
          render(summary);
          lastFingerprint = fingerprint;
        }}
      }} catch (err) {{
        document.getElementById("status").textContent = `Failed to fetch latest_summary.json: ${{err}}`;
      }}
    }}

    poll();
    setInterval(poll, REFRESH_MS);
  </script>
</body>
</html>
"""
        self.live_dashboard_path.write_text(html, encoding="utf-8")

    def _build_timing_figure(self):
        recent_chunks = list(self.chunk_history)[-6:]
        fig = go.Figure()
        legend_seen = set()

        for chunk in recent_chunks:
            label = f"Chunk {chunk['chunk_id']}"
            self._add_segment(fig, chunk["obs_time"], chunk["last_step_time"], label, "Chunk horizon", "#1f77b4", 14, legend_seen)
            self._add_segment(fig, chunk["infer_start"], chunk["infer_end"], label, "Inference", "#ff7f0e", 10, legend_seen)

            chunk_events = self._events_for_chunk(chunk["chunk_id"])
            publish_times = [event["publish_time"] for event in chunk_events if event["published"]]
            miss_times = [event["publish_time"] for event in chunk_events if not event["published"]]
            if publish_times:
                self._add_segment(fig, min(publish_times), max(publish_times), label, "Publish window", "#2ca02c", 8, legend_seen)
                fig.add_trace(
                    go.Scatter(
                        x=publish_times,
                        y=[label] * len(publish_times),
                        mode="markers",
                        marker=dict(color="#2ca02c", size=9),
                        name="Published tick",
                        showlegend="Published tick" not in legend_seen,
                        hovertemplate="publish=%{x:.3f}<extra></extra>",
                    )
                )
                legend_seen.add("Published tick")
            if miss_times:
                fig.add_trace(
                    go.Scatter(
                        x=miss_times,
                        y=[label] * len(miss_times),
                        mode="markers",
                        marker=dict(color="#d62728", size=10, symbol="x"),
                        name="No action available",
                        showlegend="No action available" not in legend_seen,
                        hovertemplate="miss=%{x:.3f}<extra></extra>",
                    )
                )
                legend_seen.add("No action available")

        fig.update_layout(
            height=max(340, 120 + 70 * len(recent_chunks)),
            margin=dict(l=40, r=20, t=40, b=40),
            template="plotly_white",
            xaxis_title="Wall Time (s)",
            yaxis_title="Chunk",
            title="Chunk lifetime, inference delay, and publish activity",
        )
        return fig

    def _build_action_figure(self, chunk: dict[str, Any]):
        groups = list(chunk["actions"].keys())
        fig = make_subplots(
            rows=max(len(groups), 1),
            cols=1,
            shared_xaxes=True,
            subplot_titles=groups if groups else ["No actions"],
            vertical_spacing=0.06,
        )
        chunk_events = self._events_for_chunk(chunk["chunk_id"])
        x_pred = np.arange(chunk["action_steps"], dtype=np.float32) * self.dt

        for row_idx, key in enumerate(groups, start=1):
            pred = np.asarray(chunk["actions"][key], dtype=np.float32)
            labels = _component_labels(key, pred.shape[1])

            for dim_idx, label in enumerate(labels):
                fig.add_trace(
                    go.Scatter(
                        x=x_pred,
                        y=pred[:, dim_idx],
                        mode="lines+markers",
                        name=f"pred {label}",
                        legendgroup=f"pred-{key}",
                        showlegend=row_idx == 1,
                        hovertemplate=f"{label}<br>t=%{{x:.3f}}s<br>value=%{{y:.4f}}<extra></extra>",
                    ),
                    row=row_idx,
                    col=1,
                )

            published_events = [event for event in chunk_events if event["published"] and key in event["action"]]
            if published_events:
                x_exec = [event["publish_time"] - chunk["obs_time"] for event in published_events]
                y_exec = np.asarray([event["action"][key] for event in published_events], dtype=np.float32)
                for dim_idx, label in enumerate(labels[: y_exec.shape[1]]):
                    fig.add_trace(
                        go.Scatter(
                            x=x_exec,
                            y=y_exec[:, dim_idx],
                            mode="markers",
                            marker=dict(size=9, symbol="diamond"),
                            name=f"cmd {label}",
                            legendgroup=f"cmd-{key}",
                            showlegend=row_idx == 1,
                            hovertemplate=f"publish dt=%{{x:.3f}}s<br>value=%{{y:.4f}}<extra></extra>",
                        ),
                        row=row_idx,
                        col=1,
                    )

            feedback_events = [
                event for event in chunk_events
                if event["published"] and key in event["feedback"]
            ]
            if feedback_events:
                x_feedback = [event["publish_time"] - chunk["obs_time"] for event in feedback_events]
                y_feedback = np.asarray([event["feedback"][key] for event in feedback_events], dtype=np.float32)
                if y_feedback.ndim == 1:
                    y_feedback = y_feedback[:, None]
                for dim_idx, label in enumerate(labels[: y_feedback.shape[1]]):
                    fig.add_trace(
                        go.Scatter(
                            x=x_feedback,
                            y=y_feedback[:, dim_idx],
                            mode="lines+markers",
                            line=dict(dash="dot"),
                            marker=dict(size=7),
                            name=f"fb {label}",
                            legendgroup=f"fb-{key}",
                            showlegend=row_idx == 1,
                            hovertemplate=f"feedback dt=%{{x:.3f}}s<br>value=%{{y:.4f}}<extra></extra>",
                        ),
                        row=row_idx,
                        col=1,
                    )

            fig.update_yaxes(title_text=key, row=row_idx, col=1)

        x_upper = max(
            [chunk["chunk_duration_s"]] +
            [event["publish_time"] - chunk["obs_time"] for event in chunk_events] +
            [0.0]
        )
        fig.update_xaxes(title_text="Seconds Since Observation Used For This Chunk", range=[0, max(0.5, x_upper + 0.2)])
        fig.update_layout(
            height=max(360, 280 * max(len(groups), 1)),
            margin=dict(l=60, r=20, t=60, b=50),
            template="plotly_white",
        )
        return fig

    def _build_infer_history_figure(self):
        chunks = list(self.chunk_history)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[chunk["chunk_id"] for chunk in chunks],
                y=[chunk["infer_cost"] for chunk in chunks],
                mode="lines+markers",
                name="Inference cost",
            )
        )
        fig.add_hline(
            y=self.last_step_horizon_s,
            line_dash="dash",
            line_color="#d62728",
            annotation_text=f"Last-step horizon ({self.last_step_horizon_s:.2f}s)",
        )
        fig.add_hline(
            y=self.chunk_duration_s,
            line_dash="dot",
            line_color="#1f77b4",
            annotation_text=f"Nominal chunk duration ({self.chunk_duration_s:.2f}s)",
        )
        fig.update_layout(
            height=320,
            margin=dict(l=50, r=20, t=50, b=40),
            template="plotly_white",
            xaxis_title="Chunk ID",
            yaxis_title="Seconds",
            title="Inference time compared with chunk horizon",
        )
        return fig

    def _build_path_figure(self, chunk: dict[str, Any]):
        fig = go.Figure()
        chunk_events = self._events_for_chunk(chunk["chunk_id"])
        latest_command = next((event for event in reversed(chunk_events) if event["published"]), None)
        obs_state = chunk.get("obs_state", {})
        arm_configs = [
            ("left_ee_pose", "Left arm", "#4c78a8", "#2ca02c", "#7f7f7f"),
            ("right_ee_pose", "Right arm", "#e45756", "#2ca02c", "#ff9d9a"),
        ]

        has_any_3d = False
        for key, arm_name, pred_color, cmd_color, fb_color in arm_configs:
            pred = np.asarray(chunk["actions"].get(key, []), dtype=np.float32)
            if pred.ndim == 2 and pred.shape[1] >= 3 and pred.shape[0] > 0:
                has_any_3d = True
                fig.add_trace(
                    go.Scatter3d(
                        x=pred[:, 0],
                        y=pred[:, 1],
                        z=pred[:, 2],
                        mode="lines",
                        name=f"{arm_name} target chunk",
                        line=dict(width=5, color=pred_color),
                    )
                )

            current_pose = np.asarray(obs_state.get(key, []), dtype=np.float32)
            if current_pose.size >= 3:
                has_any_3d = True
                fig.add_trace(
                    go.Scatter3d(
                        x=[current_pose[0]],
                        y=[current_pose[1]],
                        z=[current_pose[2]],
                        mode="markers",
                        name=f"{arm_name} current",
                        marker=dict(size=8, color=pred_color, symbol="circle"),
                    )
                )

            feedback = [
                event for event in chunk_events
                if event["published"] and key in event["feedback"]
            ]
            if feedback:
                fb_xyz = np.asarray([event["feedback"][key][:3] for event in feedback], dtype=np.float32)
                has_any_3d = True
                fig.add_trace(
                    go.Scatter3d(
                        x=fb_xyz[:, 0],
                        y=fb_xyz[:, 1],
                        z=fb_xyz[:, 2],
                        mode="lines+markers",
                        name=f"{arm_name} measured path",
                        marker=dict(size=3, color=fb_color),
                        line=dict(width=3, color=fb_color),
                    )
                )

            if latest_command and key in latest_command["action"]:
                cmd_pose = np.asarray(latest_command["action"][key], dtype=np.float32)
                if cmd_pose.size >= 3:
                    has_any_3d = True
                    fig.add_trace(
                        go.Scatter3d(
                            x=[cmd_pose[0]],
                            y=[cmd_pose[1]],
                            z=[cmd_pose[2]],
                            mode="markers",
                            name=f"{arm_name} latest command",
                            marker=dict(size=8, color=cmd_color, symbol="diamond"),
                        )
                    )
                    if current_pose.size >= 3:
                        fig.add_trace(
                            go.Scatter3d(
                                x=[current_pose[0], cmd_pose[0]],
                                y=[current_pose[1], cmd_pose[1]],
                                z=[current_pose[2], cmd_pose[2]],
                                mode="lines",
                                name=f"{arm_name} current -> command",
                                line=dict(width=4, color=cmd_color),
                            )
                        )

        if not has_any_3d:
            return None

        fig.update_layout(
            height=540,
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_white",
            title="3D robot view",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                aspectmode="data",
            ),
        )
        return fig

    def _events_for_chunk(self, chunk_id: int) -> list[dict[str, Any]]:
        events = []
        for event in self.publish_history:
            if event.get("primary_chunk_id") == chunk_id:
                events.append(event)
                continue
            selected_chunk_ids = event.get("manager_debug", {}).get("selected_chunk_ids", [])
            if chunk_id in selected_chunk_ids:
                events.append(event)
                continue
            queue_chunk_ids = event.get("manager_debug", {}).get("queue_chunk_ids", [])
            if chunk_id in queue_chunk_ids:
                events.append(event)
        return events

    def _add_segment(
        self,
        fig,
        start: float,
        end: float,
        y_label: str,
        legend_name: str,
        color: str,
        width: int,
        legend_seen: set[str],
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=[start, end],
                y=[y_label, y_label],
                mode="lines",
                line=dict(color=color, width=width),
                name=legend_name,
                showlegend=legend_name not in legend_seen,
                hoverinfo="skip",
            )
        )
        legend_seen.add(legend_name)
