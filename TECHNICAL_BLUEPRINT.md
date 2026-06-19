# System Blueprint & Technical Project Report: Smart Surveillance Platform

## 📋 Executive Summary
Modern security and safety operations are severely constrained by traditional closed-circuit television (CCTV) systems, which rely entirely on passive recording and human-in-the-loop monitoring. Human operators are highly susceptible to fatigue, cognitive overload, and delayed reaction times, which can lead to catastrophic failures during high-stakes incidents like weapon brandishing, structural fires, slip-and-fall injuries, or unauthorized area intrusions.

This project delivers a state-of-the-art **Smart Surveillance Suite**, a high-performance, fully asynchronous, edge-compatible AI operations dashboard. By fusing real-time Computer Vision (YOLOv11/v8), multi-model edge computing, and localized Generative AI (Vision-LLMs), this platform transitions surveillance from passive recording to **autonomous, real-time threat intelligence and proactive alerting**.

---

## ⚠️ 1. Problem Statement

Traditional security infrastructures suffer from four fundamental limitations:
1. **Reactive vs. Proactive Response:** Traditional systems only record incidents for post-event forensic audits rather than detecting and alerting during the critical first seconds of a threat.
2. **Cognitive Saturation:** A single security operator monitoring dozens of cameras routinely misses critical events due to attention fatigue.
3. **Inflexible Analysis Boundaries:** Traditional perimeter alerts rely on rigid hardware tripwires that cannot adapt to dynamic, complex physical environments.
4. **Lack of Contextual Reporting:** Automated alerts typically transmit raw, contextless logs (e.g., `"Motion Detected"`) that fail to convey incident severity, involved individuals, or specific hazard urgency to responding officers.

---

## 💡 2. The Solution: Smart Surveillance Platform

Our platform addresses these vulnerabilities by delivering an integrated, edge-first, AI-driven surveillance matrix:

```
  ┌────────────────────────────────────────────────────────┐
  │                 Live Camera Feed Ingestion             │
  └───────────────────────────┬────────────────────────────┘
                              │ (~25 FPS Frame Stream)
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │       Inference Fusion Layer (CUDA Accelerated)        │
  │   - YOLO Weapon, Fire, Fall, & Human Detectors         │
  │   - Dual-Point restricted polygon intrusion test       │
  └───────────────────────────┬────────────────────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼ (Local WebSocket)               ▼ (On Anomaly Trigger)
  ┌──────────────────────┐          ┌──────────────────────┐
  │  React Dashboard UI  │          │  Asynchronous Alert  │
  │  - ~12ms Ingestion   │          │  Telemetry Pipeline  │
  │  - Overlay toggles   │          └──────────┬───────────┘
  │  - Interactive zone  │                     │
  │  - Event Replay      │                     ▼ (Ollama API)
  └──────────────────────┘          ┌──────────────────────┐
                                    │ Local Vision-LLM     │
                                    │ (Incident Analysis)  │
                                    └──────────┬───────────┘
                                               │
                                               ▼ (Notifier)
                                    ┌──────────────────────┐
                                    │ Contextualized Email │
                                    │ & Dispatch Alert     │
                                    └──────────────────────┘
```

* **Automated Multi-Threat Detection:** Runs highly optimized YOLO custom models concurrently to detect weapons, structural fires, smoke, falls, and crowds.
* **Dynamic Intrusion Detection:** Allows operators to interactively draw custom polygon restricted zones directly on the dashboard video canvas, translating pixel-coordinates to real-world boundaries.
* **AI Context Enrichment:** On critical triggers, the system automatically routes the frame to a localized Vision-LLM (Moondream2/Ollama) to generate detailed, natural language incident reports.
* **Instant Priority Notifications:** Dispatches SMTP-based email routing with descriptive context and the exact JPEG snapshot within seconds of detection, bypassing slow manual alerting.
* **Zero-Lag Visual Control:** Delivers a premium, glassmorphic dark-themed React dashboard driven by high-speed WebSockets transmitting telemetry at ~25 FPS.

---

## ⚙️ 3. Architecture & System Methodology

The system is engineered as an asynchronous, highly separated two-tier architecture:

### A. Core Backend Engine (FastAPI + PyTorch + OpenCV)
The backend operates a highly optimized, multi-threaded framework written in Python:
1. **Frame Ingestion Layer:** Uses a dedicated background singleton worker thread (`VideoCaptureWorker`) to capture frames from hardware cameras or media sources continuously without blocking the primary event loop.
2. **Inference Fusion Pipeline:** Frame bytes are directed through the unified `DetectionEngine`, which executes active computer vision modules concurrently using PyTorch with dynamic hardware auto-detection.
3. **Perimeter Calculation Layer:** Evaluates coordinates using a dual-point checking algorithm (verifying both the base/feet and center points of people objects against the restricted polygon), ensuring perspective-safe trespassing alerts.
4. **Asynchronous Notification Routing:** Manages database logging and schedules Vision-LLM analysis and SMTP routing inside non-blocking asynchronous pools to prevent server slowdowns.

### B. Modern Frontend Operations Matrix (Vite + React + TypeScript + Tailwind)
A high-depth dashboard built using modern web standards:
1. **Telemetry Synchronization:** Ingests base64-encoded frames and structured anomaly alerts via high-speed WebSockets at ~25 FPS.
2. **Interactive Boundary Editor:** Renders a clean SVG overlay above the viewport canvas. Clicking the screen scales coordinates to the native capture resolution, drawing restriction polygon vertices dynamically.
3. **Event Snapshot Replayer:** Clicking items in the recent anomalies log intercepts the active WebSocket feed and loads the corresponding historical image from the database. A custom teal return button restores the live stream instantly.

---

## 🧠 4. Deep-Dive Model Specifications & Accuracy Tuning

To resolve common real-world false negatives and box flicker issues, the AI pipelines incorporate several advanced modifications:

| Model Domain | Weight File | Inspected Class Naming Map | Optimized Def. Conf. | Key Algorithmic Tuning |
| :--- | :--- | :--- | :--- | :--- |
| **Threat Detection** | `weapon.pt` | `0: 'weapon'` | `0.45` | Substring checks `("weapon", "gun", "pistol", "knife")` |
| **Fire & Smoke** | `fire.pt` | `0: 'Fire'`, `1: 'default'`, `2: 'smoke'` | `0.45` | Maps the custom `default` label to fire warnings |
| **Fall Assessment** | `fall_model.pt` | `0: 'Fall-Detected'` | `0.45` | Broadened matching to `("fall")` for label robusting |
| **Crowd & Intruder** | `yolo11n.pt` | `0: 'person'`, `1: 'bicycle'`, `2: 'car'` ... | `0.40` | Checks human coordinates in trespassing bounds |

### 🛠️ Key Neural Engine Tuning parameters:
* **Image Size Alignment (`imgsz=640`):** Enforces static input dimension scaling during inference, aligning the neural net layers to the raw camera dimensions to prevent scale-induced accuracy drops.
* **Intersection-over-Union Optimization (`iou=0.45`):** Suppresses overlapping double-bounding boxes on adjacent targets by locking NMS IoU margins to `0.45`.
* **Dynamic Hardware Auto-Detection:** Automatically instantiates the `DetectionEngine` on **CUDA (GPU)** if available, bringing individual model inference speeds down to **~12ms** for near-instant real-time triggers.

---

## 🎨 5. Premium UI/UX Design System

The user interface adopts a high-depth **Midnight Teal Glassmorphism** design language inspired by modern, premium control center interfaces:

* **Dark Color Palette:** Base background using deep midnight teal (`#040c12`) paired with glassmorphic cards (`bg-[#0c1a24]/30 backdrop-blur-xl border border-white/5 shadow-2xl`).
* **Sidebar Views Router:** Swaps main content stages dynamically (`Dashboard`, `Analytics`, `Cameras`, `Settings`) using visual hover transitions, a vertical neon highlight bar, and glowing interactive fallback placeholders for modules under development.
* **Clean OSD Corners:** Avoids center-screen text clutter. Metadata pills (Latency: `42ms`, status: `Recording`) and technical metrics (`Inference Speed: 12ms | 24 FPS | YOLOv11-custom`) are kept strictly in the corners.
* **Interactive Dropdown Overlays:** Responsive header action icons open absolute floating dropdown panels (Search bar input, Notifications status log, and diagnostics menu dropdown) that close each other automatically when toggled.
* **Neon Alert Flash Overlay:** Instantly pulses the video viewport card with a warm, glowing gradient red/amber overlay on critical threat alerts, alerting the operator immediately.
* **Custom Teal Toast Notifications:** Floating action confirmations (`APPLIED CHANGES`, `PARAMETERS RESET`, `RESTRICTED ZONE SAVED`, `🚨 DISPATCH INITIATED`) display at the bottom-center utilizing the theme's core teal branding.

---

## 🚀 6. Execution Manual

To deploy the platform locally, execute the following commands:

### Step 1: Start the FastAPI AI Backend
Open a terminal in the backend directory (`F:\minipro2\backend`) and run:
```powershell
# Activate the environment
.venv\Scripts\activate

# Launch the central uvicorn server
python main.py
```
* Serves the live WebSocket telemetry at `ws://localhost:8000/api/ws/telemetry` and registers database connections.

### Step 2: Start the Vite + React Frontend
Open a separate terminal in the frontend directory (`F:\minipro2\frontend`) and run:
```powershell
# Run the local development server
npm run dev
```
* Open `http://localhost:5173` in your browser to experience the premium Smart Surveillance Operations Matrix!

---

## 🔮 7. Future Scope & Scale Opportunities

The platform's highly decoupled, modular architecture allows for simple, powerful future expansion:
1. **Multi-Camera RTMP/RTSP Ingestion:** Scale the backend worker thread to manage an array of live RTSP IP cameras, switching active video buffers dynamically inside the telemetry stream.
2. **On-Edge Vision-LLM Models:** Migrate the vision description logic to run directly on the local edge hardware using local INT4 quantized weights (e.g., LLaVA-NeXT or Moondream2 on ONNX/TensorRT).
3. **Advanced Biometric Profiling:** Integrate deep-sort tracking layers and lightweight facial-recognition models (`FaceNet` / `InsightFace`) to catalog white-listed vs. black-listed individuals.
4. **Autonomous Alarm Systems:** Wire the backend alerts to trigger physical alert beacons or motorized gates using simple hardware-integrated APIs (e.g., Raspberry Pi GPIO control relays).

---

## 8. Phase 1 Implementation Plan: Stable AI Core

**Duration:** 2-3 weeks

**Goal:** Reduce false positives and stabilize detections before expanding the system.

The first implementation phase separates detection responsibility by threat category. A universal detector creates avoidable problems: poor specialization, class confusion, and unstable confidence scores. The stable-core architecture should route each event family through a detector and validation strategy designed for that event.

### Step 1 - Separate Detection Models

| Detector | Purpose | Recommended Models | Training / Logic Focus |
| :--- | :--- | :--- | :--- |
| **Weapon Detection** | Pistols, rifles, knives | YOLOv11m, RT-DETR | Tiny objects, low-light CCTV, side angles |
| **Fire & Smoke** | Flames, smoke plumes | EfficientNet classifier, YOLO fire detector | Texture understanding and temporal persistence; not object detection alone |
| **Fall Detection** | Elderly collapse, injury detection | YOLO Pose, OpenPose, MoveNet | Body angle, sudden collapse, no movement afterward |
| **Intrusion Detection** | Restricted-area crossing | YOLO person detection, polygon zone logic | Person localization plus zone intersection and persistence checks |

### Target `ai_core/` Layout

```text
ai_core/
|
+-- detectors/
|   +-- weapon_detector/
|   +-- fire_detector/
|   +-- fall_detector/
|   +-- intrusion_detector/
|
+-- tracking/
+-- temporal_validation/
+-- threat_engine/
+-- llm/
+-- agents/
+-- notifications/
+-- dashboard/
```

This structure creates a cleaner boundary between raw detection, object tracking, temporal validation, threat scoring, scene understanding, LLM narration, notification dispatch, and dashboard replay.
