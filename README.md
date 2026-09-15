# 🤖 Intel Physical AI Challenge
## Bimanual VLA Manipulation with Multi-Modal Reasoning

<div align="center">

![Intel Core Ultra](https://img.shields.io/badge/Optimized_for-Intel_Core_Ultra-0068B5?style=for-the-badge&logo=intel)
![OpenVINO](https://img.shields.io/badge/Powered_by-OpenVINO_INT8-4A25AA?style=for-the-badge&logo=intel)
![MuJoCo](https://img.shields.io/badge/Simulation-MuJoCo-orange?style=for-the-badge)
![LeRobot](https://img.shields.io/badge/Policy-LeRobot_ACT-FF5733?style=for-the-badge)
![Speechmatics](https://img.shields.io/badge/Bonus-Speechmatics_ASR-1ABC9C?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-39%20Passed-brightgreen?style=for-the-badge&logo=pytest)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions)

### 🎬 Full Demo Video (10 Seeds · OpenVINO · Speechmatics Bonus)

[![Watch Demo on YouTube](https://img.shields.io/badge/Watch%20Demo-YouTube-FF0000?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=ugT-6m7i8ls)

</div>

> **End-to-end Physical AI submission for the [Intel Physical AI Challenge](https://lablab.ai/event/intel-physical-ai-challenge) on lablab.ai.**
> Real-time bimanual robot control, multi-modal reasoning, and Intel edge deployment —
> including the optional **Speechmatics Voice-to-Action Bonus Award**.

---

## 🛑 The Problem

Modern Physical AI faces three hard bottlenecks for complex real-world tasks:

1. **Multi-Modal Grounding:** Interpreting abstract natural language commands and grounding them in dynamic visual scenes requires massive VLA transformers too large for edge hardware.
2. **Bimanual Coordination Complexity:** Coordinating two robotic arms for synchronized hand-offs, drawer opening, and collision-aware sequencing is exponentially harder than single-arm control.
3. **Edge Deployment Latency:** Running heavy VLA policies requires cloud GPUs, making real-time closed-loop control at the edge impossible without specialized optimization.

---

## 💡 Our Solution

A complete **perception-to-action pipeline** that tightly couples a Temporal VLA architecture with Intel's edge optimization toolkits:

- **Dual SO-101 arms** in MuJoCo coordinate on a multi-step dinner-table-setting task
- A **Temporal Action Chunking Transformer (ACT)** fuses camera pixels, language embeddings, and action history for multi-step reasoning
- Compiled to **Intel OpenVINO IR (INT8 PTQ via NNCF)** and routed to **Core Ultra NPU/iGPU**
- A **Gradio Web Dashboard** with **Speechmatics live ASR** lets you speak commands and watch the robot execute them instantly

---

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| 🎙️ **Speechmatics Voice-to-Action** | Real-time speech-to-instruction via Speechmatics Batch API — qualifies for Bonus Award |
| 🧠 **Temporal VLA Architecture** | 4-stage CNN backbone + type-embedded Cross-Modal Transformer + separate bimanual action heads |
| 🛡️ **Dynamic Collision Avoidance** | Sinusoidally oscillating MuJoCo obstacle forces policy to reason around moving objects |
| ⚡ **Intel OpenVINO INT8 PTQ** | NNCF Post-Training Quantization reduces model ~4x and boosts throughput on NPU/iGPU |
| 🔍 **Intelligent Hardware Discovery** | Auto-detects NPU → iGPU → CPU and routes workload for peak efficiency |
| 🎲 **6-Axis Domain Randomization** | Randomizes placement, mass, friction, lighting, shape scale, floor texture independently |
| 🎮 **Teleoperation Data Collector** | Record expert `.h5` demonstrations via `record_teleop.py` for Behavioral Cloning |
| 📊 **Latency Profiler** | Multi-device p50/p95/p99 analysis + bar-chart PNG + JSON report |
| 🤖 **ROS 2 Deployment Node** | Production-ready wrapper for real physical robot deployment |
| 🐳 **Docker 1-Click Deploy** | `docker build + run` gives a working environment instantly |
| ⚙️ **Centralized Config YAML** | All hyperparameters in `configs/task_config.yaml` |
| ✅ **GitHub Actions CI** | Automated smoke tests on every push |

---

## 🏗️ Architecture Workflow

```mermaid
graph TD
    A["🎙️ Voice / Text Command"] --> B["Speechmatics ASR"]
    B --> C["Sentence Embedding"]
    D["📷 Dual Cameras"] --> E["CNN Vision Encoder"]
    F["🕹️ Joint State History"] --> G["History Encoder"]

    C --> H["Cross-Modal Transformer"]
    E --> H
    G --> H

    H --> I["Bimanual Action Head"]
    I --> J["Left SO-101 Arm"]
    I --> K["Right SO-101 Arm"]

    J --> L["MuJoCo Physics"]
    K --> L
    L -->|"Next Observation"| D

    H --> M["Intel OpenVINO INT8"]
    M --> N["Core Ultra NPU / iGPU"]
```

---

## 🎬 Live Seed Demonstrations (10 Randomized Seeds)

> Each seed applies independent 6-axis domain randomization — object positions, masses, friction, lighting, shape scale, and background texture all vary.

<div align="center">

| Seed 0 | Seed 1 | Seed 2 |
|:------:|:------:|:------:|
| ![Seed 0](data/gifs/seed_0.gif) | ![Seed 1](data/gifs/seed_1.gif) | ![Seed 2](data/gifs/seed_2.gif) |

| Seed 3 | Seed 4 | Seed 5 |
|:------:|:------:|:------:|
| ![Seed 3](data/gifs/seed_3.gif) | ![Seed 4](data/gifs/seed_4.gif) | ![Seed 5](data/gifs/seed_5.gif) |

| Seed 6 | Seed 7 | Seed 8 |
|:------:|:------:|:------:|
| ![Seed 6](data/gifs/seed_6.gif) | ![Seed 7](data/gifs/seed_7.gif) | ![Seed 8](data/gifs/seed_8.gif) |

<br>

| Seed 9 |
|:------:|
| ![Seed 9](data/gifs/seed_9.gif) |

</div>

---

## 🗂️ Task Suite

We implement **4 bimanual task variations** (see `simulation/task_suite.py`):

| Task | Instruction | Challenge |
|------|-------------|-----------|
| `set_table` | Place the plate and cup in correct positions | Bimanual coordination |
| `retrieve_spoon` | Open drawer, retrieve spoon, place beside plate | Drawer manipulation |
| `handoff` | Pick up cup with Arm A, pass to Arm B, place on table | Cross-arm hand-off |
| `collision_avoid` | Place fork beside plate while avoiding moving obstacle | Dynamic collision avoidance |

---

## 📁 Repository Structure

```text
sentineledge-core/
├── app.py                         # Gradio Web Dashboard (Speechmatics + OpenVINO)
├── Dockerfile                     # 1-click reproducible container
├── environment.yml                # Deterministic Conda environment (Python 3.10)
├── requirements.txt               # pip dependencies
├── CHALLENGE_CHECKLIST.md         # Official submission checklist with scoring rubric
├── CONTRIBUTING.md                # Developer guide
│
├── configs/
│   └── task_config.yaml           # All hyperparameters (sim, policy, training, inference)
│
├── simulation/
│   ├── env.py                     # Gymnasium env (6-axis randomization, obstacle physics)
│   ├── scene.xml                  # MuJoCo world (table, drawer, dynamic obstacle, cameras)
│   ├── objects.xml                # Dinnerware geometry definitions
│   └── task_suite.py              # 4 bimanual task definitions
│
├── models/
│   └── vla_policy.py              # Temporal VLA: CNN + Cross-Modal Transformer + Bimanual heads
│
├── scripts/
│   ├── train_imitation.py         # BC training: HDF5/synthetic, AMP, cosine LR, val split
│   ├── evaluate.py                # 10-seed evaluation + OpenCV HUD video rendering
│   ├── generate_dataset.py        # Synthetic expert dataset generator (HDF5)
│   ├── record_teleop.py           # Teleoperation data collection (.h5 trajectories)
│   ├── metrics.py                 # Per-seed metrics logging + JSON/CSV export
│   ├── stitch_demo.py             # Combine seed videos into one submission reel
│   └── generate_gifs.py           # Convert seed MP4s to optimized GIFs for README
│
├── inference/
│   ├── export_openvino.py         # PyTorch to OpenVINO IR + NNCF INT8 PTQ
│   ├── benchmark_intel.py         # NPU/iGPU/CPU throughput & latency benchmark
│   └── latency_profile.py         # p50/p95/p99 profiling + bar-chart + JSON report
│
├── deployment/
│   └── ros2_vla_node.py            # ROS 2 node for real physical robot deployment
│
└── data/
    ├── videos/                    # Per-seed demo MP4s (demo_seed_0.mp4 ... demo_seed_9.mp4)
    ├── gifs/                      # Per-seed animated GIFs (seed_0.gif ... seed_9.gif)
    └── demo_final.mp4             # Full stitched submission reel (1.4 min)
```

---

## 🚀 Getting Started

### Option A — Conda
```bash
conda env create -f environment.yml
conda activate intel-vla-challenge
```

### Option B — Docker (1-Click)
```bash
docker build -t intel-vla-challenge .
docker run -p 7860:7860 intel-vla-challenge
# Open http://localhost:7860
```

---

## ▶️ Running the Pipeline

### Interactive Web Dashboard
```bash
python app.py
```

### Generate Expert Dataset
```bash
python scripts/generate_dataset.py --episodes 20 --steps 100
```

### Train the VLA Policy
```bash
# Synthetic (first-run / CI):
python scripts/train_imitation.py --epochs 5

# Real HDF5 dataset:
python scripts/train_imitation.py --dataset data/bc_dataset.h5 --epochs 10
```

### Export & Optimize for Intel Core Ultra
```bash
python inference/export_openvino.py        # INT8 PTQ export
python inference/benchmark_intel.py        # Throughput benchmark
python inference/latency_profile.py        # p50/p95/p99 profiling
```

### Official 10-Seed Evaluation
```bash
python scripts/evaluate.py --seeds 10
python scripts/stitch_demo.py              # Stitch into one submission reel
```

---

## 📊 Scoring Rubric Coverage

| Category | Max Pts | Our Implementation |
|----------|---------|--------------------|
| Bimanual Task Completion | 25 | Dual SO-101 + hand-off detection + 4 task types + drawer manipulation |
| Robustness & Generalization | 15 | 6-axis domain randomization (position, mass, friction, lighting, shape, background) |
| VLA / Multi-Modal Reasoning | 20 | Temporal ACT: Vision + Language + History → Cross-Modal Transformer → Bimanual heads |
| OpenVINO & Intel Core Ultra | 20 | INT8 PTQ via NNCF + NPU/iGPU auto-discovery + p95/p99 latency profiling |
| Technical Quality | 10 | Conda + Docker + GitHub Actions CI + Config YAML + HDF5 data pipeline |
| Innovation & Demo | 5 | Gradio UI + HUD video + 4 task suite + dynamic obstacle |
| **Speechmatics Bonus** | **+Bonus** | Full Speechmatics Batch ASR API with polling + microphone UI |

---

## 🎙️ Speechmatics Voice-to-Action (Bonus Award)

Our Gradio dashboard includes a **live microphone widget** backed by the official Speechmatics Batch Transcription API:

1. Record your voice command in the UI
2. Audio is submitted to `asr.api.speechmatics.com/v2/jobs/`
3. Job is polled until complete, returning an accurate English transcript
4. Transcribed instruction is fed directly into the VLA policy

This transforms our project into a true **Voice-to-Physical-Action** pipeline.

---

## 📋 Pre-Submission Checklist

- [x] Reproducible repository (Conda + Docker + CI)
- [x] MuJoCo dual-arm simulation with drawer and dynamic obstacle
- [x] 6-axis domain randomization for robustness
- [x] OpenVINO INT8 PTQ export + benchmark + latency profile
- [x] Training + evaluation code with HDF5 data pipeline
- [x] Teleoperation data collection pipeline
- [x] Gradio Web UI with Speechmatics ASR
- [x] ROS 2 deployment node
- [x] Demo video uploaded → **[Watch on YouTube](https://www.youtube.com/watch?v=ugT-6m7i8ls)**
- [ ] Paste `benchmark_intel.py` results from Intel Core Ultra hardware here
