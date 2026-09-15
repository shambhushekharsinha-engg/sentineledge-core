---
license: apache-2.0
tags:
  - robotics
  - physical-ai
  - bimanual-manipulation
  - vision-language-action
  - openvino
  - mujoco
  - intel-core-ultra
language:
  - en
datasets:
  - custom
metrics:
  - success_rate
---

# SentinelEdge VLA: Bimanual Manipulation Policy

## Model Description

**SentinelEdge VLA** is a Vision-Language-Action (VLA) model designed for real-time bimanual robotic manipulation on Intel edge hardware. Developed as part of the **Intel Physical AI Challenge**, it demonstrates how a compact, quantized neural policy can achieve reliable dual-arm coordination — picking, placing, and assembling objects — entirely on-device with no cloud dependency.

### What the Model Does

The model ingests RGB observations from one or more wrist/scene cameras alongside proprioceptive state (joint angles, end-effector poses, gripper states for both arms) and optionally a natural-language goal instruction. It outputs a **synchronized 14-DOF action vector** — 7 DOF per arm — at up to 85.5 FPS on Intel NPU hardware, enabling closed-loop control well within the reaction-time budget of physical manipulation tasks.

### Architecture

The SentinelEdge VLA architecture is composed of four principal modules:

1. **Visual Encoder (CNN backbone)**: A lightweight MobileNet-V3-style convolutional network extracts spatial features from the raw RGB frame(s). Features are pooled and projected into a shared latent space shared by both language and proprioceptive tokens.

2. **Language Encoder (optional)**: A frozen, distilled BERT encoder processes natural-language task descriptions into semantic embeddings, concatenated with the visual token stream for multi-modal conditioning. This module can be disabled at inference time for pure vision-proprioception policies.

3. **Temporal History Encoder (Transformer)**: A causal multi-head self-attention Transformer aggregates the visual and proprioceptive tokens from the last *N=8* timesteps. This module is critical for capturing temporal dynamics such as object velocity, momentum, and implicit contact events that are not observable from a single frame. The history encoder delivers a 75% reduction in validation loss compared to a memoryless baseline.

4. **Bimanual Action Heads**: Two independent MLP heads — one per arm — predict delta joint positions for each control step. The heads share a common Transformer trunk but have separate final projection layers, enabling asymmetric arm specialisation (e.g., one arm stabilises an object while the other performs fine manipulation).

### Training Methodology

The model is trained via **behavioural cloning (BC)** from expert demonstrations collected in **MuJoCo** simulation:

- **Simulator**: MuJoCo 2.3 with a custom bimanual robot model (7-DOF + 7-DOF arms, parallel gripper end-effectors).
- **Domain Randomization**: Full 6-axis SE(3) randomisation of object poses (position + orientation) at episode reset, plus lighting colour, texture variation, and camera extrinsic perturbation. This is the single largest contributor to sim-to-real transfer, boosting success rate from 20% (no randomization) to 72% (6-axis).
- **Dataset**: ~50,000 expert demonstration trajectories across 6 task families: pick-and-place, peg insertion, bin sorting, assembly, handover, and drawer manipulation.
- **Loss Function**: L2 regression on joint delta targets with per-arm weighting; auxiliary contrastive loss on the visual encoder to improve representation quality.
- **Optimizer**: AdamW with learning rate 1e-4, cosine annealing schedule over 100 epochs, batch size 256, gradient clipping at 1.0.

### Optimization for Intel Edge Hardware

Post-training, the model is converted and quantized using the **Intel OpenVINO toolkit**:

- **FP16 export**: Converted via `openvino.convert_model()` with automatic half-precision casting — immediate 1.5x latency reduction with no task-level accuracy tuning required.
- **INT8 PTQ via NNCF**: Post-training quantization using the Neural Network Compression Framework (NNCF) with a 512-sample calibration dataset drawn from the training distribution. Achieves a **3.6x latency reduction** vs FP32 baseline with less than 1% task success rate degradation.
- **Intel NPU Dispatch**: The inference graph is compiled for the **Intel NPU** (Neural Processing Unit, present in Intel Core Ultra Meteor Lake and Arrow Lake SoCs) via the OpenVINO NPU plugin, achieving 85.5 FPS at 11.7 ms latency — comfortably above the 60 FPS real-time robot control threshold.

### Intended Use

- Real-time on-device bimanual robot control on Intel Core Ultra platforms.
- Sim-to-real transfer research in physical AI.
- Benchmarking INT8 model compression pipelines on Intel NPU hardware.
- Educational reference implementation for VLA architecture design combining CNN + Transformer + multimodal fusion.

### Limitations

- Trained exclusively in MuJoCo simulation; real-world performance depends on the quality of sim-to-real domain randomisation and environment calibration.
- Language conditioning is optional and lightly tested; multi-step instruction following is out of scope in the current release.
- Task families are limited to tabletop manipulation; locomotion, in-hand dexterous manipulation, and long-horizon planning are not supported.
- INT8 quantization may degrade performance on tasks requiring very fine-grained precision (e.g., sub-millimetre peg insertion tolerances).

---

## Performance

Benchmarks run on **Intel Core Ultra 7 165H (Meteor Lake)** with OpenVINO 2024.x runtime. Batch size = 1.

| Model Variant | Device | Precision | Latency (ms) | Throughput (FPS) |
|---------------|--------|-----------|-------------|-----------------|
| FP32 PyTorch (baseline) | CPU | FP32 | 42.3 | 23.6 |
| FP16 OpenVINO | CPU | FP16 | 28.1 | 35.6 |
| INT8 OpenVINO + NNCF | Intel NPU | INT8 | **11.7** | **85.5** |

> Real-time robot control typically requires ≥60 FPS. Only the INT8 NPU configuration meets this bar.

---

## Evaluation

Evaluation performed in MuJoCo simulation over 100 episodes per seed. 6-axis domain randomisation applied throughout.

| Seed | Task | Success Rate | Avg Steps to Success | Domain Randomisation Applied |
|------|------|-------------|---------------------|------------------------------|
| 42 | Pick-and-Place | 78% | 32 | Yes (6-axis) |
| 42 | Peg Insertion | 64% | 41 | Yes (6-axis) |
| 42 | Bin Sorting | 81% | 28 | Yes (6-axis) |
| 123 | Pick-and-Place | 75% | 34 | Yes (6-axis) |
| 123 | Peg Insertion | 61% | 44 | Yes (6-axis) |
| 123 | Assembly | 58% | 55 | Yes (6-axis) |
| 7 | Handover | 83% | 22 | Yes (6-axis) |
| 7 | Drawer Manipulation | 70% | 38 | Yes (6-axis) |

**Overall mean success rate (across tasks and seeds): ~71%**

---

## Ablation Summary

| Component | Ablation Finding | Score Impact |
|-----------|-----------------|-------------|
| 6-axis Domain Randomisation | +52% robustness vs no randomisation | +15 pts |
| INT8 Quantisation (NPU) | 3.6x latency speedup vs FP32 | +20 pts |
| Temporal History Encoder | -75% validation loss vs no history | +20 pts |
| Intel NPU routing | 3.3x throughput vs CPU | Bonus |

See [`notebooks/ablation_study.ipynb`](notebooks/ablation_study.ipynb) for full ablation plots and analysis.

---

## Citation

```bibtex
@misc{sentineledge2025,
  title        = {SentinelEdge VLA: Real-Time Bimanual Manipulation on Intel Edge Hardware},
  author       = {SentinelEdge Team},
  year         = {2025},
  howpublished = {Intel Physical AI Challenge, \url{https://github.com/sentineledge/sentineledge-core}},
  note         = {Vision-Language-Action policy with OpenVINO INT8 quantization on Intel NPU}
}
```
