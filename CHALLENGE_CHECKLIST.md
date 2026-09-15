# Intel Physical AI Challenge - Submission Checklist

## Required Deliverables

| # | Deliverable | Status | File/Location |
|---|-------------|--------|---------------|
| 1 | Reproducible GitHub Repository | ✅ Complete | This Repo |
| 2 | MuJoCo Simulation (Dual SO-101 Arms) | ✅ Complete | simulation/ |
| 3 | Domain Randomization (10+ seeds) | ✅ Complete | simulation/env.py |
| 4 | Intel OpenVINO Inference Benchmark | ✅ Complete | inference/benchmark_intel.py |
| 5 | Demonstration Video (10 seeds) | ⬜ Record & Upload | scripts/evaluate.py |
| 6 | Technical README / Architecture Summary | ✅ Complete | README.md |
| 7 | Training / Fine-tuning Code | ✅ Complete | scripts/train_imitation.py |
| 8 | Evaluation Code | ✅ Complete | scripts/evaluate.py |

## Scoring Rubric Coverage

| Category | Max Points | Our Coverage |
|----------|-----------|-------|
| Bimanual Task Completion | 25 | Dual SO-101 + hand-off detection + 4 task types |
| Robustness & Generalization | 15 | 6-axis domain randomization (shape, mass, friction, light, bg, placement) |
| VLA / Multi-Modal Reasoning | 20 | Temporal action history + Vision + Language fusion |
| OpenVINO & Intel Core Ultra Optimization | 20 | INT8 PTQ + NPU/iGPU auto-discovery + latency profiling |
| Technical Quality & Reproducibility | 10 | Conda env + Docker + CI + Config YAML |
| Innovation & Demo Quality | 5 | Gradio UI + HUD video + Voice-to-Action |
| **Speechmatics Bonus Award** | **+Bonus** | Speechmatics ASR integrated in app.py |

## Pre-Submission Steps
- [ ] Run `python inference/export_openvino.py` to generate INT8 model
- [ ] Run `python inference/benchmark_intel.py` on Intel Core Ultra hardware
- [x] Run `python scripts/evaluate.py --seeds 10` to generate demo videos
- [x] Upload best demo video to YouTube / Hugging Face
- [x] Add video link to README.md
- [ ] Submit on lablab.ai portal
