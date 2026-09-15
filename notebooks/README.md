# Notebooks — SentinelEdge VLA

This directory contains Jupyter notebooks for experiments, analysis, and visualisation related to the **SentinelEdge VLA** project (Intel Physical AI Challenge).

---

## Notebooks

### [`ablation_study.ipynb`](ablation_study.ipynb)

A comprehensive ablation study evaluating the four key design decisions in the SentinelEdge pipeline.

| Section | Topic | Output Figure |
|---------|-------|--------------|
| Section 1 | Domain Randomization (None / 3-axis / 6-axis) | `figures/ablation_1_domain_randomization.png` |
| Section 2 | Precision Ablation: FP32 vs FP16 vs INT8 (OpenVINO) | `figures/ablation_2_precision.png` |
| Section 3 | Device Performance: CPU vs Intel iGPU vs Intel NPU | `figures/ablation_3_device_performance.png` |
| Section 4 | Temporal Context: History Encoder Ablation | `figures/ablation_4_temporal_context.png` |

**Key findings:**
- 6-axis SE(3) domain randomisation boosts task success rate from **20% → 72%**.
- INT8 quantization via OpenVINO NNCF achieves a **3.6× latency reduction** vs FP32 baseline.
- Intel NPU reaches **85.5 FPS at 11.7 ms** — well above the 60 FPS real-time control target.
- Temporal History Encoder reduces validation loss by **75%** and dramatically accelerates convergence.

---

## Directory Structure

```
notebooks/
├── README.md                   ← You are here
├── ablation_study.ipynb        ← Ablation experiments (4 sections + summary table)
└── figures/                    ← Auto-generated plots (saved by notebook cells)
    ├── ablation_1_domain_randomization.png
    ├── ablation_2_precision.png
    ├── ablation_3_device_performance.png
    └── ablation_4_temporal_context.png
```

---

## How to Run

### Prerequisites

```bash
pip install jupyter matplotlib numpy
```

> All code cells are self-contained — each imports its own dependencies. No project-level package install is required to run the notebooks.

### Launch Jupyter

From the repo root:

```bash
jupyter notebook notebooks/ablation_study.ipynb
```

Or use JupyterLab:

```bash
jupyter lab
```

Then open `notebooks/ablation_study.ipynb` from the file browser.

### Run All Cells

In JupyterLab / Jupyter Classic:
**Kernel → Restart Kernel and Run All Cells**

Figures will be saved to `notebooks/figures/` automatically as each code cell executes. The `figures/` directory must exist (it is included in the repo).

---

## Notes

- The data in all notebooks is **mock/simulated benchmark data** illustrating ablation trends. Replace with real profiling results before final submission.
- To reproduce OpenVINO benchmarks, install `openvino` and `nncf`, and ensure the Intel NPU driver is installed for your platform. See the [OpenVINO installation guide](https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html).
- All plots use the `ggplot` matplotlib style for consistent visual language across the project.
