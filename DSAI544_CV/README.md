# DSAI 544 — Computer Vision

## Final Project: Visual Telemetry in MOBA Gameplay

Predicting League of Legends game progress from single-frame screenshots using a fine-tuned ResNet-50 — without reading the in-game timer.

### Problem Formulation

| Task | Classes | Test Performance |
|------|---------|-----------------|
| Regression | Continuous (seconds) | MAE: 156.03s (~2.6 min) |
| Fine-grained classification | 7 bins (5-min intervals) | 49.53% accuracy |
| Coarse-grained classification | 3 phases (Early/Mid/Late) | **86.07% accuracy** |

### Dataset

- **7,369 frames** from 19 high-ELO Korean gameplay videos (5s intervals via FFmpeg)
- Split by video: 11 train / 4 val / 4 test, stratified by role (Top, Jungle, Mid, ADC)
- Anti-leakage masking: timer and KDA scoreboard blacked out

### Key Findings

- Model attends to minimap state and champion clusters (validated via Grad-CAM)
- Late game (25+ min) performance degrades due to long-tail data distribution (10.7% of samples)
- Mid lane shows highest error across all tasks — likely due to roaming patterns breaking visual consistency

### Training

- **Backbone**: ResNet-50 (ImageNet pretrained)
- **Optimizer**: AdamW (weight decay 1e-4)
- **LR Selection**: Grid search over {1e-4, 3e-4, 1e-3} — best: 1e-3 for all tasks
- **Augmentation**: ColorJitter only (no geometric — MOBA maps are asymmetric)
- **Logging**: Weights & Biases

### Structure

```
DSAI544_CV/
├── cv_final.py          # Full training pipeline (all 3 tasks + Grad-CAM)
├── cv_final.pdf         # Project report
└── results/             # Metrics, confusion matrices, Grad-CAM visualizations
```

### Requirements

```bash
pip install torch torchvision polars wandb scikit-learn seaborn pytorch-grad-cam
```