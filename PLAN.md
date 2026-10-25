# Project Plan — Quantised Face Detection & Recognition on FPGA

> **Goal:** an end-to-end face detection + recognition system built from low-bit CNNs trained with
> quantisation-aware training (QAT), sized to fit entirely in the on-chip memory/logic of a small
> FPGA (Zynq-7020) for low-latency on-device inference.
>
> **Constraint:** no physical FPGA board. Everything up to and including bit-exact hardware
> simulation and synthesis/resource estimates is in scope; only on-board measurement is out.
>
> **Compute:** Kaggle notebooks (T4 GPU, 12 h/session, ~30 GPU-h/week) — Lean plan ≈ 10–12 GPU-h. Hardware flow runs
> locally / in Docker (Kaggle has no Docker).

---

## 0. Starting point (audit of `notebooks/FaceNet__SVM.ipynb`)

LFW (deep-funneled), identities with ≥5 images (423 people, 5,985 images) → random 80/20 split of
*images* → frozen FaceNet (InceptionResnetV1, VGGFace2) 512-d embeddings → `StandardScaler + linear SVC`
→ 94.8 % top-1.

| Problem | Why it matters |
|---|---|
| Same identities in train and test; LFW used for training | LFW is a *test* benchmark; real FR is evaluated on unseen identities (verification). Biggest weakness. |
| No model is trained | Pretrained embeddings + SVM — no DL contribution. |
| FP32, ~23.5 M params (~90 MB) | ~150× over Zynq-7020 on-chip BRAM (~4.9 Mb ≈ 630 KB). |
| Wrong normalisation (`[0,1]` instead of `(x−127.5)/128`) | Degrades FaceNet accuracy. |
| No detection/alignment (MTCNN installed, unused) | Whole image resized; no face pipeline. |
| No L2-norm, closed-set top-1 only, Colab/Drive-specific code | Not comparable to literature, not reproducible. |

The notebook is kept as the historical baseline; the fixed version is re-run in NB01.

---

## 1. Target system

```
camera frame
   │
   ▼
┌──────────────────────────────┐   FPGA fabric (FINN streaming dataflow)
│ Tiny face detector (QAT, 4b) │   boxes + 5 landmarks
└──────────────┬───────────────┘
               ▼  5-point similarity-transform alignment → 112×112 (ARM side)
┌──────────────────────────────┐
│ MobileFaceNet-style recogniser│  mixed-precision QAT (avg ≈3–4 b), all weights on-chip
└──────────────┬───────────────┘
               ▼  128-d integer embedding
  cosine NN vs enrolled gallery → identity / "unknown"   (+ quality score, temporal aggregation)
```

## 2. Models, data, training (decisions)

### Datasets
| Use | Dataset | Notes |
|---|---|---|
| Recognition training | **CASIA-WebFace** (~10.5 k ids, ~490 k imgs) | Use the InsightFace `faces_webface_112x112` package: already 5-pt aligned at 112×112. Research-only licence. |
| Recognition eval (verification) | **LFW, CFP-FP, AgeDB-30, CALFW, CPLFW** | InsightFace `.bin` pair files; 10-fold accuracy, TAR@FAR. Identity-disjoint from training. |
| Detection train/eval | **WIDER FACE** (+ RetinaFace 5-pt landmark annotations) | Report AP on easy / medium / hard. |
| Robustness | Synthetic corruptions on the eval sets (+ optional TinyFace) | Low-res, blur, noise, low light, JPEG. |
| Video (stretch) | YouTube Faces subset / own webcam clips | Temporal aggregation. |
| Demo gallery | LFW people with ≥5 images (your original subset) | Open-set enrol/recognise demo only — never for training. |

### Teacher (FP32, frozen)
* **Primary:** InsightFace iResNet-50 ArcFace (ONNX, `buffalo_l` → `w600k_r50`), ~99.8 % LFW.
* **Reference:** FaceNet InceptionResnetV1 (VGGFace2) — your original model, fixed preprocessing.

### Students
| Model | Role | Notes |
|---|---|---|
| **MobileFaceNet-FPGA** (S/M/L width multipliers) | Main recogniser | PReLU→ReLU, keep GDC (global depthwise conv) head, 128-d embedding, 112×112 (64×64 ablation). ~0.3–1.0 M params. |
| VGG-style plain CNN | Baseline | conv3×3–BN–ReLU–maxpool; easiest FINN mapping — shows HW-friendliness vs accuracy trade-off. |
| **Tiny detector** (BlazeFace / SCRFD-0.5G style, anchor-free) | Face + 5 landmarks | 320×320 input, ReLU only, no exotic ops, so it streams through FINN. |

### Training recipe
* Loss: **ArcFace** (main); CosFace and **AdaFace** compared (AdaFace = quality-adaptive, suits edge cameras).
* **Embedding distillation** from teacher (cosine loss on cached teacher embeddings, incl. flipped view) + margin loss.
* Augmentation: flip, random crop/scale, colour jitter, **random down-sampling / blur** (low-res robustness).
* AMP, cosine LR, checkpoint every epoch → resume across Kaggle sessions.

### Quantisation
* PTQ INT8 reference → **Brevitas QAT** W4A4, W2A2 (W3A3/W1A1 only if quota allows); per-channel weight
  scales, learned step size (LSQ-style), BN folding, first/last layer 8-bit; distillation FP32→quantised.
* **Contribution — hardware-aware mixed precision:** per-layer sensitivity (fast bit-sweep),
  then bit allocation that maximises accuracy subject to a Zynq-7020 BRAM budget (knapsack/ILP),
  then QAT of the chosen config. Compare vs uniform QAT on an accuracy–memory–throughput Pareto front.
* Detector: W4A4 QAT.

---

## 3. Phases (Lean plan)

> Lean = same scientific story as the full plan at ~1/3 of the GPU cost. Dropped: W3A3, W1A1,
> Hessian-based sensitivity, a second mixed-precision run, full-scale ablations. Ablations run on a
> 2 k-identity CASIA subset (relative comparisons only). Any of these can be added back if quota allows.

### Phase 1 — Repo, data, reproducible baselines
1. `src/fpgaface` package, configs (YAML), scripts, tests, CI; notebooks are thin wrappers that
   `pip install` the repo from GitHub and call it.
2. Data: CASIA `.rec` reader (no MXNet dependency), eval `.bin` loader, LFW gallery with MTCNN alignment
   to the ArcFace 112×112 template.
3. Metrics: 10-fold verification, TAR@FAR, closed-/open-set identification, params/bytes/MACs.
4. Fixed FaceNet baseline + iResNet-50 teacher evaluated on all benchmarks.
5. Cache teacher embeddings (original view only, fp16) for all CASIA images.

### Phase 2 — FP32 students
6. On-chip budget calculator (weights + thresholds + line/skip buffers per bit-width) → size S/M/L.
   **Result:** S = width 0.5 (fits Z7020 at W4A4, 88 %), M = width 0.75 (142 % on Z7020, 92 % on ZU3EG),
   L = width 1.0. Thresholds dominate MobileFaceNet's on-chip memory → motivates mixed precision.
7. Main run: MobileFaceNet-M, ~15 epochs, one-cycle LR, ArcFace + KD, 2×T4 DDP if available.
8. Ablations on the 2 k-identity subset, ~8 epochs each: KD on/off, ArcFace/CosFace/AdaFace,
   112 vs 64 input, width S/L, VGG baseline.

### Phase 3 — Quantisation
9. PTQ INT8 (no training) as the 8-bit point.
10. QAT W4A4 and W2A2, ~4 epochs each from the FP32 student, 50 % of CASIA, KD from FP32.
11. Layer sensitivity by fast per-layer bit-sweep (eval only) → BRAM-constrained bit allocation →
    one mixed-precision QAT run → accuracy–memory–throughput Pareto plot.

### Phase 4 — Face detector
12. Tiny detector FP32 on WIDER FACE (boxes + landmarks), ~40 epochs @320; QAT W4; AP easy/med/hard.
13. Error propagation: recognition accuracy with GT vs predicted landmarks, FP32 vs W4 detector.

### Phase 5 — Computer-vision analysis + demo
14. Robustness on LFW / CFP-FP / AgeDB: resolution, blur, noise, low light, JPEG (3 severities) —
    FP32 vs INT8 vs W4 vs mixed vs W2. Pose/age read from CFP-FP/AgeDB (+ CALFW/CPLFW if available).
15. Alignment ablation (none / centre crop / 5-pt); Grad-CAM, t-SNE, failure gallery; embedding-norm
    quality score; open-set threshold calibration.
16. Demo (CPU): image/webcam/video → detector → align → integer golden recogniser → gallery match /
    "unknown"; enrolment; IoU tracking + quality-weighted temporal aggregation.

### Phase 6 — Hardware flow (no board, local/Docker)
17. Brevitas → QONNX export (detector + recogniser); golden NumPy integer model.
18. **Bit-exact checks:** PyTorch-QAT == NumPy-int == FINN cppsim == rtlsim (few images).
19. FINN folding + estimate reports (LUT/BRAM/DSP, latency, FPS); small folding sweep.
20. Optional Vivado OOC synthesis → real utilisation/Fmax/power; PYNQ package (board-ready, untested).

### Phase 7 — Write-up
21. README with diagrams, tables, Pareto/robustness plots, reproduction commands, honest limitations;
    `docs/report.md`; pytest + GitHub Actions (CPU smoke configs).

---

## 4. Notebooks & Kaggle T4 runtime (Lean)

Single T4, AMP, `channels_last`, tuned data loaders; ±50 % until the first epoch is timed. Every
training script checkpoints each epoch and resumes across sessions.

| # | Notebook | Runs on | Main work | Est. runtime |
|---|---|---|---|---|
| 00 | `00_data_prep` | CPU | Locate/verify CASIA `.rec` + eval bins, LFW gallery MTCNN alignment | ~0.5–1 h (no GPU quota) |
| 01 | `01_baselines_teacher` | T4 | Fixed FaceNet baseline, teacher eval, cache teacher embeddings (~490 k fwd) | ~0.5 h |
| 02 | `02_train_student` | T4 (2×T4 ok) | Main MobileFaceNet-M run (~15 ep) + 7 subset ablations | ~1.5–2 h + ~1.5 h |
| 03 | `03_quantisation` | T4 | PTQ INT8, QAT W4/W2, sensitivity sweep, 1 mixed-precision QAT | ~3.5–4 h |
| 04 | `04_detector` | T4 | WIDER FACE download, detector FP32 + QAT W4, AP, error propagation | ~2 h |
| 05 | `05_analysis_demo` | T4 + CPU | Robustness, alignment ablation, Grad-CAM/t-SNE, open-set, demo, figures | ~1–1.5 h |
| 06 | `06_hw_export_verify` | Local / Docker | QONNX, golden int model, FINN estimates, cppsim/rtlsim, (Vivado) | ~2–3 h CPU (+2–6 h Vivado) |

**Totals:** 7 notebooks · **~10–12 T4 GPU-hours** · ~3–4 h CPU/local — fits in one week of Kaggle
quota with headroom for re-runs.

---

## 5. Repo layout

```
face-detection/
├── README.md, PLAN.md, requirements.txt, pyproject.toml
├── configs/            # teacher, student_{s,m,l}, ablations/, qat_w{8,4,3,2}, mixed, detector, finn_z7020
├── src/fpgaface/
│   ├── data/           # casia .rec reader, eval bins, wider face, lfw gallery, alignment, corruptions
│   ├── models/         # teacher wrappers, mobilefacenet, vgg, detector, brevitas quant variants
│   ├── train/          # recogniser (margin losses + KD), detector, qat, mixed-precision search
│   ├── eval/           # verification, open-set, wider AP, robustness, cost (params/bytes/BOPs)
│   ├── hw/             # budget calc, qonnx export, golden int model, finn build
│   ├── analysis/       # grad-cam, embeddings viz, quality score
│   └── app/            # tracking, gallery, enrol, recognise, demo
├── scripts/            # CLI entry points used by the notebooks
├── notebooks/          # 00–06 + original FaceNet__SVM.ipynb
├── tests/
└── docs/               # report.md, figures/
```

## 6. Tooling

| Need | Choice |
|---|---|
| Training | PyTorch, AMP, onnxruntime-gpu (teacher) |
| QAT | Brevitas |
| HW compiler | FINN (pinned release, Docker) + QONNX; Verilator for rtlsim |
| Synthesis (optional) | Vivado / Vitis HLS (WebPACK, Zynq-7020) |
| Target parts | Zynq-7020 (PYNQ-Z2) primary; ZU3EG (Ultra96 / KV260) secondary |

## 7. Risks

* **Accuracy collapse ≤2 bits** → 8-bit first/last layers, KD, progressive QAT, mixed precision. W4A4 committed; W2A2 best-effort.
* **Recogniser + detector together exceed Zynq-7020** → time-multiplex (reconfigure) or report detector on
  ZU3EG; mixed precision is exactly the tool for this.
* **Depthwise/residual ops in FINN** → supported but fiddlier; VGG baseline is the fallback that always maps.
* **Kaggle limits (12 h sessions, quota)** → resumable checkpoints saved as Kaggle dataset versions; reduced-schedule ablations.
* **Dataset availability/licensing** → research-only datasets; document sources; LFW-only fallback path.
* **No Vivado** → FINN analytical estimates + Verilator rtlsim; clearly label estimated vs synthesised numbers.
