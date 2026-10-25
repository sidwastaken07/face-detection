# Project Plan — Quantised Face Detection & Recognition on FPGA

> **Goal:** an end-to-end face detection + recognition system built from low-bit CNNs trained with
> quantisation-aware training (QAT), sized to fit entirely in the on-chip memory/logic of a small
> FPGA (Zynq-7020) for low-latency on-device inference.
>
> **Constraint:** no physical FPGA board. Everything up to and including bit-exact hardware
> simulation and synthesis/resource estimates is in scope; only on-board measurement is out.
>
> **Compute:** Kaggle notebooks (T4 GPU, 12 h/session, ~30 GPU-h/week). Hardware flow runs
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
* PTQ INT8 reference → **Brevitas QAT** W8A8, W4A4, W3A3, W2A2 (W1A1 stretch); per-channel weight
  scales, learned step size (LSQ-style), BN folding, first/last layer 8-bit; distillation FP32→quantised.
* **Contribution — hardware-aware mixed precision:** per-layer sensitivity (bit-sweep + Hessian-trace),
  then bit allocation that maximises accuracy subject to a Zynq-7020 BRAM budget (knapsack/ILP),
  then QAT of the chosen config. Compare vs uniform QAT on an accuracy–memory–throughput Pareto front.
* Detector: W8A8 and W4A4 QAT.

---

## 3. Phases

### Phase 1 — Repo, data, reproducible baselines
1. `src/fpgaface` package, configs (YAML), scripts, tests, CI; notebooks are thin wrappers that
   `pip install` the repo from GitHub and call it.
2. Data: CASIA `.rec` reader (no MXNet dependency), eval `.bin` loader, WIDER FACE loader, LFW gallery
   with MTCNN alignment to the ArcFace 112×112 template.
3. Metrics: 10-fold verification, TAR@FAR, open-set DIR@FAR, WIDER AP, params/bytes/MACs/BOPs.
4. Fixed FaceNet baseline + iResNet-50 teacher evaluated on all 5 benchmarks.
5. Cache teacher embeddings (orig + flip) for all CASIA images.

### Phase 2 — FP32 students
6. On-chip budget calculator (weights + FINN FIFO/line buffers per bit-width) → size S/M/L to fit.
7. Train MobileFaceNet-M main run; ablations on reduced schedule: KD on/off, ArcFace/CosFace/AdaFace,
   112 vs 64 input, width S/L, VGG baseline.

### Phase 3 — Quantisation
8. PTQ INT8; uniform QAT W8/W4/W3/W2 (W1 stretch).
9. Layer sensitivity → BRAM-constrained mixed-precision search → QAT → Pareto plots.

### Phase 4 — Face detector
10. Train tiny detector FP32 on WIDER FACE (boxes + landmarks); QAT W8/W4; AP easy/med/hard.
11. Measure detection/landmark error propagation into recognition accuracy (aligned with GT vs
    predicted landmarks, FP32 vs quantised detector).

### Phase 5 — Computer-vision analysis
12. Robustness suite: resolution (112→24 px), blur, noise, low light, JPEG, pose (CPLFW), age (CALFW) —
    FP32 vs W8 vs W4 vs mixed vs W2. Key question: *do low-bit models fail differently on degraded images?*
13. Alignment ablation: none vs centre crop vs 5-pt.
14. Explainability: Grad-CAM on embedding similarity, t-SNE/UMAP of embeddings, failure-case gallery,
    FP32 vs low-bit.
15. Embedding norm as face-quality score; open-set threshold calibration.

### Phase 6 — Hardware flow (no board)
16. Brevitas → QONNX export (detector + recogniser).
17. Golden NumPy integer-only model; **bit-exact checks:** PyTorch-QAT == NumPy-int == FINN cppsim == rtlsim.
18. FINN folding for target FPS; estimate reports (LUT/BRAM/DSP, cycles, latency, FPS); folding sweep.
19. Optional (if Vivado/Vitis HLS available, WebPACK covers Zynq-7020): OOC synthesis + P&R → real
    utilisation/Fmax/power; build PYNQ bitstream + driver package (board-ready, untested).

### Phase 7 — End-to-end application
20. Demo: image/webcam/video → detector → align → recogniser (integer golden model = what the FPGA computes)
    → gallery match / "unknown"; enrolment CLI.
21. Video: IoU tracking + quality-weighted temporal embedding aggregation (stretch: YTF subset eval).
22. Latency table: FaceNet FP32 CPU vs students CPU vs FINN-estimated FPGA (clearly labelled).

### Phase 8 — Write-up
23. README with diagrams, tables, Pareto/robustness plots, reproduction commands, honest limitations.
24. `docs/report.md` technical report; pytest + GitHub Actions (CPU smoke configs).

---

## 4. Notebook breakdown & Kaggle T4 runtime estimates

Estimates are for a **single T4** with AMP, data already in a Kaggle dataset, and are ±50 % until the
first epoch is timed (they will be recalibrated then). Kaggle's 2×T4 option with DDP can cut the long
training runs by ~1.6× if not data-loader-bound (4 vCPU).

| # | Notebook | Hardware | Main work | Est. runtime |
|---|---|---|---|---|
| 00 | `00_data_prep` | CPU | Download/convert CASIA, eval bins, WIDER FACE, LFW gallery align (MTCNN) | ~1 h (no GPU quota) |
| 01 | `01_baselines_teacher` | T4 | Fixed FaceNet + SVM/cosine baseline, teacher eval on 5 benchmarks, cache teacher embeddings (~980 k fwd passes) | ~0.75–1 h |
| 02 | `02_student_fp32` | T4 | MobileFaceNet-M, 25 epochs @112 (~5–8 min/epoch) | ~2.5–3.5 h |
| 03 | `03_fp32_ablations` | T4 | ~7 reduced-schedule runs (KD, losses, 64 px, S/L, VGG) at ~1 h each | ~6–8 h (2 sessions) |
| 04 | `04_qat_uniform` | T4 | PTQ INT8 + QAT W8/W4/W3/W2, ~8 epochs each (QAT ≈2–3× slower/epoch) | ~7–9 h (2 sessions); +2–3 h W1A1 |
| 05 | `05_mixed_precision` | T4 | Layer sensitivity + Hessian trace, budgeted search, 2 mixed-precision QAT runs | ~5 h |
| 06 | `06_detector` | T4 | Tiny detector FP32 ~80 epochs @320 + QAT W8/W4 + WIDER eval | ~4–5 h |
| 07 | `07_cv_analysis` | T4 | Robustness sweep (~90 model×condition evals), alignment ablation, Grad-CAM, t-SNE, quality score | ~2–3 h |
| 08 | `08_hw_export_verify` | Local / Docker (CPU) | QONNX export, golden int model, FINN estimates, cppsim/rtlsim bit-exact checks, folding sweep | ~2–4 h; +2–6 h if Vivado synth |
| 09 | `09_demo_video` | T4 or CPU | End-to-end pipeline, enrolment, open-set calibration, tracking + temporal aggregation | ~1–2 h |
| 10 | `10_results_report` | CPU | Collect metrics, final tables and figures | ~15 min |

**Totals:** 11 notebooks · **~29–37 T4 GPU-hours** (core) + ~2–3 h stretch · ~3–5 h CPU/local.
That is **~1–1.5 weeks of Kaggle GPU quota**; plan for ~2 weeks including debugging and re-runs.

Ways to cut cost if quota is tight: train at 64×64 (~2–3× faster), run ablations on a 2 k-identity
CASIA subset, drop W3A3, or use 2×T4 DDP for NB02/NB04.

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
├── notebooks/          # 00–10 + original FaceNet__SVM.ipynb
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

* **Accuracy collapse ≤2 bits** → 8-bit first/last layers, KD, progressive QAT, mixed precision. W4A4 committed; W2/W1 stretch.
* **Recogniser + detector together exceed Zynq-7020** → time-multiplex (reconfigure) or report detector on
  ZU3EG; mixed precision is exactly the tool for this.
* **Depthwise/residual ops in FINN** → supported but fiddlier; VGG baseline is the fallback that always maps.
* **Kaggle limits (12 h sessions, quota)** → resumable checkpoints saved as Kaggle dataset versions; reduced-schedule ablations.
* **Dataset availability/licensing** → research-only datasets; document sources; LFW-only fallback path.
* **No Vivado** → FINN analytical estimates + Verilator rtlsim; clearly label estimated vs synthesised numbers.
