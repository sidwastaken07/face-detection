# Project Plan — FPGA Deployment of Quantised CNNs for Real-Time Face Recognition

> **Goal (from the CV line):** quantisation-aware training of low-bit CNN models that fit tight
> on-chip memory and logic budgets, for low-latency on-device inference.
>
> **Constraint:** no physical FPGA board. Everything up to and including bit-accurate hardware
> simulation and synthesis/resource estimates is in scope. Only on-board measurement is out.

---

## 0. Where we are (audit of `FaceNet__SVM.ipynb`)

What exists: LFW (deep-funneled) → identities with ≥5 images (423 people, 5,985 images) →
80/20 stratified split → pretrained FaceNet (InceptionResnetV1, VGGFace2) 512-d embeddings →
`StandardScaler + linear SVC` → **94.8 % top-1 accuracy** (closed-set, 423 classes).

That is a solid FP32 **baseline / teacher**, but none of the CV claim exists yet:

| CV claim | Status |
|---|---|
| Low-bit CNN | ❌ FaceNet is FP32, ~23.5 M params (~90 MB) — ~150× over a Zynq-7020's on-chip BRAM (~630 KB) |
| Quantisation-aware training | ❌ no training at all, only a frozen pretrained model |
| Fits on-chip memory/logic | ❌ no resource analysis |
| FPGA deployment / low latency | ❌ no hardware flow |

Issues to fix in the existing code:

1. **Wrong input normalisation** — FaceNet expects `(x·255 − 127.5)/128` (`fixed_image_standardization`);
   the notebook feeds `[0,1]` from `ToTensor()`. Fixing this alone should raise accuracy.
2. **No face detection/cropping** — MTCNN is installed but unused; the whole 250×250 image is resized
   to 160×160 (deep-funneled images are aligned, but the face is only the central ~half).
3. Embeddings not L2-normalised before the classifier.
4. Labels gathered via `[lbl for _, lbl in dataset]` → decodes every image just to read labels.
5. Only closed-set accuracy reported; no standard **LFW verification (pairs) protocol**, no open-set
   ("unknown person") rejection — which a real recogniser needs.
6. Notebook-only, Colab-specific (`google.colab`, Drive), mixed `clf`/`svm_clf`, stale comments
   (`>=2` vs `>=5`), duplicated saves. Not reproducible as a repo.
7. The SVM (423 one-vs-rest weight vectors) is an awkward thing to put on an FPGA; cosine nearest-
   neighbour against an enrolment gallery is simpler in hardware and supports adding new people
   without retraining.

---

## 1. Target system (what "done" looks like)

```
 camera frame ─► face detect/crop (CPU / ARM, MTCNN or Haar)
                     │  64×64 (or 112×112) uint8 face
                     ▼
            ┌───────────────────────────┐
            │  FPGA fabric (FINN)       │   W4A4 (stretch: W2A2 / W1A1) student CNN
            │  streaming dataflow CNN   │   all weights in on-chip BRAM/LUTRAM
            └───────────┬───────────────┘
                        │  128-d int embedding
                        ▼
            cosine NN vs enrolled gallery (ARM, or tiny on-fabric dot-product unit)
                        ▼
                identity / "unknown"
```

* **Teacher:** FaceNet (FP32) — kept as the accuracy reference.
* **Student:** small FPGA-friendly CNN (conv–BN–ReLU blocks, no depthwise/residual initially,
  power-of-two channel counts) trained by **embedding distillation** from FaceNet + an
  identity loss (ArcFace / CosFace margin softmax).
* **Quantisation:** Brevitas QAT at W8A8 → W4A4 → W2A2 (→ W1A1 stretch).
* **Hardware flow:** Brevitas → QONNX → **FINN** → HLS/RTL IP + Vivado synthesis reports.
  Target part for estimates: **Zynq-7020 (PYNQ-Z2)** — cheap, classic, tight budget
  (53 k LUT, 106 k FF, 140 BRAM36 ≈ 4.9 Mb, 220 DSP). Secondary: ZU3EG (Ultra96 / Kria KV260).
* **Verification without a board:** bit-exact NumPy integer reference model + FINN
  `cppsim`/`rtlsim` (Verilator) must match the PyTorch QAT model output-for-output.

---

## 2. Step-by-step plan

### Phase 1 — Repo & reproducible baseline (≈1–2 days)
1. Repo skeleton: `src/` package, `configs/` (YAML), `scripts/`, `notebooks/`, `tests/`,
   `requirements.txt`, `README.md`, `.gitignore` (data, checkpoints).
2. Data module: LFW download (Kaggle *or* the official tarball fallback), identity filter
   (≥5 imgs), deterministic split saved to `splits/*.json`, plus the official LFW `pairs.txt`
   for verification. Labels read from paths, not by decoding images.
3. Face preprocessing: MTCNN crop+align once, cache crops to disk (160×160 for teacher,
   64×64/112×112 for student) — so no FPGA-side detector is required.
4. Port the FaceNet+SVM notebook into `scripts/baseline_facenet.py` with the fixes from §0
   (correct normalisation, L2-norm, cosine-NN and SVM heads).
5. **Metrics module (used by every later phase):**
   closed-set top-1, LFW 10-fold verification accuracy + TAR@FAR=1e-3, open-set
   (DIR@FAR), params, model size (bits), MACs/BOPs.
6. Cache teacher embeddings for all crops (`.npy`) → distillation targets.

*Exit criteria:* `python scripts/baseline_facenet.py` reproduces ≥94.8 % (expect higher) and LFW
verification ≈ 99 %.

### Phase 2 — FPGA-friendly FP32 student (≈2–3 days)
7. Budget first: compute the maximum weight count for 4-bit weights to fit in, say, 70 % of
   Zynq-7020 BRAM → ~1 M params @4b ≈ 0.5 MB. Design students under that.
8. Student architectures (configurable): e.g. `TinyFaceNet-S/M/L`
   (64×64 input, 5–6 conv blocks, 3×3 kernels, max-pool, channels 32→256, GAP, FC→128-d).
   Optional: MobileFaceNet-lite variant for comparison.
9. Training: distillation loss (cosine/MSE to teacher embedding) + ArcFace head on train
   identities; augmentation (flip, small crop/colour jitter). Runs on Colab GPU;
   a CPU "smoke" config for CI.
10. Evaluate with the Phase-1 metrics; pick the size/accuracy sweet spot.

*Exit criteria:* FP32 student within a few points of the teacher on closed-set and LFW pairs,
under the parameter budget.

### Phase 3 — Quantisation (the core of the project) (≈3–4 days)
11. Baseline **PTQ** (post-training INT8) for reference — shows why QAT is needed at low bits.
12. **QAT with Brevitas**: `QuantConv2d`/`QuantReLU`/`QuantIdentity` with per-channel weight
    scales, 8-bit input & output layers, low-bit inside. Sweep W8A8, W4A4, W3A3, W2A2 (W1A1 stretch).
    Init from FP32 student, distillation from FP32 teacher continues during QAT.
13. Tricks to try & ablate: BN folding, learned scales (LSQ-style), progressive bit-width
    reduction, keeping first/last layers at 8 bit.
14. Results table + Pareto plot: accuracy vs. model size (KB) vs. BOPs per configuration.

*Exit criteria:* a W4A4 (or lower) model that fits on-chip with ≤ ~3–5 pt drop vs FP32 student.

### Phase 4 — Hardware generation & verification without a board (≈3–5 days)
15. Export QAT model → **QONNX**; run FINN transformations (streamline, convert to HW layers,
    folding/PE-SIMD config chosen for a target FPS, e.g. ≥1,000 faces/s @100 MHz).
16. **Golden model:** pure NumPy integer-only inference from the QONNX graph (thresholds,
    integer weights). Test: PyTorch-QAT == NumPy-int == FINN `cppsim` == `rtlsim`
    (bit-exact on a test batch). This is the substitute for on-board validation.
17. FINN **estimate reports** (no Vivado needed): LUT/BRAM/DSP per layer, cycles per layer,
    estimated FPS & latency.
18. If Vivado/Vitis HLS (free WebPACK covers Zynq-7020) is available locally/Docker:
    out-of-context synthesis + place & route → **post-implementation** utilisation, Fmax, power
    estimate; generate the PYNQ bitstream + driver package so it's "board-ready".
19. Folding sweep: throughput vs. resources (show the design space, pick configs for 7020 and ZU3EG).

*Exit criteria:* bit-exact match across all simulators; resource report showing fit on Zynq-7020;
latency/throughput numbers; a deployable (but untested-on-board) PYNQ package.

### Phase 5 — End-to-end application on CPU emulation (≈1–2 days)
20. `scripts/demo.py`: image/webcam → detect → crop → quantised student (PyTorch QAT or
    integer golden model, i.e. exactly what the FPGA would compute) → cosine NN → name/"unknown".
21. Enrolment CLI: add a person from a few photos to the gallery; threshold tuned on
    a validation set for open-set rejection.
22. Throughput comparison table: FaceNet FP32 on CPU vs student FP32/INT on CPU vs
    FINN-estimated FPGA (clearly labelled *estimated / simulated*).

### Phase 6 — Packaging & write-up (≈1–2 days)
23. README: motivation, architecture diagram, results tables, Pareto plots, reproduction
    commands, honest limitations ("hardware numbers from synthesis/simulation, not measured on board").
24. Tests (`pytest`): data split determinism, metric correctness, quant-model export,
    golden-model bit-exactness (small model, CPU-only); GitHub Actions running the CPU tests.
25. Short technical report (`docs/report.md`) + clean results notebook for figures.
26. Update the CV line with real numbers (e.g. "W4A4 CNN, 0.4 MB, fits Zynq-7020 on-chip,
    98.x % LFW, ~N k FPS est.").

---

## 3. Proposed repo layout

```
face-detection/
├── README.md, PLAN.md, requirements.txt, pyproject.toml
├── configs/            # baseline.yaml, student_*.yaml, qat_w4a4.yaml, finn_z7020.yaml
├── src/fpgaface/
│   ├── data/           # lfw download, split, mtcnn crop cache, pairs
│   ├── models/         # facenet teacher, fp32 students, brevitas quant students
│   ├── train/          # distill + arcface loops, qat loop
│   ├── eval/           # closed-set, LFW verification, open-set, cost (params/BOPs/bytes)
│   ├── hw/             # qonnx export, finn build, golden int model, resource budget calc
│   └── app/            # gallery, enrol, recognise
├── scripts/            # baseline_facenet.py, train_student.py, train_qat.py, export.py,
│                       # build_finn.py, verify_bitexact.py, demo.py
├── notebooks/          # FaceNet__SVM.ipynb (original), results.ipynb
├── tests/
└── docs/               # report.md, figures/
```

## 4. Tooling & compute

| Need | Choice | Notes |
|---|---|---|
| Training | PyTorch + facenet-pytorch | GPU (Colab) for real runs; CPU smoke configs |
| QAT | **Brevitas** | native FINN path via QONNX |
| HW compiler | **FINN** (Docker) | estimates + cppsim without Vivado; rtlsim needs Verilator |
| Synthesis | Vivado / Vitis HLS 2022.2+ (WebPACK, free for Zynq-7020) | optional, run locally |
| Alt. flow (optional comparison) | hls4ml + QKeras/HGQ | bit-accurate C emulation without Vivado |

## 5. Risks

* **Accuracy collapse at ≤2 bits** → keep first/last layers 8-bit, distillation, progressive QAT;
  W4A4 is the committed target, W2A2/W1A1 are stretch.
* **FINN version churn / Docker setup** → pin a FINN release; fall back to hls4ml if blocked.
* **No Vivado** → FINN analytical estimates + Verilator rtlsim still give credible numbers; state
  clearly which numbers are estimated.
* **Small dataset (LFW)** → distillation from a VGGFace2 teacher does most of the work; optionally
  pretrain the student on a larger set (CASIA-WebFace / VGGFace2 subset) if time permits.
