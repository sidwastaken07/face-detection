# Quantised Face Detection & Recognition on FPGA

Low-bit CNNs for face detection and recognition, trained with quantisation-aware training (QAT) and
sized so that every weight fits in the on-chip memory of a small FPGA (Xilinx Zynq-7020). Hardware is
generated with FINN and verified bit-exactly in simulation (no physical board).

See [`PLAN.md`](PLAN.md) for the full plan, design decisions and compute budget.

## Status

| Phase | Status |
|---|---|
| 1 · Repo, data pipeline, metrics, FP32 baselines, teacher cache | ✅ code done, awaiting Kaggle run |
| 2 · FP32 student (MobileFaceNet-FPGA) + ablations | ⏳ |
| 3 · Quantisation (PTQ INT8, QAT W4/W2, mixed precision) | ⏳ |
| 4 · Quantised face detector | ⏳ |
| 5 · Robustness / explainability analysis + demo | ⏳ |
| 6 · FINN hardware flow + bit-exact verification | ⏳ |

## Pipeline

```
frame → tiny detector (QAT) → 5-pt alignment → MobileFaceNet-FPGA (mixed-precision QAT) → 128-d embedding
      → cosine match against enrolled gallery → identity / unknown
```

* **Training data:** CASIA-WebFace (InsightFace `faces_webface_112x112`, 10.5k identities, ~490k images)
* **Evaluation:** LFW, CFP-FP, AgeDB-30, CALFW, CPLFW verification (identity-disjoint from training);
  LFW ≥5-image gallery for closed/open-set identification
* **Teacher:** InsightFace iResNet-50 ArcFace (FP32); FaceNet (VGGFace2) as the original baseline

## Running (Kaggle)

| Notebook | Accelerator | Inputs |
|---|---|---|
| `notebooks/00_data_prep.ipynb` | None (CPU) | CASIA `faces_webface_112x112`, `jessicali9530/lfw-dataset` |
| `notebooks/01_baselines_teacher.ipynb` | GPU T4 | same + output of 00 |

Each notebook clones this repository and calls the scripts in `scripts/`; all logic lives in
`src/fpgaface`. Enable Internet in the notebook settings.

## Local development

```bash
pip install -e ".[dev,teachers]"
pip install --no-deps facenet-pytorch   # pins old torch otherwise
pytest -q
```

## Layout

```
src/fpgaface/
  data/     recordio (.rec reader, no MXNet), casia, bins (verification sets), align, lfw gallery
  models/   teachers (FaceNet, iResNet-50 ArcFace via onnx2torch)
  eval/     verification (10-fold, TAR@FAR), identification (closed/open-set), cost (params/MACs/bytes)
scripts/    prepare_data.py, baselines.py, cache_teacher.py
notebooks/  Kaggle notebooks + the original FaceNet__SVM.ipynb
tests/      unit tests on synthetic data (CPU)
```

## Original baseline

`notebooks/FaceNet__SVM.ipynb` (frozen FaceNet + SVM on an image-level LFW split, 94.8 % top-1) is kept
for reference. `scripts/baselines.py` reproduces it and shows the effect of fixing its preprocessing
(FaceNet standardisation, MTCNN face crops, L2-normalised embeddings).
