# Deepfake Detection System

Full-stack web application for detecting AI-manipulated images and videos, built as the capstone project for an MS in Software Engineering. Users register, upload media, and get a real/fake verdict with a confidence score; results are stored per user and browsable through a React UI.

## How it works

```
React SPA (frontend/, :3000)
        │  JWT-authenticated REST
        ▼
FastAPI backend (app/, :8000)
        │  upload → validate → store row in SQLite
        ▼
Detector: Hugging Face ViT  ──►  detection_results (confidence, verdict, timing, metadata)
(images directly; videos via
 OpenCV frame sampling)
```

- **Images**: `POST /api/upload` validates type/size (JPEG/PNG, ≤10 MB, verified with Pillow), then `POST /api/detection/analyze/{file_id}` runs the classifier and persists a result row.
- **Videos**: `POST /api/video/upload` (MP4/AVI/MOV/MKV/WMV/FLV, ≤500 MB) reads metadata with OpenCV, samples ~10 evenly spaced frames, classifies each frame, and averages the per-frame confidence into an overall verdict.
- **Auth**: JWT (HS256 via python-jose) with bcrypt password hashing; file listings and results are scoped per user account (with one known exception noted under limitations).
- **Storage**: SQLite through SQLAlchemy ORM: `users`, `media_files`, `detection_results`.

The detector wired into the API is the pretrained Vision Transformer [`prithivMLmods/deepfake-detector-model-v1`](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1), loaded through `transformers` and run on CPU. It is fetched from the Hugging Face Hub on first startup.

## Ensemble research codebase

Alongside the serving path, `app/models/` contains a multi-model ensemble subsystem developed for the capstone research component. It is exercised through the scripts in `scripts/` rather than the HTTP API: `app/main.py` attempts to mount the `/advanced-ensemble` routes, but the import path is broken relative to the repo root, so they silently fail to load (a known defect, left as-is).

- **Detectors**: ResNet-50 and EfficientNet-B4 binary classifiers (torchvision backbones with custom heads), a frequency-domain F3Net (8×8 block DCT, fixed high-pass filtering over DCT coefficients, channel attention, dual spatial+frequency branches), and a from-scratch MesoNet. An Xception wrapper exists but depends on a model torchvision does not provide.
- **Fusion**: `EnsembleManager` implements weighted average, majority voting, soft voting, confidence-softmax attention fusion, and max/min confidence, with variance-based uncertainty and a grid search for member weights. The advanced variant adds multi-head-attention merging, LBFGS temperature calibration, and Monte-Carlo-dropout uncertainty.
- **Weights**: trained checkpoints for ResNet-50, EfficientNet-B4, and F3Net ship in `models/`, together with training-history JSONs for the ResNet and EfficientNet runs.
- **Training & evaluation**: `training/` holds the training pipelines, dataset management, adversarial-training/contrastive/distillation utilities, and an evaluation framework; `scripts/` holds dataset generation, training drivers, and the ad-hoc test/demo scripts.

## Running it

Local development:

```bash
pip install -r requirements_fastapi.txt bcrypt   # bcrypt isn't pinned in the file, but passlib's bcrypt backend needs it
python run.py                      # uvicorn on :8000, Swagger at /docs

cd frontend && npm install && npm start   # React dev server on :3000
```

Docker (API plus Postgres, Redis, nginx, Prometheus, Grafana):

```bash
DB_PASSWORD=... SECRET_KEY=... GRAFANA_PASSWORD=... docker compose up -d --build
```

Python 3.11 (pinned in the Dockerfile); the first backend start downloads the ViT weights from the Hugging Face Hub, so network access is required once.

## Datasets

No dataset media is committed. To reproduce the experiments:

- **Celeb-DF-v2**: request access through the official form linked from [yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics), then place the videos under `Celeb-DF-v2/{Celeb-real,Celeb-synthesis,YouTube-real}/` at the repo root, which is where `scripts/test_celeb_df_v2.py` and `scripts/find_demo_images.py` expect them.
- **Synthetic smoke-test set**: `python scripts/create_test_dataset_simple.py` generates a small labeled `test_data/` tree of images and placeholder videos. The copy committed at `scripts/test_data/` holds the metadata and placeholders only, and its `batch_test.py` predates the current API routes, so regenerate before use.

## Status and known limitations

Read honestly, this is a working capstone system with clearly marked rough edges:

- The single pretrained ViT is the only detector wired into the API; the ensemble is research code driven by scripts.
- No benchmark evaluation on a real deepfake dataset is recorded in this repo. The training-history JSONs in `models/` report validation accuracy against a locally generated synthetic dataset, and accuracy figures quoted in code docstrings come from upstream model cards and the literature (DeepfakeBench and the cited papers), not from measurements here. <!-- TODO(colton): run and commit a real evaluation, e.g. on the Celeb-DF-v2 test split -->
- The compose stack provisions Postgres/Redis, but the application code currently hardcodes SQLite and a development `SECRET_KEY` in `app/config.py`; fine for a demo box, not for a real deployment.
- `GET /api/files/public/{file_id}` serves uploaded files without authentication or an ownership check, a known gap in the per-user isolation.
- Model weights are committed directly to git as exceptions to the repo's own `*.pth` ignore rule (`models/resnet_weights.pth` is ~99 MB); Git LFS would be the right home for them.
- `backend/` is a vestigial early scaffold; the live package is `app/`.

## Repository layout

| Path          | Contents                                                        |
| ------------- | --------------------------------------------------------------- |
| `app/`        | FastAPI application: routes, auth, ORM models, detector code    |
| `frontend/`   | React single-page app (Create React App)                        |
| `models/`     | Trained ensemble checkpoints and training histories             |
| `training/`   | Training pipelines, dataset management, evaluation framework    |
| `scripts/`    | Dataset generation, training drivers, ad-hoc test/demo scripts  |
| `deployment/` | Kubernetes manifests, nginx, Prometheus/Grafana, deploy script  |

## References

- Chollet, *Xception: Deep Learning with Depthwise Separable Convolutions*, 2017
- Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, 2019
- Qian et al., *Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues* (F3Net), 2020
- Li et al., *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*, 2020
- [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench); [`prithivMLmods/deepfake-detector-model-v1`](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1)
