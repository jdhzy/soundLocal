current data and results are in scc /restricted/projectnb/cs599dg/jd/ego4d_mc3_temporal


# SoundingActions Temporal Localization  
### (MC3 Evaluation on the AVE Dataset)

This repository evaluates cross-modal temporal alignment between audio and video using the **MC3 backbone** from  
*SoundingActions: Learning How Actions Sound from Narrated Egocentric Videos (Chen et al., CVPR 2024)*.

We test how well a **frozen MC3 encoder** can locate when an event’s sound occurs within a 10-second video clip from the **AVE dataset**, without any fine-tuning.

---

## Environment Setup (Makefile)

The provided Makefile automates all setup steps on SCC or a local Linux system.

| Command | Description |
|----------|-------------|
| `make install` | Creates `.venv` using Python 3.12, installs dependencies from `requirements.txt` (CPU). |
| `make gpu` | Loads Python + CUDA modules, installs PyTorch (CUDA 12.1 wheels), and prepares a GPU-ready environment. |
| `make gpu-login` | Requests a 4-hour interactive GPU node (1 GPU, 24 GB RAM). |
| `make clean` | Deletes `.venv` to reset the environment. |
| `make reinstall` | Combines `clean` + `install` to rebuild everything from scratch. |

**Tip:**  
Run `make gpu-login` first to get a GPU shell, then inside that session run `make gpu` so CUDA wheels install correctly.

---

## Dataset Setup — AVE (Audio-Visual Event)

The AVE dataset contains 4,097 10-second clips annotated with the start and end times of the sound-producing event.

### Step 1: Create directories
```bash
mkdir -p data/ave/raw data/ave/ave_wav

Step 2: Download and extract videos

cd data/ave/raw
wget http://www.robots.ox.ac.uk/~vgg/data/ave/AVE_Dataset.zip
unzip AVE_Dataset.zip -d AVE
cd ../../..

Step 3: Download annotations

wget http://www.robots.ox.ac.uk/~vgg/data/ave/annotations/AVE_annotation.csv \
  -O data/ave/ave_annotations.csv

Expected structure

data/ave/
├── raw/AVE/                # Original .mp4 videos
├── ave_annotations.csv     # Ground-truth (start_s, end_s)
└── ave_wav/                # Audio tracks (.wav) generated later
```


## Source Directory Overview

### Core Modules

- **mc3_frozen.py** — Defines the frozen MC3 video encoder used for feature extraction. Handles device/dtype management and normalization.  
- **video_windows.py** — Video I/O helpers for frame sampling, sliding windows, and center-time generation.

---

### Embedding Extraction

- **extract_video_emb.py** — Encodes each MP4 into per-window video embeddings `(T, D)` using the frozen MC3 model. Saves `.npz` per video in `cache/vid_emb/`.  
- **extract_audio_emb.py** — Simple single-scale audio embedding extractor.  
- **extract_audio_multiscale.py** — Multi-scale audio extraction (0.25–5 s windows, variable stride). Saves `.npz` per `(video,length)` in `cache/aud_emb/`.

---

### Evaluation and Refinement

- **eval_temporal_alignment.py** — Baseline temporal alignment: computes cosine similarity between audio and video embeddings over time → `softmax(sim / τ)` → temporal PDF → peak = predicted sound time.  
- **train_refiner.py** — Builds a dataset of baseline PDFs and trains a lightweight MLP refiner (`Linear 128→256→128 + Softmax`) to sharpen them.  
- **eval_with_refiner.py** — Runs evaluation using the trained refiner network to improve temporal precision.  
- **refiner.py** — Defines the standalone refiner network architecture.

---

### Hyperparameter and Ablation

- **hparam_sweep.py** — Grid-search over audio window length (L) and temperature (τ); writes per-run results under `reports/grid/`.  
- **hparam_2.py**, **hparam3.py** — Extended sweeps adding stride/window combinations.  
- **ablate_small.py** — Quick small-set ablations for z-score, smoothing, and fusion strategies.

---

### Analysis and Utilities

- **analyze_results.py** — Reads `summary.csv`, canonicalizes metrics, computes MAE, Hit@δ, Inside, and generates histograms + calibration plots in timestamped `reports/analysis/<run_tag>/`.  
- **parse_ave_txt_annotations.py** — Converts AVE’s text annotations into CSV format if needed.  
- **cache/** — Local directory for storing intermediate embeddings (`vid_emb/`, `aud_emb/`). Safe to delete or rebuild.

