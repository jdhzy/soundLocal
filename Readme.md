Awesome—here’s a tidy README section you can drop in that explicitly covers (1) what the Makefile does, (2) Dataset setup, and (3) a concise file-by-file glossary of src/.

⸻

Project Guide

What the Makefile Does

The Makefile is there to make setup on SCC painless:
	•	make install
Creates a Python virtualenv (.venv), upgrades pip, and installs everything from requirements.txt for CPU use.
	•	make gpu
Loads SCC modules for Python and CUDA, creates .venv (if missing), installs PyTorch/cu121 wheels, then installs your requirements.txt. Result: a GPU-ready environment.
	•	make clean
Deletes the .venv to reset your environment.
	•	make reinstall
clean + install in one step.


⸻

Dataset Setup (AVE)

We use the Audio-Visual Event (AVE) dataset (4,097 clips, 10 s each) and the official annotations.
	1.	Make folders:

mkdir -p data/ave/raw data/ave/ave_wav

	2.	Download & unzip videos:

cd data/ave/raw
wget http://www.robots.ox.ac.uk/~vgg/data/ave/AVE_Dataset.zip
unzip AVE_Dataset.zip -d AVE
cd ../../..

	3.	Download annotations:

wget http://www.robots.ox.ac.uk/~vgg/data/ave/annotations/AVE_annotation.csv \
  -O data/ave/ave_annotations.csv

Expected structure:

data/ave/
├── raw/AVE/                # .mp4 videos
├── ave_annotations.csv     # GT intervals (start_s, end_s)
└── ave_wav/                # (you’ll generate .wav tracks here)


⸻

Absolutely 👍 — here’s a polished GitHub-ready README.md version, formatted with sections, code blocks, and tables.
It fully explains the Makefile, dataset setup, and each source file, while staying readable and professional.

⸻

🎬 SoundingActions Temporal Localization

(MC3 Evaluation on the AVE Dataset)

This repository evaluates cross-modal temporal alignment between audio and video using the MC3 backbone from
🧠 [SoundingActions: Learning How Actions Sound from Narrated Egocentric Videos (Chen et al., CVPR 2024)].

We test how well a frozen MC3 encoder can locate when an event’s sound occurs within a 10-second video clip from the AVE dataset—without any fine-tuning.

⸻

🧰 Environment Setup (Makefile)

The provided Makefile automates all setup steps on SCC or local Linux systems.

Command	Description
make install	Creates .venv using Python 3.12, installs dependencies from requirements.txt (CPU).
make gpu	Loads Python + CUDA modules, installs PyTorch (CUDA 12.1 wheels), and prepares a GPU-ready environment.
make gpu-login	Requests a 4-hour interactive GPU node (1 GPU, 24 GB RAM) for experimentation.
make clean	Deletes .venv to reset the environment.
make reinstall	Combines clean + install to rebuild everything from scratch.

💡 Tip:
Run make gpu-login first to get a GPU shell, then inside that session run make gpu so CUDA wheels install correctly.

⸻

📦 Dataset Setup — AVE (Audio-Visual Event)

The AVE dataset contains 4,097 10-second clips annotated with the start and end times of the sound-producing event.

1️⃣ Create directories

mkdir -p data/ave/raw data/ave/ave_wav

2️⃣ Download and extract videos

cd data/ave/raw
wget http://www.robots.ox.ac.uk/~vgg/data/ave/AVE_Dataset.zip
unzip AVE_Dataset.zip -d AVE
cd ../../..

3️⃣ Download annotations

wget http://www.robots.ox.ac.uk/~vgg/data/ave/annotations/AVE_annotation.csv \
  -O data/ave/ave_annotations.csv

Expected structure

data/ave/
├── raw/AVE/                # Original .mp4 videos
├── ave_annotations.csv     # Ground-truth (start_s, end_s)
└── ave_wav/                # Audio tracks (.wav) generated later


⸻

🧩 Source Directory Overview

🧠 Core Modules

File	Purpose
mc3_frozen.py	Defines the frozen MC3 video encoder used for feature extraction. Handles device/dtype management and normalization.
video_windows.py	Video I/O helpers for frame sampling, sliding windows, and center-time generation.


⸻

🎥 Embedding Extraction

File	Description
extract_video_emb.py	Encodes each MP4 into per-window video embeddings (T, D) using the frozen MC3 model. Saves .npz per video in cache/vid_emb/.
extract_audio_emb.py	Simple single-scale audio embedding extractor.
extract_audio_multiscale.py	Multi-scale audio extraction (0.25–5 s windows, variable stride). Saves .npz per (video,length) in cache/aud_emb/.


⸻

📈 Evaluation and Refinement

File	Description
eval_temporal_alignment.py	Baseline temporal alignment: computes cosine similarity between audio + video embeddings over time → softmax( sim / τ ) → temporal PDF → peak = predicted sound time.
train_refiner.py	Builds a dataset of baseline PDFs and trains a lightweight MLP refiner (Linear 128→256→128 + Softmax) to sharpen them.
eval_with_refiner.py	Runs evaluation using the trained refiner network to improve temporal precision.
refiner.py	Defines the standalone refiner network architecture.


⸻

⚙️ Hyperparameter & Ablation

File	Description
hparam_sweep.py	Grid-search over audio window length (L) and temperature (τ); writes per-run results under reports/grid/.
hparam_2.py, hparam3.py	Extended sweeps adding stride/window combinations.
ablate_small.py	Quick small-set ablations for z-score, smoothing, and fusion strategies.


⸻

📊 Analysis & Utilities

File	Description
analyze_results.py	Reads summary.csv, canonicalizes metrics, computes MAE, Hit@δ, Inside, and generates histograms + calibration plots in timestamped reports/analysis/<run_tag>/.
parse_ave_txt_annotations.py	Converts AVE’s text annotations into CSV format if needed.
cache/	Local directory for storing intermediate embeddings (vid_emb/, aud_emb/). Safe to delete/rebuild.


⸻

🔁 Typical Workflow

# 1. Extract video embeddings
python src/extract_video_emb.py --vid_dir data/ave/raw/AVE --out_dir cache/vid_emb

# 2. Extract multi-scale audio embeddings
python src/extract_audio_multiscale.py \
  --wav_dir data/ave/ave_wav --out_dir cache/aud_emb \
  --lengths "0.25,0.5,1.0,2.0,5.0" --stride_sec 0.25 --device cuda

# 3. Evaluate baseline temporal alignment
python src/eval_temporal_alignment.py \
  --vid_emb_dir cache/vid_emb --aud_emb_dir cache/aud_emb \
  --annotations_csv data/ave/ave_annotations.csv \
  --summary_csv reports/summary.csv --curve_dir reports/curves

# 4. Hyperparameter sweeps (optional)
python src/hparam_sweep.py --lengths "0.5,1.0,2.0" --taus "0.03,0.05,0.07,0.10"

# 5. Train refiner network
python src/train_refiner.py \
  --vid_emb_dir cache/vid_emb --aud_emb_dir cache/aud_emb \
  --annotations_csv data/ave/ave_annotations.csv --epochs 5 --batch_size 256

# 6. Evaluate with refiner
python src/eval_with_refiner.py \
  --vid_emb_dir cache/vid_emb --aud_emb_dir cache/aud_emb \
  --refiner_ckpt checkpoints/refiner/refiner.pt \
  --summary_csv reports/summary_refiner.csv --curve_dir reports/curves_refiner

# 7. Analyze results
python src/analyze_results.py --summary_csv reports/summary_refiner.csv \
  --out_dir reports/analysis --run_tag refiner_final


⸻

🧠 Key Ideas
	•	Cosine similarity quantifies audio–video alignment in feature space.
	•	Softmax temperature (τ) tunes the sharpness of the temporal probability curve.
	•	Adaptive τ (entropy target) automatically balances certainty and coverage.
	•	Multi-crop audio fusion averages/max-pools PDFs across ± 0.25 s offsets.
	•	Z-score + Gaussian smoothing stabilize noisy similarities.
	•	Refiner MLP post-processes PDFs to produce sharper temporal localization.

⸻

📊 Example Results

Metric	Baseline	After Refiner
MAE ↓	2.86 s	0.77 s
Hit@0.25 ↑	0.04	0.67
Hit@0.5 ↑	0.08	0.68
Hit@1.0 ↑	0.17	0.73
Inside Interval ↑	0.81	0.84


⸻

📎 Citation

Chen et al., SoundingActions: Learning How Actions Sound from Narrated Egocentric Videos.
CVPR 2024.
https://arxiv.org/abs/2401.00054

⸻

Would you like me to append a short “Project Structure Tree” (showing the hierarchy under src/, data/, cache/, and reports/) at the end to make it even more complete for GitHub?What Each File Does (Source Glossary)

Core modeling / IO
	•	mc3_frozen.py
Wraps the MC3 video backbone (frozen) + a lightweight audio projector. Handles dtype/device logic, normalization, and safe inference.
	•	video_windows.py
Video reading utilities (decord/OpenCV) and sliding-window logic for clips; returns center timestamps per window.

Embedding extraction
	•	extract_video_emb.py
Reads each MP4, samples at target FPS/size, makes (T, D) video embeddings with window centers, and saves one NPZ per video to cache/vid_emb/.
	•	extract_audio_emb.py (simple variant)
Extracts fixed-length audio crops → embeddings → NPZ.
	•	extract_audio_multiscale.py (recommended)
Extracts multi-scale audio embeddings (e.g., 0.25, 0.5, 1.0, 2.0, 5.0 s windows) with stride; batches on GPU/CPU safely; saves NPZ per (video,length) under cache/aud_emb/.

Evaluation (baseline + refiner)
	•	eval_temporal_alignment.py
Baseline: for each clip, pick an audio crop (middle or multi-offsets), compute cosine similarity vs. video embeddings over time, convert to a PDF via softmax(·/τ), and predict the peak time (hard argmax or soft-argmax). Writes a summary CSV and optional per-clip plots.
	•	train_refiner.py
Builds a dataset of baseline PDFs, resampled to a fixed grid (D), and trains a tiny MLP refiner (Linear 128→256→128 + Softmax) to sharpen PDFs. Saves checkpoints/refiner/refiner.pt.
	•	eval_with_refiner.py
Loads embeddings + refiner checkpoint, refines the baseline PDF, maps back to the original time grid, predicts refined peak time, and writes CSV + plots. (Also supports soft-argmax, entropy-targeted τ, z-score, Gaussian smoothing, multi-crop fusion.)
	•	refiner.py
The refiner MLP definition (clean, importable class used by training/eval).

Sweeps & ablations
	•	hparam_sweep.py
Runs a grid over audio length L and temperature τ (optionally window/stride), evaluates, and writes per-run folders under reports/grid/ + an aggregate CSV.
	•	hparam_2.py, hparam3.py
Additional sweep scripts for broader grids (e.g., window/stride) and different reporting layouts.
	•	ablate_small.py
Lightweight sanity/ablation runner—e.g., different fusion rules, z-score on/off—on a small subset for quick checks.

Analysis & utilities
	•	analyze_results.py
Reads a summary CSV, canonicalizes column names, computes MAE/Hit@δ/Inside, optional calibration plot, class-wise tables, MAE histogram, and writes them to a timestamped reports/analysis/<run_tag>/.
	•	parse_ave_txt_annotations.py
Helper to convert AVE’s text-style annotations into the CSV format used here (if needed).
	•	cache/
Local cache holder (e.g., cache/vid_emb/). Created automatically; safe to delete/rebuild.
