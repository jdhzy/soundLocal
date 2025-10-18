# =========================================================
# Makefile for environment setup on SCC (Python 3.12.4 + optional CUDA)
# =========================================================
.RECIPEPREFIX := >
SHELL := /bin/bash -l
# ---- Configuration ----
PY_MODULE := python3/3.12.4
CUDA_MODULE := cuda/12.2
PY_BIN    := python3.12
VENV      := .venv
REQ       := requirements.txt

.PHONY: install clean gpu reinstall

# ---- Default install (CPU setup) ----
install:
> echo "→ Loading Python module: $(PY_MODULE)"
> module load $(PY_MODULE) || { echo "✗ Failed to load $(PY_MODULE)"; exit 1; }
> if ! command -v $(PY_BIN) >/dev/null 2>&1; then \
>   echo "✗ $(PY_BIN) not found after module load. Check available modules."; \
>   exit 1; \
> fi
> echo "→ Using Python binary: $(PY_BIN)"
> test -x "$(VENV)/bin/python" || $(PY_BIN) -m venv "$(VENV)"
> . "$(VENV)/bin/activate"; pip install --upgrade pip
> . "$(VENV)/bin/activate"; pip install -r "$(REQ)"
> echo "✓ CPU environment setup complete."

# ---- GPU setup (CUDA + PyTorch CU121 wheels) ----
gpu:
> echo "→ Loading modules for GPU environment..."
> module load $(PY_MODULE)
> module load $(CUDA_MODULE)
> echo "✓ CUDA module loaded: $(CUDA_MODULE)"
> test -x "$(VENV)/bin/python" || $(PY_BIN) -m venv "$(VENV)"
> . "$(VENV)/bin/activate"; pip install --upgrade pip
> echo "→ Installing PyTorch with CUDA 12.1 support..."
> . "$(VENV)/bin/activate"; pip uninstall -y torch torchvision torchaudio || true
> . "$(VENV)/bin/activate"; pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
>   torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
> . "$(VENV)/bin/activate"; pip install -r "$(REQ)"
> echo "✓ GPU environment ready with CUDA + PyTorch."

# ---- Launch a 4-hour interactive GPU session on SCC ----
gpu-login:
> echo "→ Requesting interactive GPU node (4 hours, 24GB RAM)..."
> srun --pty --gres=gpu:1 --mem=24G -t 4:00:00 --partition=gpu bash -l
> echo "✓ Entered GPU interactive shell."
# ---- Clean environment ----
clean:
> echo "→ Removing virtual environment at $(VENV)"
> rm -rf "$(VENV)"
> echo "✓ Clean complete."

# ---- Reinstall everything ----
reinstall: clean install
