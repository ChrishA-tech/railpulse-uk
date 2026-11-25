"""RailPulse UK package root."""

import os

# Avoid macOS arm64 OpenMP thread-pool conflicts when LightGBM and PyTorch share a process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
