"""Device selection, timing/memory measurement and run metadata shared by all scripts."""

from __future__ import annotations

import importlib.metadata
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

TRACKED_PACKAGES = [
    "colpali-engine",
    "torch",
    "transformers",
    "peft",
    "datasets",
    "huggingface-hub",
    "pillow",
    "numpy",
]


def get_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    """bfloat16 is what the authors use; fall back to float16 on GPUs without native bf16 (e.g. T4, P100)."""
    if requested != "auto":
        return getattr(torch, requested)
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class Timer:
    """Wall-clock timer that synchronizes the accelerator so async kernels are counted."""

    def __init__(self, device: str):
        self.device = device
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        synchronize(self.device)
        self.seconds = time.perf_counter() - self._start


class MemoryTracker:
    """Peak accelerator memory in MB.

    CUDA keeps a true peak counter. MPS only exposes current usage, so we sample it
    after each step (`sample()`) and keep the maximum; this can miss short spikes.
    """

    def __init__(self, device: str):
        self.device = device
        self._mps_peak = 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def sample(self) -> None:
        if self.device == "mps":
            self._mps_peak = max(self._mps_peak, torch.mps.driver_allocated_memory())

    def report(self) -> dict:
        mb = 1024**2
        if self.device == "cuda":
            return {
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / mb,
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / mb,
                "method": "torch.cuda.max_memory_*",
            }
        if self.device == "mps":
            self.sample()
            return {"peak_driver_allocated_mb": self._mps_peak / mb, "method": "sampled torch.mps.driver_allocated_memory"}
        # ru_maxrss is bytes on macOS, kilobytes on Linux
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss / mb if sys.platform == "darwin" else rss / 1024
        return {"peak_process_rss_mb": rss_mb, "method": "resource.getrusage ru_maxrss"}


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _package_versions() -> dict:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _hardware(device: str) -> dict:
    info = {"device": device, "platform": platform.platform(), "processor": platform.processor()}
    if device == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
        info["cuda_version"] = torch.version.cuda
    return info


def hub_revision(repo_id: str, revision: str | None, repo_type: str) -> str | None:
    """Resolve a Hub branch/tag to the exact commit sha, so the run is pinned even if 'main' moves."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo_id, revision=revision, repo_type=repo_type)
        return info.sha
    except Exception:  # offline or unavailable: record that we could not resolve it
        return None


def collect_metadata(config: dict, config_path: str, device: str, seed: int) -> dict:
    status = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": config_path,
        "config": config,
        "seed": seed,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "hardware": _hardware(device),
        "command": " ".join(sys.argv),
    }
