"""Hardware GPU Profiler and Telemetry Runner.

Compiles and runs lowered FDIR PyTorch modules on the physical GPU,
recording real hardware execution latency using CUDA Events and peak
active memory using GPU memory tracking APIs.
"""

from __future__ import annotations
import time
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn


class GPUProfiler:
    """Telemetry collector running execution workloads directly on physical NVIDIA GPUs."""

    @staticmethod
    def is_gpu_available() -> bool:
        """Check if CUDA device is ready for profiling workloads."""
        return torch.cuda.is_available()

    @staticmethod
    def profile_module(module: nn.Module,
                       inputs: List[torch.Tensor],
                       num_warmup: int = 10,
                       num_steps: int = 50) -> Dict[str, Any]:
        """Profile a lowered PyTorch module on physical GPU using CUDA Events.

        Returns:
          Dict containing hardware metrics: latency_ms, peak_memory_mb, dynamic_flops
        """
        if not torch.cuda.is_available():
            return {
                "latency_ms": 0.0,
                "peak_memory_mb": 0.0,
                "cuda_device": "None",
                "error": "CUDA not available on this system."
            }

        device = torch.device("cuda:0")
        module = module.to(device)
        module.eval()

        # Send input tensors to GPU device
        gpu_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

        # Reset peak memory tracking
        torch.cuda.reset_peak_memory_stats(device)

        # 1. Warm-up steps
        try:
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = module(*gpu_inputs)
        except Exception as e:
            return {"error": f"Warm-up failed: {e}"}

        torch.cuda.synchronize(device)

        # 2. Timing loops using CUDA Events for hardware precision
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        try:
            with torch.no_grad():
                for _ in range(num_steps):
                    _ = module(*gpu_inputs)
        except Exception as e:
            return {"error": f"Execution loop failed: {e}"}

        end_event.record()
        torch.cuda.synchronize(device)

        latency_ms = start_event.elapsed_time(end_event) / num_steps
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)

        return {
            "latency_ms": latency_ms,
            "peak_memory_mb": peak_memory_bytes / (1024 * 1024),
            "cuda_device": torch.cuda.get_device_name(device),
        }
