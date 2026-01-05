"""
System metrics service for CPU, memory, GPU, and temperature monitoring.
"""

import subprocess
import psutil
from typing import Dict, Any, Optional
from datetime import datetime


async def get_system_metrics() -> Dict[str, Any]:
    """
    Get comprehensive system metrics including CPU, memory, GPU, and temperature.
    Optimized for Apple Silicon (M4 Mac).
    """
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu": await _get_cpu_metrics(),
        "memory": _get_memory_metrics(),
        "gpu": await _get_gpu_metrics(),
        "temperature": await _get_temperature_metrics(),
        "processes": _get_process_metrics()
    }
    return metrics


async def _get_cpu_metrics() -> Dict[str, Any]:
    """Get CPU usage and info."""
    return {
        "usage": psutil.cpu_percent(interval=0.1),
        "cores": psutil.cpu_count(),
        "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        "perCore": psutil.cpu_percent(interval=0.1, percpu=True)
    }


def _get_memory_metrics() -> Dict[str, Any]:
    """Get memory usage."""
    mem = psutil.virtual_memory()
    return {
        "used": round(mem.used / (1024 ** 3), 2),  # GB
        "total": round(mem.total / (1024 ** 3), 2),  # GB
        "available": round(mem.available / (1024 ** 3), 2),  # GB
        "percent": mem.percent
    }


async def _get_gpu_metrics() -> Dict[str, Any]:
    """
    Get Apple Silicon GPU metrics.
    Uses ioreg for GPU activity estimation on M-series chips.
    """
    gpu_usage = None
    gpu_name = "Apple M4"
    gpu_cores = "10"

    try:
        # Try to get GPU info from system_profiler
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            if displays:
                gpu_info = displays[0]
                gpu_name = gpu_info.get("sppci_model", "Apple GPU")
                gpu_cores = gpu_info.get("sppci_cores", "10")
    except Exception:
        pass

    # Try to estimate GPU usage via ioreg IOAccelerator
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"],
            capture_output=True,
            text=True,
            timeout=3
        )
        if result.returncode == 0:
            output = result.stdout
            # Look for performance statistics
            if "PerformanceStatistics" in output:
                # GPU is being used if we have activity
                # Parse Device Utilization if available
                import re
                match = re.search(r'"Device Utilization %"\s*=\s*(\d+)', output)
                if match:
                    gpu_usage = int(match.group(1))
                else:
                    # Check if GPU accelerator is active
                    match = re.search(r'"GPU Activity"\s*=\s*(\d+)', output)
                    if match:
                        gpu_usage = int(match.group(1))
    except Exception:
        pass

    # Fallback: Check if neural server is using GPU by its process activity
    if gpu_usage is None:
        try:
            for proc in psutil.process_iter(['name', 'cmdline', 'cpu_percent']):
                cmdline = " ".join(proc.info.get('cmdline') or [])
                if 'neural_server.py' in cmdline or 'mlx' in cmdline.lower():
                    cpu = proc.cpu_percent(interval=0.1)
                    # If neural server is active, GPU is likely being used
                    if cpu > 5:
                        gpu_usage = min(int(cpu * 0.8), 100)  # Estimate GPU from CPU
                    break
        except Exception:
            pass

    return {
        "name": gpu_name,
        "cores": gpu_cores,
        "available": True,
        "usage": gpu_usage if gpu_usage is not None else 0,
        "memory": None
    }


async def _get_temperature_metrics() -> Dict[str, Any]:
    """
    Get temperature readings.
    On macOS, uses ioreg or osx-cpu-temp if available.
    """
    temps = {
        "cpu": None,
        "gpu": None,
        "available": False
    }

    try:
        # Try ioreg for thermal info
        result = subprocess.run(
            ["ioreg", "-r", "-c", "AppleSmartBattery", "-a"],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Note: Getting accurate temps on Apple Silicon is tricky without sudo
        # For now, we mark as unavailable unless using powermetrics

        # Check if we have the thermal monitoring script
        import os
        thermal_script = "/Volumes/AI_SSD/ai-local/luddo-ai-service/bin/thermal_monitor"
        if os.path.exists(thermal_script):
            result = subprocess.run(
                [thermal_script],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse output if available
                temps["available"] = True

    except Exception:
        pass

    return temps


def _get_process_metrics() -> Dict[str, Any]:
    """Get metrics for Luddo AI processes."""
    processes = {
        "aiEngine": None,
        "neuralServer": None,
        "trainingManager": None
    }

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            cmdline = " ".join(proc.info.get('cmdline') or [])

            if 'luddo-ai-engine' in cmdline or 'dist/index.js' in cmdline:
                processes["aiEngine"] = {
                    "pid": proc.info['pid'],
                    "cpu": proc.cpu_percent(),
                    "memory": round(proc.memory_info().rss / (1024 ** 2), 1)  # MB
                }
            elif 'neural_server.py' in cmdline:
                processes["neuralServer"] = {
                    "pid": proc.info['pid'],
                    "cpu": proc.cpu_percent(),
                    "memory": round(proc.memory_info().rss / (1024 ** 2), 1)
                }
            elif 'training_manager' in cmdline:
                processes["trainingManager"] = {
                    "pid": proc.info['pid'],
                    "cpu": proc.cpu_percent(),
                    "memory": round(proc.memory_info().rss / (1024 ** 2), 1)
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return processes
