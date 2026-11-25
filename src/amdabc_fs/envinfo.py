"""Hardware and software information, saved at the start of every run.

Written to ``results_dir/environment/<timestamp>.json`` (one file per start, so
a resumed run keeps the record of every machine/session) and logged.
Only the standard library is required; psutil and torch are used if present.
"""
from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from .io import write_json

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "catboost",
            "torch", "matplotlib", "seaborn", "PyYAML"]


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _cpu_name():
    system = platform.system()
    if system == "Windows":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            return platform.processor()
    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def _ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 2**30, 1)
        if system == "Darwin":
            return round(int(_run(["sysctl", "-n", "hw.memsize"])) / 2**30, 1)
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 2**20, 1)
    except Exception:
        pass
    return None


def _physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def _gpu():
    info = {"torch_cuda_available": False}
    try:
        import torch
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["devices"] = [{"name": torch.cuda.get_device_name(i),
                                "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                               for i in range(torch.cuda.device_count())]
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["mps_available"] = True
    except Exception as e:
        info["torch_error"] = repr(e)
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if smi:
        info["nvidia_smi"] = smi.splitlines()
    return info


def collect() -> dict:
    pk = {}
    for p in PACKAGES:
        try:
            pk[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pk[p] = None
    return {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "physical_cores": _physical_cores(),
        "ram_gb": _ram_gb(),
        "gpu": _gpu(),
        "python": sys.version.split()[0],
        "packages": pk,
        "mealpy": "vendored subset of 3.0.3 (src/amdabc_fs/optim/mealpy_core)",
    }


def save(results_dir, log=None) -> dict:
    info = collect()
    stamp = info["time"].replace(":", "-")
    write_json(Path(results_dir) / "environment" / f"{stamp}.json", info)
    if log:
        g = info["gpu"].get("devices") or info["gpu"].get("nvidia_smi") or "none"
        log.info("machine: %s | %s logical / %s physical cores | %s GB RAM | GPU: %s | %s | Python %s",
                 info["cpu"], info["logical_cores"], info["physical_cores"], info["ram_gb"], g,
                 info["os"], info["python"])
    return info
