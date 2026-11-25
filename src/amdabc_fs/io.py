"""Atomic result files.

Every unit of work writes exactly one result file. The file is written to a
temporary name in the same directory, flushed to disk, then renamed with
``os.replace`` (atomic on POSIX and Windows). A power cut therefore leaves
either the old state (no file) or a complete file, never a half-written one;
on restart the run skips every task whose file exists and redoes the rest.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np


def _retry(fn, *args, tries=40):
    """On Windows a sync client (Dropbox, OneDrive) or an antivirus scanner may
    hold a just-written file for a moment; renaming/removing it then fails with
    PermissionError. Retry for up to ~20 s before giving up."""
    for i in range(tries):
        try:
            return fn(*args)
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(min(0.05 * (i + 1), 1.0))


def _atomic(path: Path, write) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        _retry(os.replace, tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                _retry(os.remove, tmp, tries=5)
            except OSError:
                pass                      # removed by clean_tmp() at the next start
        raise


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_json(path, obj) -> None:
    data = json.dumps(obj, cls=_Enc, indent=1).encode("utf-8")
    _atomic(path, lambda fh: fh.write(data))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_npz(path, **arrays) -> None:
    _atomic(path, lambda fh: np.savez_compressed(fh, **arrays))


def read_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_csv(path, df, **kw) -> None:
    data = df.to_csv(**kw).encode("utf-8")
    _atomic(path, lambda fh: fh.write(data))


def clean_tmp(root) -> int:
    """Remove leftovers of interrupted writes (*.tmp)."""
    n = 0
    for p in list(Path(root).rglob("*.tmp")) + list(Path(root).rglob("*.started")):
        p.unlink(missing_ok=True)
        n += 1
    return n
