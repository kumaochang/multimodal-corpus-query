"""Shared local execution guards and provenance; no research inputs loaded at import."""

import ast
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import psutil
from threadpoolctl import threadpool_info


def record(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(clean(obj), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def check_profile():
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}
    expected = {"numpy": "1.26.4", "scipy": "1.15.2", "scikit-learn": "1.6.1"}
    if platform.python_version() != "3.12.3" or versions != expected:
        raise RuntimeError("unadopted numerical environment")
    pools = threadpool_info()
    if any(p["num_threads"] != 1 for p in pools):
        raise RuntimeError("numeric thread pool not limited to one")
    mkl = [p for p in pools if p.get("internal_api") == "mkl"]
    if len(mkl) != 1 or not mkl[0]["version"].startswith("2023.1"):
        raise RuntimeError("unadopted BLAS profile")
    return {"python": platform.python_version(), **versions, "threadpools": pools,
            "logical_CPUs": psutil.cpu_count(), "RAM_bytes": psutil.virtual_memory().total,
            "platform": platform.platform()}


def check_sources(root, inventory):
    inputs = json.loads(Path(inventory).read_bytes())["sources"]
    for relative, pin in inputs.items():
        if record(Path(root) / relative) != pin:
            raise RuntimeError(f"frozen source mismatch: {relative}")
    return inputs


def extract_functions(path, names, constants=None):
    """Compile selected frozen pure functions without importing their main/file code."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise RuntimeError("frozen function missing")
    ns = {"np": np, **(constants or {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return {name: ns[name] for name in names}


class Budget:
    def __init__(self, seconds=600, bytes_limit=4 * 1024**3):
        self.start = time.perf_counter()
        self.seconds = seconds
        self.bytes_limit = bytes_limit
        self.peak = 0

    def tick(self):
        mem = psutil.Process().memory_info()
        self.peak = max(self.peak, getattr(mem, "peak_wset", mem.rss))
        wall = time.perf_counter() - self.start
        if self.peak > self.bytes_limit or wall > self.seconds:
            raise RuntimeError("adopted task budget exceeded")
        return {"wall_seconds": wall, "peak_working_set_bytes": self.peak,
                "budget_seconds": self.seconds, "budget_bytes": self.bytes_limit,
                "OS_hard_memory_cap": False, "measurement": "Windows peak_wset or sampled RSS"}
