#!/usr/bin/env python
"""Write the software versions used in this project (for the Methods 'Software' paragraph).
Run inside the analysis env:   conda activate scvi-env ; python software_versions.py
Output: software_versions.txt (next to this script)."""
import sys, platform, importlib
from importlib import metadata

PKGS = ["numpy", "scipy", "pandas", "scikit-learn", "statsmodels", "matplotlib", "anndata", "scanpy",
        "scvi-tools", "torch", "scrublet", "infercnvpy", "squidpy", "liana", "lifelines", "nimfa",
        "requests", "openpyxl", "h5py", "leidenalg", "igraph", "umap-learn", "pynndescent", "decoupler"]

lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}"]
for p in PKGS:
    try:
        lines.append(f"{p} {metadata.version(p)}")
    except metadata.PackageNotFoundError:
        try:                                   # some packages register under a different dist name
            m = importlib.import_module(p.replace("-", "_"))
            lines.append(f"{p} {getattr(m, '__version__', 'installed (version unknown)')}")
        except Exception:
            lines.append(f"{p} not installed")
try:
    import torch
    lines.append(f"CUDA available: {torch.cuda.is_available()}"
                 + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
except Exception:
    pass
open("software_versions.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
