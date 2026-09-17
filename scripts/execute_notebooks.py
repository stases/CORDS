"""Execute default notebook cells with this Python interpreter, offline.

Pass --write to retain executed outputs in the source notebooks. Temporary
kernelspecs live outside the repository and never modify the user's kernels.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = args.notebooks or sorted((root / "notebooks").glob("*.ipynb"))
    with tempfile.TemporaryDirectory(prefix="cords-kernel-") as directory:
        kernel = Path(directory) / "cords"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "CORDS validation", "language": "python"}))
        manager = KernelSpecManager(kernel_dirs=[directory])
        for path in paths:
            started = time.perf_counter()
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            km = AsyncKernelManager(kernel_name="cords", kernel_spec_manager=manager)
            NotebookClient(notebook, km=km, timeout=120, allow_errors=False,
                           resources={"metadata": {"path": str(path.resolve().parent)}}).execute(cleanup_kc=True)
            if args.write:
                nbformat.write(notebook, path)
            print(f"{path.name}: executed in {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
