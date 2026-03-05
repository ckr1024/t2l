"""
Download T2I-CompBench validation prompts from the official GitHub repository.

Source: https://github.com/Karine-Huang/T2I-CompBench
Paper: "T2I-CompBench: A Comprehensive Benchmark for Open-world
        Compositional Text-to-image Generation" (NeurIPS 2023)

Usage:
    python data/download_t2i_compbench.py
"""

import os
import subprocess
import urllib.request
import ssl

BASE_URL = "https://raw.githubusercontent.com/Karine-Huang/T2I-CompBench/main/examples/dataset"
SUBSETS = ["color_val.txt", "shape_val.txt", "texture_val.txt"]

TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2i_compbench")


def _download_file(url: str, target_path: str):
    """Download with urllib first, fall back to curl on SSL errors."""
    try:
        ctx = ssl.create_default_context()
        urllib.request.urlretrieve(url, target_path)
        return
    except Exception:
        pass

    try:
        ctx = ssl._create_unverified_context()
        urllib.request.urlretrieve(url, target_path, context=ctx)
        return
    except Exception:
        pass

    result = subprocess.run(
        ["curl", "-sL", url, "-o", target_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download {url}: {result.stderr}")


def download_dataset(target_dir: str = TARGET_DIR):
    os.makedirs(target_dir, exist_ok=True)

    for filename in SUBSETS:
        target_path = os.path.join(target_dir, filename)
        if os.path.exists(target_path):
            with open(target_path, "r") as f:
                n = sum(1 for line in f if line.strip())
            if n >= 200:
                print(f"  {filename} already exists ({n} prompts)")
                continue

        url = f"{BASE_URL}/{filename}"
        print(f"  Downloading {filename} from {url} ...")
        _download_file(url, target_path)
        with open(target_path, "r") as f:
            n = sum(1 for line in f if line.strip())
        print(f"  Saved {filename} ({n} prompts)")

    print(f"\nAll T2I-CompBench validation files saved to: {target_dir}")
    return target_dir


if __name__ == "__main__":
    download_dataset()
