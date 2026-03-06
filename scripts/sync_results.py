"""
Extract and summarize experiment results, prepare for git sync.

Usage (run on server after experiments complete):
    python scripts/sync_results.py paper_results paper_results_s2

This script:
1. Reads all_results.json / checkpoint.json from each results directory
2. Merges and prints a readable summary of all metrics  
3. Saves merged_results.json for analysis
4. Shows which files git will sync (thanks to .gitignore)

The .gitignore is configured to:
  SYNC:   *.json, *.tex, *.pdf, *.py, figures/qualitative/, figures/attention/
  IGNORE: */images/ (bulk generated images, 99%+ of size)
"""

import json
import os
import sys


def load_results(results_dir: str) -> dict:
    for name in ["all_results.json", "checkpoint.json"]:
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                return data.get("results", data) if name == "checkpoint.json" else data
    return {}


def merge_results(dirs: list) -> dict:
    merged = {}
    for d in dirs:
        r = load_results(d)
        for exp, data in r.items():
            if exp not in merged or not merged[exp] or (isinstance(merged[exp], dict) and "error" in merged[exp]):
                merged[exp] = data
    return merged


def fmt(val, width=8):
    if isinstance(val, float):
        return f"{val:>{width}.4f}"
    return f"{str(val):>{width}}"


def print_blip_table(data: dict, title: str):
    if not data:
        print(f"\n  {title}: NO DATA\n")
        return
    print(f"\n  {title}")
    print(f"  {'Config':<25} {'Color':>8} {'Shape':>8} {'Texture':>8} {'  Avg':>8} {'   N':>6}")
    print(f"  {'-'*65}")
    for cfg, subsets in data.items():
        if not isinstance(subsets, dict) or "color" not in subsets:
            continue
        c = subsets["color"].get("blip_vqa", 0)
        s = subsets["shape"].get("blip_vqa", 0)
        t = subsets["texture"].get("blip_vqa", 0)
        avg = (c + s + t) / 3
        n = subsets["color"].get("num_images", subsets["color"].get("num_eval_pairs", "?"))
        print(f"  {cfg:<25} {c:>8.4f} {s:>8.4f} {t:>8.4f} {avg:>8.4f} {str(n):>6}")


def print_hyp_table(data: dict, title: str):
    if not data:
        print(f"\n  {title}: NO DATA\n")
        return
    print(f"\n  {title}")
    print(f"  {'Config':<25} {'Color':>8} {'Shape':>8} {'Texture':>8} {'  Avg':>8} {'  Δ':>8}")
    print(f"  {'-'*67}")

    eucl = data.get("euclidean", {})
    hyp = data.get("hyperbolic", {})
    eucl_avgs = {}
    for cfg, sub in eucl.items():
        if isinstance(sub, dict) and "color" in sub:
            c, s, t = sub["color"].get("blip_vqa", 0), sub["shape"].get("blip_vqa", 0), sub["texture"].get("blip_vqa", 0)
            avg = (c + s + t) / 3
            eucl_avgs[cfg] = avg
            print(f"  {cfg:<25} {c:>8.4f} {s:>8.4f} {t:>8.4f} {avg:>8.4f}")

    print(f"  {'--- Hyperbolic ---':<25} {'':>8} {'':>8} {'':>8} {'':>8}")
    for cfg, sub in hyp.items():
        if isinstance(sub, dict) and "color" in sub:
            c, s, t = sub["color"].get("blip_vqa", 0), sub["shape"].get("blip_vqa", 0), sub["texture"].get("blip_vqa", 0)
            avg = (c + s + t) / 3
            base_cfg = cfg.replace("Hyp-", "")
            delta = avg - eucl_avgs.get(base_cfg, avg)
            delta_str = f"{delta:>+8.4f}"
            print(f"  {cfg:<25} {c:>8.4f} {s:>8.4f} {t:>8.4f} {avg:>8.4f} {delta_str}")


def print_time_table(data: dict, title: str):
    if not data:
        print(f"\n  {title}: NO DATA\n")
        return
    print(f"\n  {title}")
    print(f"  {'Config':<25} {'Per-image (s)':>14} {'   Std':>8} {'  Overhead':>10}")
    print(f"  {'-'*60}")
    base_time = None
    for cfg, vals in data.items():
        if isinstance(vals, dict) and "per_image" in vals:
            t = vals["per_image"]
            if base_time is None:
                base_time = t
            overhead = f"{t/base_time:.1f}x" if base_time else "1.0x"
            print(f"  {cfg:<25} {t:>14.2f} {vals.get('per_image_std', 0):>8.3f} {overhead:>10}")


def count_images(results_dir: str) -> int:
    count = 0
    for root, dirs, files in os.walk(results_dir):
        count += sum(1 for f in files if f.endswith((".png", ".jpg")))
    return count


def main():
    dirs = sys.argv[1:] if len(sys.argv) > 1 else ["paper_results"]
    existing = [d for d in dirs if os.path.exists(d)]

    if not existing:
        print(f"ERROR: None of {dirs} found. Available directories:")
        for item in os.listdir("."):
            if os.path.isdir(item) and "result" in item.lower():
                print(f"  {item}/")
        return

    print("=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)

    for d in existing:
        size = os.popen(f"du -sh {d} 2>/dev/null").read().strip().split()[0]
        n_img = count_images(d)
        print(f"  {d}: {size}, {n_img} images")

    merged = merge_results(existing)
    experiments_found = [k for k in merged if merged[k]]
    print(f"  Experiments found: {experiments_found}")

    print_blip_table(merged.get("main", {}), "Table 1: Main Comparison")
    print_blip_table(merged.get("ablation", {}), "Table 2: Ablation Study")
    print_hyp_table(merged.get("hyp_comp", {}), "Table 3: Euclidean vs Hyperbolic")
    print_blip_table(merged.get("curvature", {}), "Table 4: Curvature Sensitivity")
    print_time_table(merged.get("time", {}), "Table 5: Inference Time")

    gpt4o = merged.get("gpt4o", {})
    if gpt4o:
        total = sum(v.get("num_images", 0) for v in gpt4o.values() if isinstance(v, dict))
        print(f"\n  Table 6: GPT-4o ({total} images generated, scoring needs API)")

    qual = merged.get("qualitative", {})
    if qual:
        print(f"\n  Qualitative: {qual.get('status', 'unknown')}")

    attn = merged.get("attention", {})
    if attn and "error" in attn:
        print(f"\n  Attention: ERROR - {attn['error']}")
    elif attn:
        print(f"\n  Attention: {attn.get('status', 'unknown')}")

    # Quick diagnostic
    print(f"\n{'='*70}")
    print("DIAGNOSTIC")
    print("=" * 70)
    main_data = merged.get("main", {})
    if main_data:
        n = 0
        for cfg, sub in main_data.items():
            if isinstance(sub, dict) and "color" in sub:
                n = sub["color"].get("num_images", 0)
                break
        if n and n < 50:
            print(f"  ⚠ Only {n} images per config — this appears to be a quick/test run")
            print(f"    Full T2I-CompBench uses 300 prompts × seeds")
        elif n >= 100:
            print(f"  ✓ {n} images per config — looks like a full run")
        else:
            print(f"  ? {n} images per config")

    # Save merged
    out_path = "merged_results.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\n  Saved: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

    # Git instructions
    print(f"\n{'='*70}")
    print("GIT SYNC COMMANDS (run on server):")
    print("=" * 70)
    dirs_args = " ".join(existing)
    print(f"""
  # .gitignore already excludes bulk images. Just run:
  git add .gitignore scripts/ *.py configs/ experiments/ utils/ data/
  git add merged_results.json""")
    for d in existing:
        print(f"  git add {d}/all_results.json {d}/checkpoint.json 2>/dev/null")
        print(f"  git add {d}/latex/ {d}/figures/ 2>/dev/null")
    print(f"""
  git commit -m "Experiment results: {', '.join(experiments_found)}"
  git push
""")


if __name__ == "__main__":
    main()
