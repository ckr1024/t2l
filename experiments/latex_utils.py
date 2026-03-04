"""
LaTeX table and figure generation for paper-ready output.

Produces formatted LaTeX tables from experiment results that can be
directly inserted into the paper.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def format_score(mean: float, std: float = None, bold: bool = False) -> str:
    """Format a score for LaTeX, optionally bold for best result."""
    if std is not None and std > 0:
        s = f"{mean:.4f}$\\pm${std:.4f}"
    else:
        s = f"{mean:.4f}"
    return f"\\textbf{{{s}}}" if bold else s


def find_best_in_column(results: Dict, key: str) -> str:
    """Find the config name with the best score for a given key."""
    best_name, best_score = None, -float("inf")
    for name, data in results.items():
        val = data.get(key)
        if val is not None and val > best_score:
            best_score = val
            best_name = name
    return best_name


def generate_main_comparison_table(results: Dict, output_path: str) -> str:
    """
    Generate Table 1: Main comparison (SDXL vs ToMe-Euclidean vs ToMe-Hyperbolic).

    Expected results structure:
    {
        "SDXL": {"color": {"blip_vqa": ..., "blip_std": ..., "image_reward": ...}, ...},
        "ToMe (Eucl.)": {...},
        "ToMe (Hyp.)": {...}
    }
    """
    subsets = ["color", "shape", "texture"]
    methods = list(results.keys())

    flat = {}
    for m in methods:
        flat[m] = {}
        for s in subsets:
            d = results[m].get(s, {})
            flat[m][f"blip_{s}"] = d.get("blip_vqa", 0)
            flat[m][f"blip_{s}_std"] = d.get("blip_std", 0)
        scores = [flat[m].get(f"blip_{s}", 0) for s in subsets]
        flat[m]["blip_avg"] = np.mean(scores)
        ir_scores = [results[m].get(s, {}).get("image_reward") for s in subsets]
        ir_scores = [x for x in ir_scores if x is not None]
        flat[m]["image_reward"] = np.mean(ir_scores) if ir_scores else None

    best_cols = {}
    for col_key in [f"blip_{s}" for s in subsets] + ["blip_avg", "image_reward"]:
        best_cols[col_key] = find_best_in_column(flat, col_key)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Attribute binding performance comparison on T2I-CompBench. "
        r"We report BLIP-VQA scores for color, shape, and texture binding, "
        r"along with the ImageReward score. Best results are in \textbf{bold}.}",
        r"\label{tab:main_comparison}",
        r"\begin{tabular}{l|ccc|c|c}",
        r"\toprule",
        r"Method & Color$\uparrow$ & Shape$\uparrow$ & Texture$\uparrow$ & Avg$\uparrow$ & ImgRwd$\uparrow$ \\",
        r"\midrule",
    ]

    for m in methods:
        parts = [m.replace("_", r"\_")]
        for s in subsets:
            k = f"blip_{s}"
            is_best = best_cols.get(k) == m
            parts.append(format_score(flat[m][k], flat[m].get(f"{k}_std"), bold=is_best))
        is_best_avg = best_cols.get("blip_avg") == m
        parts.append(format_score(flat[m]["blip_avg"], bold=is_best_avg))
        ir = flat[m].get("image_reward")
        if ir is not None:
            is_best_ir = best_cols.get("image_reward") == m
            parts.append(format_score(ir, bold=is_best_ir))
        else:
            parts.append("--")
        lines.append(" & ".join(parts) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    latex = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    return latex


def generate_ablation_table(results: Dict, output_path: str) -> str:
    """
    Generate Table 2: Ablation study.

    Expected results structure: same as main comparison but with configs A-F + Ours.
    """
    subsets = ["color", "shape", "texture"]
    config_order = ["A", "B", "C", "D", "E", "F", "Ours"]
    component_flags = {
        "A":    (False, False, False, False),
        "B":    (True,  True,  False, False),
        "C":    (True,  True,  True,  False),
        "D":    (False, False, True,  True),
        "E":    (False, False, True,  False),
        "F":    (True,  True,  False, True),
        "Ours": (True,  True,  True,  True),
    }

    flat = {}
    for m in config_order:
        if m not in results:
            continue
        flat[m] = {}
        for s in subsets:
            d = results[m].get(s, {})
            flat[m][f"blip_{s}"] = d.get("blip_vqa", 0)
        scores = [flat[m].get(f"blip_{s}", 0) for s in subsets]
        flat[m]["blip_avg"] = np.mean(scores)

    best_cols = {}
    for col_key in [f"blip_{s}" for s in subsets] + ["blip_avg"]:
        best_cols[col_key] = find_best_in_column(flat, col_key)

    ck = lambda b: r"\checkmark" if b else ""

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study of ToMe components. ToMe: Token Merging, "
        r"ETS: End Token Substitution, $\mathcal{L}_{ent}$: Entropy loss, "
        r"$\mathcal{L}_{sem}$: Semantic binding loss.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l|cccc|ccc|c}",
        r"\toprule",
        r"Config & ToMe & ETS & $\mathcal{L}_{ent}$ & $\mathcal{L}_{sem}$ "
        r"& Color$\uparrow$ & Shape$\uparrow$ & Texture$\uparrow$ & Avg$\uparrow$ \\",
        r"\midrule",
    ]

    for m in config_order:
        if m not in flat:
            continue
        tome, ets, lent, lsem = component_flags[m]
        parts = [m, ck(tome), ck(ets), ck(lent), ck(lsem)]
        for s in subsets:
            k = f"blip_{s}"
            is_best = best_cols.get(k) == m
            parts.append(format_score(flat[m][k], bold=is_best))
        is_best_avg = best_cols.get("blip_avg") == m
        parts.append(format_score(flat[m]["blip_avg"], bold=is_best_avg))
        lines.append(" & ".join(parts) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    return latex


def generate_hyperbolic_comparison_table(eucl_results: Dict, hyp_results: Dict, output_path: str) -> str:
    """
    Generate Table 3: Euclidean vs Hyperbolic for each ablation config.

    Shows paired comparison to isolate the effect of hyperbolic operations.
    """
    subsets = ["color", "shape", "texture"]
    config_pairs = [("B", "Hyp-B"), ("C", "Hyp-C"), ("F", "Hyp-F"), ("Ours", "Hyp-Ours")]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Effect of hyperbolic operations on each configuration. "
        r"$\Delta$ shows the improvement from replacing Euclidean operations "
        r"with their Poincar\'{e} ball counterparts.}",
        r"\label{tab:hyperbolic_comparison}",
        r"\begin{tabular}{l|l|ccc|c}",
        r"\toprule",
        r"Config & Space & Color & Shape & Texture & Avg \\",
        r"\midrule",
    ]

    all_results = {**eucl_results, **hyp_results}

    for eucl_name, hyp_name in config_pairs:
        if eucl_name not in all_results or hyp_name not in all_results:
            continue

        for label, name in [(f"{eucl_name}", eucl_name), (f"{eucl_name}+Hyp", hyp_name)]:
            parts = [label, "Eucl." if name == eucl_name else "Hyp."]
            scores = []
            for s in subsets:
                d = all_results[name].get(s, {})
                v = d.get("blip_vqa", 0)
                scores.append(v)
                parts.append(f"{v:.4f}")
            parts.append(f"{np.mean(scores):.4f}")
            lines.append(" & ".join(parts) + r" \\")

        eucl_avgs = [all_results[eucl_name].get(s, {}).get("blip_vqa", 0) for s in subsets]
        hyp_avgs = [all_results[hyp_name].get(s, {}).get("blip_vqa", 0) for s in subsets]
        deltas = [h - e for h, e in zip(hyp_avgs, eucl_avgs)]
        delta_parts = [r"$\Delta$", ""]
        for d in deltas:
            sign = "+" if d >= 0 else ""
            delta_parts.append(f"{sign}{d:.4f}")
        avg_delta = np.mean(deltas)
        sign = "+" if avg_delta >= 0 else ""
        delta_parts.append(f"{sign}{avg_delta:.4f}")
        lines.append(" & ".join(delta_parts) + r" \\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines += [r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    return latex


def generate_curvature_table(results: Dict, output_path: str) -> str:
    """
    Generate Table 4: Curvature sensitivity analysis.
    """
    subsets = ["color", "shape", "texture"]
    curvatures = sorted(results.keys(), key=lambda x: float(x.split("=")[1]))

    flat = {}
    for c_name in curvatures:
        flat[c_name] = {}
        for s in subsets:
            d = results[c_name].get(s, {})
            flat[c_name][f"blip_{s}"] = d.get("blip_vqa", 0)
        scores = [flat[c_name].get(f"blip_{s}", 0) for s in subsets]
        flat[c_name]["blip_avg"] = np.mean(scores)

    best_cols = {}
    for col_key in [f"blip_{s}" for s in subsets] + ["blip_avg"]:
        best_cols[col_key] = find_best_in_column(flat, col_key)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity analysis of Poincar\'{e} ball curvature parameter $c$. "
        r"Best results are in \textbf{bold}.}",
        r"\label{tab:curvature}",
        r"\begin{tabular}{c|ccc|c}",
        r"\toprule",
        r"$c$ & Color$\uparrow$ & Shape$\uparrow$ & Texture$\uparrow$ & Avg$\uparrow$ \\",
        r"\midrule",
    ]

    for c_name in curvatures:
        parts = [c_name.replace("c=", "")]
        for s in subsets:
            k = f"blip_{s}"
            is_best = best_cols.get(k) == c_name
            parts.append(format_score(flat[c_name][k], bold=is_best))
        is_best_avg = best_cols.get("blip_avg") == c_name
        parts.append(format_score(flat[c_name]["blip_avg"], bold=is_best_avg))
        lines.append(" & ".join(parts) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    return latex


def generate_time_table(results: Dict, output_path: str) -> str:
    """
    Generate Table 5: Time complexity comparison.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inference time comparison. Overhead is relative to SDXL baseline.}",
        r"\label{tab:time_complexity}",
        r"\begin{tabular}{l|cc}",
        r"\toprule",
        r"Method & Time/image (s) & Overhead \\",
        r"\midrule",
    ]

    baseline_time = None
    for name, data in results.items():
        t = data.get("per_image", data.get("avg_time_per_image", 0))
        if "baseline" in name.lower() or "sdxl" in name.lower():
            baseline_time = t
            break

    for name, data in results.items():
        t = data.get("per_image", data.get("avg_time_per_image", 0))
        if baseline_time and baseline_time > 0:
            overhead = (t / baseline_time - 1) * 100
            overhead_str = f"+{overhead:.1f}\\%" if overhead > 0 else f"{overhead:.1f}\\%"
        else:
            overhead_str = "--"
        lines.append(f"{name} & {t:.1f} & {overhead_str}" + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    return latex


def generate_gpt4o_table(results: Dict, output_path: str) -> str:
    """
    Generate Table 6: GPT-4o object binding scores.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{GPT-4o object binding benchmark scores (0-100 scale, "
        r"normalized to 0-1). Higher is better.}",
        r"\label{tab:gpt4o}",
        r"\begin{tabular}{l|cc}",
        r"\toprule",
        r"Method & GPT-4o Score$\uparrow$ & Std \\",
        r"\midrule",
    ]

    best_name = max(results.keys(), key=lambda k: results[k].get("mean_score", 0))
    for name, data in results.items():
        mean = data.get("mean_score", 0)
        std = data.get("std_score", 0)
        is_best = name == best_name
        score_str = format_score(mean, std, bold=is_best)
        lines.append(f"{name} & {score_str} & {std:.4f}" + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    return latex


def generate_curvature_plot_script(results: Dict, output_path: str) -> str:
    """
    Generate a matplotlib script for the curvature sensitivity plot.
    """
    subsets = ["color", "shape", "texture"]
    curvatures = []
    scores_by_subset = {s: [] for s in subsets}
    avg_scores = []

    for c_name in sorted(results.keys(), key=lambda x: float(x.split("=")[1])):
        c_val = float(c_name.split("=")[1])
        curvatures.append(c_val)
        subset_scores = []
        for s in subsets:
            v = results[c_name].get(s, {}).get("blip_vqa", 0)
            scores_by_subset[s].append(v)
            subset_scores.append(v)
        avg_scores.append(np.mean(subset_scores))

    script = f"""import matplotlib.pyplot as plt
import numpy as np

curvatures = {curvatures}
color_scores = {scores_by_subset['color']}
shape_scores = {scores_by_subset['shape']}
texture_scores = {scores_by_subset['texture']}
avg_scores = {avg_scores}

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(curvatures, color_scores, 'o-', label='Color', color='#e74c3c', linewidth=2)
ax.plot(curvatures, shape_scores, 's-', label='Shape', color='#3498db', linewidth=2)
ax.plot(curvatures, texture_scores, '^-', label='Texture', color='#2ecc71', linewidth=2)
ax.plot(curvatures, avg_scores, 'D--', label='Average', color='#8e44ad', linewidth=2.5)

ax.set_xlabel('Curvature $c$', fontsize=14)
ax.set_ylabel('BLIP-VQA Score', fontsize=14)
ax.set_title('Curvature Sensitivity Analysis', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('{output_path}', dpi=300, bbox_inches='tight')
plt.close()
print(f"Curvature plot saved to {output_path}")
"""

    script_path = output_path.replace(".pdf", ".py").replace(".png", ".py")
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)
    return script
