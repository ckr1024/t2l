"""
Cross-Attention Map Visualization (Figure 7 in paper)

Visualizes how cross-attention maps change across different ablation
configurations, demonstrating the effect of ToMe and the auxiliary losses.

Generates attention heatmaps for:
  - Individual token attention maps (e.g., [cat], [sunglasses], [dog], [hat])
  - Composite token attention maps (e.g., [cat*], [dog*])
  - Comparison across configurations

Usage:
    python -m experiments.visualize_attention [--prompt "a cat wearing sunglasses and a dog wearing hat"]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ptp_utils import AttentionStore, aggregate_attention
from configs.experiment_config import ABLATION_CONFIGS


def extract_attention_maps(
    attention_store: AttentionStore,
    res: int = 32,
    from_where: tuple = ("up", "down", "mid"),
    select: int = 0,
) -> torch.Tensor:
    """
    Extract aggregated cross-attention maps.

    Returns:
        Tensor of shape (H, W, num_tokens)
    """
    return aggregate_attention(
        attention_store=attention_store,
        res=res,
        from_where=from_where,
        is_cross=True,
        select=select,
    )


def visualize_token_attention(
    attention_maps: torch.Tensor,
    tokenizer,
    prompt: str,
    token_indices: List[int],
    output_dir: str,
    prefix: str = "",
    image_size: int = 256,
):
    """
    Visualize cross-attention maps for specified tokens.

    Args:
        attention_maps: (H, W, num_tokens) tensor
        tokenizer: CLIP tokenizer
        prompt: Text prompt
        token_indices: Indices of tokens to visualize
        output_dir: Directory to save visualizations
        prefix: Filename prefix for saved images
        image_size: Output image resolution
    """
    os.makedirs(output_dir, exist_ok=True)

    tokens = tokenizer.encode(prompt)
    h, w, seq_len = attention_maps.shape

    images_with_labels = []

    for idx in token_indices:
        if idx >= seq_len:
            continue

        attn_map = attention_maps[:, :, idx].detach().cpu().float()

        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        attn_np = attn_map.numpy()
        attn_resized = np.array(
            Image.fromarray((attn_np * 255).astype(np.uint8)).resize(
                (image_size, image_size), Image.BILINEAR
            )
        )

        import cv2

        heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        if idx < len(tokens):
            token_text = tokenizer.decode([tokens[idx]])
        else:
            token_text = f"token_{idx}"

        pil_img = Image.fromarray(heatmap_rgb)
        img_path = os.path.join(output_dir, f"{prefix}token_{idx}_{token_text.strip()}.png")
        pil_img.save(img_path)
        images_with_labels.append((pil_img, token_text.strip()))

    return images_with_labels


def create_attention_comparison_grid(
    images_dict: Dict[str, List[tuple]],
    output_path: str,
    cell_size: int = 256,
):
    """
    Create a grid comparing attention maps across configurations.

    Args:
        images_dict: {config_name: [(image, label), ...]}
        output_path: Path to save the grid image
        cell_size: Size of each cell in the grid
    """
    if not images_dict:
        return

    configs = list(images_dict.keys())
    max_tokens = max(len(v) for v in images_dict.values())

    label_height = 40
    config_label_width = 120
    grid_width = config_label_width + max_tokens * cell_size
    grid_height = label_height + len(configs) * (cell_size + label_height)

    grid = Image.new("RGB", (grid_width, grid_height), "white")

    import cv2

    for col_idx in range(max_tokens):
        first_config = configs[0]
        if col_idx < len(images_dict[first_config]):
            label = images_dict[first_config][col_idx][1]
        else:
            label = ""

        label_img = Image.new("RGB", (cell_size, label_height), "white")
        label_np = np.array(label_img)
        cv2.putText(
            label_np, f"[{label}]",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
        )
        grid.paste(Image.fromarray(label_np), (config_label_width + col_idx * cell_size, 0))

    for row_idx, config_name in enumerate(configs):
        y_offset = label_height + row_idx * (cell_size + label_height)

        config_label = Image.new("RGB", (config_label_width, cell_size), "white")
        config_np = np.array(config_label)
        cv2.putText(
            config_np, config_name,
            (10, cell_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )
        grid.paste(Image.fromarray(config_np), (0, y_offset))

        for col_idx, (img, label) in enumerate(images_dict[config_name]):
            img_resized = img.resize((cell_size, cell_size))
            grid.paste(img_resized, (config_label_width + col_idx * cell_size, y_offset))

    grid.save(output_path)
    print(f"Attention comparison grid saved to {output_path}")


def run_attention_visualization(
    prompt: str,
    config_names: List[str],
    model,
    prompt_parser,
    seed: int = 42,
    output_dir: str = "./experiment_results/attention_viz",
):
    """
    Generate attention visualizations for multiple configurations.

    This reproduces Figure 7 from the paper, showing how cross-attention
    maps differ across ablation configurations.
    """
    from run_demo import run_on_prompt, filter_text

    os.makedirs(output_dir, exist_ok=True)

    try:
        import en_core_web_trf
        nlp = en_core_web_trf.load()
    except ImportError:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(prompt)
    prompt_parser.set_doc(doc)
    token_indices = prompt_parser._get_indices(prompt)
    prompt_anchor = prompt_parser._split_prompt(doc)
    token_indices, prompt_anchor = filter_text(token_indices, prompt_anchor)

    all_token_ids = model.tokenizer.encode(prompt)

    visualize_indices = []
    for idx_group in token_indices:
        for idx_list in idx_group:
            if isinstance(idx_list, list):
                visualize_indices.extend(idx_list)
            else:
                visualize_indices.append(idx_list)
    visualize_indices = sorted(set(visualize_indices))

    fallback_needed = not token_indices or not prompt_anchor

    all_attention_images = {}

    for config_name in config_names:
        print(f"\n--- Config {config_name} ---")

        config_cls = ABLATION_CONFIGS[config_name]
        config = config_cls()
        config.prompt = prompt

        nouns = [chunk.root.text for chunk in doc.noun_chunks]
        merged_parts = []
        for chunk in doc.noun_chunks:
            root = chunk.root.text
            if root not in merged_parts:
                merged_parts.append(root)
        config.prompt_merged = " and ".join([f"a {n}" for n in merged_parts[:2]]) or prompt
        config.prompt_length = len(model.tokenizer(prompt)["input_ids"]) - 2

        orig_std = config.run_standard_sd
        if fallback_needed:
            config.run_standard_sd = True

        g = torch.Generator("cuda").manual_seed(seed)
        controller = AttentionStore(save_global_store=True)

        use_indices = token_indices if token_indices else [[0], [0]]
        use_anchor = prompt_anchor if prompt_anchor else [prompt]

        image = run_on_prompt(
            prompt=prompt,
            model=model,
            controller=controller,
            token_indices=use_indices,
            prompt_anchor=use_anchor,
            seed=g,
            config=config,
        )

        if fallback_needed:
            config.run_standard_sd = orig_std

        config_dir = os.path.join(output_dir, f"config_{config_name}")
        os.makedirs(config_dir, exist_ok=True)
        image.save(os.path.join(config_dir, "generated_image.png"))

        try:
            attention_maps = extract_attention_maps(controller, res=32)

            token_images = visualize_token_attention(
                attention_maps=attention_maps,
                tokenizer=model.tokenizer,
                prompt=prompt,
                token_indices=visualize_indices,
                output_dir=config_dir,
                prefix=f"config{config_name}_",
            )

            all_attention_images[f"Config {config_name}"] = token_images
        except Exception as e:
            print(f"  Skipping attention extraction for {config_name}: {e}")
            continue

    grid_path = os.path.join(output_dir, "attention_comparison.png")
    create_attention_comparison_grid(all_attention_images, grid_path)

    print(f"\nVisualization complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Attention Visualization")
    parser.add_argument("--prompt", type=str,
                       default="a cat wearing sunglasses and a dog wearing hat")
    parser.add_argument("--configs", type=str, nargs="+",
                       default=["A", "C", "Ours"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                       default="./experiment_results/attention_viz")
    parser.add_argument("--model_path", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from configs.experiment_config import BaseExperimentConfig
    base_config = BaseExperimentConfig()
    base_config.model_path = args.model_path

    from run_demo import load_model
    model, prompt_parser = load_model(base_config, device)

    run_attention_visualization(
        prompt=args.prompt,
        config_names=args.configs,
        model=model,
        prompt_parser=prompt_parser,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
