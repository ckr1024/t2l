import pprint
from typing import List

import torch
from PIL import Image, ImageDraw, ImageFont

from configs.demo_config import RunConfig1, RunConfig2
from pipe_tome import tomePipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
from utils.hyperbolic_utils import TokenMergerWithAttnHyperspace
from prompt_utils import PromptParser
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def read_prompt(path):
    with open(path, "r") as f:
        prompt_ls = f.readlines()
    all_prompt = []
    for idx, prompt in enumerate(prompt_ls):
        prompt = prompt.replace("\n", "")
        all_prompt.append([idx, prompt])
    return all_prompt


def load_model(config, device):
    stable_diffusion_version = "stabilityai/stable-diffusion-xl-base-1.0"
    if hasattr(config, "model_path") and config.model_path is not None:
        stable_diffusion_version = config.model_path
    stable = tomePipeline.from_pretrained(
        stable_diffusion_version,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
    ).to(device)
    stable.unet.requires_grad_(False)
    stable.vae.requires_grad_(False)
    prompt_parser = PromptParser(stable_diffusion_version)
    return stable, prompt_parser


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {
        idx: stable.tokenizer.decode(t)
        for idx, t in enumerate(stable.tokenizer(prompt)["input_ids"])
        if 0 < idx < len(stable.tokenizer(prompt)["input_ids"]) - 1
    }
    pprint.pprint(token_idx_to_word)
    token_indices = input(
        "Please enter the a comma-separated list indices of the tokens you wish to "
        "alter (e.g., 2,5): "
    )
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(
    prompt: List[str],
    model: tomePipeline,
    controller: AttentionStore,
    token_indices: List[int],
    prompt_anchor: List[str],
    seed: torch.Generator,
    config,
    run_standard_sd: bool = None,
    use_hyperbolic: bool = False,
    hyper_merger=None,
) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)

    _run_standard = (
        run_standard_sd if run_standard_sd is not None else config.run_standard_sd
    )

    outputs = model(
        prompt=prompt,
        guidance_scale=config.guidance_scale,
        generator=seed,
        num_inference_steps=config.n_inference_steps,
        attention_store=controller,
        indices_to_alter=token_indices,
        prompt_anchor=prompt_anchor,
        attention_res=config.attention_res,
        run_standard_sd=_run_standard,
        thresholds=config.thresholds,
        scale_factor=config.scale_factor,
        scale_range=config.scale_range,
        prompt3=config.prompt_merged,
        prompt_length=config.prompt_length,
        token_refinement_steps=config.token_refinement_steps,
        attention_refinement_steps=config.attention_refinement_steps,
        tome_control_steps=config.tome_control_steps,
        eot_replace_step=config.eot_replace_step,
        use_pose_loss=config.use_pose_loss,
        negative_prompt="low res, ugly, blurry, artifact, unreal",
        use_hyperbolic=use_hyperbolic,
        hyper_merger=hyper_merger,
    )
    image = outputs.images[0]
    return image


def filter_text(token_indices, prompt_anchor):
    final_idx = []
    final_prompt = []
    for i, idx in enumerate(token_indices):
        if len(idx[1]) == 0:
            continue
        final_idx.append(idx)
        final_prompt.append(prompt_anchor[i])
    return final_idx, final_prompt


def create_comparison_grid(
    images_dict: dict, seeds: list, method_names: list, prompt: str
) -> Image.Image:
    """
    Creates a labeled comparison grid:
      - Columns = methods (SDXL, ToMe, ToMe_Hyper)
      - Rows = seeds
      - Top row = method labels; left column = seed labels
    """
    img_w, img_h = images_dict[method_names[0]][0].size
    label_h = 48
    seed_label_w = 120
    pad = 4

    cols = len(method_names)
    rows = len(seeds)
    total_w = seed_label_w + cols * (img_w + pad) - pad
    total_h = label_h + rows * (img_h + pad) - pad

    grid = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    for j, name in enumerate(method_names):
        x = seed_label_w + j * (img_w + pad) + img_w // 2
        draw.text((x, label_h // 2), name, fill="black", anchor="mm", font=font)

    for i, seed in enumerate(seeds):
        y = label_h + i * (img_h + pad) + img_h // 2
        draw.text(
            (seed_label_w // 2, y),
            f"Seed\n{seed}",
            fill="black",
            anchor="mm",
            font=font_small,
        )

        for j, name in enumerate(method_names):
            x = seed_label_w + j * (img_w + pad)
            yy = label_h + i * (img_h + pad)
            grid.paste(images_dict[name][i], (x, yy))

    return grid


def main():
    config = RunConfig2()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stable, prompt_parser = load_model(config, device)

    # ---- parse prompt ----
    if config.use_nlp:
        import en_core_web_trf

        nlp = en_core_web_trf.load()
        doc = nlp(config.prompt)
        prompt_parser.set_doc(doc)
        token_indices = prompt_parser._get_indices(config.prompt)
        prompt_anchor = prompt_parser._split_prompt(doc)
        token_indices, prompt_anchor = filter_text(token_indices, prompt_anchor)
    else:
        token_indices = config.token_indices
        prompt_anchor = config.prompt_anchor

    # ---- hyperbolic merger ----
    hyper_merger = TokenMergerWithAttnHyperspace(
        embed_dim=2048, num_heads=8
    ).to(device).eval()

    # ---- define methods to compare ----
    methods = [
        ("SDXL", True, False),
        ("ToMe", False, False),
        ("ToMe_Hyper", False, True),
    ]
    method_names = [m[0] for m in methods]
    all_images = {name: [] for name in method_names}

    prompt_output_path = config.output_path / config.prompt
    prompt_output_path.mkdir(exist_ok=True, parents=True)

    for seed in config.seeds:
        for method_name, run_std, use_hyper in methods:
            print(f"\n{'='*50}")
            print(f"  Method: {method_name}  |  Seed: {seed}")
            print(f"  Prompt: {config.prompt}")
            print(f"{'='*50}")

            g = torch.Generator(device).manual_seed(seed)
            controller = AttentionStore()

            image = run_on_prompt(
                prompt=config.prompt,
                model=stable,
                controller=controller,
                token_indices=token_indices,
                prompt_anchor=prompt_anchor,
                seed=g,
                config=config,
                run_standard_sd=run_std,
                use_hyperbolic=use_hyper,
                hyper_merger=hyper_merger if use_hyper else None,
            )

            image.save(prompt_output_path / f"{seed}_{method_name}.png")
            all_images[method_name].append(image)

    # ---- comparison grid ----
    grid = create_comparison_grid(
        all_images, config.seeds, method_names, config.prompt
    )
    grid.save(
        config.output_path / f"{config.prompt}_comparison.png"
    )
    print(f"\nComparison grid saved to {config.output_path / f'{config.prompt}_comparison.png'}")

    # ---- also save per-method grids for backward compat ----
    for name in method_names:
        joined = vis_utils.get_image_grid(all_images[name])
        joined.save(config.output_path / f"{config.prompt}_{name}.png")


if __name__ == "__main__":
    main()
