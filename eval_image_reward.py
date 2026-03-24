#!/usr/bin/env python
"""
ImageReward Standalone Evaluator
=================================
Evaluate pre-generated images using the ImageReward model.
No generation — only scoring.

Expected directory layout:
    <image_root>/<method>/<subset>/samples/<prompt>_<idx>.png

Usage
-----
    # Auto-detect all methods & subsets under the directory
    python eval_image_reward.py --image_root eval_results

    # Specify methods / subsets explicitly
    python eval_image_reward.py --image_root eval_results --methods GeoBind --subsets color texture

    # Force re-evaluate (ignore cached scores)
    python eval_image_reward.py --image_root eval_results --force

    # Custom ImageReward checkpoint
    python eval_image_reward.py --image_root eval_results \
        --reward_path /path/to/ImageReward.pt \
        --med_config /path/to/med_config.json
"""

import os
import sys
import json
import logging
import argparse
import traceback
import re
import importlib
import types
from datetime import datetime

import torch
from tqdm import tqdm

SUBSETS = ["color", "shape", "texture"]

# ─────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"ir_eval_log_{ts}.txt")

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger = logging.getLogger("ir_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Logging to {log_path}")
    return logger

log = logging.getLogger("ir_eval")

# ─────────────────────────────────────────────────────────
#  ImageReward import compatibility
# ─────────────────────────────────────────────────────────

def _local_transformers_shims():
    """
    Backward-compatible fallbacks for symbols removed from
    transformers.modeling_utils in newer transformers versions.
    """
    import torch.nn as nn

    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        if len(input_tensors) == 0:
            raise ValueError("input_tensors has to be a tuple/list of tensors")

        if chunk_size > 0:
            tensor_shape = input_tensors[0].shape[chunk_dim]
            for input_tensor in input_tensors:
                if input_tensor.shape[chunk_dim] != tensor_shape:
                    raise ValueError(
                        "All input tensors must have the same shape at chunk_dim"
                    )
            if tensor_shape % chunk_size != 0:
                raise ValueError(
                    "The dimension to be chunked must be a multiple of chunk_size"
                )

            num_chunks = tensor_shape // chunk_size
            input_tensors_chunks = tuple(
                input_tensor.chunk(num_chunks, dim=chunk_dim)
                for input_tensor in input_tensors
            )
            output_chunks = [
                forward_fn(*chunked_inputs)
                for chunked_inputs in zip(*input_tensors_chunks)
            ]
            return torch.cat(output_chunks, dim=chunk_dim)

        return forward_fn(*input_tensors)

    def find_pruneable_heads_and_indices(
        heads, n_heads, head_size, already_pruned_heads
    ):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0

        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask), dtype=torch.long)[mask]
        return heads, index

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()

        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()

        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(
            new_size[1], new_size[0], bias=layer.bias is not None
        ).to(layer.weight.device)

        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True

        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True

        return new_layer

    return {
        "apply_chunking_to_forward": apply_chunking_to_forward,
        "find_pruneable_heads_and_indices": find_pruneable_heads_and_indices,
        "prune_linear_layer": prune_linear_layer,
    }

def import_image_reward():
    """
    Import ImageReward with a compatibility shim for newer transformers.
    """
    missing_re = re.compile(
        r"cannot import name '([^']+)' from 'transformers\.modeling_utils'"
    )
    provider_modules = [
        "transformers.pytorch_utils",
        "transformers.modeling_utils",
        "transformers.modeling_attn_mask_utils",
        "transformers.utils",
    ]
    local_shims = _local_transformers_shims()

    def patch_symbol(symbol_name):
        import transformers.modeling_utils as modeling_utils
        if hasattr(modeling_utils, symbol_name):
            return True

        for mod_name in provider_modules:
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            if hasattr(mod, symbol_name):
                setattr(modeling_utils, symbol_name, getattr(mod, symbol_name))
                return True
        if symbol_name in local_shims:
            setattr(modeling_utils, symbol_name, local_shims[symbol_name])
            return True
        return False

    last_error = None
    for _ in range(6):
        try:
            # Prefer importing ImageReward.utils directly to avoid
            # ImageReward.__init__ side-effects (e.g., ReFL/diffusers imports).
            utils_module = importlib.import_module("ImageReward.utils")
            if hasattr(utils_module, "load"):
                return types.SimpleNamespace(load=utils_module.load)
            raise ImportError("ImageReward.utils loaded but no `load` function found")
        except ImportError as e:
            last_error = e
            msg = str(e)
            m = missing_re.search(msg)
            if not m:
                raise

            missing_symbol = m.group(1)
            ok = patch_symbol(missing_symbol)
            if not ok:
                break

            # Remove partially-imported ImageReward modules before retrying.
            for mod_name in list(sys.modules.keys()):
                if mod_name == "ImageReward" or mod_name.startswith("ImageReward."):
                    sys.modules.pop(mod_name, None)

    raise ImportError(
        "ImageReward import failed due to incompatible transformers API. "
        "Please install a compatible version, e.g. "
        "`pip install \"transformers==4.30.2\"`."
    ) from last_error


def patch_image_reward_internals(reward_module):
    """
    Apply all runtime compatibility patches to ImageReward's internals
    so it works with newer transformers versions.

    Patches applied:
      1. Replace init_tokenizer so it doesn't rely on additional_special_tokens_ids.
      2. Add all_tied_weights_keys to PreTrainedModel if missing.
      3. Neutralise init_weights on the BLIP BertModel if tie_weights still fails.
    """
    # ── Patch 1: init_tokenizer ────────────────────────────────────────────
    try:
        import ImageReward.models.BLIP.blip as blip_module
        from transformers import BertTokenizer

        def patched_init_tokenizer():
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
            # Derive enc_token_id without relying on additional_special_tokens_ids.
            try:
                enc_id = tokenizer.additional_special_tokens_ids[0]
            except (AttributeError, IndexError, TypeError):
                enc_id = tokenizer.convert_tokens_to_ids("[ENC]")
                if enc_id == tokenizer.unk_token_id:
                    # Fallback: use vocab size - 1 as a harmless placeholder.
                    enc_id = len(tokenizer) - 1
            tokenizer.enc_token_id = enc_id
            return tokenizer

        blip_module.init_tokenizer = patched_init_tokenizer
        log.info("  [compat] Patched ImageReward.models.BLIP.blip.init_tokenizer")
    except Exception as e:
        log.warning(f"  [compat] Could not patch init_tokenizer: {e}")

    # ── Patch 2: PreTrainedModel.all_tied_weights_keys ────────────────────
    try:
        from transformers.modeling_utils import PreTrainedModel
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = property(lambda self: [])
            log.info("  [compat] Added PreTrainedModel.all_tied_weights_keys")
    except Exception as e:
        log.warning(f"  [compat] Could not patch all_tied_weights_keys: {e}")

    # ── Patch 3: BLIP BertModel.init_weights safety net ───────────────────
    try:
        import ImageReward.models.BLIP.med as med_module

        original_init_weights = getattr(med_module.BertModel, "init_weights", None)

        def safe_init_weights(self):
            try:
                if original_init_weights:
                    original_init_weights(self)
                else:
                    super(med_module.BertModel, self).init_weights()
            except AttributeError as exc:
                if "all_tied_weights_keys" in str(exc):
                    # tie_weights references all_tied_weights_keys; skip gracefully.
                    self._init_weights(self)
                    log.warning("  [compat] Skipped tie_weights (all_tied_weights_keys missing)")
                else:
                    raise

        med_module.BertModel.init_weights = safe_init_weights
        log.info("  [compat] Patched ImageReward.models.BLIP.med.BertModel.init_weights")
    except Exception as e:
        log.warning(f"  [compat] Could not patch BertModel.init_weights: {e}")

# ─────────────────────────────────────────────────────────
#  Result persistence
# ─────────────────────────────────────────────────────────

def _results_path(image_root):
    return os.path.join(image_root, "image_reward_results.json")

def load_results(image_root):
    p = _results_path(image_root)
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {}

def save_results(results, image_root):
    p = _results_path(image_root)
    with open(p, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved -> {p}")

# ─────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────

def load_prompts(data_dir, subset):
    path = os.path.join(data_dir, f"{subset}_val.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

# ─────────────────────────────────────────────────────────
#  ImageReward scoring
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_one(images_dir, prompts, model, detail_save_path=None):
    """Score all images in a (method, subset) pair and return the mean score."""
    scores = []
    per_image_details = []

    for k, prompt in enumerate(tqdm(prompts, desc="    ImageReward")):
        img_path = os.path.join(images_dir, f"{prompt}_{k}.png")
        detail = {"prompt": prompt, "index": k, "image_path": img_path}

        if not os.path.exists(img_path):
            detail["score"] = None
            detail["note"] = "missing"
            per_image_details.append(detail)
            continue

        try:
            score = model.score(prompt, str(img_path))
            detail["score"] = round(float(score), 6)
            scores.append(float(score))
        except Exception as e:
            detail["score"] = None
            detail["note"] = f"error: {e}"

        per_image_details.append(detail)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    n_valid = len(scores)
    n_total = len(prompts)

    if detail_save_path:
        os.makedirs(os.path.dirname(detail_save_path), exist_ok=True)
        with open(detail_save_path, "w") as f:
            json.dump({
                "image_reward_mean": round(mean_score, 6),
                "n_valid": n_valid,
                "n_total": n_total,
                "per_image": per_image_details,
            }, f, indent=2, ensure_ascii=False)
        log.info(f"    Detail saved -> {detail_save_path}")

    return mean_score, n_valid, n_total

# ─────────────────────────────────────────────────────────
#  Auto-detect available methods / subsets
# ─────────────────────────────────────────────────────────

def detect_available(image_root, methods_filter=None, subsets_filter=None):
    pairs = []
    if not os.path.isdir(image_root):
        return pairs
    for method in sorted(os.listdir(image_root)):
        method_dir = os.path.join(image_root, method)
        if not os.path.isdir(method_dir):
            continue
        if methods_filter and method not in methods_filter:
            continue
        for subset in sorted(os.listdir(method_dir)):
            if subsets_filter and subset not in subsets_filter:
                continue
            samples_dir = os.path.join(method_dir, subset, "samples")
            if os.path.isdir(samples_dir):
                n_images = len([f for f in os.listdir(samples_dir) if f.endswith(".png")])
                if n_images > 0:
                    pairs.append((method, subset, n_images))
    return pairs

# ─────────────────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────────────────

def print_result_table(results):
    subsets = [s for s in SUBSETS if s in results]
    methods_set = set()
    for s in subsets:
        methods_set.update(results[s].keys())
    methods = sorted(methods_set)

    log.info("")
    log.info("=" * 62)
    log.info("  T2I-CompBench  ImageReward Scores")
    log.info("=" * 62)
    header = f"  {'Method':<16}"
    for s in subsets:
        header += f"{s.capitalize():<14}"
    log.info(header)
    log.info("  " + "-" * 56)
    for m in methods:
        row = f"  {m:<16}"
        for s in subsets:
            val = results.get(s, {}).get(m)
            if val is not None:
                row += f"{val:<14.4f}"
            else:
                row += f"{'N/A':<14}"
        log.info(row)
    log.info("=" * 62)

# ─────────────────────────────────────────────────────────
#  CLI + main
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone ImageReward evaluator for T2I-CompBench images")
    p.add_argument("--image_root", required=True,
                   help="Root dir containing <method>/<subset>/samples/")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Methods to evaluate (auto-detect if omitted)")
    p.add_argument("--subsets", nargs="+", default=None,
                   help="Subsets to evaluate (auto-detect if omitted)")
    p.add_argument("--data_dir", default="data/t2i_compbench",
                   help="Directory containing *_val.txt prompt files")
    p.add_argument("--reward_path", default="ImageReward-v1.0",
                   help="ImageReward checkpoint name or path")
    p.add_argument("--med_config", default=None,
                   help="Path to med_config.json (optional, for local checkpoints)")
    p.add_argument("--force", action="store_true",
                   help="Re-evaluate even if cached results exist")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.image_root)

    pairs = detect_available(args.image_root, args.methods, args.subsets)
    if not pairs:
        log.error(f"No images found under {args.image_root}. "
                  f"Expected layout: <image_root>/<method>/<subset>/samples/*.png")
        return

    log.info(f"Found {len(pairs)} (method, subset) pairs to evaluate:")
    for method, subset, n in pairs:
        log.info(f"  {method}/{subset}  -- {n} images")

    results = {} if args.force else load_results(args.image_root)

    try:
        reward = import_image_reward()
    except Exception:
        log.error(f"Failed to import ImageReward:\n{traceback.format_exc()}")
        return

    log.info("Applying ImageReward compatibility patches ...")
    patch_image_reward_internals(reward)

    log.info(f"Loading ImageReward model: {args.reward_path} ...")
    load_kwargs = {"name": args.reward_path}
    if args.med_config:
        load_kwargs["med_config"] = args.med_config

    try:
        ir_model = reward.load(**load_kwargs)
    except Exception:
        log.error(f"Failed to load ImageReward:\n{traceback.format_exc()}")
        return

    for method, subset, _ in pairs:
        if subset not in results:
            results[subset] = {}
        existing = results[subset].get(method)
        if existing is not None and not args.force:
            log.info(f"  [{method}/{subset}] cached: {existing} -- skip (use --force to re-eval)")
            continue

        images_dir = os.path.join(args.image_root, method, subset, "samples")
        prompts = load_prompts(args.data_dir, subset)

        log.info(f"{'─'*50}")
        log.info(f"  Evaluating  {method} / {subset}")
        log.info(f"{'─'*50}")

        detail_path = os.path.join(args.image_root, method, subset, "ir_detail.json")
        try:
            score, n_valid, n_total = evaluate_one(
                images_dir, prompts, ir_model, detail_save_path=detail_path,
            )
            results[subset][method] = round(score, 4)
            log.info(f"  -> ImageReward score = {score:.4f}  ({n_valid}/{n_total} valid)")
        except Exception:
            log.error(f"  FAILED {method}/{subset}:\n{traceback.format_exc()}")
            results[subset][method] = None

        save_results(results, args.image_root)

    del ir_model
    torch.cuda.empty_cache()

    print_result_table(results)
    log.info(f"Final results -> {_results_path(args.image_root)}")


if __name__ == "__main__":
    main()
