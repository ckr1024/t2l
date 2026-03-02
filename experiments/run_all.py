"""
Master Experiment Runner

Runs all experiments from the paper in sequence:
  1. T2I-CompBench evaluation (Table 1: BLIP-VQA + ImageReward)
  2. Ablation study (Table 2: configs A-F + Ours)
  3. GPT-4o object binding benchmark (Table 1: GPT-4o column)
  4. Cross-attention visualization (Figure 7)
  5. Time complexity measurement (Table 3)

Usage:
    python -m experiments.run_all [--experiments all] [--quick]

    --quick: Run with reduced prompts for faster testing
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_metrics import save_evaluation_results


def run_time_complexity_experiment(model, prompt_parser, output_dir: str):
    """
    Time complexity measurement (Table 3 in paper).

    Measures inference time for different configurations and step counts.
    """
    from configs.experiment_config import ConfigA, ConfigC, ConfigOurs
    from experiments.run_t2i_compbench import generate_images_for_prompts

    os.makedirs(output_dir, exist_ok=True)

    test_prompts = [
        ("a cat wearing sunglasses and a dog wearing hat", []),
        ("a red apple and a green pear", []),
        ("a boy with hat and a girl with sunglasses", []),
    ]

    configs_and_steps = [
        ("SDXL_20", ConfigA, 20),
        ("ToMe_ConfigC_20", ConfigC, 20),
        ("ToMe_Ours_20", ConfigOurs, 20),
        ("SDXL_50", ConfigA, 50),
        ("ToMe_ConfigC_50", ConfigC, 50),
        ("ToMe_Ours_50", ConfigOurs, 50),
    ]

    results = {}

    for name, config_cls, n_steps in configs_and_steps:
        config = config_cls()
        config.n_inference_steps = n_steps
        config.seeds = [42]

        print(f"\n--- Timing: {name} (steps={n_steps}) ---")

        start_time = time.time()
        gen_results = generate_images_for_prompts(
            test_prompts, config, model, prompt_parser, [42],
            os.path.join(output_dir, name, "images"),
        )
        elapsed = time.time() - start_time

        avg_time = elapsed / max(len(gen_results), 1)
        results[name] = {
            "total_time": elapsed,
            "avg_time_per_image": avg_time,
            "n_steps": n_steps,
            "num_images": len(gen_results),
        }
        print(f"  Total: {elapsed:.1f}s, Per image: {avg_time:.1f}s")

    print("\n" + "=" * 60)
    print("Time Complexity Results (Table 3)")
    print("=" * 60)
    print(f"{'Method':<25} {'Steps':>6} {'Time/img':>10}")
    print("-" * 45)
    for name, res in results.items():
        print(f"{name:<25} {res['n_steps']:>6} {res['avg_time_per_image']:>9.1f}s")

    save_evaluation_results(results, os.path.join(output_dir, "time_results.json"))
    return results


def main():
    parser = argparse.ArgumentParser(description="Run All Experiments")
    parser.add_argument("--experiments", type=str, nargs="+",
                       default=["all"],
                       choices=["all", "compbench", "ablation", "gpt4o", "attention", "time"],
                       help="Which experiments to run")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: fewer prompts for testing")
    parser.add_argument("--model_path", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output_dir", type=str, default="./experiment_results")
    args = parser.parse_args()

    experiments = args.experiments
    if "all" in experiments:
        experiments = ["compbench", "ablation", "gpt4o", "attention", "time"]

    num_prompts = 5 if args.quick else 20

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from configs.experiment_config import BaseExperimentConfig
    base_config = BaseExperimentConfig()
    base_config.model_path = args.model_path

    from run_demo import load_model
    print("Loading model...")
    model, prompt_parser = load_model(base_config, device)
    print("Model loaded.\n")

    all_results = {}
    total_start = time.time()

    # 1. T2I-CompBench
    if "compbench" in experiments:
        print("\n" + "#" * 60)
        print("# Experiment 1: T2I-CompBench Evaluation")
        print("#" * 60)
        from experiments.run_t2i_compbench import evaluate_subset
        from configs.experiment_config import T2ICompBenchConfig

        config = T2ICompBenchConfig()
        config.seeds = args.seeds
        config.model_path = args.model_path

        compbench_results = {}
        for subset in ["color", "shape", "texture"]:
            results = evaluate_subset(
                subset=subset,
                config=config,
                model=model,
                prompt_parser=prompt_parser,
                num_prompts=num_prompts,
                seeds=args.seeds,
                use_image_reward=not args.quick,
            )
            compbench_results[subset] = results

        all_results["t2i_compbench"] = compbench_results

    # 2. Ablation Study
    if "ablation" in experiments:
        print("\n" + "#" * 60)
        print("# Experiment 2: Ablation Study")
        print("#" * 60)
        from experiments.run_ablation import run_ablation_study, print_ablation_table

        ablation_results = run_ablation_study(
            config_names=["A", "B", "C", "D", "E", "F", "Ours"],
            subsets=["color", "shape", "texture"],
            model=model,
            prompt_parser=prompt_parser,
            num_prompts=num_prompts,
            seeds=args.seeds,
        )
        print_ablation_table(ablation_results)
        all_results["ablation"] = ablation_results

    # 3. GPT-4o Benchmark
    if "gpt4o" in experiments:
        print("\n" + "#" * 60)
        print("# Experiment 3: GPT-4o Object Binding Benchmark")
        print("#" * 60)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Skipping GPT-4o scoring (no OPENAI_API_KEY). Generating images only.")

        from experiments.run_gpt4o_benchmark import generate_gpt4o_benchmark_images
        from experiments.eval_gpt4o import get_gpt4o_prompts, GPT4oEvaluator
        from configs.experiment_config import GPT4oBenchmarkConfig

        config = GPT4oBenchmarkConfig()
        config.seeds = args.seeds
        config.model_path = args.model_path

        prompts = get_gpt4o_prompts()[:num_prompts]
        output_dir = os.path.join(args.output_dir, "gpt4o_benchmark", "images")

        gen_results = generate_gpt4o_benchmark_images(
            prompts, config, model, prompt_parser, args.seeds, output_dir
        )

        gpt4o_result = {"num_generated": len(gen_results)}

        if api_key:
            evaluator = GPT4oEvaluator(api_key=api_key)
            gpt4o_scores = evaluator.evaluate_batch(
                [r["image_path"] for r in gen_results],
                [r["prompt"] for r in gen_results],
            )
            gpt4o_result["scores"] = gpt4o_scores
            print(f"GPT-4o Mean Score: {gpt4o_scores['mean_score']:.4f}")

        all_results["gpt4o"] = gpt4o_result

    # 4. Attention Visualization
    if "attention" in experiments:
        print("\n" + "#" * 60)
        print("# Experiment 4: Cross-Attention Visualization")
        print("#" * 60)
        from experiments.visualize_attention import run_attention_visualization

        run_attention_visualization(
            prompt="a cat wearing sunglasses and a dog wearing hat",
            config_names=["A", "C", "Ours"],
            model=model,
            prompt_parser=prompt_parser,
            seed=42,
            output_dir=os.path.join(args.output_dir, "attention_viz"),
        )
        all_results["attention_viz"] = {"status": "completed"}

    # 5. Time Complexity
    if "time" in experiments:
        print("\n" + "#" * 60)
        print("# Experiment 5: Time Complexity")
        print("#" * 60)
        time_results = run_time_complexity_experiment(
            model, prompt_parser,
            os.path.join(args.output_dir, "time_complexity"),
        )
        all_results["time_complexity"] = time_results

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print(f"All experiments completed in {total_time:.1f}s")
    print("=" * 60)

    summary_path = os.path.join(args.output_dir, "all_results.json")
    all_results["total_time_seconds"] = total_time
    save_evaluation_results(all_results, summary_path)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
