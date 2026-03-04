from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class BaseExperimentConfig:
    """Base configuration shared by all experiment configs."""
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    use_nlp: bool = True
    n_inference_steps: int = 50
    guidance_scale: float = 7.5
    attention_res: int = 32
    scale_factor: int = 3
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    save_cross_attention_maps: bool = False
    output_path: Path = Path("./experiment_results")
    seeds: List[int] = field(default_factory=lambda: [42])
    num_images_per_prompt: int = 1

    # These will be set per-prompt by the experiment runner
    prompt: str = ""
    token_indices: List[int] = field(default_factory=list)
    prompt_anchor: List[str] = field(default_factory=list)
    prompt_merged: str = ""
    prompt_length: int = 0

    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26, 1: 25, 2: 24, 3: 23, 4: 22.5,
            5: 22, 6: 21.5, 7: 21, 8: 21, 9: 21,
        }
    )

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)


# ============================================================
# Ablation Configs (Table 2 in paper)
# ============================================================

@dataclass
class ConfigA(BaseExperimentConfig):
    """Config A: Baseline SDXL (no ToMe, no Lent, no Lsem)"""
    run_standard_sd: bool = True
    use_token_merge: bool = False
    use_ets: bool = False
    tome_control_steps: List[int] = field(default_factory=lambda: [0, 0])
    token_refinement_steps: int = 0
    attention_refinement_steps: List[int] = field(default_factory=lambda: [0, 0])
    eot_replace_step: int = 999
    use_pose_loss: bool = False
    output_path: Path = Path("./experiment_results/ablation/config_A")


@dataclass
class ConfigB(BaseExperimentConfig):
    """Config B: ToMe + ETS only (no entropy loss, no semantic binding loss)"""
    run_standard_sd: bool = False
    use_token_merge: bool = True
    use_ets: bool = True
    tome_control_steps: List[int] = field(default_factory=lambda: [0, 0])
    token_refinement_steps: int = 3
    attention_refinement_steps: List[int] = field(default_factory=lambda: [0, 0])
    eot_replace_step: int = 0
    use_pose_loss: bool = False
    output_path: Path = Path("./experiment_results/ablation/config_B")


@dataclass
class ConfigC(BaseExperimentConfig):
    """Config C: ToMe + ETS + Lent (no semantic binding loss)"""
    run_standard_sd: bool = False
    use_token_merge: bool = True
    use_ets: bool = True
    tome_control_steps: List[int] = field(default_factory=lambda: [0, 7])
    token_refinement_steps: int = 3
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    eot_replace_step: int = 0
    use_pose_loss: bool = False
    output_path: Path = Path("./experiment_results/ablation/config_C")


@dataclass
class ConfigD(BaseExperimentConfig):
    """Config D: Lent + Lsem only (no ToMe, no ETS)"""
    run_standard_sd: bool = False
    use_token_merge: bool = False
    use_ets: bool = False
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    token_refinement_steps: int = 3
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    eot_replace_step: int = 999
    use_pose_loss: bool = False
    output_path: Path = Path("./experiment_results/ablation/config_D")


@dataclass
class ConfigE(BaseExperimentConfig):
    """Config E: Lent only (no ToMe, no ETS, no Lsem)"""
    run_standard_sd: bool = False
    use_token_merge: bool = False
    use_ets: bool = False
    tome_control_steps: List[int] = field(default_factory=lambda: [0, 7])
    token_refinement_steps: int = 3
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    eot_replace_step: int = 999
    use_pose_loss: bool = False
    output_path: Path = Path("./experiment_results/ablation/config_E")


@dataclass
class ConfigF(BaseExperimentConfig):
    """Config F: ToMe + ETS + Lsem (no entropy loss)"""
    run_standard_sd: bool = False
    use_token_merge: bool = True
    use_ets: bool = True
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 0])
    token_refinement_steps: int = 3
    attention_refinement_steps: List[int] = field(default_factory=lambda: [0, 0])
    eot_replace_step: int = 0
    use_pose_loss: bool = False
    output_path: Path = Path("./experiment_results/ablation/config_F")


@dataclass
class ConfigOurs(BaseExperimentConfig):
    """Full ToMe: ToMe + ETS + Lent + Lsem"""
    run_standard_sd: bool = False
    use_token_merge: bool = True
    use_ets: bool = True
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    token_refinement_steps: int = 3
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    eot_replace_step: int = 0
    use_pose_loss: bool = False
    output_path: Path = Path("./experiment_results/ablation/config_Ours")


ABLATION_CONFIGS = {
    "A": ConfigA,
    "B": ConfigB,
    "C": ConfigC,
    "D": ConfigD,
    "E": ConfigE,
    "F": ConfigF,
    "Ours": ConfigOurs,
}


# ============================================================
# T2I-CompBench Evaluation Config
# ============================================================

@dataclass
class T2ICompBenchConfig(ConfigOurs):
    """Config for T2I-CompBench benchmark evaluation."""
    output_path: Path = Path("./experiment_results/t2i_compbench")
    seeds: List[int] = field(default_factory=lambda: [42])
    num_prompts: int = 300
    benchmark_subsets: List[str] = field(
        default_factory=lambda: ["color", "shape", "texture"]
    )


# ============================================================
# GPT-4o Benchmark Config
# ============================================================

@dataclass
class GPT4oBenchmarkConfig(ConfigOurs):
    """Config for GPT-4o object binding benchmark."""
    output_path: Path = Path("./experiment_results/gpt4o_benchmark")
    seeds: List[int] = field(default_factory=lambda: [42])
    use_pose_loss: bool = True

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"


# ============================================================
# Hyperbolic Variants (Improvement over original ToMe)
# ============================================================

@dataclass
class HyperbolicConfigOurs(ConfigOurs):
    """Full ToMe with hyperbolic operations (our improvement)."""
    use_hyperbolic: bool = True
    hyperbolic_curvature: float = 1.0
    output_path: Path = Path("./experiment_results/hyperbolic/config_Ours")


@dataclass
class HyperbolicConfigB(ConfigB):
    """ToMe + ETS only, with hyperbolic token merging."""
    use_hyperbolic: bool = True
    hyperbolic_curvature: float = 1.0
    output_path: Path = Path("./experiment_results/hyperbolic/config_B")


@dataclass
class HyperbolicConfigC(ConfigC):
    """ToMe + ETS + Lent, with hyperbolic operations."""
    use_hyperbolic: bool = True
    hyperbolic_curvature: float = 1.0
    output_path: Path = Path("./experiment_results/hyperbolic/config_C")


@dataclass
class HyperbolicConfigF(ConfigF):
    """ToMe + ETS + Lsem, with hyperbolic operations."""
    use_hyperbolic: bool = True
    hyperbolic_curvature: float = 1.0
    output_path: Path = Path("./experiment_results/hyperbolic/config_F")


HYPERBOLIC_CONFIGS = {
    "Hyp-B": HyperbolicConfigB,
    "Hyp-C": HyperbolicConfigC,
    "Hyp-F": HyperbolicConfigF,
    "Hyp-Ours": HyperbolicConfigOurs,
}

# Merge all configs for unified access
ALL_CONFIGS = {**ABLATION_CONFIGS, **HYPERBOLIC_CONFIGS}
