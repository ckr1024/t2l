"""
Evaluation metrics for T2I semantic binding experiments.
Implements BLIP-VQA scoring and ImageReward scoring.

BLIP-VQA: Measures attribute binding accuracy by asking VQA questions
           about object-attribute associations in generated images.
ImageReward: Measures human preference alignment between text and image.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class BLIPVQAEvaluator:
    """
    BLIP-VQA based evaluator for attribute binding assessment.

    For each generated image, asks VQA questions about attribute-object bindings
    (e.g., "What is the color of the cat?") and computes the probability that
    the answer matches the expected attribute.

    This follows the T2I-CompBench evaluation protocol.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        """Lazy-load BLIP VQA model to save memory."""
        if self.model is not None:
            return

        from transformers import BlipProcessor, BlipForQuestionAnswering

        print("Loading BLIP-VQA model...")
        model_name = "Salesforce/blip-vqa-base"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("BLIP-VQA model loaded.")

    def ask_vqa(self, image: Image.Image, question: str) -> Dict[str, float]:
        """
        Ask a VQA question and return answer probabilities.

        Returns:
            Dict mapping answer text to probability score.
        """
        self.load_model()

        inputs = self.processor(image, question, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)

        return answer.strip().lower()

    def compute_vqa_score(
        self,
        image: Image.Image,
        attribute: str,
        object_name: str,
        attribute_type: str,
    ) -> float:
        """
        Compute BLIP-VQA score for a single attribute-object pair.

        Args:
            image: Generated PIL image
            attribute: Expected attribute (e.g., "red")
            object_name: Object name (e.g., "cat")
            attribute_type: Type of attribute ("color", "shape", or "texture")

        Returns:
            Score between 0 and 1
        """
        self.load_model()

        question_templates = {
            "color": f"What is the color of the {object_name}?",
            "shape": f"What is the shape of the {object_name}?",
            "texture": f"What is the texture of the {object_name}?",
        }

        question = question_templates.get(
            attribute_type, f"What is the {attribute_type} of the {object_name}?"
        )

        inputs = self.processor(image, question, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=10,
                output_scores=True,
                return_dict_in_generate=True,
            )

        answer = self.processor.decode(out.sequences[0], skip_special_tokens=True).strip().lower()
        attribute_lower = attribute.strip().lower()

        if attribute_lower in answer or answer in attribute_lower:
            return 1.0

        if out.scores:
            first_token_logits = out.scores[0][0]
            probs = torch.softmax(first_token_logits, dim=-1)

            attr_tokens = self.processor.tokenizer.encode(
                attribute_lower, add_special_tokens=False
            )
            if attr_tokens:
                score = probs[attr_tokens[0]].item()
                return max(score, 0.0)

        return 0.0

    def evaluate_batch(
        self,
        image_paths: List[str],
        prompts_data: List[Dict],
        attribute_type: str,
    ) -> Dict[str, float]:
        """
        Evaluate a batch of images for attribute binding.

        Args:
            image_paths: List of paths to generated images
            prompts_data: List of dicts with keys: "prompt", "attribute", "object"
            attribute_type: "color", "shape", or "texture"

        Returns:
            Dict with "mean_score" and individual scores
        """
        scores = []
        for img_path, pdata in zip(image_paths, prompts_data):
            image = Image.open(img_path).convert("RGB")
            score = self.compute_vqa_score(
                image, pdata["attribute"], pdata["object"], attribute_type
            )
            scores.append(score)

        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "individual_scores": scores,
            "num_samples": len(scores),
        }


class ImageRewardEvaluator:
    """
    ImageReward evaluator for human preference scoring.

    Uses the ImageReward model to compute human preference scores
    that comprehensively measure image quality and prompt alignment.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    def load_model(self):
        """Lazy-load ImageReward model."""
        if self.model is not None:
            return

        import ImageReward as RM

        print("Loading ImageReward model...")
        self.model = RM.load("ImageReward-v1.0")
        print("ImageReward model loaded.")

    def compute_score(self, image_path: str, prompt: str) -> float:
        """Compute ImageReward score for a single image-prompt pair."""
        self.load_model()
        score = self.model.score(prompt, image_path)
        return score

    def evaluate_batch(
        self,
        image_paths: List[str],
        prompts: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate a batch of images with ImageReward.

        Returns:
            Dict with mean score and individual scores
        """
        scores = []
        for img_path, prompt in zip(image_paths, prompts):
            score = self.compute_score(img_path, prompt)
            scores.append(score)

        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "individual_scores": scores,
            "num_samples": len(scores),
        }


def parse_attribute_prompt(prompt: str, attribute_type: str) -> List[Dict]:
    """
    Parse a T2I-CompBench prompt to extract object-attribute pairs.

    For color prompts like "a red car and a blue bus":
    Returns [{"object": "car", "attribute": "red"}, {"object": "bus", "attribute": "blue"}]
    """
    import spacy

    try:
        import en_core_web_trf
        nlp = en_core_web_trf.load()
    except ImportError:
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(prompt)
    pairs = []

    modifiers = {"amod", "nmod", "compound", "npadvmod", "advmod", "acomp"}

    for token in doc:
        if token.pos_ not in ["NOUN", "PROPN"] or token.dep_ in modifiers:
            continue

        for child in token.children:
            if child.dep_ in modifiers:
                if attribute_type == "color" and child.pos_ == "ADJ":
                    pairs.append({"object": token.text, "attribute": child.text})
                elif attribute_type == "texture" and child.pos_ == "ADJ":
                    pairs.append({"object": token.text, "attribute": child.text})
                elif attribute_type == "shape" and child.pos_ == "ADJ":
                    pairs.append({"object": token.text, "attribute": child.text})

    if not pairs:
        pairs.append({"object": "object", "attribute": "", "prompt": prompt})

    return pairs


def save_evaluation_results(results: Dict, output_path: str):
    """Save evaluation results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")
