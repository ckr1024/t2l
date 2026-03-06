"""
Evaluation metrics for T2I semantic binding experiments.

BLIP-VQA: Follows the T2I-CompBench evaluation protocol exactly:
  - Extract noun phrases via spaCy (e.g., "a green bench")
  - Ask BLIP-VQA: "{noun_phrase}?" → get P(yes)
  - Per-image score = product of P(yes) across all noun phrases
  - Final score = mean of per-image scores

ImageReward: Measures human preference alignment between text and image.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Optional


def extract_noun_phrases(prompt: str) -> List[str]:
    """
    Extract noun phrases from a prompt using spaCy,
    matching T2I-CompBench's noun_chunks extraction.
    """
    try:
        import en_core_web_trf
        nlp = en_core_web_trf.load()
    except ImportError:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.load("en_core_web_trf")

    doc = nlp(prompt.rstrip("."))
    skip = {"top", "the side", "the left", "the right"}
    noun_phrases = []
    for chunk in doc.noun_chunks:
        if chunk.text.lower() not in skip:
            noun_phrases.append(chunk.text)
    return noun_phrases


class BLIPVQAEvaluator:
    """
    BLIP-VQA evaluator matching the T2I-CompBench protocol.

    For each image:
      1. Extract noun phrases from the prompt
      2. For each noun phrase, ask BLIP: "{noun_phrase}?" 
      3. Get P(yes) from the model's output distribution
      4. Image score = product of all P(yes) values
    Final metric = mean of image scores across the dataset.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self._yes_token_id = None

    def load_model(self):
        if self.model is not None:
            return

        from transformers import BlipProcessor, BlipForQuestionAnswering

        print("Loading BLIP-VQA model...")
        model_name = "Salesforce/blip-vqa-base"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self._yes_token_id = self.processor.tokenizer.encode(
            "yes", add_special_tokens=False
        )[0]
        print("BLIP-VQA model loaded.")

    def compute_vqa_prob(self, image: Image.Image, question: str) -> float:
        """
        Ask a yes/no question and return P(yes).
        Matches T2I-CompBench's 'vqa_prob' inference mode.
        """
        self.load_model()

        inputs = self.processor(image, question, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1,
            )

        logits = outputs.scores[0][0]
        probs = F.softmax(logits, dim=-1)
        p_yes = probs[self._yes_token_id].item()
        return p_yes

    def evaluate_image(
        self, image: Image.Image, noun_phrases: List[str]
    ) -> float:
        """
        Evaluate one image: product of P(yes) for each noun phrase question.
        Matches T2I-CompBench's per-image scoring.
        """
        if not noun_phrases:
            return 0.0

        score = 1.0
        for np_text in noun_phrases:
            question = f"{np_text}?"
            p_yes = self.compute_vqa_prob(image, question)
            score *= p_yes
        return score

    def evaluate_batch(
        self,
        image_paths: List[str],
        prompts: List[str],
        attribute_type: str = "color",
    ) -> Dict[str, float]:
        """
        Evaluate a batch of images following T2I-CompBench protocol.

        Args:
            image_paths: Paths to generated images
            prompts: Text prompts (one per image)
            attribute_type: Not used in new protocol but kept for API compat

        Returns:
            Dict with mean_score, std_score, num_samples
        """
        scores = []
        for img_path, prompt in zip(image_paths, prompts):
            image = Image.open(img_path).convert("RGB")
            noun_phrases = extract_noun_phrases(prompt)
            score = self.evaluate_image(image, noun_phrases)
            scores.append(score)

        return {
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "std_score": float(np.std(scores)) if scores else 0.0,
            "individual_scores": scores,
            "num_samples": len(scores),
        }


class ImageRewardEvaluator:
    """ImageReward evaluator for human preference scoring."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    def load_model(self):
        if self.model is not None:
            return

        import ImageReward as RM

        print("Loading ImageReward model...")
        self.model = RM.load("ImageReward-v1.0")
        print("ImageReward model loaded.")

    def compute_score(self, image_path: str, prompt: str) -> float:
        self.load_model()
        return self.model.score(prompt, image_path)

    def evaluate_batch(
        self,
        image_paths: List[str],
        prompts: List[str],
    ) -> Dict[str, float]:
        scores = []
        for img_path, prompt in zip(image_paths, prompts):
            score = self.compute_score(img_path, prompt)
            scores.append(score)

        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "individual_scores": scores,
            "num_samples": len(scores),
        }


def save_evaluation_results(results: Dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")
