"""
GPT-4o based evaluation for object binding benchmark.

Uses the GPT-4o multimodal model to assess whether generated images
correctly bind objects with their associated attributes/sub-objects.

Scoring follows the 9-level rubric described in the paper (Fig. 10):
  100:  Both subjects only possess their own attributes
  87.5: Both subjects possess their attributes, but one also has the other's
  75:   Both subjects possess their own + each other's attributes
  62.5: One correct, one has only the other's attributes
  50:   One correct, the other has neither
  37.5: Neither has own attributes, but one has the other's
  25:   Neither subject has any attributes
  12.5: Missing one subject
  0:    Missing both subjects
"""

import os
import json
import base64
from typing import List, Dict, Optional
from pathlib import Path


GPT4O_SCORING_PROMPT = """Based on our picture and prompt, give the score of the picture below. \
The subjects are the two subjects of the prompt words, and the attributes are the adjectives or \
nouns corresponding to the subjects in the prompt body.

The first line of the answer contains only the rating, and then the explanation is given starting \
from the second line.

The scoring criteria are as follows:
100: Both subjects only possess their own attributes, not the attributes of the other subject.
87.5: Both subjects possesses their attributes. But only one subject that possesses the attributes of another subject.
75: When two subjects possess their own attributes, they both possess the attributes of the other subject.
62.5: One subject possesses attributes of its own, without attributes of the another subject. The other subject only possesses attributes of another subject.
50: One subject possesses attributes of its own. The other subject do not possesses attributes of itself or the other party.
37.5: Both subjects not possess its own attributes. But exist one subject has the attributes of the other party.
25: Neither subject has attributes of itself or the other party.
12.5: Missing one subject
0: Missing two subject"""


GPT4O_OBJECT_BINDING_PROMPTS = [
    "a cat wearing sunglasses and a dog wearing hat",
    "a man wearing hat and a woman wearing necklace",
    "a dog with hat and a cat with scarf",
    "a boy with glasses and a girl with earrings",
    "a boy with hat and a corgi with sunglasses",
    "a cat with scarf and a dog with tie",
    "a fox with sunglasses and a deer with crown",
    "a bear with hat and a man with glasses",
    "a man with hat and a girl with necklace",
    "a tiger with glasses and a dog with hat",
    "a squirrel holding guns and a bear with hat",
    "a cat with pink hat and a dog with blue sunglasses",
    "a lion with yellow crown and a sheep with white bandanas",
    "a rabbit with bow tie and a turtle with top hat",
    "a horse with saddle and a cow with bell",
    "a penguin with scarf and a polar bear with sunglasses",
    "a monkey with hat and a parrot with necklace",
    "a wolf with crown and a fox with cape",
    "a cat with ribbon and a dog with bandana",
    "a boy with backpack and a girl with umbrella",
    "a man with tie and a woman with scarf",
    "a duck with hat and a chicken with glasses",
    "a elephant with crown and a giraffe with scarf",
    "a panda with sunglasses and a koala with hat",
    "a robot with hat and an astronaut with sunglasses",
    "a pirate with eyepatch and a knight with shield",
    "a owl with glasses and a eagle with crown",
    "a cat with bell and a dog with bone",
    "a wizard with hat and a warrior with sword",
    "a chef with hat and a doctor with stethoscope",
    "a lion with glasses and a tiger with hat",
    "a mouse with bow and a rat with hat",
    "a frog with crown and a snake with glasses",
    "a bear with scarf and a wolf with hat",
    "a deer with antlers and a elk with bell",
    "a puppy with collar and a kitten with ribbon",
    "a farmer with hat and a fisherman with rod",
    "a prince with crown and a princess with tiara",
    "a cowboy with hat and a sheriff with badge",
    "a bird with hat and a fish with crown",
    "a dragon with glasses and a unicorn with scarf",
    "a raccoon with mask and a fox with cape",
    "a hamster with wheel and a guinea pig with hat",
    "a sailor with hat and a captain with telescope",
    "a clown with hat and a magician with wand",
    "a sheep with bell and a goat with ribbon",
    "a horse with hat and a donkey with glasses",
    "a cat with crown and a dog with cape",
    "a boy with hat and a dog with sunglasses",
    "a man with glasses and a woman with hat",
]


class GPT4oEvaluator:
    """
    GPT-4o based evaluator for object binding assessment.

    Uses OpenAI's GPT-4o model to score generated images based on
    how well objects are bound to their respective attributes.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: No OpenAI API key provided. Set OPENAI_API_KEY env var.")
        self.client = None

    def _init_client(self):
        if self.client is not None:
            return

        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key)

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API upload."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def score_image(self, image_path: str, prompt: str, model: str = "gpt-4o") -> Dict:
        """
        Score a single image using GPT-4o.

        Returns:
            Dict with "score" (float) and "explanation" (str)
        """
        self._init_client()

        base64_image = self._encode_image(image_path)

        full_prompt = f"Prompt: {prompt}\n{GPT4O_SCORING_PROMPT}"

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )

        response_text = response.choices[0].message.content.strip()

        try:
            lines = response_text.split("\n")
            score_line = lines[0].strip()
            score = float(score_line)
            explanation = "\n".join(lines[1:]).strip()
        except (ValueError, IndexError):
            score = -1.0
            explanation = response_text

        normalized_score = score / 100.0 if score >= 0 else 0.0

        return {
            "score": score,
            "normalized_score": normalized_score,
            "explanation": explanation,
            "raw_response": response_text,
        }

    def evaluate_batch(
        self,
        image_paths: List[str],
        prompts: List[str],
        model: str = "gpt-4o",
    ) -> Dict:
        """
        Evaluate a batch of images with GPT-4o.

        Returns:
            Dict with mean score and individual results
        """
        import numpy as np

        results = []
        scores = []

        for i, (img_path, prompt) in enumerate(zip(image_paths, prompts)):
            print(f"  Scoring image {i+1}/{len(image_paths)}: {prompt[:50]}...")
            try:
                result = self.score_image(img_path, prompt, model)
                results.append(result)
                if result["score"] >= 0:
                    scores.append(result["normalized_score"])
            except Exception as e:
                print(f"  Error scoring image {i+1}: {e}")
                results.append({"score": -1, "error": str(e)})

        return {
            "mean_score": np.mean(scores) if scores else 0.0,
            "std_score": np.std(scores) if scores else 0.0,
            "individual_results": results,
            "num_valid": len(scores),
            "num_total": len(image_paths),
        }


def get_gpt4o_prompts() -> List[str]:
    """Return the 50 GPT-4o object binding benchmark prompts."""
    return GPT4O_OBJECT_BINDING_PROMPTS
