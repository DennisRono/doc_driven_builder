import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import re
from collections import Counter
import math

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    loss: float
    perplexity: float
    accuracy: float
    bleu_score: float
    entity_coverage: float
    response_quality: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "accuracy": self.accuracy,
            "bleu_score": self.bleu_score,
            "entity_coverage": self.entity_coverage,
            "response_quality": self.response_quality,
        }


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics."""

    def __init__(self):
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")

    def evaluate_model(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        tokenizer,
        device: torch.device,
        source_texts: Optional[List[str]] = None,
    ) -> EvaluationMetrics:
        """Comprehensive model evaluation."""
        model.eval()

        all_losses = []
        all_predictions = []
        all_targets = []
        generated_texts = []

        with torch.no_grad():
            for batch in data_loader:
                try:
                    input_ids = batch["input_ids"].to(device)
                    target_ids = batch["target_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    outputs = model(input_ids, attention_mask)

                    losses = self.criterion(
                        outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1)
                    )

                    predictions = torch.argmax(outputs, dim=-1)

                    all_losses.extend(losses.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(target_ids.cpu().numpy())

                    if len(generated_texts) < 10:
                        for i in range(min(2, input_ids.size(0))):
                            prompt = tokenizer.decode(input_ids[i].cpu().tolist())
                            generated = self._generate_sample(
                                model, tokenizer, prompt, device, max_length=50
                            )
                            generated_texts.append(generated)

                except Exception as e:
                    logger.warning(f"Error in evaluation batch: {e}")
                    continue

        loss = np.mean([l for l in all_losses if not np.isnan(l)])
        perplexity = math.exp(loss) if loss < 10 else float("inf")
        accuracy = self._calculate_accuracy(all_predictions, all_targets)
        bleu_score = self._calculate_bleu_score(generated_texts, source_texts)
        entity_coverage = self._calculate_entity_coverage(generated_texts, source_texts)
        response_quality = self._calculate_response_quality(generated_texts)

        return EvaluationMetrics(
            loss=loss,
            perplexity=perplexity,
            accuracy=accuracy,
            bleu_score=bleu_score,
            entity_coverage=entity_coverage,
            response_quality=response_quality,
        )

    def _generate_sample(
        self,
        model: nn.Module,
        tokenizer,
        prompt: str,
        device: torch.device,
        max_length: int = 50,
    ) -> str:
        """Generate a sample text for evaluation."""
        try:
            model.eval()
            tokens = tokenizer.encode(prompt, max_length=128)
            input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

            generated_tokens = []

            with torch.no_grad():
                for _ in range(max_length):
                    outputs = model(input_ids)
                    next_token_logits = outputs[0, -1, :]
                    next_token = torch.argmax(next_token_logits).item()

                    if next_token == 3:
                        break

                    generated_tokens.append(next_token)
                    next_token_tensor = torch.tensor(
                        [[next_token]], dtype=torch.long
                    ).to(device)
                    input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

                    if input_ids.size(1) >= 128:
                        input_ids = input_ids[:, 1:]

            return tokenizer.decode(generated_tokens)

        except Exception as e:
            logger.warning(f"Error generating sample: {e}")
            return ""

    def _calculate_accuracy(self, predictions: List, targets: List) -> float:
        """Calculate token-level accuracy."""
        if not predictions or not targets:
            return 0.0

        correct = 0
        total = 0

        for pred_seq, target_seq in zip(predictions, targets):
            if isinstance(pred_seq, np.ndarray):
                pred_seq = pred_seq.tolist()
            if isinstance(target_seq, np.ndarray):
                target_seq = target_seq.tolist()

            for pred, target in zip(pred_seq, target_seq):
                if target != 0:
                    total += 1
                    if pred == target:
                        correct += 1

        return correct / max(total, 1)

    def _calculate_bleu_score(
        self, generated_texts: List[str], reference_texts: Optional[List[str]]
    ) -> float:
        """Calculate simplified BLEU score."""
        if not generated_texts or not reference_texts:
            return 0.0

        try:
            total_score = 0.0
            count = 0

            for gen_text in generated_texts:
                if not gen_text.strip():
                    continue

                best_score = 0.0
                gen_tokens = gen_text.lower().split()

                for ref_text in reference_texts:
                    ref_tokens = ref_text.lower().split()
                    score = self._compute_bleu_1gram(gen_tokens, ref_tokens)
                    best_score = max(best_score, score)

                total_score += best_score
                count += 1

            return total_score / max(count, 1)

        except Exception as e:
            logger.warning(f"Error calculating BLEU score: {e}")
            return 0.0

    def _compute_bleu_1gram(self, candidate: List[str], reference: List[str]) -> float:
        """Compute 1-gram BLEU score."""
        if not candidate or not reference:
            return 0.0

        candidate_counts = Counter(candidate)
        reference_counts = Counter(reference)

        overlap = 0
        for word, count in candidate_counts.items():
            overlap += min(count, reference_counts.get(word, 0))

        precision = overlap / len(candidate) if candidate else 0.0

        bp = 1.0
        if len(candidate) < len(reference):
            bp = math.exp(1 - len(reference) / len(candidate))

        return bp * precision

    def _calculate_entity_coverage(
        self, generated_texts: List[str], source_texts: Optional[List[str]]
    ) -> float:
        """Calculate how well generated text covers entities from source."""
        if not generated_texts or not source_texts:
            return 0.0

        try:

            source_entities = set()
            for text in source_texts:
                entities = self._extract_entities(text)
                source_entities.update(entities)

            if not source_entities:
                return 1.0

            covered_entities = set()
            for text in generated_texts:
                entities = self._extract_entities(text)
                covered_entities.update(entities)

            coverage = len(covered_entities & source_entities) / len(source_entities)
            return coverage

        except Exception as e:
            logger.warning(f"Error calculating entity coverage: {e}")
            return 0.0

    def _extract_entities(self, text: str) -> set:
        """Extract entities from text (simplified approach)."""
        entities = set()

        capitalized = re.findall(r"\b[A-Z][a-z]+\b", text)
        entities.update(capitalized)

        technical = re.findall(r"\b\w*[._]\w*\b", text)
        entities.update(technical)

        code_patterns = re.findall(r"`[^`]+`", text)
        entities.update([p.strip("`") for p in code_patterns])

        return entities

    def _calculate_response_quality(self, generated_texts: List[str]) -> float:
        """Calculate response quality based on various heuristics."""
        if not generated_texts:
            return 0.0

        try:
            total_score = 0.0
            count = 0

            for text in generated_texts:
                if not text.strip():
                    continue

                score = 0.0

                length = len(text.split())
                if 5 <= length <= 100:
                    score += 0.3
                elif length > 3:
                    score += 0.1

                words = text.lower().split()
                if len(set(words)) / max(len(words), 1) > 0.7:
                    score += 0.3

                if text.strip().endswith((".", "!", "?", ":")):
                    score += 0.2

                if any(
                    term in text.lower()
                    for term in ["function", "class", "import", "def", "use", "create"]
                ):
                    score += 0.2

                total_score += score
                count += 1

            return total_score / max(count, 1)

        except Exception as e:
            logger.warning(f"Error calculating response quality: {e}")
            return 0.0


class InstructionTestSuite:
    """Test suite for instruction-following capabilities."""

    def __init__(self):
        self.test_cases = [
            {
                "instruction": "Create a function",
                "expected_keywords": ["def", "function", "return"],
                "weight": 1.0,
            },
            {
                "instruction": "Import a library",
                "expected_keywords": ["import", "from", "library"],
                "weight": 1.0,
            },
            {
                "instruction": "Define a class",
                "expected_keywords": ["class", "def", "__init__"],
                "weight": 1.0,
            },
            {
                "instruction": "Use torch",
                "expected_keywords": ["torch", "tensor", "model"],
                "weight": 1.0,
            },
        ]

    def evaluate_instruction_following(
        self, model: nn.Module, tokenizer, device: torch.device
    ) -> float:
        """Evaluate instruction-following accuracy."""
        total_score = 0.0
        total_weight = 0.0

        for test_case in self.test_cases:
            try:
                instruction = test_case["instruction"]
                expected_keywords = test_case["expected_keywords"]
                weight = test_case["weight"]

                response = self._generate_response(
                    model, tokenizer, instruction, device
                )

                score = self._score_response(response, expected_keywords)

                total_score += score * weight
                total_weight += weight

                logger.debug(
                    f"Instruction: '{instruction}' -> Response: '{response}' -> Score: {score:.2f}"
                )

            except Exception as e:
                logger.warning(f"Error in instruction test: {e}")
                continue

        return total_score / max(total_weight, 1)

    def _generate_response(
        self,
        model: nn.Module,
        tokenizer,
        instruction: str,
        device: torch.device,
        max_length: int = 30,
    ) -> str:
        """Generate response to instruction."""
        try:
            model.eval()
            tokens = tokenizer.encode(instruction, max_length=128)
            input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

            generated_tokens = []

            with torch.no_grad():
                for _ in range(max_length):
                    outputs = model(input_ids)
                    next_token_logits = outputs[0, -1, :]
                    next_token = torch.argmax(next_token_logits).item()

                    if next_token == 3:
                        break

                    generated_tokens.append(next_token)
                    next_token_tensor = torch.tensor(
                        [[next_token]], dtype=torch.long
                    ).to(device)
                    input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

                    if input_ids.size(1) >= 128:
                        input_ids = input_ids[:, 1:]

            return tokenizer.decode(generated_tokens)

        except Exception as e:
            logger.warning(f"Error generating response: {e}")
            return ""

    def _score_response(self, response: str, expected_keywords: List[str]) -> float:
        """Score response based on expected keywords."""
        if not response:
            return 0.0

        response_lower = response.lower()
        matches = sum(1 for keyword in expected_keywords if keyword in response_lower)

        return matches / len(expected_keywords)
