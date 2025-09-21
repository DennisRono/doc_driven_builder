import argparse
import logging
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
import torch
from dataclasses import asdict
from torch.serialization import add_safe_globals
from improved_training import ImprovedTrainingConfig

from dataset import (
    MultiFormatDataProcessor,
    Tokenizer,
    DataConfig,
    create_data_loaders,
)
from model import DocumentationModel, ModelConfig
from training import Trainer, TrainingConfig
from evaluation import ModelEvaluator, InstructionTestSuite

add_safe_globals([ModelConfig])

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper())

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


class ConfigManager:
    """Manage configuration files and defaults."""

    @staticmethod
    def save_config(config_dict: Dict, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, "r") as f:
            return json.load(f)

    @staticmethod
    def get_default_configs() -> Dict:
        """Get default configurations."""
        return {
            "model": asdict(ModelConfig()),
            "training": asdict(TrainingConfig()),
            "data": asdict(DataConfig()),
        }


class TextGenerator:
    """Enhanced text generation with multiple sampling strategies."""

    def __init__(
        self,
        model: DocumentationModel,
        tokenizer: Tokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        strategy: str = "top_p",
    ) -> str:
        """Generate text with various sampling strategies."""
        self.model.eval()

        tokens = self.tokenizer.encode(prompt, max_length=256)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                next_token_logits = outputs[0, -1, :] / temperature

                if strategy == "greedy":
                    next_token = torch.argmax(next_token_logits).item()
                elif strategy == "top_k":
                    next_token = self._sample_top_k(next_token_logits, top_k)
                elif strategy == "top_p":
                    next_token = self._sample_top_p(next_token_logits, top_p)
                else:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                if next_token == 3:
                    break

                generated_tokens.append(next_token)

                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(
                    self.device
                )
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

                if input_ids.size(1) >= 256:
                    input_ids = input_ids[:, 1:]

        return self.tokenizer.decode(generated_tokens)

    def _sample_top_k(self, logits: torch.Tensor, k: int) -> int:
        """Sample from top-k tokens."""
        top_k_logits, top_k_indices = torch.topk(logits, k)
        probs = torch.softmax(top_k_logits, dim=-1)
        sampled_index = torch.multinomial(probs, 1).item()
        return top_k_indices[sampled_index].item()

    def _sample_top_p(self, logits: torch.Tensor, p: float) -> int:
        """Sample from top-p (nucleus) sampling."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0

        sorted_logits[sorted_indices_to_remove] = float("-inf")

        probs = torch.softmax(sorted_logits, dim=-1)
        sampled_index = torch.multinomial(probs, 1).item()
        return sorted_indices[sampled_index].item()


def load_documentation_files(file_paths: List[str]) -> List[str]:
    """Load documentation from multiple files or folders."""
    import os

    texts = []
    for path in file_paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    full_path = os.path.join(root, file)
                    try:
                        content = MultiFormatDataProcessor.load_file(full_path)
                        if content.strip():
                            texts.append(content)
                            logging.info(
                                f"Loaded {len(content)} characters from {full_path}"
                            )
                    except Exception as e:
                        logging.warning(f"Failed to load {full_path}: {e}")
        else:
            try:
                content = MultiFormatDataProcessor.load_file(path)
                if content.strip():
                    texts.append(content)
                    logging.info(f"Loaded {len(content)} characters from {path}")
            except Exception as e:
                logging.warning(f"Failed to load {path}: {e}")
    return texts


def create_sample_documentation() -> List[str]:
    """Create sample documentation for testing."""
    return [
        "To create a function in Python, use the def keyword followed by the function name and parameters. Functions help organize code and make it reusable.",
        "Import libraries using the import statement. For example, import torch for PyTorch or import numpy for numerical computing.",
        "Define a class using the class keyword. Classes encapsulate data and methods. Add methods using def inside the class definition.",
        "Use torch.nn.Module as base class for neural networks. Override the forward method to define the computation graph.",
        "Train models using optimizer.step() after loss.backward() to update weights. This implements gradient descent optimization.",
        "Create datasets by inheriting from torch.utils.data.Dataset and implementing __getitem__ and __len__ methods.",
        "Use DataLoader to batch and shuffle data during training. This improves training efficiency and convergence.",
        "Apply torch.compile to optimize model performance in PyTorch 2.0+. This can significantly speed up training and inference.",
        "Initialize weights using torch.nn.init functions for better training stability. Proper initialization prevents vanishing gradients.",
        "Use CrossEntropyLoss for classification tasks and MSELoss for regression. Choose the loss function based on your problem type.",
        "Implement attention mechanisms using torch.nn.MultiheadAttention. Attention helps models focus on relevant parts of the input.",
        "Use learning rate scheduling to improve training dynamics. Common schedules include cosine annealing and step decay.",
        "Apply dropout and batch normalization for regularization. These techniques help prevent overfitting in deep networks.",
        "Save and load model checkpoints using torch.save and torch.load. This enables resuming training and model deployment.",
        "Use mixed precision training with torch.cuda.amp for faster training on modern GPUs. This reduces memory usage and training time.",
    ]


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Documentation-Driven Builder")
    parser.add_argument(
        "command",
        choices=["train", "generate", "evaluate", "config"],
        help="Command to execute",
    )
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--data", type=str, nargs="+", help="Documentation file paths")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--prompt", type=str, help="Generation prompt")
    parser.add_argument(
        "--max-length", type=int, default=100, help="Max generation length"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Generation temperature"
    )
    parser.add_argument(
        "--strategy",
        choices=["greedy", "top_k", "top_p", "random"],
        default="top_p",
        help="Sampling strategy",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument("--log-file", type=str, help="Log file path")

    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    if args.config and Path(args.config).exists():
        config_dict = ConfigManager.load_config(args.config)
    else:
        config_dict = ConfigManager.get_default_configs()
        if args.config:
            ConfigManager.save_config(config_dict, args.config)
            logger.info(f"Created default config at {args.config}")

    model_config = ModelConfig(**config_dict["model"])
    training_config = TrainingConfig(**config_dict["training"])
    data_config = DataConfig(**config_dict["data"])

    if args.command == "config":
        print(json.dumps(config_dict, indent=2))
        return

    if args.data:
        texts = load_documentation_files(args.data)
    else:
        texts = create_sample_documentation()

    if not texts:
        logger.error("No documentation texts loaded")
        return

    tokenizer = Tokenizer(model_config.vocab_size)
    tokenizer.build_vocab(texts)

    model = DocumentationModel(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.command == "train":
        train_loader, val_loader, test_loader = create_data_loaders(
            texts, tokenizer, data_config
        )

        trainer = Trainer(model, model_config, training_config, args.output)

        if args.checkpoint:
            trainer.resume_from_checkpoint(args.checkpoint)

        history = trainer.train(train_loader, val_loader, test_loader)

        results_path = Path(args.output) / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Training completed. Results saved to {results_path}")

    elif args.command == "generate":
        if not args.prompt:
            logger.error("Prompt required for generation")
            return

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

        generator = TextGenerator(model, tokenizer, device)
        generated_text = generator.generate(
            args.prompt, args.max_length, args.temperature, strategy=args.strategy
        )

        print(f"Prompt: {args.prompt}")
        print(f"Generated: {generated_text}")

    elif args.command == "evaluate":
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

        _, _, test_loader = create_data_loaders(texts, tokenizer, data_config)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, test_loader, tokenizer, device, texts)

        instruction_suite = InstructionTestSuite()
        instruction_score = instruction_suite.evaluate_instruction_following(
            model, tokenizer, device
        )

        print("Evaluation Results:")
        print(f"Loss: {metrics.loss:.4f}")
        print(f"Perplexity: {metrics.perplexity:.2f}")
        print(f"Accuracy: {metrics.accuracy:.4f}")
        print(f"BLEU Score: {metrics.bleu_score:.4f}")
        print(f"Entity Coverage: {metrics.entity_coverage:.4f}")
        print(f"Response Quality: {metrics.response_quality:.4f}")
        print(f"Instruction Following: {instruction_score:.4f}")


if __name__ == "__main__":
    main()
