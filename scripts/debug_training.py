import torch
import logging
from pathlib import Path

from dataset import create_data_loaders, Tokenizer, DataConfig
from model import EnhancedDocumentationModel, ModelConfig
from training import Trainer, TrainingConfig


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_sample_data():
    """Create minimal sample data for testing."""
    return [
        "Create a function using def keyword in Python.",
        "Import torch for PyTorch neural networks.",
        "Define a class with methods and attributes.",
        "Use optimizer.step() to update model weights.",
        "Apply loss.backward() for gradient computation.",
    ]


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    texts = create_sample_data()

    model_config = ModelConfig(
        vocab_size=1000, d_model=128, nhead=4, num_layers=2, max_length=64
    )

    training_config = TrainingConfig(
        epochs=2, learning_rate=1e-3, warmup_steps=10, accumulation_steps=1
    )

    data_config = DataConfig(
        max_length=64, batch_size=2, train_split=0.6, val_split=0.2, test_split=0.2
    )

    tokenizer = Tokenizer(model_config.vocab_size)
    tokenizer.build_vocab(texts)

    train_loader, val_loader, test_loader = create_data_loaders(
        texts, tokenizer, data_config
    )

    model = EnhancedDocumentationModel(model_config)
    trainer = Trainer(model, model_config, training_config, "debug_checkpoints")

    trainer.debug_single_batch(train_loader)

    logger.info("Starting short training run...")
    history = trainer.train(train_loader, val_loader, test_loader)

    logger.info("Training completed successfully!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final learning rate: {history['learning_rate'][-1]:.2e}")


if __name__ == "__main__":
    main()
