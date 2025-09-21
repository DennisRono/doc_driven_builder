import torch
from torch.utils.data import Dataset, DataLoader
import re
import json
import markdown
from pathlib import Path
from typing import List, Dict, Tuple, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data processing."""

    max_length: int = 256
    batch_size: int = 16
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    min_text_length: int = 10
    max_text_length: int = 2000


class Tokenizer:
    """Enhanced tokenizer with subword support and better handling."""

    def __init__(self, vocab_size: int = 8000, use_subword: bool = True):
        self.vocab_size = vocab_size
        self.use_subword = use_subword
        self.word_to_idx = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3, "<mask>": 4}
        self.idx_to_word = {0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>", 4: "<mask>"}
        self.word_counts = {}
        self.subword_units = {}

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary with subword tokenization support."""
        logger.info("Building enhanced vocabulary...")

        for text in texts:
            words = self._basic_tokenize(text)
            for word in words:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1

        if self.use_subword:
            self._build_subword_vocab()
        else:
            self._build_word_vocab()

        logger.info(f"Built vocabulary with {len(self.word_to_idx)} tokens")

    def _basic_tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization preserving important patterns."""

        text = re.sub(r"(https?://\S+)", r" \1 ", text)
        text = re.sub(r"(`[^`]+`)", r" \1 ", text)
        text = re.sub(r"(\w+\.\w+)", r" \1 ", text)

        text = text.lower()
        text = re.sub(r"[^\w\s\.\-_@/:]", " ", text)
        tokens = text.split()

        return [t for t in tokens if len(t) > 0]

    def _build_subword_vocab(self) -> None:
        """Build subword vocabulary using simple BPE-like approach."""

        chars = set()
        for word in self.word_counts:
            chars.update(word)

        for char in sorted(chars):
            if char not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[char] = idx
                self.idx_to_word[idx] = char

        sorted_words = sorted(
            self.word_counts.items(), key=lambda x: x[1], reverse=True
        )
        for word, count in sorted_words:
            if len(self.word_to_idx) >= self.vocab_size:
                break
            if count >= 3 and word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word

    def _build_word_vocab(self) -> None:
        """Build traditional word-level vocabulary."""
        sorted_words = sorted(
            self.word_counts.items(), key=lambda x: x[1], reverse=True
        )
        for word, _ in sorted_words[: self.vocab_size - len(self.word_to_idx)]:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

    def encode(self, text: str, max_length: int = 256) -> List[int]:
        """Enhanced encoding with better OOV handling."""
        if not text or len(text.strip()) == 0:
            return [0] * max_length

        words = self._basic_tokenize(text)
        tokens = []

        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            elif self.use_subword:

                for char in word:
                    tokens.append(self.word_to_idx.get(char, 1))
            else:
                tokens.append(1)

        tokens = [2] + tokens + [3]

        if len(tokens) > max_length:
            tokens = tokens[: max_length - 1] + [3]
        else:
            tokens.extend([0] * (max_length - len(tokens)))

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Enhanced decoding with better formatting."""
        words = []
        for token in tokens:
            if token == 3:
                break
            if token not in [0, 2]:
                word = self.idx_to_word.get(token, "<unk>")
                if word != "<unk>":
                    words.append(word)

        result = " ".join(words)

        result = re.sub(r"\s+([.,!?;:])", r"\1", result)
        result = re.sub(r"\s+", " ", result).strip()
        return result


class MultiFormatDataProcessor:
    """Process multiple documentation formats."""

    @staticmethod
    def load_file(file_path: Union[str, Path]) -> str:
        """Load and normalize file content."""
        file_path = Path(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()

        content = content.replace("\r\n", "\n").replace("\r", "\n")

        if file_path.suffix.lower() == ".md":
            return MultiFormatDataProcessor._process_markdown(content)
        elif file_path.suffix.lower() == ".rst":
            return MultiFormatDataProcessor._process_rst(content)
        elif file_path.suffix.lower() == ".json":
            return MultiFormatDataProcessor._process_json(content)
        else:
            return MultiFormatDataProcessor._process_plain_text(content)

    @staticmethod
    def _process_markdown(content: str) -> str:
        """Process Markdown content."""

        html = markdown.markdown(content)

        text = re.sub(r"<[^>]+>", "", html)
        return MultiFormatDataProcessor._normalize_text(text)

    @staticmethod
    def _process_rst(content: str) -> str:
        """Process reStructuredText content."""

        content = re.sub(r"\.\. \w+::", "", content)
        content = re.sub(r"::\s*$", "", content, flags=re.MULTILINE)
        content = re.sub(r'^\s*[-=~`#"^+*]{3,}\s*$', "", content, flags=re.MULTILINE)
        return MultiFormatDataProcessor._normalize_text(content)

    @staticmethod
    def _process_json(content: str) -> str:
        """Process JSON content (extract text fields)."""
        try:
            data = json.loads(content)
            texts = []
            MultiFormatDataProcessor._extract_text_from_json(data, texts)
            return MultiFormatDataProcessor._normalize_text(" ".join(texts))
        except json.JSONDecodeError:
            return MultiFormatDataProcessor._normalize_text(content)

    @staticmethod
    def _extract_text_from_json(obj, texts: List[str]) -> None:
        """Recursively extract text from JSON structure."""
        if isinstance(obj, dict):
            for value in obj.values():
                MultiFormatDataProcessor._extract_text_from_json(value, texts)
        elif isinstance(obj, list):
            for item in obj:
                MultiFormatDataProcessor._extract_text_from_json(item, texts)
        elif isinstance(obj, str) and len(obj.strip()) > 0:
            texts.append(obj.strip())

    @staticmethod
    def _process_plain_text(content: str) -> str:
        """Process plain text content."""
        return MultiFormatDataProcessor._normalize_text(content)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text content."""

        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()
        return text


class EnhancedDocumentationDataset(Dataset):
    """Enhanced dataset with robust preprocessing and validation."""
    def __init__(
        self,
        texts: List[str],
        tokenizer: Tokenizer,
        config: DataConfig,
        mode: str = "train",
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode

        self.texts = self._filter_texts(texts)
        logger.info(f"Dataset ({mode}): {len(self.texts)} samples")

        if len(self.texts) == 0:
            raise ValueError(f"No valid texts found for {mode} dataset")

    def _filter_texts(self, texts: List[str]) -> List[str]:
        """Filter texts based on length and content quality."""
        filtered = []
        for text in texts:
            if not text or not text.strip():
                continue

            text = text.strip()
            if (
                self.config.min_text_length <= len(text) <= self.config.max_text_length
                and len(text.split()) >= 3
            ):
                filtered.append(text)

        return filtered

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            text = self.texts[idx]
            tokens = self.tokenizer.encode(text, self.config.max_length)

            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            target_ids = torch.tensor(tokens[1:], dtype=torch.long)

            attention_mask = (input_ids != 0).float()

            return {
                "input_ids": input_ids,
                "target_ids": target_ids,
                "attention_mask": attention_mask,
                "text": (text[:100] + "..." if len(text) > 100 else text),
            }
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            dummy_tokens = [2, 1, 3] + [0] * (self.config.max_length - 3)
            return {
                "input_ids": torch.tensor(dummy_tokens[:-1], dtype=torch.long),
                "target_ids": torch.tensor(dummy_tokens[1:], dtype=torch.long),
                "attention_mask": torch.ones(len(dummy_tokens) - 1, dtype=torch.float),
                "text": "<error>",
            }


def create_data_loaders(
    texts: List[str], tokenizer: Tokenizer, config: DataConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    n = len(texts)

    if n == 0:
        raise ValueError("No texts provided")

    train_end = int(config.train_split * n)
    val_end = train_end + int(config.val_split * n)

    if train_end >= n:
        train_end = n - 1

    val_end = min(val_end, n)

    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]

    if len(train_texts) == 0 and n > 0:
        train_texts = [texts[0]]

    if len(val_texts) == 0 and n > 1:
        val_texts = [texts[min(len(train_texts), n - 1)]]
        if val_texts[0] in train_texts:
            val_texts = [texts[-1]]

    if len(test_texts) == 0 and n > 2:
        test_texts = val_texts[-1:]
        val_texts = val_texts[:-1] or train_texts[-1:]

    train_dataset = EnhancedDocumentationDataset(
        train_texts, tokenizer, config, "train"
    )
    val_dataset = EnhancedDocumentationDataset(val_texts, tokenizer, config, "val")
    test_dataset = EnhancedDocumentationDataset(test_texts, tokenizer, config, "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
