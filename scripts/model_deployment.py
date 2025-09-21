import json
import logging
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from dataset import Tokenizer
from model import DocumentationModel, ModelConfig
from model_inference import (
    GenerationConfig,
    ModelLoader,
    TextGenerator,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment/package creation."""

    model_name: str = "documentation_model"
    version: str = "1.0.0"
    description: str = "Documentation-driven text generation model"
    max_batch_size: int = 8
    max_sequence_length: int = 256
    default_generation_config: Optional[GenerationConfig] = None
    optimization_level: str = "standard"  # can be -"minimal", "standard", "aggressive"
    include_analysis_tools: bool = True


class ModelPackager:
    """Package trained models for deployment."""

    def __init__(self, checkpoint_path: str, tokenizer_path: Optional[str] = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None

        # Load model and tokenizer via your ModelLoader interface
        logger.info("Loading model from checkpoint: %s", self.checkpoint_path)
        self.model, self.model_config, self.metadata = (
            ModelLoader.load_model_from_checkpoint(str(self.checkpoint_path))
        )

        # Tokenizer: if provided and exists, load it, otherwise create minimal one
        if self.tokenizer_path and self.tokenizer_path.exists():
            try:
                self.tokenizer = Tokenizer.load(str(self.tokenizer_path))
            except Exception as e:
                logger.warning(
                    "Failed to load tokenizer (%s) — creating minimal tokenizer. Error: %s",
                    self.tokenizer_path,
                    e,
                )
                self.tokenizer = Tokenizer()
        else:
            logger.warning(
                "Tokenizer path not provided or does not exist; creating minimal tokenizer."
            )
            self.tokenizer = Tokenizer()

    def create_deployment_package(
        self, output_path: str, config: Optional[DeploymentConfig] = None
    ) -> str:
        """Create a complete deployment package zip at output_path."""
        if config is None:
            config = DeploymentConfig()

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        package_name = f"{config.model_name}_v{config.version}.zip"
        package_path = output_path / package_name

        # create a temporary folder to collect files (safer for arc names)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 1. Optimized model checkpoint file
            model_file = self._create_optimized_model(config)
            model_file_path = Path(model_file)
            model_dest = tmpdir_path / "model.pt"
            model_dest.write_bytes(model_file_path.read_bytes())

            # 2. Tokenizer file
            tokenizer_file = self._save_tokenizer()
            tokenizer_file_path = Path(tokenizer_file)
            tokenizer_dest = tmpdir_path / "tokenizer.json"
            tokenizer_dest.write_bytes(tokenizer_file_path.read_bytes())

            # 3. Config JSON
            package_info = {
                "created_at": time.time(),
                "pytorch_version": torch.__version__,
            }

            # model parameters count (try method then fallback)
            try:
                model_parameters = self.model.get_num_params()
            except Exception:
                model_parameters = sum(p.numel() for p in self.model.parameters())

            try:
                model_size_mb = self._estimate_model_size()
            except Exception:
                model_size_mb = None

            config_data = {
                "deployment_config": asdict(config),
                "model_config": (
                    asdict(self.model_config)
                    if hasattr(self.model_config, "__dict__")
                    or isinstance(self.model_config, dict)
                    else asdict(self.model_config)
                ),
                "model_metadata": self.metadata,
                "package_info": {
                    **package_info,
                    "model_parameters": int(model_parameters),
                    "model_size_mb": (
                        float(model_size_mb) if model_size_mb is not None else None
                    ),
                },
            }

            config_json_path = tmpdir_path / "config.json"
            config_json_path.write_text(json.dumps(config_data, indent=2, default=str))

            # 4. Deployment script
            deployment_script_path = tmpdir_path / "deploy.py"
            deployment_script_path.write_text(self._create_deployment_script(config))

            # 5. Examples script
            examples_script_path = tmpdir_path / "examples.py"
            examples_script_path.write_text(self._create_examples_script(config))

            # 6. README
            readme_path = tmpdir_path / "README.md"
            readme_path.write_text(
                self._create_readme(config, model_parameters, model_size_mb)
            )

            # Build zip
            with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in tmpdir_path.iterdir():
                    zf.write(file_path, arcname=file_path.name)

        logger.info("Deployment package created: %s", package_path)
        return str(package_path)

    def _create_optimized_model(self, config: DeploymentConfig) -> str:
        """Create an optimized, deployment-ready checkpoint file and return filepath."""
        # Work on a copy of the model reference (do not in-place destroy user's model)
        model_for_deploy = self.model

        # Apply aggressive optimizations if requested
        if config.optimization_level == "aggressive":
            try:
                model_for_deploy = torch.quantization.quantize_dynamic(
                    model_for_deploy,
                    {nn.Linear},
                    dtype=getattr(torch, "qint8", torch.int8),
                )
                logger.info("Applied dynamic quantization.")
            except Exception as e:
                logger.warning("Dynamic quantization failed: %s", e)

        # Try to compile when supported (PyTorch >= 2.x)
        try:
            # Keep original model reference on failure; compile returns a compiled module
            compiled = torch.compile(model_for_deploy, mode="reduce-overhead")
            model_for_deploy = compiled
            logger.info("Model compiled with torch.compile.")
        except Exception as e:
            logger.warning(
                "Model compilation failed or not supported in this runtime: %s", e
            )

        # Save a deployment checkpoint with necessary metadata
        deployment_checkpoint = {
            "model_state_dict": model_for_deploy.state_dict(),
            "model_config": (
                asdict(self.model_config)
                if hasattr(self.model_config, "__dict__")
                or isinstance(self.model_config, dict)
                else asdict(self.model_config)
            ),
            "deployment_config": asdict(config),
            "optimization_applied": config.optimization_level,
            "pytorch_version": torch.__version__,
            "created_at": time.time(),
            "metadata": self.metadata,
        }

        tmp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp_file_path = Path(tmp_file.name)
        tmp_file.close()
        torch.save(deployment_checkpoint, str(tmp_file_path))
        logger.info("Saved optimized checkpoint to %s", tmp_file_path)
        return str(tmp_file_path)

    def _save_tokenizer(self) -> str:
        """Save tokenizer to a temporary file and return its path."""
        tmp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp_path = Path(tmp_file.name)
        tmp_file.close()

        try:
            # Try to call standard saver
            if hasattr(self.tokenizer, "save"):
                self.tokenizer.save(str(tmp_path))
            elif hasattr(self.tokenizer, "to_json") and callable(
                self.tokenizer.to_json
            ):
                tmp_path.write_text(self.tokenizer.to_json())
            else:
                # Fallback: attempt to serialize repr or dict
                try:
                    tmp_path.write_text(
                        json.dumps(self.tokenizer.__dict__, default=str)
                    )
                except Exception:
                    tmp_path.write_text(str(self.tokenizer))
        except Exception as e:
            logger.warning(
                "Failed to save tokenizer cleanly: %s. Writing repr instead.", e
            )
            tmp_path.write_text(repr(self.tokenizer))

        return str(tmp_path)

    def _estimate_model_size(self) -> float:
        """Estimate model size in MB from parameters and buffers."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def _create_deployment_script(self, config: DeploymentConfig) -> str:
        """Return content for a minimal deployment script (deploy.py)."""
        # The script below is intentionally simple: it expects model.pt, tokenizer.json, config.json
        # to be present in the same directory when deployed.
        content = f'''"""
            Deployment wrapper for {config.model_name} v{config.version}

            This is a minimal, self-contained loader + generator. You may need to adapt the import paths
            to match your production package layout (the packager does not include your model source file).
            """
            import json
            from pathlib import Path
            from typing import List

            import torch

            # NOTE: change these imports to the correct module names if your model & tokenizer
            # classes are stored in a package in the deployment bundle.
            try:
                from model import DocumentationModel, ModelConfig
                from dataset import Tokenizer
            except Exception:
                # Fallback: try the original project module names used during packaging
                try:
                    from scripts.model import DocumentationModel, ModelConfig
                    from scripts.dataset import Tokenizer
                except Exception:
                    raise ImportError("Could not import model/tokenizer classes. Adjust import paths in deploy.py.")

            class DeployedModel:
                def __init__(self, model_path: str = "model.pt", tokenizer_path: str = "tokenizer.json", config_path: str = "config.json"):
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    # load package config (for metadata)
                    with open(config_path, "r") as f:
                        self.package_config = json.load(f)

                    # load checkpoint
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    model_cfg_dict = checkpoint.get("model_config", {{}})
                    model_config = ModelConfig(**model_cfg_dict) if model_cfg_dict else ModelConfig()

                    # instantiate model and load state
                    self.model = DocumentationModel(model_config)
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    self.model.to(self.device)
                    self.model.eval()

                    # tokenizer
                    self.tokenizer = Tokenizer.load(tokenizer_path)

                    # simple constants
                    self.eos_token_id = getattr(self.tokenizer, "eos_token_id", 2)

                    print(f"Loaded model with {{self.package_config.get('package_info', {{}}).get('model_parameters', 'unknown')}} parameters.")

                def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.8, top_p: float = 0.9) -> str:
                    tokens = self.tokenizer.encode(prompt)
                    input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
                    generated = input_ids.clone()

                    with torch.no_grad():
                        for _ in range(max_length):
                            logits = self.model(generated)[:, -1, :]  # (batch=1, vocab)
                            logits = logits / max(1e-8, float(temperature))

                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            probs = torch.softmax(sorted_logits, dim=-1)
                            cumulative_probs = torch.cumsum(probs, dim=-1)

                            # top-p filtering on sorted probabilities
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0

                            # scatter back to original indexing
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            logits[indices_to_remove] = float("-inf")

                            probs = torch.softmax(logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)

                            # stop if EOS
                            if next_token.item() == self.eos_token_id:
                                break

                            generated = torch.cat([generated, next_token], dim=1)

                    output_tokens = generated[0].cpu().tolist()[len(tokens):]
                    return self.tokenizer.decode(output_tokens)

                def batch_generate(self, prompts: List[str], max_length: int = 100, temperature: float = 0.8, top_p: float = 0.9):
                    return [self.generate(p, max_length=max_length, temperature=temperature, top_p=top_p) for p in prompts]


            if __name__ == "__main__":
                dm = DeployedModel()
                examples = [
                    "To implement a neural network in PyTorch,",
                    "The key components of a transformer are",
                    "When training deep learning models,"
                ]
                for p in examples:
                    print("=== Prompt ===")
                    print(p)
                    print("=== Generated ===")
                    print(dm.generate(p, max_length=50))
                    print("-" * 40)
            '''
        return content

    def _create_examples_script(self, config: DeploymentConfig) -> str:
        """Return a small examples script (examples.py)."""
        content = f'''
        """
            Example usage of {config.model_name} v{config.version}
            """

            from deploy import DeployedModel

            def basic_generation_example():
                model = DeployedModel()
                prompts = [
                    "PyTorch is a deep learning framework that",
                    "To train a neural network, you need",
                    "The attention mechanism in transformers"
                ]
                for p in prompts:
                    print("INPUT:", p)
                    print("OUTPUT:", model.generate(p, max_length=50, temperature=0.8))
                    print()

            def batch_example():
                model = DeployedModel()
                prompts = [
                    "Machine learning is",
                    "Deep learning models",
                    "Neural networks can"
                ]
                results = model.batch_generate(prompts, max_length=30)
                for p, r in zip(prompts, results):
                    print(p, "->", r)

            if __name__ == "__main__":
                basic_generation_example()
                print("\\n" + "="*60 + "\\n")
                batch_example()
            '''
        return content

    def _create_readme(
        self,
        config: DeploymentConfig,
        num_params: Optional[int],
        model_size_mb: Optional[float],
    ) -> str:
        """Return README content for the package."""
        num_params_str = f"{num_params:,}" if num_params is not None else "unknown"
        model_size_str = (
            f"{model_size_mb:.1f} MB" if model_size_mb is not None else "unknown"
        )

        return f"""
        # {config.model_name} v{config.version}

        {config.description}

        ## Model Information

        - **Parameters**: {num_params_str}
        - **Model Size**: {model_size_str}
        - **PyTorch Version**: {torch.__version__}
        - **Optimization Level**: {config.optimization_level}

        ## Quick Start

        ```python
        from deploy import DeployedModel

        model = DeployedModel()
        result = model.generate("PyTorch is a", max_length=50)
        print(result)
        ````

        ## Files included

        * `model.pt` - deployment checkpoint containing model weights and metadata
        * `tokenizer.json` - tokenizer (format depends on your tokenizer implementation)
        * `config.json` - package configuration and metadata
        * `deploy.py` - minimal loader & generation wrapper
        * `examples.py` - example usage scripts
        """
