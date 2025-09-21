import torch
import torch.nn.functional as F
from pathlib import Path
import json
import logging
from typing import Dict, List, Union, Tuple
from dataclasses import dataclass
import time
import numpy as np

from model import DocumentationModel, ModelConfig
from dataset import Tokenizer

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    early_stopping: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 2
    min_length: int = 10

class ModelLoader:
    """Utility class for loading trained models from checkpoints."""
    
    @staticmethod
    def load_model_from_checkpoint(checkpoint_path: str, device: str = 'auto') -> Tuple[DocumentationModel, ModelConfig, Dict]:
        """Load model from checkpoint with full configuration."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model_config_dict = checkpoint.get('model_config', {})
        model_config = ModelConfig(**model_config_dict)
        
        model = DocumentationModel(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        try:
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
        
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {}),
            'training_config': checkpoint.get('training_config', {}),
            'timestamp': checkpoint.get('timestamp', 0)
        }
        
        logger.info(f"Model loaded successfully with {model.get_num_params():,} parameters")
        return model, model_config, metadata

class TextGenerator:
    """Advanced text generation with multiple sampling strategies."""
    
    def __init__(self, model: DocumentationModel, tokenizer: Tokenizer, 
                 device: str = 'auto'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def generate(self, prompt: str, config: GenerationConfig = None) -> Dict[str, Union[str, float, int]]:
        """Generate text with comprehensive configuration options."""
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        
        input_tokens = self.tokenizer.encode(prompt)
        if len(input_tokens) > self.model.config.max_length - config.max_length:
            input_tokens = input_tokens[-(self.model.config.max_length - config.max_length):]
        
        input_ids = torch.tensor([input_tokens], device=self.device)
        
        if config.num_beams > 1:
            generated_ids = self._beam_search(input_ids, config)
        else:
            generated_ids = self._sampling_generate(input_ids, config)
        
        generated_tokens = generated_ids[0].cpu().tolist()
        generated_text = self.tokenizer.decode(generated_tokens[len(input_tokens):])
        
        generation_time = time.time() - start_time
        
        return {
            'generated_text': generated_text,
            'full_text': prompt + generated_text,
            'input_length': len(input_tokens),
            'output_length': len(generated_tokens) - len(input_tokens),
            'generation_time': generation_time,
            'tokens_per_second': (len(generated_tokens) - len(input_tokens)) / generation_time
        }
    
    def _sampling_generate(self, input_ids: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """Generate using sampling strategies (top-k, top-p, temperature)."""
        generated = input_ids.clone()
        past_tokens = set()
        
        with torch.no_grad():
            for _ in range(config.max_length):
                logits = self.model(generated)[:, -1, :]
                
                if config.repetition_penalty != 1.0:
                    for token_id in past_tokens:
                        logits[0, token_id] /= config.repetition_penalty
                
                if config.temperature != 1.0:
                    logits = logits / config.temperature
                
                if config.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, config.top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                if config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                if config.do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                if next_token.item() == config.eos_token_id:
                    if generated.size(1) >= config.min_length:
                        break
                
                generated = torch.cat([generated, next_token], dim=1)
                past_tokens.add(next_token.item())
                
                if generated.size(1) >= self.model.config.max_length:
                    break
        
        return generated
    
    def _beam_search(self, input_ids: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """Generate using beam search for more coherent outputs."""
        batch_size = input_ids.size(0)
        beam_size = config.num_beams
        vocab_size = self.model.config.vocab_size
        
        beams = input_ids.repeat(beam_size, 1)
        beam_scores = torch.zeros(beam_size, device=self.device)
        beam_scores[1:] = float('-inf')
        
        finished_beams = []
        
        with torch.no_grad():
            for step in range(config.max_length):
                logits = self.model(beams)[:, -1, :]
                
                if config.length_penalty != 1.0:
                    length_penalty = ((5 + beams.size(1)) / 6) ** config.length_penalty
                    scores = F.log_softmax(logits, dim=-1) / length_penalty
                else:
                    scores = F.log_softmax(logits, dim=-1)
                
                scores = beam_scores.unsqueeze(1) + scores
                
                scores_flat = scores.view(-1)
                top_scores, top_indices = torch.topk(scores_flat, beam_size)
                
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size
                
                new_beams = []
                new_scores = []
                
                for i in range(beam_size):
                    beam_idx = beam_indices[i]
                    token_idx = token_indices[i]
                    score = top_scores[i]
                    
                    new_beam = torch.cat([beams[beam_idx], token_idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    
                    if token_idx == config.eos_token_id and new_beam.size(1) >= config.min_length:
                        finished_beams.append((new_beam, score))
                    else:
                        new_beams.append(new_beam)
                        new_scores.append(score)
                
                if len(new_beams) == 0:
                    break
                
                beams = torch.stack(new_beams[:beam_size])
                beam_scores = torch.tensor(new_scores[:beam_size], device=self.device)
                
                if config.early_stopping and len(finished_beams) >= beam_size:
                    break
        
        if finished_beams:
            best_beam, _ = max(finished_beams, key=lambda x: x[1])
            return best_beam.unsqueeze(0)
        else:
            best_idx = torch.argmax(beam_scores)
            return beams[best_idx].unsqueeze(0)

class ModelAnalyzer:
    """Comprehensive model analysis and introspection tools."""
    
    def __init__(self, model: DocumentationModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def analyze_attention_patterns(self, text: str, layer_idx: int = -1) -> Dict:
        """Analyze attention patterns for given text."""
        tokens = self.tokenizer.encode(text)
        input_ids = torch.tensor([tokens], device=self.device)
        
        attention_weights = []
        
        def attention_hook(module, input, output):
            if hasattr(module, 'attention'):
                attention_weights.append(output)
        
        hooks = []
        if layer_idx == -1:
            for block in self.model.blocks:
                hooks.append(block.attention.register_forward_hook(attention_hook))
        else:
            hooks.append(self.model.blocks[layer_idx].attention.register_forward_hook(attention_hook))
        
        with torch.no_grad():
            _ = self.model(input_ids)
        
        for hook in hooks:
            hook.remove()
        
        return {
            'tokens': [self.tokenizer.decode([t]) for t in tokens],
            'attention_weights': attention_weights,
            'layer_analyzed': layer_idx if layer_idx != -1 else 'all'
        }
    
    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity for given text."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 2:
            return float('inf')
        
        input_ids = torch.tensor([tokens[:-1]], device=self.device)
        target_ids = torch.tensor([tokens[1:]], device=self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction='mean'
            )
        
        return torch.exp(loss).item()
    
    def get_model_statistics(self) -> Dict:
        """Get comprehensive model statistics."""
        total_params = self.model.get_num_params()
        trainable_params = self.model.get_num_trainable_params()
        
        param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in self.model.buffers())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'parameter_memory_mb': param_memory / (1024 * 1024),
            'buffer_memory_mb': buffer_memory / (1024 * 1024),
            'total_memory_mb': (param_memory + buffer_memory) / (1024 * 1024),
            'model_config': self.model.config.__dict__,
            'device': str(self.device)
        }

class BatchInference:
    """Efficient batch inference for processing multiple texts."""
    
    def __init__(self, model: DocumentationModel, tokenizer: Tokenizer, 
                 batch_size: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
    
    def process_batch(self, texts: List[str], config: GenerationConfig = None) -> List[Dict]:
        """Process multiple texts in batches for efficiency."""
        if config is None:
            config = GenerationConfig()
        
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self._process_single_batch(batch_texts, config)
            results.extend(batch_results)
        
        return results
    
    def _process_single_batch(self, texts: List[str], config: GenerationConfig) -> List[Dict]:
        """Process a single batch of texts."""
        tokenized_inputs = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > self.model.config.max_length - config.max_length:
                tokens = tokens[-(self.model.config.max_length - config.max_length):]
            tokenized_inputs.append(tokens)
        
        max_len = max(len(tokens) for tokens in tokenized_inputs)
        padded_inputs = []
        attention_masks = []
        
        for tokens in tokenized_inputs:
            padding_length = max_len - len(tokens)
            padded_tokens = [config.pad_token_id] * padding_length + tokens
            attention_mask = [0] * padding_length + [1] * len(tokens)
            
            padded_inputs.append(padded_tokens)
            attention_masks.append(attention_mask)
        
        input_ids = torch.tensor(padded_inputs, device=self.device)
        attention_mask = torch.tensor(attention_masks, device=self.device)
        
        results = []
        with torch.no_grad():
            for i in range(len(texts)):
                single_input = input_ids[i:i+1]
                single_mask = attention_mask[i:i+1]
                
                generated = single_input.clone()
                original_length = single_mask.sum().item()
                
                for _ in range(config.max_length):
                    logits = self.model(generated, single_mask)[:, -1, :]
                    
                    if config.do_sample:
                        probs = F.softmax(logits / config.temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    if next_token.item() == config.eos_token_id:
                        break
                    
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    single_mask = torch.cat([single_mask, torch.ones(1, 1, device=self.device)], dim=1)
                
                generated_tokens = generated[0].cpu().tolist()
                input_tokens = generated_tokens[:original_length]
                output_tokens = generated_tokens[original_length:]
                
                input_text = self.tokenizer.decode(input_tokens)
                output_text = self.tokenizer.decode(output_tokens)
                
                results.append({
                    'input_text': input_text,
                    'generated_text': output_text,
                    'full_text': input_text + output_text,
                    'input_length': len(input_tokens),
                    'output_length': len(output_tokens)
                })
        
        return results

def load_and_use_model(checkpoint_path: str, prompt: str, 
                      generation_config: GenerationConfig = None) -> Dict:
    """Convenience function to load model and generate text in one call."""
    model, model_config, metadata = ModelLoader.load_model_from_checkpoint(checkpoint_path)
    
    tokenizer_path = Path(checkpoint_path).parent / 'tokenizer.json'
    if tokenizer_path.exists():
        tokenizer = Tokenizer.load(str(tokenizer_path))
    else:
        logger.warning("Tokenizer not found, creating new one")
        tokenizer = Tokenizer()
    
    generator = TextGenerator(model, tokenizer)
    result = generator.generate(prompt, generation_config)
    
    result['model_metadata'] = metadata
    result['model_config'] = model_config.__dict__
    
    return result
