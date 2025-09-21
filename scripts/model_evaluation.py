import torch
import torch.nn.functional as F
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
import time
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from model import DocumentationModel, ModelConfig
from dataset import Tokenizer
from model_inference import ModelLoader, TextGenerator, GenerationConfig

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    perplexity: float
    bleu_score: float
    rouge_scores: Dict[str, float]
    generation_speed: float
    memory_usage: float
    accuracy_metrics: Dict[str, float]
    sample_generations: List[Dict[str, str]]

class ModelEvaluator:
    """Comprehensive model evaluation suite."""
    
    def __init__(self, model: DocumentationModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def evaluate_perplexity(self, test_texts: List[str]) -> float:
        """Evaluate model perplexity on test texts."""
        total_loss = 0.0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for text in test_texts:
                tokens = self.tokenizer.encode(text)
                if len(tokens) < 2:
                    continue
                
                # Split into input and target
                input_tokens = tokens[:-1]
                target_tokens = tokens[1:]
                
                # Process in chunks if too long
                chunk_size = self.model.config.max_length - 1
                for i in range(0, len(input_tokens), chunk_size):
                    input_chunk = input_tokens[i:i + chunk_size]
                    target_chunk = target_tokens[i:i + chunk_size]
                    
                    if len(input_chunk) == 0:
                        continue
                    
                    input_ids = torch.tensor([input_chunk], device=self.device)
                    target_ids = torch.tensor([target_chunk], device=self.device)
                    
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        target_ids.reshape(-1),
                        reduction='sum'
                    )
                    
                    total_loss += loss.item()
                    total_tokens += len(target_chunk)
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def evaluate_generation_quality(self, prompts: List[str], 
                                  reference_texts: List[str] = None,
                                  config: GenerationConfig = None) -> Dict:
        """Evaluate generation quality with multiple metrics."""
        if config is None:
            config = GenerationConfig()
        
        generator = TextGenerator(self.model, self.tokenizer)
        
        generations = []
        generation_times = []
        
        for prompt in prompts:
            start_time = time.time()
            result = generator.generate(prompt, config)
            generation_time = time.time() - start_time
            
            generations.append(result['generated_text'])
            generation_times.append(generation_time)
        
        # Calculate metrics
        avg_generation_time = np.mean(generation_times)
        avg_tokens_per_second = np.mean([
            len(self.tokenizer.encode(gen)) / time 
            for gen, time in zip(generations, generation_times)
        ])
        
        results = {
            'average_generation_time': avg_generation_time,
            'tokens_per_second': avg_tokens_per_second,
            'generations': generations,
            'prompts': prompts
        }
        
        # Add reference-based metrics if available
        if reference_texts:
            results.update(self._compute_reference_metrics(generations, reference_texts))
        
        return results
    
    def _compute_reference_metrics(self, generations: List[str], 
                                 references: List[str]) -> Dict:
        """Compute reference-based metrics like BLEU and ROUGE."""
        # Simplified BLEU calculation (would use nltk.translate.bleu_score in practice)
        bleu_scores = []
        for gen, ref in zip(generations, references):
            gen_tokens = set(gen.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if len(ref_tokens) == 0:
                bleu_scores.append(0.0)
            else:
                precision = len(gen_tokens & ref_tokens) / max(len(gen_tokens), 1)
                recall = len(gen_tokens & ref_tokens) / len(ref_tokens)
                
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
                
                bleu_scores.append(f1)
        
        return {
            'bleu_score': np.mean(bleu_scores),
            'bleu_scores_individual': bleu_scores
        }
    
    def benchmark_performance(self, test_texts: List[str], 
                            batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict:
        """Benchmark model performance across different batch sizes."""
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking batch size: {batch_size}")
            
            # Prepare batches
            batches = [test_texts[i:i + batch_size] 
                      for i in range(0, len(test_texts), batch_size)]
            
            times = []
            memory_usage = []
            
            for batch in batches:
                # Measure memory before
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated()
                
                # Time the batch processing
                start_time = time.time()
                
                with torch.no_grad():
                    for text in batch:
                        tokens = self.tokenizer.encode(text)
                        if len(tokens) > self.model.config.max_length:
                            tokens = tokens[:self.model.config.max_length]
                        
                        input_ids = torch.tensor([tokens], device=self.device)
                        _ = self.model(input_ids)
                
                batch_time = time.time() - start_time
                times.append(batch_time / len(batch))  # Time per sample
                
                # Measure memory after
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated()
                    memory_usage.append((memory_after - memory_before) / (1024 * 1024))  # MB
            
            results[batch_size] = {
                'avg_time_per_sample': np.mean(times),
                'std_time_per_sample': np.std(times),
                'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
                'throughput_samples_per_second': 1.0 / np.mean(times)
            }
        
        return results
    
    def analyze_model_behavior(self, test_prompts: List[str]) -> Dict:
        """Analyze model behavior patterns and characteristics."""
        generator = TextGenerator(self.model, self.tokenizer)
        
        # Test different generation strategies
        configs = {
            'greedy': GenerationConfig(do_sample=False, temperature=1.0),
            'sampling': GenerationConfig(do_sample=True, temperature=0.8),
            'high_temp': GenerationConfig(do_sample=True, temperature=1.2),
            'low_temp': GenerationConfig(do_sample=True, temperature=0.5),
            'top_k': GenerationConfig(do_sample=True, top_k=20, temperature=0.8),
            'top_p': GenerationConfig(do_sample=True, top_p=0.7, temperature=0.8)
        }
        
        results = {}
        
        for strategy_name, config in configs.items():
            strategy_results = []
            
            for prompt in test_prompts:
                result = generator.generate(prompt, config)
                
                # Analyze generation characteristics
                generated_text = result['generated_text']
                
                # Basic statistics
                word_count = len(generated_text.split())
                char_count = len(generated_text)
                avg_word_length = np.mean([len(word) for word in generated_text.split()]) if word_count > 0 else 0
                
                # Repetition analysis
                words = generated_text.lower().split()
                unique_words = len(set(words))
                repetition_ratio = 1 - (unique_words / max(word_count, 1))
                
                strategy_results.append({
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'word_count': word_count,
                    'char_count': char_count,
                    'avg_word_length': avg_word_length,
                    'repetition_ratio': repetition_ratio,
                    'generation_time': result['generation_time']
                })
            
            # Aggregate statistics for this strategy
            results[strategy_name] = {
                'individual_results': strategy_results,
                'avg_word_count': np.mean([r['word_count'] for r in strategy_results]),
                'avg_char_count': np.mean([r['char_count'] for r in strategy_results]),
                'avg_repetition_ratio': np.mean([r['repetition_ratio'] for r in strategy_results]),
                'avg_generation_time': np.mean([r['generation_time'] for r in strategy_results])
            }
        
        return results

def comprehensive_model_evaluation(checkpoint_path: str, test_data_path: str = None) -> Dict:
    """Run comprehensive evaluation on a trained model."""
    logger.info(f"Starting comprehensive evaluation of {checkpoint_path}")
    
    # Load model
    model, model_config, metadata = ModelLoader.load_model_from_checkpoint(checkpoint_path)
    
    # Load tokenizer
    tokenizer_path = Path(checkpoint_path).parent / 'tokenizer.json'
    if tokenizer_path.exists():
        tokenizer = Tokenizer.load(str(tokenizer_path))
    else:
        logger.warning("Tokenizer not found, creating new one")
        tokenizer = Tokenizer()
    
    evaluator = ModelEvaluator(model, tokenizer)
    
    # Load test data
    if test_data_path and Path(test_data_path).exists():
        with open(test_data_path, 'r') as f:
            test_texts = [line.strip() for line in f if line.strip()]
    else:
        # Use sample test texts
        test_texts = [
            "PyTorch is a machine learning framework",
            "Neural networks consist of layers of interconnected nodes",
            "Training a model requires data, loss function, and optimizer",
            "Transformers use attention mechanisms for sequence processing",
            "Documentation should be clear and comprehensive"
        ]
    
    # Test prompts for generation evaluation
    test_prompts = [
        "To train a neural network, you need to",
        "The attention mechanism works by",
        "PyTorch tensors are",
        "When debugging machine learning models",
        "Best practices for documentation include"
    ]
    
    results = {
        'model_metadata': metadata,
        'model_config': model_config.__dict__,
        'evaluation_timestamp': time.time()
    }
    
    # 1. Perplexity evaluation
    logger.info("Evaluating perplexity...")
    results['perplexity'] = evaluator.evaluate_perplexity(test_texts)
    
    # 2. Generation quality evaluation
    logger.info("Evaluating generation quality...")
    results['generation_quality'] = evaluator.evaluate_generation_quality(test_prompts)
    
    # 3. Performance benchmarking
    logger.info("Benchmarking performance...")
    results['performance_benchmark'] = evaluator.benchmark_performance(test_texts[:20])
    
    # 4. Behavior analysis
    logger.info("Analyzing model behavior...")
    results['behavior_analysis'] = evaluator.analyze_model_behavior(test_prompts[:3])
    
    # 5. Model statistics
    from model_inference import ModelAnalyzer
    analyzer = ModelAnalyzer(model, tokenizer)
    results['model_statistics'] = analyzer.get_model_statistics()
    
    logger.info("Comprehensive evaluation completed")
    return results

def save_evaluation_report(results: Dict, output_path: str):
    """Save evaluation results to a comprehensive report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create markdown report
    md_path = output_path.with_suffix('.md')
    with open(md_path, 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        
        # Model info
        f.write("## Model Information\n")
        f.write(f"- **Checkpoint**: {results.get('model_metadata', {}).get('epoch', 'N/A')} epochs\n")
        f.write(f"- **Parameters**: {results.get('model_statistics', {}).get('total_parameters', 'N/A'):,}\n")
        f.write(f"- **Model Size**: {results.get('model_statistics', {}).get('total_memory_mb', 'N/A'):.1f} MB\n\n")
        
        # Performance metrics
        f.write("## Performance Metrics\n")
        f.write(f"- **Perplexity**: {results.get('perplexity', 'N/A'):.2f}\n")
        
        gen_quality = results.get('generation_quality', {})
        f.write(f"- **Generation Speed**: {gen_quality.get('tokens_per_second', 'N/A'):.1f} tokens/sec\n")
        f.write(f"- **Average Generation Time**: {gen_quality.get('average_generation_time', 'N/A'):.3f}s\n\n")
        
        # Sample generations
        f.write("## Sample Generations\n")
        generations = gen_quality.get('generations', [])
        prompts = gen_quality.get('prompts', [])
        
        for prompt, generation in zip(prompts[:3], generations[:3]):
            f.write(f"**Prompt**: {prompt}\n")
            f.write(f"**Generated**: {generation}\n\n")
    
    logger.info(f"Evaluation report saved to {json_path} and {md_path}")
