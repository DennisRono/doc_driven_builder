import sys
from pathlib import Path
from model_inference import ModelLoader, TextGenerator
from model_evaluation import ModelEvaluator
from model_deployment import ModelPackager
from dataset import Tokenizer

CHECKPOINT = Path('output/best_checkpoint.pt')
TOKENIZER = "path/to/tokenizer.json"
OUTPUT = "output_package"
TEST_TEXTS = ["This is a test.", "Another test sentence."]
PROMPT = "Hello world, generate text about AI"

def run_inference():
    model, config, meta = ModelLoader.loadmodelfromcheckpoint(CHECKPOINT)
    tokenizer = Tokenizer.loadstr(TOKENIZER)
    gen = TextGenerator(model, tokenizer)
    text, _, _, _ = gen.generate(PROMPT)
    print("Generated Text:", text)

def run_evaluation():
    model, config, meta = ModelLoader.loadmodelfromcheckpoint(CHECKPOINT)
    tokenizer = Tokenizer.loadstr(TOKENIZER)
    evaluator = ModelEvaluator(model, tokenizer)
    perplexity = evaluator.evaluateperplexity(TEST_TEXTS)
    print("Perplexity:", perplexity)

def run_package():
    packager = ModelPackager(CHECKPOINT, TOKENIZER)
    path = packager.createdeploymentpackage(OUTPUT)
    print("Deployment package created at:", path)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/usage.py [infer|evaluate|package]")
        sys.exit(1)
    command = sys.argv[1].lower()
    if command == "infer":
        run_inference()
    elif command == "evaluate":
        run_evaluation()
    elif command == "package":
        run_package()
    else:
        print("Unknown command:", command)
        print("Usage: python usage.py [infer|evaluate|package]")
        sys.exit(1)

if __name__ == "__main__":
    main()
