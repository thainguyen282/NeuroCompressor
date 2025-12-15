import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurocompressor import TextCompressor, TextDecompressor
from neurocompressor.utils import setup_logging, load_config, save_results, read_text_file
from neurocompressor.metrics import calculate_metrics, print_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Run compression experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input text file")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    config = load_config(args.config)
    input_text = read_text_file(args.input)
    print(f"Loaded input text: {len(input_text)} characters")
    
    # Load model and tokenizer externally
    model_config = config.get("model", {})
    model_name = model_config.get("model_name", "gpt2")  # Default fallback
    print(f"Loading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Model loaded successfully.")
    
    # Initialize compressor with external model and tokenizer
    compression_config = config.get("compression", {})
    compressor = TextCompressor(
        model=model,
        tokenizer=tokenizer,
        **compression_config
    )
    
    print("\nCompressing text...")
    compression_result = compressor.compress(
        input_text,
        compression_ratio=compression_config.get("compression_ratio", 0.5),
        preserve_format=compression_config.get("preserve_format", True)
    )
    
    print(f"Compression completed:")
    print(f"  Original length: {compression_result['original_length']} characters")
    print(f"  Compressed length: {compression_result['compressed_length']} characters")
    print(f"  Compression ratio: {compression_result['compression_ratio']:.4f}")
    
    # Initialize decompressor (Note: TextDecompressor may need updating to accept model/tokenizer)
    decompression_config = config.get("decompression", {})
    # For now, decompressor can use the compressor's decompress method
    # Or update TextDecompressor similarly if needed
    
    # Decompress text using compressor's decompress method
    print("\nDecompressing text...")
    decompressed_text = compressor.decompress(
        compression_result,
        max_new_tokens=None
    )
    
    print(f"Decompression completed:")
    print(f"  Decompressed length: {len(decompressed_text)} characters")
    
    # Evaluate reconstruction
    print("\nEvaluating reconstruction quality...")
    metrics = calculate_metrics(input_text, decompressed_text)
    print_metrics(metrics)
    
    # Save results
    results = {
        "experiment_config": config,
        "compression_result": compression_result,
        "decompressed_text": decompressed_text,
        "evaluation_metrics": metrics,
        "input_file": args.input
    }
    
    save_results(results, args.output)
    
    # Save compressed and decompressed texts
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "compressed.txt", 'w', encoding='utf-8') as f:
        f.write(compression_result["compressed"])
    
    with open(output_path / "decompressed.txt", 'w', encoding='utf-8') as f:
        f.write(decompressed_text)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()

