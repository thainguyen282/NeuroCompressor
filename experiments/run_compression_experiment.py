import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurocompressor import TextCompressor, TextDecompressor, LLMModel
from neurocompressor.utils import setup_logging, load_config, save_results, read_text_file
from neurocompressor.metrics import calculate_metrics, print_metrics


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
    model = LLMModel(**config.get("model", {}))
    compressor = TextCompressor(model=model, **config.get("compression", {}))
    
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
    
    # Initialize decompressor
    decompression_config = config.get("decompression", {})
    decompressor = TextDecompressor(
        model=model,
        decompression_strategy=decompression_config.get("strategy", "semantic")
    )
    
    # Decompress text
    print("\nDecompressing text...")
    decompressed_text = decompressor.decompress(
        compression_result,
        target_length=compression_result.get("original_length"),
        preserve_style=decompression_config.get("preserve_style", True)
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

