from neurocompressor import TextCompressor, TextDecompressor, LLMModel
from neurocompressor.utils import setup_logging
from neurocompressor.metrics import calculate_metrics, print_metrics


def main():
    # Sample text to compress
    sample_text = """
    Large language models have revolutionized natural language processing by demonstrating
    remarkable capabilities in understanding and generating human-like text. These models,
    trained on vast amounts of textual data, can perform a wide range of tasks including
    translation, summarization, question answering, and creative writing. The key to their
    success lies in their ability to capture complex patterns and relationships in language,
    enabling them to generate coherent and contextually appropriate responses.
    """
    print("Original text:")
    print(sample_text)
    print(f"Length: {len(sample_text)} characters\n")
    
    # Initialize LLM model
    # Note: You'll need to set your API key via environment variable or pass it directly
    model = LLMModel(
        backend="openai",
        model_name="gpt-3.5-turbo",
        api_key=None  # Will use OPENAI_API_KEY environment variable
    )
    # Initialize compressor
    compressor = TextCompressor(
        model=model,
        compression_strategy="semantic",
        chunk_size=500,
        overlap=50
    )
    # Compress text
    print("Compressing text...")
    compression_result = compressor.compress(
        sample_text,
        compression_ratio=0.5,
        preserve_format=True
    )
    
    print(f"\nCompressed text:")
    print(compression_result["compressed"])
    print(f"Length: {compression_result['compressed_length']} characters")
    print(f"Compression ratio: {compression_result['compression_ratio']:.4f}\n")
    
    # Initialize decompressor
    decompressor = TextDecompressor(
        model=model,
        decompression_strategy="semantic"
    )
    
    # Decompress text
    print("Decompressing text...")
    decompressed_text = decompressor.decompress(
        compression_result,
        target_length=compression_result.get("original_length"),
        preserve_style=True
    )
    
    print(f"\nDecompressed text:")
    print(decompressed_text)
    print(f"Length: {len(decompressed_text)} characters\n")
    
    # Evaluate quality
    print("Evaluating reconstruction quality...")
    metrics = calculate_metrics(sample_text, decompressed_text)
    print_metrics(metrics)
    
    # Save compressed data
    compressor.save(compression_result, "compressed_data.json")
    print("Saved compressed data to compressed_data.json")


if __name__ == "__main__":
    main()

