from neurocompressor import TextCompressor, TextDecompressor, LLMModel
from neurocompressor.utils import setup_logging
from neurocompressor.metrics import calculate_metrics, print_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        
        # Load model and tokenizer externally
        model_name = "/project/phan/tqn/safety_alignment/Qwen3-8b"
        print(f"Loading model: {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Model loaded successfully.\n")
        
        # Initialize compressor with external model and tokenizer
        compressor = TextCompressor(
            model=model,
            tokenizer=tokenizer,
            compression_strategy="hierarchical",
        )
        # Compress text
        print("Compressing text...")
        compression_result = compressor.compress(
            sample_text,
            compression_ratio=0.5,
            preserve_format=True
        )
        # export to text file
        compressed_file = "compressed_text_hierarchical.txt"
        with open(compressed_file, "w", encoding="utf-8") as f:
            f.write(compression_result["compressed"])
        print(f"\nCompressed text:")
        print(f"Compressed text: {compression_result['compressed']}")
        print(f"Compression ratio: {compression_result['compression_ratio']:.4f}")
        print(f"Saved compressed text to {compressed_file}\n")
        
        # Initialize decompressor
        decompressor = TextDecompressor(
            model=model,
            tokenizer=tokenizer,
            decompression_strategy="sequence"
        )
        
        # Option 1: Decompress from compression_result dict
        print("Decompressing from compression_result dict...")
        decompressed_text = decompressor.decompress(
            {
                "compressed": compression_result["compressed"],
                "strategy": "sequence",
                "metadata": {},
            }
        )
        
        # # Option 2: Decompress from text file
        # print("\nDecompressing from text file...")
        # with open(compressed_file, "r", encoding="utf-8") as f:
        #     compressed_text_from_file = f.read()
        
        # decompressed_text_from_file = decompressor.decompress(
        #     {
        #         "compressed": compressed_text_from_file,
        #         "strategy": "sequence",
        #         "metadata": {},
        #     }
        # )
        # export to text file
        decompressed_file = "decompressed_text_hierarchical.txt"
        with open(decompressed_file, "w", encoding="utf-8") as f:
            f.write(decompressed_text)
        print(f"\nDecompressed text (from file):")
        print(decompressed_text)
        print(f"Length: {len(decompressed_text)} characters")
        print(f"Saved decompressed text to {decompressed_file}\n")
        exit()
        # Evaluate quality
        print("Evaluating reconstruction quality...")
        metrics = calculate_metrics(sample_text, decompressed_text)
        print_metrics(metrics)
        
        # Save compressed data
        compressor.save(compression_result, "compressed_data.json")
        print("Saved compressed data to compressed_data.json")
if __name__ == "__main__":
    main()
