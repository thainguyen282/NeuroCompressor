"""
Text Decompression Module

This module implements text decompression using large language models
to reconstruct original text from compressed representations.
"""

import logging
from typing import Optional, Dict, Any
import torch
import re

logger = logging.getLogger(__name__)


class TextDecompressor:
    """
    Decompresses text from LLM-based compressed representations.
    
    The decompressor uses LLMs to reconstruct original text from
    compressed representations with high fidelity.
    """
    
    def __init__(self, model, tokenizer, decompression_strategy: str = "sequence", chunk_size: int = 50,
                 mask_token: str = "[]"):
        """
        Initialize the text decompressor.
        
        Args:
            model: LLM model instance for decompression
            tokenizer: Tokenizer instance (should match the one used in compression)
            decompression_strategy: Strategy to use ('semantic', 'token', 'hybrid')
            mask_token: Custom mask token string to use for decompression (default: "[MASK]")
                       Should match the mask token used during compression
        """
        self.model = model
        self.tokenizer = tokenizer
        self.decompression_strategy = decompression_strategy
        self.chunk_size = chunk_size
        self.mask_token = mask_token
        
        # Add mask token to tokenizer if it doesn't exist (should already exist from compression)
        self._add_mask_token_to_tokenizer()
        
        logger.info(f"Initialized TextDecompressor with strategy: {decompression_strategy}")
    
    def decompress(self, compressed_data: Dict[str, Any], 
                   target_length: Optional[int] = None,
                   preserve_style: bool = True, **kwargs) -> str:
        """
        Decompress text from compressed representation.
        
        Args:
            compressed_data: Dictionary containing compressed text and metadata.
                             Expected keys:
                               - 'compressed': the compressed / masked text
                               - 'strategy' : compression strategy ('semantic', 'sequence', ...)
                               - 'metadata' : optional dict with extra info
            target_length: Target length for decompressed text (if known)
            preserve_style: Whether to preserve original writing style
            **kwargs: Additional decompression parameters
            
        Returns:
            Decompressed text
        """
        strategy = compressed_data.get("strategy", self.decompression_strategy)
        compressed_text = compressed_data["compressed"]
        metadata = compressed_data.get("metadata", {})
        
        if strategy == "tree":
            return self._decompress_semantic(
                compressed_text, target_length, preserve_style, metadata, **kwargs
            )
        elif strategy == "sequence":
            return self._decompress_sequence(
                compressed_text, **kwargs
            )

    def _add_mask_token_to_tokenizer(self):
        """
        Add the mask token to the tokenizer if it doesn't exist.
        This ensures the mask token is available for decompression.
        
        Note: The mask token should match the one used during compression.
        If loading a saved tokenizer, it should already contain the mask token.
        """
        # Check if mask token already exists
        if hasattr(self.tokenizer, 'mask_token') and self.tokenizer.mask_token == self.mask_token:
            # Token already exists, just get its ID
            return
        
        # Add the mask token to the tokenizer
        num_added = self.tokenizer.add_tokens([self.mask_token], special_tokens=True)
        
        if num_added > 0:
            # Resize model embeddings if needed (for models that use token embeddings)
            try:
                if hasattr(self.model, 'resize_token_embeddings'):
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    logger.info(f"Resized model embeddings to accommodate new mask token")
            except Exception as e:
                logger.warning(f"Could not resize model embeddings: {e}")
            
            logger.info(f"Added mask token '{self.mask_token}' to tokenizer")
        else:
            logger.info(f"Mask token '{self.mask_token}' already exists in tokenizer")

    def _get_mask_token_id(self) -> int:
        """
        Helper to get the token id used as the mask token in compressed text.
        Uses the dedicated mask token that was added to the tokenizer.
        Mirrors the logic in TextCompressor.
        """
        try:
            # Get the mask token ID from the tokenizer
            mask_token_id = self.tokenizer.convert_tokens_to_ids(self.mask_token)
            if mask_token_id != self.tokenizer.unk_token_id:
                return mask_token_id
        except Exception:
            pass
        
        # Fallback: try encoding the mask token directly
        try:
            mask_encoded = self.tokenizer.encode(self.mask_token, add_special_tokens=False)
            if len(mask_encoded) > 0:
                return mask_encoded[0]
        except Exception:
            pass

        # Final safety fallbacks
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            return self.tokenizer.pad_token_id
        if getattr(self.tokenizer, "unk_token_id", None) is not None:
            return self.tokenizer.unk_token_id
        return 0

    def _get_special_token_ids(self) -> set:
        """
        Collect all special token ids we want to preserve (not change).
        Mirrors the logic in TextCompressor.
        """
        special_token_ids = set()
        if getattr(self.tokenizer, "bos_token_id", None) is not None:
            special_token_ids.add(self.tokenizer.bos_token_id)
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            special_token_ids.add(self.tokenizer.eos_token_id)
        if getattr(self.tokenizer, "cls_token_id", None) is not None:
            special_token_ids.add(self.tokenizer.cls_token_id)
        if getattr(self.tokenizer, "sep_token_id", None) is not None:
            special_token_ids.add(self.tokenizer.sep_token_id)
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            special_token_ids.add(self.tokenizer.pad_token_id)
        return special_token_ids

    def _is_void_token(self, token_id: int) -> bool:
        """
        Check if a token is a void token (whitespace, punctuation, formatting).
        These tokens should not be treated as content tokens.
        Mirrors the logic in TextCompressor.
        """
        try:
            # Never treat the mask token as void
            if token_id == self._get_mask_token_id():
                return False
            
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            stripped = token_text.strip()
            if not stripped:
                return True

            if re.match(r'^[\s\.,;:!?\-\(\)\{\}\'\"\n\r\t]+$', token_text):
                return True

            if token_text.startswith((' ', '\n', '\t', '\r')) and len(stripped) <= 2:
                if re.match(r'^[\s\.,;:!?\-\(\)\[\]\{\}\'\"\n\r\t]+$', stripped):
                    return True

            return False
        except Exception:
            return False

    def _decompress_sequence(self, text: str, **kwargs) -> str:
        """
        Sequential decompression:
        For each token position, if the token is a mask token, generate the next token from the prefix.
        - If the token is not the mask token, keep it.
        - If the token is the mask token, generate the next token from the already reconstructed prefix.
        """
        device = next(self.model.parameters()).device
        tokenized_text = self.tokenizer(text, return_tensors="pt")
        input_ids = tokenized_text.input_ids.to(device)

        # Determine mask token id and special tokens (reuse helpers)
        mask_token_id = self._get_mask_token_id()
        special_token_ids = self._get_special_token_ids()

        # Ensure batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # We modify input_ids in-place; no need for a separate copy

        with torch.no_grad():
            for batch_idx in range(input_ids.shape[0]):
                seq = input_ids[batch_idx]
                seq_len = seq.shape[0]

                # Skip leading/trailing special tokens
                start_idx = 0
                end_idx = seq_len
                while start_idx < seq_len and seq[start_idx].item() in special_token_ids:
                    start_idx += 1
                while end_idx > start_idx and seq[end_idx - 1].item() in special_token_ids:
                    end_idx -= 1

                for i in range(start_idx, end_idx):
                    comp_token_id = seq[i].item()

                    # Skip specials / void tokens
                    if comp_token_id in special_token_ids or self._is_void_token(comp_token_id):
                        continue

                    # Prefix up to (but excluding) position i
                    prefix = seq[max(0, i - self.chunk_size):i].unsqueeze(0)  # [1, i]
                    if prefix.numel() == 0:
                        # No context to predict from; can't safely mask
                        continue

                    attention_mask = torch.ones_like(prefix, device=device)
                    if comp_token_id == mask_token_id:
                        # Generate the next token (greedy) using the already reconstructed prefix
                        generated = self.model.generate(
                            prefix,
                            attention_mask=attention_mask,
                            max_new_tokens=1,
                            do_sample=False,
                            pad_token_id=(
                                self.tokenizer.pad_token_id
                                if self.tokenizer.pad_token_id is not None
                                else self.tokenizer.eos_token_id
                            ),
                        )
                        seq[i] = generated[0, -1].item()

        # Assume a single sequence and decode it back to text
        reconstructed_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return reconstructed_text
    
    def _create_decompression_prompt(self, compressed_text: str,
                                    target_length: int, preserve_style: bool) -> str:
        """Create a prompt for LLM-based decompression."""
        style_instruction = "Maintain the original writing style, tone, and voice." if preserve_style else ""
        
        prompt = f"""Expand the following compressed text to approximately {target_length} characters, restoring all details and information while maintaining coherence and readability. {style_instruction}

Compressed text:
{compressed_text}

Expanded text:"""
        
        return prompt
    
    def evaluate_reconstruction(self, original: str, decompressed: str) -> Dict[str, float]:
        """
        Evaluate the quality of decompression.
        
        Args:
            original: Original text
            decompressed: Decompressed text
            
        Returns:
            Dictionary with evaluation metrics
        """
        from neurocompressor.metrics import calculate_metrics
        
        return calculate_metrics(original, decompressed)

