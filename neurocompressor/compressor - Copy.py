import logging
import torch
import re
from typing import Optional, Dict, Any, List, Tuple
import json
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class TextCompressor:
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                 compression_strategy: str = "tree", chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize TextCompressor with external model and tokenizer.
        
        Args:
            model: Pre-loaded AutoModelForCausalLM instance
            tokenizer: Pre-loaded AutoTokenizer instance
            compression_strategy: Strategy to use ('tree', 'semantic', etc.)
            chunk_size: Size of text chunks for processing
            overlap: Overlap between chunks
        """
        self.model = model
        self.tokenizer = tokenizer
        self.compression_strategy = compression_strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        logger.info(f"Initialized TextCompressor with strategy: {compression_strategy}")
    def compress(self, text: str, compression_ratio: float = 0.5,
                 preserve_format: bool = True, **kwargs) -> Dict[str, Any]:

        if self.compression_strategy == "tree":
            return self._compress_tree(text)
        elif self.compression_strategy == "sequence":
            return self._compress_sequence(text)
        
        # if self.compression_strategy == "semantic":
        #     return self._compress_semantic(text, compression_ratio, preserve_format, **kwargs)
        
        raise ValueError(f"Unknown compression strategy: {self.compression_strategy}")

    def _get_underscore_token_id(self) -> int:
        """
        Helper to get the token id used as the mask/underscore token in compressed text.
        This version always prefers the single-token encoding of "_" and falls back to
        simple, stable tokenizer IDs (pad/unk/0). It does NOT try to use model-native
        mask tokens like [MASK]; the literal underscore character is the primary choice.
        """
        # 1) Try literal underscore as a single token
        try:
            underscore_encoded = self.tokenizer.encode("_", add_special_tokens=False)
            if len(underscore_encoded) > 0:
                return underscore_encoded[0]
        except Exception:
            pass

        # 2) Final safety fallbacks
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        if getattr(self.tokenizer, "unk_token_id", None) is not None:
            return self.tokenizer.unk_token_id
        return 0

    def _get_special_token_ids(self) -> set:
        """
        Helper to collect all special token ids we want to preserve.
        Reused across multiple compression strategies.
        """
        special_token_ids = set()
        if self.tokenizer.bos_token_id is not None:
            special_token_ids.add(self.tokenizer.bos_token_id)
        if self.tokenizer.eos_token_id is not None:
            special_token_ids.add(self.tokenizer.eos_token_id)
        if self.tokenizer.cls_token_id is not None:
            special_token_ids.add(self.tokenizer.cls_token_id)
        if self.tokenizer.sep_token_id is not None:
            special_token_ids.add(self.tokenizer.sep_token_id)
        if self.tokenizer.pad_token_id is not None:
            special_token_ids.add(self.tokenizer.pad_token_id)
        return special_token_ids

    # write function to compress in sequence. If the next token generation is the same as ground truth, then mask it. If not keep original.
    def _compress_sequence(self, text: str) -> Dict[str, Any]:
        """
        Sequential compression:
        For each token position, let the model predict the next token from the prefix.
        - If the predicted token == ground-truth token, we mask it (replace with underscore token)
        - Otherwise, we keep the original token.
        """
        device = next(self.model.parameters()).device
        tokenized_text = self.tokenizer(text, return_tensors="pt")
        input_ids = tokenized_text.input_ids.to(device)

        # Determine underscore (mask) token id and special tokens (reuse helpers)
        underscore_token_id = self._get_underscore_token_id()
        special_token_ids = self._get_special_token_ids()

        # Ensure batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        compressed_ids = input_ids.clone()

        with torch.no_grad():
            for batch_idx in range(input_ids.shape[0]):
                seq = input_ids[batch_idx]
                comp_seq = compressed_ids[batch_idx]
                seq_len = seq.shape[0]

                # Skip leading/trailing special tokens
                start_idx = 0
                end_idx = seq_len
                while start_idx < seq_len and seq[start_idx].item() in special_token_ids:
                    start_idx += 1
                while end_idx > start_idx and seq[end_idx - 1].item() in special_token_ids:
                    end_idx -= 1

                for i in range(start_idx, end_idx):
                    gt_token_id = seq[i].item()

                    # Skip specials / void tokens
                    if gt_token_id in special_token_ids or self._is_void_token(gt_token_id):
                        continue

                    # Prefix up to (but excluding) position i
                    prefix = seq[:i].unsqueeze(0)  # [1, i]
                    if prefix.numel() == 0:
                        # No context to predict from; can't safely mask
                        continue

                    attention_mask = torch.ones_like(prefix, device=device)

                    # Generate the next token (greedy)
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

                    pred_token_id = generated[0, -1].item()

                    # If prediction matches ground truth, mask it
                    if pred_token_id == gt_token_id:
                        comp_seq[i] = underscore_token_id

        # Decode compressed sequence (assume single batch for now)
        compressed_text = self.tokenizer.decode(compressed_ids[0], skip_special_tokens=True)

        compression_ratio = (
            len(compressed_text) / len(text)
        )

        return {
            "compressed": compressed_text,
            "compression_ratio": compression_ratio,
            "masked_tokens": masked_count,
        }
    
    def _compress_tree(self, text: str) -> Dict[str, Any]:
        tokenized_text = self.tokenizer(text, return_tensors="pt")
        input_ids = tokenized_text.input_ids
        
        # Create both versions in a single pass (more efficient)
        compressed_text, compression_ratio = self._replace_bigram_tokens_both(input_ids.clone(), text)
        return {
            "compressed": compressed_text,
            "compression_ratio": compression_ratio,
        }
    
    def _replace_bigram_tokens_both(self, input_ids: torch.Tensor, text: str) -> Tuple[str, float]:
        """
        Create both v1 and v2 in a single loop for efficiency.
        v1: masks first token of each bigram
        v2: masks second token of each bigram
        """
        underscore_token_id = self._get_underscore_token_id()
        # Get special token IDs to preserve (reuse helper)
        special_token_ids = self._get_special_token_ids()
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        compressed_ids_v1 = input_ids.clone()
        compressed_ids_v2 = input_ids.clone()
        
        for batch_idx in range(compressed_ids_v1.shape[0]):
            seq = compressed_ids_v1[batch_idx]
            seq_len = seq.shape[0]
            
            # Find start and end indices, skipping special tokens
            start_idx = 0
            end_idx = seq_len
            
            # Skip special tokens at the start
            while start_idx < seq_len and seq[start_idx].item() in special_token_ids:
                start_idx += 1
            
            # Skip special tokens at the end
            while end_idx > start_idx and seq[end_idx - 1].item() in special_token_ids:
                end_idx -= 1
            
            # Process bigrams: mask first token for v1, second token for v2
            for i in range(start_idx, end_idx - 1, 2):
                # v1: mask first token of bigram (position i)
                token_id_v1 = seq[i].item()
                if not self._is_void_token(token_id_v1):
                    compressed_ids_v1[batch_idx, i] = underscore_token_id
                
                # v2: mask second token of bigram (position i+1)
                if i + 1 < end_idx:
                    token_id_v2 = seq[i + 1].item()
                    if not self._is_void_token(token_id_v2):
                        compressed_ids_v2[batch_idx, i + 1] = underscore_token_id
        
        if squeeze_output:
            compressed_ids_v1 = compressed_ids_v1.squeeze(0)
            compressed_ids_v2 = compressed_ids_v2.squeeze(0)
        
        # Handle tensor dimensions for decoding
        ids_v1 = compressed_ids_v1[0] if compressed_ids_v1.dim() > 1 else compressed_ids_v1
        ids_v2 = compressed_ids_v2[0] if compressed_ids_v2.dim() > 1 else compressed_ids_v2
        compressed_text_v1 = self.tokenizer.decode(ids_v1, skip_special_tokens=True)
        compressed_text_v2 = self.tokenizer.decode(ids_v2, skip_special_tokens=True)

        # unmask the tokens that cannot be reconstructed
        compressed_text_v1, compression_ratio_v1 = self._lossless_gurantee(compressed_text_v1, compressed_ids_v1, text, input_ids)
        compressed_text_v2, compression_ratio_v2 = self._lossless_gurantee(compressed_text_v2, compressed_ids_v2, text, input_ids)
        if compression_ratio_v1 < compression_ratio_v2:
            return compressed_text_v1, compression_ratio_v1
        else:
            return compressed_text_v2, compression_ratio_v2
    
    def _is_void_token(self, token_id: int) -> bool:
        """
        Check if a token is a void token (whitespace, punctuation, formatting).
        These tokens should not be replaced with underscores.
        """
        try:
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            # Check if token is only whitespace, punctuation, or formatting characters
            # Strip and check if empty or only contains whitespace/punctuation
            stripped = token_text.strip()
            if not stripped:
                return True  # Only whitespace
            
            # Check if it's only punctuation/whitespace characters
            if re.match(r'^[\s\.,;:!?\-\(\)\[\]\{\}\'\"\n\r\t]+$', token_text):
                return True
            
            # Check if it starts with whitespace and the rest is punctuation
            if token_text.startswith((' ', '\n', '\t', '\r')) and len(stripped) <= 2:
                if re.match(r'^[\s\.,;:!?\-\(\)\[\]\{\}\'\"\n\r\t]+$', stripped):
                    return True
            
            return False
        except:
            # If we can't decode, assume it's not a void token
            return False
    
    def _reconstruct_text(self, compressed_text: str, max_new_tokens: Optional[int] = None, 
                         do_sample: bool = False, **kwargs) -> Tuple[str, torch.Tensor]:
        """
        Reconstruct text from compressed version using the model.
        Reusable helper method for both decompress and _lossless_gurantee.
        
        Args:
            compressed_text: The compressed text to reconstruct
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling (default: False for deterministic)
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (reconstructed_text, reconstructed_input_ids)
        """
        # Determine the mask token *text* used in the compressed representation
        mask_token_id = self._get_underscore_token_id()
        try:
            mask_token_text = self.tokenizer.decode([mask_token_id], skip_special_tokens=False).strip()
        except Exception:
            mask_token_text = ""
        if not mask_token_text:
            # Fallback textual representation shown to the model
            mask_token_text = "[MASK]"

        # Format prompt using Qwen chat template: instruct model to fill mask tokens
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in text reconstruction and error-correction. "
                    "Given partially masked text, you precisely infer the missing tokens "
                    "and output a fluent, faithful reconstruction of the original."
                ),
            },
            {
                "role": "user",
                "content": (
                    "You are given a text where certain tokens are masked with the special token "
                    f"'{mask_token_text}'. Fill in every masked token so that the text is fully "
                    "reconstructed and as close as possible to the original.\n\n"
                    f"Masked text:\n{compressed_text}\n\n"
                    "Return only the fully reconstructed text, with no explanations or comments."
                ),
            },
        ]
        
        # Apply chat template if available (Qwen3-8b format)
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True, 
                enable_thinking=False
            )
        else:
            # Fallback to simple prompt if chat template not available
            prompt = (
                "You are an expert in text reconstruction and error-correction. "
                "Given partially masked text, you precisely infer the missing tokens "
                "and output a fluent, faithful reconstruction of the original.\n\n"
                "You are given a text where certain tokens are masked with the special token "
                f"'{mask_token_text}'. Fill in every masked token so that the text is fully "
                "reconstructed and as close as possible to the original.\n\n"
                f"Masked text:\n{compressed_text}\n\n"
                "Return only the fully reconstructed text, with no explanations or comments."
            )
        
        # Tokenize the prompt
        device = next(self.model.parameters()).device
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
        prompt_input_ids = prompt_tokens.input_ids.to(device)
        prompt_attention_mask = prompt_tokens.attention_mask.to(device)
        
        # Generate input_ids using the model
        with torch.no_grad():
            generated_output = self.model.generate(
                prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Extract only the newly generated tokens (excluding the prompt)
        reconstructed_input_ids = generated_output[:, prompt_input_ids.shape[-1]:]
        reconstructed_text = self.tokenizer.decode(reconstructed_input_ids[0], skip_special_tokens=True)
        
        return reconstructed_text, reconstructed_input_ids
    
    def _lossless_gurantee(self, compressed_text: str, compressed_input_ids: torch.Tensor, text: str, original_input_ids: torch.Tensor) -> Tuple[str, float]:
        """
        Lossless guarantee: Ensure the compressed text is lossless.
        Reuses _reconstruct_text for reconstruction.
        """
        # Reconstruct text using the shared helper method
        max_new_tokens = original_input_ids.shape[-1] if original_input_ids.dim() > 0 else len(original_input_ids)
        reconstructed_text, reconstructed_input_ids = self._reconstruct_text(
            compressed_text, 
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        
        print(f"start validating lossless guarantee")
        print(f"reconstruction text: {reconstructed_text}")
        
        # Compare original with reconstructed to find unmatched tokens
        orig = original_input_ids.squeeze() if original_input_ids.dim() > 1 else original_input_ids
        recon = reconstructed_input_ids.squeeze() if reconstructed_input_ids.dim() > 1 else reconstructed_input_ids[0]
        comp = compressed_input_ids.squeeze() if compressed_input_ids.dim() > 1 else compressed_input_ids
        
        # Find tokens that don't match between original and reconstructed
        unmatched = {i for i in range(min(len(orig), len(recon))) if orig[i].item() != recon[i].item()}
        
        # Create lossless version: use original tokens for unmatched positions, compressed tokens otherwise
        lossless_version = [orig[i].item() if i in unmatched else comp[i].item() for i in range(len(orig))]
        lossless_text = self.tokenizer.decode(lossless_version, skip_special_tokens=True)
        compression_ratio = len(lossless_text) / len(text)
        
        return lossless_text, compression_ratio
    
    def save(self, compressed_data: Dict[str, Any], filepath: str):
        """Save compressed data to a file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(compressed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved compressed data to {filepath}")
    
    def load(self, filepath: str) -> Dict[str, Any]:
        """Load compressed data from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded compressed data from {filepath}")
        return data

