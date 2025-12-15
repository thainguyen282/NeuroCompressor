import logging
import torch
import re
from typing import Optional, Dict, Any, List, Tuple
import json
import hashlib
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TextCompressor:
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                compression_strategy: str = "tree", chunk_size: int = 50, overlap: int = 5,
                mask_token: str = "[]"):
        """
        Initialize TextCompressor with external model and tokenizer.
        
        Args:
            model: Pre-loaded AutoModelForCausalLM instance
            tokenizer: Pre-loaded AutoTokenizer instance
            compression_strategy: Strategy to use ('tree', 'semantic', etc.)
            chunk_size: Size of text chunks for processing
            overlap: Overlap between chunks
            mask_token: Custom mask token string to use for compression (default: "[MASK]")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.compression_strategy = compression_strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Add mask token to tokenizer if it doesn't exist
        self.mask_token = mask_token
        self._add_mask_token_to_tokenizer()
        
        logger.info(f"Initialized TextCompressor with strategy: {compression_strategy}")
    def compress(self, text: str, compression_ratio: float = 0.5,
                preserve_format: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Compress text using the specified compression strategy.
        
        Returns:
            Dictionary containing:
                - 'compressed': compressed text
                - 'compression_ratio': compression ratio
                - 'compression_time': time taken in seconds
        """
        start_time = time.time()
        
        if self.compression_strategy == "hierarchical":
            result = self._compress_hierarchical(text)
        elif self.compression_strategy == "sequence":
            result = self._compress_sequence(text)
        else:
            raise ValueError(f"Unknown compression strategy: {self.compression_strategy}")
        
        end_time = time.time()
        compression_time = end_time - start_time
        
        # Add compression time to result
        result['compression_time'] = compression_time
        print(f"compression time: {compression_time}")
        return result

    def _add_mask_token_to_tokenizer(self):
        """
        Add the mask token to the tokenizer if it doesn't exist.
        This ensures the mask token is saved with the tokenizer and can be reused.
        The mask token is added but removed from special_tokens_map so it won't be skipped during decoding.
        
        Note: To save the tokenizer with the mask token for later reuse, call:
            tokenizer.save_pretrained(save_directory)
        """
        # Check if mask token already exists in vocabulary
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.mask_token)
        if mask_token_id != self.tokenizer.unk_token_id:
            # Token already exists in vocabulary
            logger.info(f"Mask token '{self.mask_token}' already exists in tokenizer")
        else:
            # Add the mask token to the tokenizer
            # Using add_tokens() ensures it's added as a single atomic token
            num_added = self.tokenizer.add_tokens([self.mask_token], special_tokens=False)
            
            if num_added > 0:
                # Resize model embeddings if needed (for models that use token embeddings)
                try:
                    if hasattr(self.model, 'resize_token_embeddings'):
                        self.model.resize_token_embeddings(len(self.tokenizer))
                        logger.info(f"Resized model embeddings to accommodate new mask token")
                except Exception as e:
                    logger.warning(f"Could not resize model embeddings: {e}")
                
                logger.info(f"Added mask token '{self.mask_token}' to tokenizer")
        
        # Remove mask token from special_tokens_map so it won't be skipped during decode
        # This allows skip_special_tokens=True to work while preserving mask tokens
        if hasattr(self.tokenizer, 'special_tokens_map') and self.tokenizer.special_tokens_map is not None:
            if 'mask_token' in self.tokenizer.special_tokens_map:
                del self.tokenizer.special_tokens_map['mask_token']
                logger.info(f"Removed mask token from special_tokens_map")
        
        # Also remove from mask_token attribute if it exists
        if hasattr(self.tokenizer, 'mask_token') and self.tokenizer.mask_token == self.mask_token:
            self.tokenizer.mask_token = None

    def _get_mask_token_id(self) -> int:
        """
        Helper to get the token id used as the mask token in compressed text.
        Uses the dedicated mask token that was added to the tokenizer.
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
        - If the predicted token == ground-truth token, we mask it (replace with mask token)
        - Otherwise, we keep the original token.
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

        compressed_ids = input_ids.clone()

        with torch.no_grad():
            # Iterate over each sequence in the batch (usually 1)
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

                # Use KV caching: feed tokens one by one and use the logits for the
                # next token prediction. At step `pos`, we feed token at `pos` and
                # compare the predicted next token to ground truth at `pos + 1`.
                past_key_values = None

                # We can only predict from position 0 up to end_idx-2 (next token is pos+1)
                for pos in tqdm(
                    range(0, end_idx - 1),
                    desc=f"Compressing sequence {batch_idx + 1}/{input_ids.shape[0]}",
                    leave=False,
                ):
                    # Single-token input at current position
                    cur_input = seq[pos : pos + 1].unsqueeze(0)  # shape [1, 1]

                    outputs = self.model(
                        input_ids=cur_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :]  # logits for next token
                    past_key_values = outputs.past_key_values

                    target_pos = pos + 1
                    # Only consider positions inside the main content window
                    if target_pos < start_idx or target_pos >= end_idx:
                        continue

                    gt_token_id = seq[target_pos].item()

                    # Skip specials / void tokens
                    if gt_token_id in special_token_ids or self._is_void_token(gt_token_id):
                        continue

                    # Greedy prediction for the next token
                    pred_token_id = int(torch.argmax(logits, dim=-1).item())

                    # If prediction matches ground truth, mask it
                    if pred_token_id == gt_token_id:
                        comp_seq[target_pos] = mask_token_id

        # Decode compressed sequence (assume single batch for now)
        compressed_text = self.tokenizer.decode(compressed_ids[0], skip_special_tokens=True)

        compression_ratio = (
            len(compressed_text) / len(text)
        )

        return {
            "compressed": compressed_text,
            "compression_ratio": compression_ratio,
        }
    
    def _compress_hierarchical(self, text: str) -> Dict[str, Any]:
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
        mask_token_id = self._get_mask_token_id()
        # Get special token IDs to preserve (reuse helper)
        special_token_ids = self._get_special_token_ids()
        
        if input_ids.dim() != 1:
            input_ids = input_ids.squeeze(0)
        compressed_ids_v1 = input_ids.clone()
        compressed_ids_v2 = input_ids.clone()
        
    
        seq = compressed_ids_v1
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
                compressed_ids_v1[i] = mask_token_id
            
            # v2: mask second token of bigram (position i+1)
            if i + 1 < end_idx:
                token_id_v2 = seq[i + 1].item()
                if not self._is_void_token(token_id_v2):
                    compressed_ids_v2[i + 1] = mask_token_id
        
    

        # unmask the tokens that cannot be reconstructed
        compressed_text_v1, compression_ratio_v1 = self._lossless_gurantee(compressed_ids_v1, text, input_ids)
        compressed_text_v2, compression_ratio_v2 = self._lossless_gurantee(compressed_ids_v2, text, input_ids)
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
            # Never treat the mask token as void
            if token_id == self._get_mask_token_id():
                return False
            
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            # Check if token is only whitespace, punctuation, or formatting characters
            # Strip and check if empty or only contains whitespace/punctuation
            stripped = token_text.strip()
            if not stripped:
                return True  # Only whitespace
            
            # Check if it's only punctuation/whitespace characters
            if re.match(r'^[\s\.,;:!?\-\(\)\{\}\'\"\n\r\t]+$', token_text):
                return True
            
            # Check if it starts with whitespace and the rest is punctuation
            if token_text.startswith((' ', '\n', '\t', '\r')) and len(stripped) <= 2:
                if re.match(r'^[\s\.,;:!?\-\(\)\[\]\{\}\'\"\n\r\t]+$', stripped):
                    return True
            
            return False
        except:
            # If we can't decode, assume it's not a void token
            return False
    
    def _reconstruct_text(self, compressed_input_ids: torch.Tensor, original_input_ids: torch.Tensor) -> str:
        """
        Reconstruct text from compressed version using the model.
        For sequence-based reconstruction, this mirrors the token-level mechanism
        used in `TextDecompressor._decompress_sequence`:

        - Take the compressed text (with mask tokens) and tokenize it.
        - Loop left-to-right:
        * If the token is not the mask token, keep it.
        * If the token **is** the mask token, generate the next token from
            the already reconstructed prefix (no natural-language prompt) and use
            that prediction instead of the mask token.
        """
        device = next(self.model.parameters()).device
        reconstructed_input_ids = compressed_input_ids.clone()

        # Determine mask token id and special tokens (reuse helpers)
        mask_token_id = self._get_mask_token_id()
        special_token_ids = self._get_special_token_ids()


        # We modify input_ids in-place; no need for a separate copy
        with torch.no_grad():
            
                seq = reconstructed_input_ids
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
                        ground_truth_token = original_input_ids[i].item()
                        seq[i] = ground_truth_token
                        if generated[0, -1].item() != ground_truth_token:
                            compressed_input_ids[i] = ground_truth_token
        compressed_text = self.tokenizer.decode(
            compressed_input_ids, skip_special_tokens=True
        )
        return compressed_text
    
    def _lossless_gurantee(self, compressed_input_ids: torch.Tensor, text: str, original_input_ids: torch.Tensor) -> Tuple[str, float]:
        lossless_text = self._reconstruct_text(
            compressed_input_ids, 
            original_input_ids
        )
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

