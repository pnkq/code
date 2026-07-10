from p.dispatcher import Dispatcher

from p.tokenizer_vie import VietnameseTokenizer
from p.tokenizer_eng import EnglishTokenizer
from p.tokens import Token
from p.piece import Piece

import torch

class HybridTokenizer:
    def __init__(self, rule_based_segmenter, bpe_encoder, max_length=512):
        self.segmenter = rule_based_segmenter  # Your VN rule-based tool (e.g., Underthesea/VnCoreNLP)
        self.bpe = bpe_encoder                  # Your English BPE engine
        self.max_length = max_length
        
        # 1. Define your vocabulary mapping (Ensure special tokens are included!)
        self.vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4} 
        # ... populate self.vocab with your custom VN words and EN BPE subwords ...
        
        # 2. Required Hugging Face Special Token Properties
        self.bos_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.mask_token_id = 4
        
        self.mask_token = "<mask>"
        self.pad_token = "<pad>"

    def __len__(self):
        # Required so RobertaConfig knows the vocabulary size
        return len(self.vocab)

    def _tokenize_single_text(self, text):
        """Your custom hybrid logic goes here."""
        # Step A: Apply Vietnamese rule-based word segmentation (e.g., "học sinh" -> "học_sinh")
        vn_segmented_text = self.segmenter(text)
        
        # Step B: Run your English BPE on the resulting tokens or mixed text
        # Step C: Convert strings to IDs using your self.vocab
        # (For this example, let's assume 'raw_ids' is your list of vocabulary integers)
        raw_ids = [self.vocab.get(token, self.unk_token_id) for token in vn_segmented_text.split()]
        
        # Step D: Add RoBERTa style start (<s>) and end (</s>) boundaries
        input_ids = [self.bos_token_id] + raw_ids[:self.max_length - 2] + [self.eos_token_id]
        
        # Step E: Create the attention mask (1 for real tokens)
        attention_mask = [1] * len(input_ids)
        
        return input_ids, attention_mask

    def __call__(self, text, padding=True, truncation=True, max_length=None, return_special_tokens_mask=False, **kwargs):
        """
        The main gateway Hugging Face calls. 
        It must handle both a single string and a list of strings (batches).
        """
        if max_length is None:
            max_length = self.max_length

        # Handle batches vs single strings
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_special_tokens_mask = []

        for t in texts:
            ids, mask = self._tokenize_single_text(t)
            
            # Dynamic Truncation
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
                mask = mask[:max_length]
                
            batch_input_ids.append(ids)
            batch_attention_mask.append(mask)
            
            if return_special_tokens_mask:
                # 1 for special tokens (<s>, </s>, <pad>), 0 for regular words. Crucial for MLM data collator!
                spec_mask = [1 if x in [self.bos_token_id, self.eos_token_id, self.pad_token_id] else 0 for x in ids]
                batch_special_tokens_mask.append(spec_mask)

        # Dynamic Padding
        if padding:
            longest = max(len(ids) for ids in batch_input_ids)
            for i in range(len(batch_input_ids)):
                pad_len = longest - len(batch_input_ids[i])
                batch_input_ids[i] += [self.pad_token_id] * pad_len
                batch_attention_mask[i] += [0] * pad_len  # 0 means ignore padding tokens
                if return_special_tokens_mask:
                    batch_special_tokens_mask[i] += [1] * pad_len

        # Construct final output dictionary
        output = {
            "input_ids": batch_input_ids if is_batch else batch_input_ids[0],
            "attention_mask": batch_attention_mask if is_batch else batch_attention_mask[0]
        }
        
        if return_special_tokens_mask:
            output["special_tokens_mask"] = batch_special_tokens_mask if is_batch else batch_special_tokens_mask[0]
            
        return output