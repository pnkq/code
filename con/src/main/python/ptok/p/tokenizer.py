from p.pipeline import Pipeline
from p.vocabulary import Vocabulary

class HybridTokenizer:
    def __init__(self, vocab_file, max_length=512):
        self.pipeline = Pipeline()
        self.max_length = max_length
        
        # 1. Define your vocabulary mapping (Ensure special tokens are included!)
        self.vocab = Vocabulary.load(vocab_file)
        
        # 2. Required Hugging Face Special Token Properties
        self.bos_token_id = self.vocab.token_to_id("<s>")
        self.pad_token_id = self.vocab.token_to_id("<pad>")
        self.eos_token_id = self.vocab.token_to_id("</s>")
        self.unk_token_id = self.vocab.token_to_id("<unk>")
        self.mask_token_id = self.vocab.token_to_id("<mask>")
        
        self.mask_token = "<mask>"
        self.pad_token = "<pad>"

    def __len__(self):
        # Required so RobertaConfig knows the vocabulary size
        return len(self.vocab)

    def _tokenize_single_text(self, text):
        # Tokenize the text into pieces
        pieces = self.pipeline.tokenize(text)
        # Find token ids using the vocab
        input_ids = self.vocab.encode(pieces)        
        # Create the attention mask (1 for real tokens)
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
    
