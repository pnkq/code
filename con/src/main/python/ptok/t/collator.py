import numpy as np
import torch


class MaskedLanguageModelDataCollator:

    def __init__(self, tokenizer, mlm_probability=0.15, debug=False):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.debug = debug

        self.special_ids = {
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id
        }

    def __call__(self, examples):
        input_ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
        labels = input_ids.clone()

        probability_matrix = torch.full(
            labels.shape,
            self.mlm_probability
        )

        # Never mask special tokens
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)

        for token_id in self.special_ids:
            special_tokens_mask |= labels.eq(token_id)

        probability_matrix.masked_fill_(special_tokens_mask, 0.0)

        # Sample masked positions
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Ignore unmasked tokens
        labels[~masked_indices] = -100

        rand = torch.rand(labels.shape)
        # 80% -> <mask>
        indices_replaced = masked_indices & (rand < 0.8)
        # 10% -> random token
        indices_random = masked_indices & (rand >= 0.8) & (rand < 0.9)
        # indices_unchanged = masked_indices & (rand >= 0.9) # for clarity only

        indices = torch.randint(len(self.tokenizer.normal_token_ids),labels.shape)
        random_words = torch.tensor(self.tokenizer.normal_token_ids, dtype=torch.long)[indices]

        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        input_ids[indices_random] = random_words[indices_random]

        # Remaining 10% unchanged
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        if self.debug:
            output["masked_indices"] = masked_indices
            output["indices_replaced"] = indices_replaced
            output["indices_random"] = indices_random
            output["special_tokens_mask"] = special_tokens_mask        
            
        return output
    
