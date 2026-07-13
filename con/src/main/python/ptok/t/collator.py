import numpy as np
import torch


class MaskedLanguageModelDataCollator:

    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

        self.special_ids = {
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id
        }

        self.normal_token_ids = torch.tensor([ i
            for i in range(tokenizer.vocab_size)
            if i not in {
                tokenizer.pad_token_id,
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.mask_token_id
            }
        ])

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

        # 80% -> <mask>
        replace_prob = torch.full(labels.shape, 0.8)
        indices_replaced = (torch.bernoulli(replace_prob).bool() & masked_indices)
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% -> random token
        random_prob = torch.full(labels.shape, 0.5)
        indices_random = (torch.bernoulli(random_prob).bool() & masked_indices & ~indices_replaced)

        indices = torch.randint(len(self.normal_token_ids), labels.shape)
        random_words = self.normal_token_ids[indices]
        
        input_ids[indices_random] = random_words[indices_random]

        # Remaining 10% unchanged
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
