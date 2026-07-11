from typing import List, Union
import torch


class HybridTokenizer:

    def __init__(self, pipeline, vocab):
        self.pipeline = pipeline
        self.vocab = vocab

        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"

        self.bos_token_id = self.vocab.token_to_id(self.bos_token)
        self.eos_token_id = self.vocab.token_to_id(self.eos_token)
        self.pad_token_id = self.vocab.token_to_id(self.pad_token)
        self.unk_token_id = self.vocab.token_to_id(self.unk_token)
        self.mask_token_id = self.vocab.token_to_id(self.mask_token)

    def __len__(self):
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        pieces = self.pipeline.tokenize(text)
        return self.vocab.encode(pieces)
    
    def encode_with_special_tokens(self, text):
        token_ids = self.encode(text)
        return [ self.bos_token_id, *token_ids, self.eos_token_id ]

    def encode_batch(self, texts):
        return [ self.encode(t) for t in texts ]
    
    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            ids = [ i for i in ids if i not in { self.bos_token_id, self.eos_token_id, self.pad_token_id} ]
        pieces = self.vocab.decode(ids)
        return self.pipeline.decoder.decode(pieces)

    def build_inputs_with_special_tokens(self, token_ids):
        return [ self.bos_token_id, *token_ids, self.eos_token_id]
    
    def get_special_tokens_mask(self, token_ids):
        specials = { self.bos_token_id, self.eos_token_id, self.pad_token_id }
        return [ 1 if i in specials else 0 for i in token_ids ]

    def __call__(self, text: Union[str, List[str]], return_attention_mask=True, return_special_tokens_mask=False, return_tensors=None):
        """
        Tokenize one or more texts.

        This method is intended for inference / HuggingFace compatibility.

        Returns
        -------
        {
            "input_ids": ...,
            "attention_mask": ...,
            "special_tokens_mask": ...
        }
        """
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        batch_input_ids = []
        batch_attention_mask = []
        batch_special_tokens_mask = []

        for sentence in texts:
            ids = self.encode(sentence)
            #
            # Add <s> and </s>
            #
            ids = self.build_inputs_with_special_tokens(ids)
            batch_input_ids.append(ids)
            if return_attention_mask:
                batch_attention_mask.append([1] * len(ids))

            if return_special_tokens_mask:
                batch_special_tokens_mask.append(self.get_special_tokens_mask(ids, already_has_special_tokens=True))

        #
        # Construct output dictionary
        #
        output = {
            "input_ids": batch_input_ids
        }

        if return_attention_mask:
            output["attention_mask"] = batch_attention_mask

        if return_special_tokens_mask:
            output["special_tokens_mask"] = batch_special_tokens_mask

        #
        # Convert batch of size 1 back to a single example
        #
        if not is_batch:
            output = { key: value[0] for key, value in output.items() }

        #
        # Convert to PyTorch tensors
        #
        if return_tensors == "pt":
            output = { key: torch.tensor(value, dtype=torch.long) for key, value in output.items() }

        return output

