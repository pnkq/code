from typing import List, Union
import torch


class HybridTokenizer:

    def __init__(self, pipeline, vocab):
        self.pipeline = pipeline
        self.vocab = vocab

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.mask_token = "<mask>"

        self.pad_token_id = self.vocab.token_to_id(self.pad_token)
        self.unk_token_id = self.vocab.token_to_id(self.unk_token)
        self.bos_token_id = self.vocab.token_to_id(self.bos_token)
        self.eos_token_id = self.vocab.token_to_id(self.eos_token)
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
    
    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=True):
        specials = {
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id
        }
        return [1 if i in specials else 0 for i in token_ids]
    
    def prepare_for_model(self, token_ids, return_attention_mask=True, return_special_tokens_mask=False):
        input_ids = self.build_inputs_with_special_tokens(token_ids)

        output = { "input_ids": input_ids }

        if return_attention_mask:
            output["attention_mask"] = [1] * len(input_ids)

        if return_special_tokens_mask:
            output["special_tokens_mask"] = self.get_special_tokens_mask(input_ids, already_has_special_tokens=True)

        return output
        
    
    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.vocab.id_to_token(ids)
        
        return [self.vocab.id_to_token(i) for i in ids]

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

        outputs = [
            self.prepare_for_model(
                self.encode(sentence),
                return_attention_mask=return_attention_mask,
                return_special_tokens_mask=return_special_tokens_mask
            )
            for sentence in texts
        ]

        batch = {
            key: [o[key] for o in outputs]
            for key in outputs[0]
        }

        if not is_batch:
            batch = {k: v[0] for k, v in batch.items()}

        if return_tensors == "pt":
            batch = { k: torch.tensor(v, dtype=torch.long) for k, v in batch.items() }

        return batch
        
    def pad(self, encoded_inputs, padding=True, max_length=None, return_attention_mask=True, return_tensors=None, **kwargs):
        if isinstance(encoded_inputs, dict):
            encoded_inputs = [encoded_inputs]

        longest = max(len(item["input_ids"]) for item in encoded_inputs)
        if max_length is not None:
            longest = max_length

        batch = {
            "input_ids": [],
            "attention_mask": []
        }

        for item in encoded_inputs:
            ids = list(item["input_ids"])
            pad_len = longest - len(ids)

            batch["input_ids"].append(ids + [self.pad_token_id] * pad_len)

            if return_attention_mask:
                mask = [1] * len(ids) + [0] * pad_len
                batch["attention_mask"].append(mask)

        if return_tensors == "pt":
            batch = {
                k: torch.tensor(v, dtype=torch.long)
                for k, v in batch.items()
            }

        return batch
    
    @property
    def all_special_ids(self):
        return [
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.unk_token_id,
            self.mask_token_id
        ]
    
    @property
    def all_special_tokens(self):
        return [
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.unk_token,
            self.mask_token
        ]
    
    @property
    def special_tokens_map(self):
        return {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "mask_token": self.mask_token
        }
    
    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask"]
    
    @property
    def is_fast(self):
        return False
    
    @property
    def normal_token_ids(self):
        if not hasattr(self, "_normal_token_ids"):
            forbidden = {
                self.bos_token_id,
                self.eos_token_id,
                self.pad_token_id,
                self.mask_token_id
            }
            self._normal_token_ids = [
                i for i in range(self.vocab_size)
                if i not in forbidden
            ]
        return self._normal_token_ids