from dataclasses import dataclass
from transformers import RobertaConfig


@dataclass
class TrainingConfig:
    vocab_size: int

    #
    # RoBERTa architecture
    #
    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 256

    #
    # Input
    #
    max_position_embeddings: int = 514

    #
    # Dropout
    #
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    #
    # Token ids
    #
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    #
    # Optimizer
    #
    learning_rate: float = 5e-4
    weight_decay: float = 0.01

    #
    # Training
    #
    batch_size: int = 8
    epochs: int = 10
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 1

    #
    # Logging
    #
    logging_steps: int = 100
    save_steps: int = 10000

    #
    # Mixed precision
    #
    fp16: bool = False
    bf16: bool = False

    def create_model_config(self):
        return RobertaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            type_vocab_size=1
        )
    
