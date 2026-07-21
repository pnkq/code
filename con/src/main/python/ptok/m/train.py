from transformers import RobertaForMaskedLM, Trainer, TrainingArguments

import sys
from pathlib import Path

HOME_DIR = Path.home()
sys.path.append(f"{HOME_DIR}/code/con/src/main/python/ptok/")

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from m.config import TrainingConfig
from t.memmap import MemMapDataset
from t.collator import MaskedLanguageModelDataCollator
from p.tokenizer import HybridTokenizer
from p.pipeline import Pipeline
from p.vocabulary import Vocabulary


def main():

    tokenizer = HybridTokenizer(Pipeline(), Vocabulary.load("vocab.json"))

    dataset = MemMapDataset("20231101_vie.bin", sequence_length=512)
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Number of sequences: {dataset.num_sequences}")
    print(f"Shape of a sequence: {dataset[0]['input_ids'].shape}")

    cfg = TrainingConfig(
        vocab_size=len(tokenizer),
        batch_size=32,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=1024,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    model = RobertaForMaskedLM(cfg.create_model_config())
    # Returns total number of parameters
    print(f"Total Parameters: {model.num_parameters():,}")

    collator = MaskedLanguageModelDataCollator(tokenizer)

    args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator
    )

    # Pass the path to your checkpoint folder directly when starting training
    trainer.train()
    trainer.save_model(f"v-model_{cfg.hidden_size}_{cfg.num_hidden_layers}_{cfg.num_attention_heads}_{cfg.intermediate_size}")


if __name__ == "__main__":
    main()