from transformers import RobertaForMaskedLM, Trainer, TrainingArguments

import sys
from pathlib import Path

HOME_DIR = Path.home()
sys.path.append(f"{HOME_DIR}/code/con/src/main/python/ptok/")

from m.config import TrainingConfig
from t.memmap import MemMapDataset
from t.collator import MaskedLanguageModelDataCollator
from p.tokenizer import HybridTokenizer
from p.pipeline import Pipeline
from p.vocabulary import Vocabulary


def main():

    tokenizer = HybridTokenizer(Pipeline(), Vocabulary.load("vocab.json"))

    dataset = MemMapDataset("1.bin", sequence_length=514)

    cfg = TrainingConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    model = RobertaForMaskedLM(cfg.create_model_config())

    collator = MaskedLanguageModelDataCollator(tokenizer)

    args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=cfg.batch_size,
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
    trainer.train(resume_from_checkpoint="./results/checkpoint-1000")
    trainer.save_model("model")


if __name__ == "__main__":
    main()