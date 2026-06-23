"""
train.py
--------
Fine-tunes Mavkif/m2m100_rup_ur_to_rur with LoRA (via peft) using
HuggingFace Seq2SeqTrainer.

LoRA is applied to ALL attention projection layers across both the encoder
and decoder (self-attention + cross-attention):
    q_proj, k_proj, v_proj, out_proj

Hardware target: NVIDIA A5000 (24 GB VRAM)
  → fp16 mixed precision
  → per_device_train_batch_size = 16
  → gradient_accumulation_steps  = 4  (effective batch = 64)
  → base model is NOT quantised (full fp16 fine-tune via LoRA)
"""

import os
import argparse
import numpy as np

import torch
from datasets import load_from_disk
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import sacrebleu

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID          = "Mavkif/m2m100_rup_ur_to_rur"
TOKENIZER_ID      = "Mavkif/m2m100_rup_tokenizer_both"
TGT_LANG_TOKEN_ID = 128105   # __roman-ur__
SRC_LANG          = "ur"

# LoRA targets: q/k/v/out_proj in M2M100Attention.
# M2M100 uses "out_proj" (not "o_proj") — confirmed from modeling_m2m_100.py.
# PEFT matches these names as substring patterns across all named modules,
# so this single list covers encoder self-attn, decoder self-attn, and
# decoder cross-attn automatically.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune M2M100 transliteration model")
    p.add_argument(
        "--dataset_dir",
        default="./processed_dataset",
        help="Path to the tokenized HuggingFace DatasetDict (output of prepare_data.py)"
    )
    p.add_argument(
        "--output_dir",
        default="./checkpoints",
        help="Directory for checkpoints and trainer state"
    )
    p.add_argument(
        "--final_model_dir",
        default="./fine_tuned_model",
        help="Directory to save the best LoRA adapter weights after training"
    )
    # LoRA hyperparameters
    p.add_argument("--lora_r",       type=int,   default=16,
                   help="LoRA rank. Higher = more capacity, more VRAM. Start with 16.")
    p.add_argument("--lora_alpha",   type=int,   default=32,
                   help="LoRA scaling factor. Typically 2× lora_r.")
    p.add_argument("--lora_dropout", type=float, default=0.1,
                   help="Dropout applied inside LoRA adapters.")
    # Training hyperparameters
    p.add_argument("--num_epochs",        type=int,   default=30)
    p.add_argument("--batch_size",        type=int,   default=16,
                   help="Per-device train batch size. 16 is safe on A5000 24GB with fp16.")
    p.add_argument("--eval_batch_size",   type=int,   default=32)
    p.add_argument("--grad_accum",        type=int,   default=4,
                   help="Gradient accumulation steps. Effective batch = batch_size × grad_accum.")
    p.add_argument("--learning_rate",     type=float, default=5e-4,
                   help="AdamW learning rate. LoRA adapters can use a higher LR than full FT.")
    p.add_argument("--warmup_ratio",      type=float, default=0.06)
    p.add_argument("--label_smoothing",   type=float, default=0.1)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--max_new_tokens",    type=int,   default=128,
                   help="Max tokens to generate during eval (for BLEU/chrF).")
    return p.parse_args()

# ── LoRA configuration ────────────────────────────────────────────────────────

def build_lora_config(args) -> LoraConfig:
    """
    Configures LoRA for M2M100ForConditionalGeneration (encoder-decoder).

    TaskType.SEQ_2_SEQ_LM tells PEFT this is an encoder-decoder model,
    so it injects adapters into both encoder and decoder modules that
    match the target_modules names.
    """
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",                    # don't train bias terms
        target_modules=LORA_TARGET_MODULES,
        # inference_mode=False is the default; set explicitly for clarity
        inference_mode=False,
    )

# ── Metrics ───────────────────────────────────────────────────────────────────

def build_compute_metrics(tokenizer):
    """
    Returns a compute_metrics function for Seq2SeqTrainer that reports:
      - BLEU   (sacrebleu, word-level, standard NMT metric)
      - chrF   (sacrebleu, character-level F-score — more meaningful for
                transliteration because it rewards partial character matches)

    Both metrics are computed on detokenized strings (not raw token ids).
    """
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # preds may contain -100 if the model didn't generate up to max length;
        # clip to valid token id range before decoding
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)

        # labels come with -100 for padding — replace so batch_decode doesn't crash
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        # Decode token id arrays → strings
        decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,  skip_special_tokens=True)

        # Strip leading/trailing whitespace
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # sacrebleu expects references as a list-of-lists (one list per reference)
        references = [[l] for l in decoded_labels]

        bleu_score = sacrebleu.corpus_bleu(decoded_preds, list(zip(*references)))
        chrf_score = sacrebleu.corpus_chrf(decoded_preds, references)

        return {
            "bleu": round(bleu_score.score, 4),
            "chrf": round(chrf_score.score, 4),
        }

    return compute_metrics

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load tokenizer ─────────────────────────────────────────────────────
    print(f"[train] Loading tokenizer from '{TOKENIZER_ID}' ...")
    tokenizer = M2M100Tokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.src_lang = SRC_LANG  # set globally; also set per-batch in collation

    # ── 2. Load base model ────────────────────────────────────────────────────
    print(f"[train] Loading base model from '{MODEL_ID}' ...")
    model = M2M100ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # load weights in fp16 to save VRAM
    )

    # ── 3. Apply LoRA ─────────────────────────────────────────────────────────
    lora_config = build_lora_config(args)
    model = get_peft_model(model, lora_config)

    # Print trainable parameter summary
    model.print_trainable_parameters()

    # Verify PEFT found modules — if 0 trainable params something is wrong
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0, (
        "No trainable parameters found! Check that LORA_TARGET_MODULES names "
        "match the actual layer names in M2M100Attention."
    )

    # ── 4. Load tokenized dataset ─────────────────────────────────────────────
    print(f"[train] Loading dataset from '{args.dataset_dir}' ...")
    dataset = load_from_disk(args.dataset_dir)
    train_dataset = dataset["train"]
    eval_dataset  = dataset["validation"]
    print(f"[train] train: {len(train_dataset)} | validation: {len(eval_dataset)}")

    # ── 5. Data collator ──────────────────────────────────────────────────────
    # DataCollatorForSeq2Seq pads batches to the longest sequence in the batch
    # (more efficient than padding to max_length globally in prepare_data.py)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,    # keeps tensor shapes aligned for fp16 ops
    )

    # ── 6. Training arguments ─────────────────────────────────────────────────
    # Evaluation strategy: eval at every epoch end; best checkpoint kept by
    # eval_loss (lower = better) for early stopping.
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,

        # ── Epochs & batch ────────────────────────────────────────────────
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,   # effective batch = 64

        # ── Precision ─────────────────────────────────────────────────────
        fp16=True,           # A5000 has full fp16 support; faster + less VRAM
        bf16=False,          # A5000 does support bf16 but fp16 is fine here

        # ── Optimiser ─────────────────────────────────────────────────────
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",

        # ── Label smoothing ───────────────────────────────────────────────
        label_smoothing_factor=args.label_smoothing,

        # ── Evaluation & checkpointing ────────────────────────────────────
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,    # required for EarlyStoppingCallback
        metric_for_best_model="eval_loss",
        greater_is_better=False,        # lower eval_loss = better
        save_total_limit=3,             # keep only the 3 most recent checkpoints

        # ── Generation settings (used during eval) ────────────────────────
        predict_with_generate=True,
        generation_max_length=args.max_new_tokens,
        generation_config=None,         # we set forced_bos below via model config

        # ── Logging ───────────────────────────────────────────────────────
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=20,
        report_to="none",               # disable W&B / TensorBoard by default

        # ── Misc ──────────────────────────────────────────────────────────
        dataloader_num_workers=4,
        seed=42,
    )

    # Force the decoder to always start with __roman-ur__ (id 128105).
    # This must be set on the base model config so Seq2SeqTrainer picks it up
    # during generation (predict_with_generate=True).
    model.config.forced_bos_token_id = TGT_LANG_TOKEN_ID

    # ── 7. Trainer ────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stop_patience,
                # stops training if eval_loss doesn't improve for N evals
            )
        ],
    )

    # ── 8. Train ──────────────────────────────────────────────────────────────
    print("[train] Starting training ...")
    trainer.train()

    # ── 9. Save best LoRA adapter weights ────────────────────────────────────
    # save_pretrained on a PeftModel saves ONLY the adapter weights (a few MB),
    # not the frozen base model weights.
    os.makedirs(args.final_model_dir, exist_ok=True)
    model.save_pretrained(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)
    print(f"[train] LoRA adapters saved to '{args.final_model_dir}'.")

    # ── 10. Final eval ────────────────────────────────────────────────────────
    print("[train] Running final evaluation on validation set ...")
    metrics = trainer.evaluate()
    print("[train] Final metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()