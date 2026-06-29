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

Compatibility: transformers >= 5.0 + peft >= 0.10
  • Seq2SeqTrainer: tokenizer= → processing_class=
  • TrainingArguments: warmup_ratio= → warmup_steps= (float<1 = ratio in 5.x)
  • TrainingArguments: logging_dir= removed entirely

WHY task_type IS NOT SET IN LoraConfig
---------------------------------------
When task_type=TaskType.SEQ_2_SEQ_LM is given, get_peft_model() wraps the
model in PeftModelForSeq2SeqLM. That class's forward() converts
decoder_input_ids → decoder_inputs_embeds internally, then passes BOTH to
M2M100ForConditionalGeneration. M2M100Decoder.forward() raises ValueError
when it receives both simultaneously. This happens regardless of whether
we strip decoder_inputs_embeds in a Trainer subclass, because PEFT re-injects
it inside its own forward() AFTER our strip.

Without task_type, get_peft_model() returns a generic PeftModel whose
forward() is a pure pass-through with no decoder embedding conversion.
LoRA adapters are injected identically in both cases — task_type only
controls the wrapper class, not where adapters are applied.
Generation still works because PeftModel delegates .generate() to the
base model via __getattr__.
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
from peft import LoraConfig, get_peft_model
import sacrebleu

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID          = "Mavkif/m2m100_rup_ur_to_rur"
TOKENIZER_ID      = "Mavkif/m2m100_rup_tokenizer_both"
TGT_LANG_TOKEN_ID = 128105   # __roman-ur__
SRC_LANG          = "ur"

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune M2M100 transliteration model")
    p.add_argument("--dataset_dir",     default="./processed_dataset")
    p.add_argument("--output_dir",      default="./checkpoints")
    p.add_argument("--final_model_dir", default="./fine_tuned_model")
    p.add_argument("--lora_r",               type=int,   default=16)
    p.add_argument("--lora_alpha",           type=int,   default=32)
    p.add_argument("--lora_dropout",         type=float, default=0.1)
    p.add_argument("--num_epochs",           type=int,   default=30)
    p.add_argument("--batch_size",           type=int,   default=16)
    p.add_argument("--eval_batch_size",      type=int,   default=32)
    p.add_argument("--grad_accum",           type=int,   default=4)
    p.add_argument("--learning_rate",        type=float, default=5e-4)
    p.add_argument("--warmup_ratio",         type=float, default=0.06)
    p.add_argument("--label_smoothing",      type=float, default=0.1)
    p.add_argument("--early_stop_patience",  type=int,   default=3)
    p.add_argument("--max_new_tokens",       type=int,   default=128)
    p.add_argument("--dataloader_workers",   type=int,   default=2)
    return p.parse_args()

# ── LoRA configuration ────────────────────────────────────────────────────────

def build_lora_config(args) -> LoraConfig:
    # task_type intentionally omitted — see module docstring for full explanation.
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        inference_mode=False,
    )

# ── Metrics ───────────────────────────────────────────────────────────────────

def build_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds  = np.where(preds  < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        decoded_preds  = [p.strip() for p in tokenizer.batch_decode(preds,  skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        chrf = sacrebleu.corpus_chrf(decoded_preds, [[l] for l in decoded_labels])
        return {
            "bleu": round(bleu.score, 4),
            "chrf": round(chrf.score, 4),
        }
    return compute_metrics

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. Tokenizer
    print(f"[train] Loading tokenizer from '{TOKENIZER_ID}' ...")
    tokenizer = M2M100Tokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.src_lang = SRC_LANG

    # 2. Base model
    print(f"[train] Loading base model from '{MODEL_ID}' ...")
    model = M2M100ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    # Required for gradient flow through LoRA during training
    model.config.use_cache = False
    # Force decoder to start with __roman-ur__ at generation time
    model.config.forced_bos_token_id = TGT_LANG_TOKEN_ID

    # 3. Apply LoRA (generic PeftModel — no task_type, no decoder embedding conflict)
    model = get_peft_model(model, build_lora_config(args))
    model.print_trainable_parameters()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0, \
        "No trainable params — check LORA_TARGET_MODULES names."

    # 4. Dataset
    print(f"[train] Loading dataset from '{args.dataset_dir}' ...")
    dataset  = load_from_disk(args.dataset_dir)
    train_ds = dataset["train"]
    eval_ds  = dataset["validation"]
    print(f"[train] train: {len(train_ds)} | validation: {len(eval_ds)}")

    # 5. Collator
    # model= omitted intentionally: supplying model= calls
    # prepare_decoder_input_ids_from_labels() which injects decoder_input_ids
    # into the batch. M2M100ForConditionalGeneration already does this
    # internally via shift_tokens_right(), so passing it from the collator
    # creates a duplicate that causes errors.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # 6. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        fp16=True,
        bf16=False,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_ratio,   # float<1 treated as ratio in transformers 5.x
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        label_smoothing_factor=args.label_smoothing,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=args.max_new_tokens,
        generation_config=None,
        logging_steps=20,
        report_to="none",
        dataloader_num_workers=args.dataloader_workers,
        seed=42,
    )

    # 7. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
    )

    # 8. Train
    print("[train] Starting training ...")
    trainer.train()

    # 9. Save LoRA adapter weights only (not the frozen base model)
    os.makedirs(args.final_model_dir, exist_ok=True)
    model.save_pretrained(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)
    print(f"[train] LoRA adapters saved to '{args.final_model_dir}'.")

    # 10. Final eval
    print("[train] Running final evaluation ...")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()