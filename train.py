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
  • M2M100 + transformers 5.x bug: M2M100Model.forward() passes BOTH
    decoder_input_ids AND decoder_inputs_embeds to M2M100Decoder, which
    raises ValueError. Fix: M2M100Seq2SeqTrainer.compute_loss() strips
    decoder_inputs_embeds from the batch before the forward call.
    No file patching, no monkey-patching — pure Python subclass.
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

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

# ── Fix: custom Trainer that strips the conflicting key ──────────────────────

class M2M100Seq2SeqTrainer(Seq2SeqTrainer):
    """
    Thin Seq2SeqTrainer subclass that works around a transformers 5.x bug
    in M2M100Model.forward().

    THE BUG
    -------
    In transformers 5.x, M2M100Model.forward() (modeling_m2m_100.py ~L780)
    pre-computes decoder_inputs_embeds from decoder_input_ids via
    self.shared(), then passes BOTH fields to M2M100Decoder.forward().
    M2M100Decoder.forward() explicitly raises:
        ValueError: You cannot specify both decoder_input_ids and
                    decoder_inputs_embeds at the same time
    This happens on the very first training step.

    THE FIX
    -------
    Override compute_loss() to remove decoder_inputs_embeds from the batch
    dict before calling model(**inputs). M2M100ForConditionalGeneration then
    calls M2M100Model with only decoder_input_ids, and M2M100Model computes
    the embeddings internally without conflict.

    This is purely a data-flow fix — no file patching, no monkey-patching,
    no changes to installed packages. Works regardless of PEFT version.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Strip the conflicting key injected by transformers 5.x collation.
        inputs.pop("decoder_inputs_embeds", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Also strip during eval / generate steps to be safe.
        inputs.pop("decoder_inputs_embeds", None)
        return super().prediction_step(
            model, inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune M2M100 transliteration model")
    p.add_argument("--dataset_dir",    default="./processed_dataset")
    p.add_argument("--output_dir",     default="./checkpoints")
    p.add_argument("--final_model_dir",default="./fine_tuned_model")
    # LoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    # Training
    p.add_argument("--num_epochs",          type=int,   default=30)
    p.add_argument("--batch_size",          type=int,   default=16)
    p.add_argument("--eval_batch_size",     type=int,   default=32)
    p.add_argument("--grad_accum",          type=int,   default=4)
    p.add_argument("--learning_rate",       type=float, default=5e-4)
    p.add_argument("--warmup_ratio",        type=float, default=0.06)
    p.add_argument("--label_smoothing",     type=float, default=0.1)
    p.add_argument("--early_stop_patience", type=int,   default=3)
    p.add_argument("--max_new_tokens",      type=int,   default=128)
    return p.parse_args()

# ── LoRA configuration ────────────────────────────────────────────────────────

def build_lora_config(args) -> LoraConfig:
    """
    TaskType.SEQ_2_SEQ_LM is correct for encoder-decoder.
    It tells PEFT to inject adapters into both encoder and decoder modules.
    The decoder_inputs_embeds conflict is fixed in M2M100Seq2SeqTrainer above,
    not by removing task_type (which would silently break generation).
    """
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
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
    # Disable KV-cache: required for gradient flow through LoRA during training.
    model.config.use_cache = False

    # 3. Apply LoRA
    model = get_peft_model(model, build_lora_config(args))
    model.print_trainable_parameters()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0, \
        "No trainable params — check LORA_TARGET_MODULES names."

    # Force decoder BOS to __roman-ur__ for both training eval and inference.
    model.config.forced_bos_token_id = TGT_LANG_TOKEN_ID

    # 4. Dataset
    print(f"[train] Loading dataset from '{args.dataset_dir}' ...")
    dataset     = load_from_disk(args.dataset_dir)
    train_ds    = dataset["train"]
    eval_ds     = dataset["validation"]
    print(f"[train] train: {len(train_ds)} | validation: {len(eval_ds)}")

    # 5. Collator
    # model= intentionally omitted: passing model= causes the collator to call
    # model.prepare_decoder_input_ids_from_labels(), injecting decoder_input_ids
    # into the batch. Combined with transformers 5.x also computing
    # decoder_inputs_embeds inside M2M100Model.forward(), this creates the
    # dual-key conflict. Without model=, M2M100ForConditionalGeneration shifts
    # labels internally via shift_tokens_right(), which is the correct path.
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
        # warmup_steps accepts float < 1 as ratio in transformers 5.x
        # (replaces the removed warmup_ratio parameter)
        warmup_steps=args.warmup_ratio,
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
        dataloader_num_workers=4,
        seed=42,
    )

    # 7. Trainer — use the subclass that strips decoder_inputs_embeds
    trainer = M2M100Seq2SeqTrainer(
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

    # 9. Save LoRA adapter weights only (not frozen base model)
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