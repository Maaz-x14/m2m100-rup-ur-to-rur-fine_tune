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

Compatibility: transformers >= 5.0
  • Seq2SeqTrainer: tokenizer= removed → processing_class=
  • Seq2SeqTrainingArguments: warmup_ratio= removed → warmup_steps= (float < 1
    is interpreted as ratio in 5.x)
  • Seq2SeqTrainingArguments: logging_dir= removed → removed entirely; set the
    TENSORBOARD_LOGGING_DIR environment variable if TensorBoard logging is needed
"""

import os
import sys
import pathlib
import importlib
import importlib.util

# ── GPU selection ────────────────────────────────────────────────────────────
# Pin to GPU 0 (fully free, 23 GB). DataParallel + PEFT SEQ_2_SEQ_LM triggers
# the "decoder_input_ids and decoder_inputs_embeds simultaneously" crash.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ── Patch M2M100Decoder.forward (must run before transformers is imported) ─────

def _patch_m2m100_decoder():
    """
    transformers 5.x bug: M2M100Model.forward pre-computes decoder_inputs_embeds
    from decoder_input_ids via self.shared(...), then passes BOTH to self.decoder().
    M2M100Decoder.forward raises ValueError when it receives both simultaneously.

    Correct fix: in the conflict branch, set input_ids = None (not inputs_embeds).
    The subsequent elif inputs_embeds is not None: branch then runs correctly,
    setting input_shape from the embeddings and proceeding without calling
    embed_tokens (which would fail because input_ids is now None).

    Previous incorrect attempt set inputs_embeds = None, which skipped the elif
    and then called embed_tokens(input_ids) with a potentially-None input_ids.

    Also reverts any previous bad patch automatically.
    """
    spec = importlib.util.find_spec("transformers")
    if not spec or not spec.origin:
        print("[patch] Cannot locate transformers — skipped.")
        return
    m2m_file = (pathlib.Path(spec.origin).parent /
                "models" / "m2m_100" / "modeling_m2m_100.py")
    if not m2m_file.exists():
        print(f"[patch] {m2m_file} not found — skipped.")
        return

    src = m2m_file.read_text(encoding="utf-8")

    # ── Revert any old incorrect patch ────────────────────────────────────────
    old_bad = ('        inputs_embeds = None  '
               '# PATCHED: prefer input_ids when both provided (PEFT+transformers5.x bug)')
    raise_line = ('        raise ValueError("You cannot specify both decoder_input_ids'
                  ' and decoder_inputs_embeds at the same time")')
    if old_bad in src:
        src = src.replace(old_bad, raise_line, 1)
        print("[patch] Reverted old incorrect patch first.")

    # ── Apply the correct patch ────────────────────────────────────────────────
    correct_fix = ('        input_ids = None  '
                   '# PATCHED: prefer pre-computed inputs_embeds (transformers5.x M2M100Model bug)')
    correct_tag = 'PATCHED: prefer pre-computed inputs_embeds'

    if correct_tag in src:
        print("[patch] Already correctly patched.")
        return
    if raise_line not in src:
        print("[patch] WARNING: expected line not found — patch skipped.")
        return

    src = src.replace(raise_line, correct_fix, 1)
    m2m_file.write_text(src, encoding="utf-8")
    importlib.invalidate_caches()
    print(f"[patch] Correctly patched M2M100Decoder.forward in {m2m_file.name}")

_patch_m2m100_decoder()

# ── Imports ──────────────────────────────────────────────────────────────────────
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
    p.add_argument("--warmup_ratio",      type=float, default=0.06,
                   help="Fraction of total steps used for LR warmup (passed as warmup_steps "
                        "float to transformers 5.x, which accepts floats < 1 as a ratio).")
    p.add_argument("--label_smoothing",   type=float, default=0.1)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--max_new_tokens",    type=int,   default=128,
                   help="Max tokens to generate during eval (for BLEU/chrF).")
    return p.parse_args()

# ── LoRA configuration ────────────────────────────────────────────────────────

def build_lora_config(args) -> LoraConfig:
    """
    LoRA config for M2M100 — task_type intentionally omitted.

    WHY task_type is NOT set
    ────────────────────────
    When task_type=TaskType.SEQ_2_SEQ_LM is given, get_peft_model() wraps
    the model in PeftModelForSeq2SeqLM.  That class's forward() contains
    prompt-learning code that converts decoder_input_ids → decoder_inputs_embeds
    even for plain LoRA, then passes BOTH to M2M100ForConditionalGeneration.
    M2M100Decoder.forward() explicitly rejects receiving both simultaneously
    (raises ValueError / causes TypeError).

    Without task_type, get_peft_model() returns a generic PeftModel whose
    forward() is a simple pass-through:
        return self.base_model(*args, **kwargs)
    No decoder-embedding conversion, no conflict.

    LoRA adapters are applied identically in both cases — task_type only
    controls the wrapper class, not where the low-rank matrices are injected.
    Generation (predict_with_generate=True) still works because PeftModel
    delegates unknown attributes (including .generate) to the base model.
    """
    return LoraConfig(
        # task_type deliberately omitted — see docstring above
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
    tokenizer.src_lang = SRC_LANG

    # ── 2. Load base model ────────────────────────────────────────────────────
    print(f"[train] Loading base model from '{MODEL_ID}' ...")
    model = M2M100ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # load weights in fp16 to save VRAM
    )

    # Disable KV-cache: incompatible with LoRA gradient flow during training.
    # Must be set BEFORE get_peft_model wraps the model.
    model.config.use_cache = False

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
    # (more efficient than padding to max_length globally in prepare_data.py).
    #
    # NOTE: model= is intentionally omitted.  When model= is supplied the
    # collator calls model.prepare_decoder_input_ids_from_labels() and injects
    # decoder_input_ids into every batch.  PEFT >= 0.12 then converts those
    # ids to decoder_inputs_embeds internally (SEQ_2_SEQ_LM prompt paths)
    # and passes BOTH to M2M100ForConditionalGeneration, which crashes.
    # Without model=, M2M100 creates decoder_input_ids from labels via
    # shift_tokens_right() internally, bypassing the PEFT conflict entirely.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,    # keeps tensor shapes aligned for fp16 ops
    )

    # ── 6. Training arguments ─────────────────────────────────────────────────
    # Evaluation strategy: eval at every epoch end; best checkpoint kept by
    # eval_loss (lower = better) for early stopping.
    #
    # transformers 5.x API notes:
    #   • warmup_ratio= was removed in 5.x; warmup_steps= now accepts a float
    #     in [0, 1) and treats it as a ratio — functionally identical.
    #   • logging_dir= was removed in 5.x; set the TENSORBOARD_LOGGING_DIR
    #     environment variable instead if you need TensorBoard logs.
    #   • tokenizer= was removed from Seq2SeqTrainer.__init__; the replacement
    #     is processing_class= (see step 7 below).
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
        # warmup_steps accepts a float < 1 in transformers 5.x; treated as
        # warmup ratio (replaces the removed warmup_ratio parameter).
        warmup_steps=args.warmup_ratio,
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
        # logging_dir= was removed in transformers 5.x.
        # Set env var TENSORBOARD_LOGGING_DIR before running if you need it:
        #   export TENSORBOARD_LOGGING_DIR=./checkpoints/logs
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
    # transformers 5.x: `tokenizer=` was removed from Trainer.__init__.
    # The replacement is `processing_class=`, which accepts a tokenizer,
    # feature extractor, or processor. Functionally identical for our use case.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,     # replaces tokenizer= (removed in 5.x)
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