"""
train.py
--------
Fine-tunes Mavkif/m2m100_rup_ur_to_rur with LoRA (via peft) using
HuggingFace Seq2SeqTrainer.

LoRA applied to: q_proj, k_proj, v_proj, out_proj (encoder + decoder)
Hardware target: NVIDIA A5000 (24 GB VRAM), fp16, batch=16, grad_accum=4

Bugs fixed (do NOT reintroduce):
  #1 prepare_data.py label tokenisation — see prepare_data.py
  #2 Seq2SeqTrainer(tokenizer=...) → processing_class=
  #2 warmup_steps= must be an INTEGER computed from ratio × total_steps;
       passing a raw float (e.g. 0.06) is silently truncated to 0 — zero warmup
  #2 logging_dir= removed; use TENSORBOARD_LOGGING_DIR env var
  #3 M2M100 decoder conflict (training path) — M2M100Seq2SeqTrainer injects
       decoder_inputs_embeds in compute_loss ONLY so shift_tokens_right is skipped
  #3 M2M100 decoder conflict (generation path) — PatchedM2M100Model.forward()
       drops decoder_input_ids when decoder_inputs_embeds already present
  #4 task_type=TaskType.SEQ_2_SEQ_LM MUST NOT be set — PeftModelForSeq2SeqLM
       re-injects decoder_input_ids after our strip, re-triggering the conflict
  #5 model.config.forced_bos_token_id rejected in transformers 5.x —
       use model.generation_config.forced_bos_token_id instead
  #5 Trainer syncs generation_config from tokenizer AFTER model setup, wiping
       forced_bos_token_id — fixed by re-enforcing in prediction_step()
  #6 OOM during eval — beam search on 128k vocab with large batch causes spike:
       eval_batch=32 * beams=4 * seq=128 * vocab=128112 * fp16 ≈ 4.2 GB logits
       Fix: generation_num_beams=1 (greedy eval), eval_batch_size default → 8,
       expandable_segments allocator, empty_cache after each prediction_step
  #7 patch_model() memory leak — del model.model before assigning patched copy,
       otherwise both live in VRAM simultaneously (488M params duplicated)
  #8 compute_metrics referenced tokenizer as a global (NameError at eval time);
       build_compute_metrics(tokenizer) closure factory restores correct scoping
  #9 generation_max_length in TrainingArguments conflicts with max_new_tokens in
       GenerationConfig (different semantics: absolute vs relative position);
       removed generation_max_length — GenerationConfig.max_new_tokens is canonical
  #10 DataParallel multi-GPU removed — single A5000 (24 GB) is sufficient for
       2500 rows; DataParallel + fp16 + custom loss override is a minefield and
       caused catastrophic loss divergence (loss=103→125) in run 2
  #11 BLEU=0 eval bug — prediction_step was injecting decoder_inputs_embeds into
       the GENERATION path; .generate() is autoregressive and handles decoder
       inputs itself token-by-token — passing pre-computed embeddings caused it
       to generate from fixed teacher-forced context → empty PRED strings every
       eval. Fix: decoder_inputs_embeds injection is now compute_loss ONLY.
       prediction_step only re-enforces forced_bos_token_id and calls super().
"""

import os
import glob
import argparse
import numpy as np
from transformers import GenerationConfig

# ── MUST be set before torch initialises the CUDA allocator ──────────────────
# Reduces fragmentation that causes the VRAM spike during eval generation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Model
from peft import LoraConfig, get_peft_model, PeftModel
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
    p.add_argument("--eval_batch_size",      type=int,   default=8)   # bug #6: was 32
    p.add_argument("--grad_accum",           type=int,   default=4)
    p.add_argument("--learning_rate",        type=float, default=5e-4)
    p.add_argument("--warmup_ratio",         type=float, default=0.06)
    p.add_argument("--label_smoothing",      type=float, default=0.1)
    p.add_argument("--early_stop_patience",  type=int,   default=3)
    p.add_argument("--max_new_tokens",       type=int,   default=128)
    p.add_argument("--dataloader_workers",   type=int,   default=2)
    p.add_argument("--init_adapter_dir",     default=None,
        help="Path to an EXISTING, already-trained LoRA adapter (e.g. a prior "
             "run's fine_tuned_model/) to continue training from. If set, this "
             "adapter's weights are loaded onto the base model as the starting "
             "point instead of a fresh randomly-initialised adapter. This is "
             "DIFFERENT from --output_dir checkpoint resume, which only resumes "
             "a crashed run of THIS SAME training job. Use --init_adapter_dir to "
             "start a NEW training run (new data, new run name) on top of a "
             "previously completed adapter.")
    return p.parse_args()

# ── LoRA configuration ────────────────────────────────────────────────────────

def build_lora_config(args) -> LoraConfig:
    # task_type intentionally omitted — bug #4
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        inference_mode=False,
    )

# ── PatchedM2M100Model — fixes decoder conflict in generation path (bug #3) ──

class PatchedM2M100Model(M2M100Model):
    """
    M2M100Model.forward() in transformers 5.x computes decoder_inputs_embeds
    from decoder_input_ids, then passes BOTH to M2M100Decoder → ValueError.

    Fix: drop decoder_input_ids when decoder_inputs_embeds already provided.
    Covers the .generate() path which our Trainer override cannot intercept.
    """
    def forward(self, *args, **kwargs):
        if kwargs.get("decoder_inputs_embeds") is not None:
            kwargs.pop("decoder_input_ids", None)
        return super().forward(*args, **kwargs)


def patch_model(model: M2M100ForConditionalGeneration) -> M2M100ForConditionalGeneration:
    """
    Swaps model.model in-place with PatchedM2M100Model.
    del before assign is critical — bug #7 (double VRAM usage without it).
    """
    original = model.model
    patched  = PatchedM2M100Model(model.config)
    patched.load_state_dict(original.state_dict())
    patched.to(next(original.parameters()).dtype)
    patched.to(next(original.parameters()).device)
    del model.model            # free original BEFORE assigning — bug #7
    model.model = patched
    torch.cuda.empty_cache()
    return model

# ── M2M100Seq2SeqTrainer — fixes decoder conflict in training path (bug #3) ──

class M2M100Seq2SeqTrainer(Seq2SeqTrainer):
    """
    Injects decoder_inputs_embeds in compute_loss (training path) ONLY.

    WHY decoder_inputs_embeds must NOT go into prediction_step (bug #11):
    ─────────────────────────────────────────────────────────────────────
    During eval with predict_with_generate=True, super().prediction_step()
    calls model.generate() internally. .generate() is autoregressive — it
    builds the decoder sequence one token at a time, starting from BOS.
    If decoder_inputs_embeds is present in the batch, the parent Trainer
    passes it to generate() as a kwargs override, which feeds the entire
    teacher-forced embedding sequence as the decoder context BEFORE any
    generation happens. The model then "generates" zero additional tokens
    on top of an already-complete sequence → empty PRED strings → BLEU=0.

    Fix: prediction_step only re-enforces forced_bos_token_id (bug #5)
    and delegates to super() with a clean batch — no embeds injected.
    decoder_inputs_embeds injection stays in compute_loss (training) only.
    """

    @staticmethod
    def _base(model):
        # Unwrap DataParallel/DDP if present (safety guard; we run single GPU).
        # get_base_model() unwraps the PEFT wrapper to reach the underlying
        # M2M100ForConditionalGeneration.
        if hasattr(model, "module"):
            model = model.module
        return model.get_base_model() if hasattr(model, "get_base_model") else model

    def _decoder_inputs_embeds(self, model, labels):
        base     = self._base(model)
        pad_id   = base.config.pad_token_id
        start_id = base.config.decoder_start_token_id

        clean          = labels.masked_fill(labels == -100, pad_id)
        shifted        = clean.new_zeros(clean.shape)
        shifted[:, 1:] = clean[:, :-1]
        shifted[:, 0]  = start_id
        return base.model.shared(shifted)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Training path only — inject teacher-forced embeddings so
        # M2M100ForConditionalGeneration skips shift_tokens_right and
        # never passes decoder_input_ids into M2M100Model (bug #3).
        inputs = dict(inputs)
        if "labels" in inputs and "decoder_inputs_embeds" not in inputs:
            inputs["decoder_inputs_embeds"] = self._decoder_inputs_embeds(
                model, inputs["labels"]
            )
        return super().compute_loss(
            model, inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # ── Bug #5: re-enforce target BOS token before every eval generation ──
        # The Trainer re-syncs generation_config from the tokenizer after model
        # setup, wiping forced_bos_token_id. Re-apply it here on every call.
        base = self._base(model)
        base.generation_config.forced_bos_token_id = TGT_LANG_TOKEN_ID

        # ── Bug #11: do NOT inject decoder_inputs_embeds here ──────────────
        # Generation is autoregressive. Passing pre-computed embeddings to
        # generate() causes it to skip its own decoding loop → empty outputs.
        # The batch goes to super() as-is; generate() handles decoder inputs.
        result = super().prediction_step(
            model, inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )
        # Free beam-search intermediates immediately — bug #6
        torch.cuda.empty_cache()
        return result

# ── Metrics ───────────────────────────────────────────────────────────────────

def build_compute_metrics(tokenizer):
    """
    Returns compute_metrics as a closure over tokenizer.
    Must be a factory — tokenizer is a local in main(), NOT a global (bug #8).
    """
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds  = np.where(preds  < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        decoded_preds  = [p.strip() for p in tokenizer.batch_decode(preds,  skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        # ── PROBE: print first 5 REF/PRED pairs + raw token IDs ─────────────
        # PRED non-empty → generation is working.
        # PRED empty + raw IDs are all pad/EOS → forced_bos_token_id not applied.
        # PRED is Urdu script → wrong BOS token, model generating in src language.
        print("\n── Eval probe (first 5 examples) ──")
        for i in range(min(5, len(decoded_preds))):
            print(f"  REF  [{i}]: {decoded_labels[i]}")
            print(f"  PRED [{i}]: {decoded_preds[i]}")
            print(f"  RAW IDs  : {preds[i][:15].tolist()}")
            print()

        # Summary stats — catch silent empty-generation at scale
        empty_count = sum(1 for p in decoded_preds if p == "")
        if empty_count > 0:
            print(f"  [WARNING] {empty_count}/{len(decoded_preds)} PRED strings are empty — "
                  f"check forced_bos_token_id and prediction_step.")

        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        chrf = sacrebleu.corpus_chrf(decoded_preds, [[l] for l in decoded_labels])
        return {
            "bleu": round(bleu.score, 4),
            "chrf": round(chrf.score, 4),
        }
    return compute_metrics

# ── Checkpoint guard ──────────────────────────────────────────────────────────

def check_checkpoint_state(output_dir: str) -> str | None:
    """
    Detects existing checkpoints in output_dir.
    Prints a clear warning so you know whether training resumes or starts fresh.
    Returns the checkpoint path to resume from, or None to start fresh.

    IMPORTANT: if a previous run crashed mid-step (e.g. due to DataParallel
    error), the checkpoint saved at that point may contain corrupted optimizer
    state. If loss immediately looks wrong (>50 at step 1), delete checkpoints
    and restart from scratch.
    """
    checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    if not checkpoints:
        print(f"[train] No checkpoints found in '{output_dir}' — starting fresh.")
        return None

    latest = checkpoints[-1]
    print(f"[train] WARNING: Found {len(checkpoints)} checkpoint(s) in '{output_dir}'.")
    print(f"[train] Will RESUME from: {latest}")
    print(f"[train] If this is unintended (e.g. after a crashed run), delete '{output_dir}' and restart.")
    return latest

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Single GPU enforcement (bug #10) ─────────────────────────────────────
    # DataParallel + fp16 + custom compute_loss override caused catastrophic
    # loss divergence in run 2 (loss=103→125). A5000 has 24 GB — more than
    # enough for 2500 rows at batch=16. Single GPU is stable and correct.
    if torch.cuda.device_count() > 1:
        print(f"[train] {torch.cuda.device_count()} GPUs detected. "
              f"Using single GPU only (CUDA_VISIBLE_DEVICES=0). "
              f"DataParallel is disabled — see bug #10.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    model.config.use_cache = False

    # bug #5: transformers 5.x rejects forced_bos_token_id on model.config;
    # also, the Trainer re-syncs generation_config from the tokenizer after
    # setup — prediction_step() re-enforces this on every eval call.
    model.config.forced_bos_token_id = None
    model.generation_config.forced_bos_token_id = TGT_LANG_TOKEN_ID

    # 3. Patch M2M100Model — must be BEFORE attaching any LoRA adapter
    model = patch_model(model)
    print("[train] PatchedM2M100Model installed.")

    # 4. Attach LoRA adapter
    if args.init_adapter_dir:
        # Continue training an EXISTING, already-trained adapter (e.g. run2's
        # fine_tuned_model/) instead of starting from a fresh random adapter.
        # This is the correct way to do "run3 continues from run2" — the
        # checkpoint resume logic below only resumes a crashed run of THIS
        # job, it does NOT load a previously-completed adapter.
        print(f"[train] Loading EXISTING adapter from '{args.init_adapter_dir}' "
              f"to continue training (NOT starting a fresh adapter).")
        model = PeftModel.from_pretrained(
            model, args.init_adapter_dir, is_trainable=True
        )
    else:
        print("[train] No --init_adapter_dir given — starting a FRESH LoRA "
              "adapter from the base model (run2's weights will NOT be used).")
        model = get_peft_model(model, build_lora_config(args))
    model.print_trainable_parameters()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0, \
        "No trainable params — check LORA_TARGET_MODULES names."

    # 5. Dataset
    print(f"[train] Loading dataset from '{args.dataset_dir}' ...")
    dataset  = load_from_disk(args.dataset_dir)
    train_ds = dataset["train"]
    eval_ds  = dataset["validation"]
    print(f"[train] train: {len(train_ds)} | validation: {len(eval_ds)}")

    # Sanity check — warn if dataset looks like the old 1911-row version
    if len(train_ds) < 2000:
        print(f"[train] WARNING: only {len(train_ds)} training rows detected. "
              f"Expected ~2376 (full combined dataset). "
              f"Did you re-run prepare_data.py on transliteration_dataset.csv?")

    # 6. Collator — model= omitted: supplying it injects decoder_input_ids
    #    into the batch, conflicting with M2M100's internal shift_tokens_right
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # bug #9: max_new_tokens in GenerationConfig is the canonical limit.
    # Do NOT also set generation_max_length in TrainingArguments — it maps to
    # max_length (absolute position), conflicting with max_new_tokens (relative).
    gen_config = GenerationConfig(
        forced_bos_token_id=TGT_LANG_TOKEN_ID,
        max_new_tokens=args.max_new_tokens,
        num_beams=1,
    )

    # bug #2: warmup_steps must be an integer; passing a raw float (e.g. 0.06)
    # is silently truncated to 0, giving zero warmup and destabilising training.
    total_steps  = (len(train_ds) // (args.batch_size * args.grad_accum)) * args.num_epochs
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    print(f"[train] total_steps={total_steps} | warmup_steps={warmup_steps} "
          f"| lr={args.learning_rate}")

    # 7. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        fp16=True,
        bf16=False,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,         # bug #2: integer, computed above
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
        # generation_max_length intentionally omitted — bug #9
        generation_num_beams=1,            # bug #6: greedy eval — 4x less VRAM than beam=4
        generation_config=gen_config,
        logging_steps=20,
        report_to="none",
        dataloader_num_workers=args.dataloader_workers,
        seed=42,
    )

    # 8. Checkpoint guard — detect stale/corrupted checkpoints before training
    resume_from = check_checkpoint_state(args.output_dir)

    # 9. Trainer
    trainer = M2M100Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),   # bug #8: closure factory
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
    )

    # 10. Train
    print("[train] Starting training ...")
    trainer.train(resume_from_checkpoint=resume_from)

    # 11. Save LoRA adapter weights only (not frozen base model)
    os.makedirs(args.final_model_dir, exist_ok=True)
    model.save_pretrained(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)
    print(f"[train] LoRA adapters saved to '{args.final_model_dir}'.")

    # 12. Final eval
    print("[train] Running final evaluation ...")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()