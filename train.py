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
       decoder_inputs_embeds so shift_tokens_right is skipped
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
"""

import os
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
    p.add_argument("--eval_batch_size",      type=int,   default=8)   # bug #6: was 32
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
    Pre-computes decoder_inputs_embeds from labels before compute_loss /
    prediction_step so M2M100ForConditionalGeneration skips shift_tokens_right
    and never passes decoder_input_ids into M2M100Model.
    """

    @staticmethod
    def _base(model):
        if hasattr(model, "module"):        # unwrap DataParallel / DistributedDataParallel
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
        # Re-enforce target language BOS token before every eval generation.
        # The Trainer syncs generation_config from the tokenizer after model
        # setup (bug #5), wiping forced_bos_token_id — this re-applies it.
        base = self._base(model)
        base.generation_config.forced_bos_token_id = TGT_LANG_TOKEN_ID

        inputs = dict(inputs)
        if "labels" in inputs and "decoder_inputs_embeds" not in inputs:
            with torch.no_grad():
                inputs["decoder_inputs_embeds"] = self._decoder_inputs_embeds(
                    model, inputs["labels"]
                )
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
    Must be a factory — tokenizer is a local in main(), NOT a global.
    Using a bare module-level function that references `tokenizer` directly
    will crash with NameError at eval time (bug #8).
    """
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds  = np.where(preds  < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        decoded_preds  = [p.strip() for p in tokenizer.batch_decode(preds,  skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        # 🔍 PROBE — print first 5 pairs each eval to confirm script + content
        for i in range(min(5, len(decoded_preds))):
            print(f"  REF : {decoded_labels[i]}")
            print(f"  PRED: {decoded_preds[i]}")
            print()

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
    model.config.use_cache = False

    # bug #5: transformers 5.x rejects forced_bos_token_id on model.config;
    # also, the Trainer re-syncs generation_config from the tokenizer after
    # setup — prediction_step() re-enforces this on every eval call.
    model.config.forced_bos_token_id = None
    model.generation_config.forced_bos_token_id = TGT_LANG_TOKEN_ID

    # 3. Patch M2M100Model — must be BEFORE get_peft_model()
    model = patch_model(model)
    print("[train] PatchedM2M100Model installed.")

    # 4. Apply LoRA
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

    # 8. Trainer
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

    # 9. Train
    print("[train] Starting training ...")
    trainer.train()

    # 10. Save LoRA adapter weights only (not frozen base model)
    os.makedirs(args.final_model_dir, exist_ok=True)
    model.save_pretrained(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)
    print(f"[train] LoRA adapters saved to '{args.final_model_dir}'.")

    # 11. Final eval
    print("[train] Running final evaluation ...")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()