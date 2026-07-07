PROJECT CONTEXT — M2M100 Urdu → Roman Urdu Transliteration Fine-tune (Run 3 complete)

REPO: https://github.com/Maaz-x14/m2m100-rup-ur-to-rur-fine_tune
BASE MODEL: Mavkif/m2m100_rup_ur_to_rur
TOKENIZER: Mavkif/m2m100_rup_tokenizer_both
HARDWARE: NVIDIA A5000 (24GB VRAM), single GPU only
TASK: Fine-tune M2M100 with LoRA — Urdu Arabic script → Roman Urdu transliteration

---

CURRENT MODEL STATE (Run 3 — the good model)

Training completed at epoch 16 via early stopping (patience=3).
Final metrics: eval_loss=2.111 | BLEU=67.36 | chrF=67.29
Adapter saved to: ./fine_tuned_model (adapter_model.safetensors, ~18MB)
Dataset used: transliteration_dataset.csv (2501 rows combined, 2376 train / 125 val after 95/5 split)

Inference quality is good — loanwords (exercise, exam, laptop, crash, delete, backup, 
screenshot, server, password) all transliterate correctly. General transliteration is 
very good.

KNOWN REMAINING ISSUE: iss/uss disambiguation is ~50/50 accuracy — not reliable.

---

ALL BUGS FIXED IN train.py (do NOT reintroduce)

#1  prepare_data.py label tokenisation — labels tokenized with src_lang="ur" then 
    position 0 manually replaced with __roman-ur__ (id 128105)
#2  warmup_steps must be integer; processing_class= not tokenizer=; logging_dir= removed
#3  M2M100 decoder conflict — decoder_inputs_embeds injected in compute_loss ONLY 
    (NOT in prediction_step generation path); PatchedM2M100Model.forward() drops 
    decoder_input_ids when embeds present
#4  task_type=TaskType.SEQ_2_SEQ_LM must NOT be set in LoraConfig
#5  model.config.forced_bos_token_id rejected in transformers 5.x — use 
    model.generation_config.forced_bos_token_id; re-enforced in prediction_step()
#6  OOM during eval — generation_num_beams=1, eval_batch_size=8, 
    expandable_segments:True, empty_cache() after each prediction_step
#7  patch_model() memory leak — del model.model before assigning patched copy
#8  compute_metrics NameError — use build_compute_metrics(tokenizer) closure factory
#9  generation_max_length removed from TrainingArguments — conflicts with max_new_tokens
#10 DataParallel REMOVED — single GPU only. Multi-GPU + fp16 + custom loss override 
    caused catastrophic loss divergence (loss=103→125) in run 2
#11 BLEU=0 eval bug FIXED — prediction_step was injecting decoder_inputs_embeds into 
    generation path causing empty PRED strings. Fix: injection in compute_loss only. 
    prediction_step only re-enforces forced_bos_token_id and calls super() clean.

---

KEY TECHNICAL DETAILS

Token IDs: __ur__ = 128095, __roman-ur__ = 128105
LoRA: r=16, alpha=32, targets q_proj, k_proj, v_proj, out_proj, no task_type
Only LoRA adapter weights saved (adapter_model.safetensors, ~18MB)
inference.py uses merge_and_unload() to bake adapters into base weights
PatchedM2M100Model patch required in both train.py and inference.py

---

DATASET HISTORY

- final_transliteration_dataset.csv: original 2012 rows
- iss_uss_dataset_cleaned.csv: 489 rows iss/uss disambiguation, native-speaker validated
- transliteration_dataset.csv: combined 2501 rows (current training CSV)

IMPORTANT iss/uss finding: اس نے is essentially always "uss" in natural spoken Urdu.
AI-generated iss/uss examples are unreliable — native speaker validation is mandatory.
The model currently achieves ~50/50 on iss/uss — this is the main remaining problem.

---

CURRENT TASK: Improve iss/uss disambiguation

Three approaches to implement (in priority order):

APPROACH 1 (immediate, no retraining): Rule-based postprocessor
- Run on Urdu INPUT before/after model
- uss signals: past tense suffixes (تھا، تھی، تھے، گیا، گئی، آیا) + temporal adverbs 
  (کل، پہلے، برسوں، کبھی)
- iss signals: imperative/present mood + proximal markers (یہاں، ابھی، سامنے)
- Default for ambiguous bare اس نے constructions: uss
- Implement as a postprocessor on model's Roman Urdu output (find "iss"/"uss" tokens, 
  check Urdu source for signal words, override if confident)

APPROACH 2 (after postprocessor validated): Expand iss/uss training data
- Need 2000-3000 more contrastive minimal pair examples
- Structure: same base sentence, different temporal context → different iss/uss
- Must be native-speaker validated (AI-generated pairs are unreliable)
- Retrain with r=32, alpha=64 (increased LoRA rank for longer-range context modeling)

APPROACH 3 (if 1+2 insufficient): Separate binary iss/uss classifier
- Small standalone model: Urdu BERT → iss/uss decision
- Post-processes transliteration output
- Only pursue if rule-based + data approaches don't reach >80% accuracy

---

SECONDARY ISSUES (lower priority than iss/uss)

1. Code-switching loanwords: "message" → "barad", "spreadsheet" → "bread-sheet"
   Model hallucinating on English words embedded in Urdu sentences.
   Root cause: training data has few code-switching examples.
   Fix: add 200-300 code-switching examples to dataset.

2. iss/uss for malika example:
   REF: iss malika se mulaqat karo (proximal, present)
   PRED: uss malka se mulaqaat karo
   This is an iss/uss error + a vowel omission (malika → malka). Two separate bugs.

---

MENTOR STYLE INSTRUCTIONS (carry these into all responses)

- Be ruthless — stress-test every idea until it's bulletproof, call out wrong thinking 
  immediately, no sugarcoating
- Gen Z communication — direct, blunt, witty, get to the point fast
- Structured layered responses (bullets, formulas, outlines) but concise
- No unnecessary elaboration, no fluff
- Pakistan context, single A5000 GPU, production deployment target is DigitalOcean GPU 
  server with FastAPI serving

---

FILES TO ATTACH WITH THIS PROMPT:
- train.py (current version with all 11 bugs fixed)
- inference.py
- prepare_data.py
- transliteration_dataset.csv (or path reference)
- iss_uss_dataset_cleaned.csv

Start by implementing Approach 1 (rule-based postprocessor) as a standalone 
postprocess.py script that can wrap inference.py output.