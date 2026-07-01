# M2M100 LoRA Fine-Tune — Urdu → Roman Urdu Transliteration

Fine-tunes [`Mavkif/m2m100_rup_ur_to_rur`](https://huggingface.co/Mavkif/m2m100_rup_ur_to_rur) with LoRA adapters (via `peft`) to improve transliteration of rare and domain-specific words (e.g. "exercise", "doctor", "computer").

---

## Project structure

```
M2M100-FINETUNE/
├── data/
│   └── final_transliteration_dataset.csv   # urdu + roman_urdu columns
├── prepare_data.py     # tokenise CSV → HF DatasetDict on disk
├── train.py            # LoRA fine-tuning with Seq2SeqTrainer
├── inference.py        # load adapter + run transliteration
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Python 3.10+ required.** The type hint `list[str]` in `inference.py` requires ≥3.9; tested on 3.12.

---

## Quick start

### 1 — Prepare the dataset

```bash
python prepare_data.py \
    --csv data/transliteration_dataset.csv \
    --output_dir ./processed_dataset
```

Tokenises the CSV using `Mavkif/m2m100_rup_tokenizer_both`, performs a 95/5 train/validation split, and saves in HuggingFace Arrow format.

Expected output:
```
[prepare_data] Token IDs confirmed → __ur__: 128095 | __roman-ur__: 128105
[prepare_data] Dataset saved to './processed_dataset'
  first label token: 128105 (__roman-ur__ ✓)
```

### 2 — Fine-tune

```bash
python train.py \
    --dataset_dir ./processed_dataset \
    --output_dir ./checkpoints \
    --final_model_dir ./fine_tuned_model
```

Default hyperparameters target an **NVIDIA A5000 (24 GB)**:

| Flag | Default | Notes |
|---|---|---|
| `--batch_size` | 16 | Per-device train batch size |
| `--grad_accum` | 4 | Effective batch = `batch_size × grad_accum` = 64 |
| `--learning_rate` | 5e-4 | Higher than full FT is fine for LoRA adapters |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | Keep at `2 × lora_r` |
| `--num_epochs` | 30 | Early stopping (patience 3) will usually fire first |

For smaller GPUs (e.g. 8–12 GB): `--batch_size 4 --grad_accum 16`.

Only the LoRA adapter weights are saved to `--final_model_dir` (a few MB, not the frozen base model).

### 3 — Inference

```bash
python inference.py --model_dir ./fine_tuned_model

python inference.py \
    --model_dir ./fine_tuned_model \
    --sentences "ڈاکٹر نے کہا" "کمپیوٹر بند ہے" "ورزش کرو"

python inference.py \
    --model_dir ./fine_tuned_model \
    --input_file urdu_sentences.txt
```

---

## How LoRA is applied

LoRA adapters are injected into all four attention projection layers across
**both encoder and decoder** (self-attention and cross-attention):

    q_proj, k_proj, v_proj, out_proj

`task_type` is intentionally NOT set in `LoraConfig`. Setting
`task_type=TaskType.SEQ_2_SEQ_LM` wraps the model in `PeftModelForSeq2SeqLM`,
whose `forward()` converts `decoder_input_ids → decoder_inputs_embeds` and
passes both to `M2M100Decoder` — crashing with a ValueError. Without
`task_type`, a generic `PeftModel` is returned with a pure pass-through
`forward()`. Adapters are injected identically either way; `.generate()` still
works via `__getattr__` delegation to the base model.

With default settings (`r=16`, `alpha=32`), ~4.7 M of 488 M parameters are
trainable (~0.97%).
---

## Tokeniser notes

Always use **`Mavkif/m2m100_rup_tokenizer_both`**, never `facebook/m2m100_418M`.

| Token | ID |
|---|---|
| `__ur__` | 128095 |
| `__roman-ur__` | 128105 |

In `prepare_data.py`, labels are tokenised with `src_lang="ur"` (prepending
`__ur__`, id 128095 at position 0), then position 0 is manually replaced with
`__roman-ur__` (id 128105). Setting `tokenizer.src_lang = "roman-ur"` directly
raises a KeyError because `"roman-ur"` is not in `lang_code_to_token`, and
`as_target_tokenizer()` was removed in transformers 5.x — neither approach works.

---

## Known compatibility issues and fixes

### transformers 5.x API changes (all fixed in current code)

| Old (4.x) | New (5.x) | File |
|---|---|---|
| `Seq2SeqTrainer(tokenizer=...)` | `Seq2SeqTrainer(processing_class=...)` | `train.py` |
| `warmup_ratio=0.06` | `warmup_steps=0.06` (float < 1 = ratio) | `train.py` |
| `logging_dir=...` | Set `TENSORBOARD_LOGGING_DIR` env var | `train.py` |

### M2M100 + transformers 5.x decoder conflict (fixed in current code)

`M2M100Model.forward()` in transformers 5.x pre-computes `decoder_inputs_embeds` from `decoder_input_ids` via `self.shared()`, then passes **both** to `M2M100Decoder`, which raises:
```
ValueError: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time
```

**Fix**: `M2M100Seq2SeqTrainer` (a thin subclass of `Seq2SeqTrainer` in `train.py`) overrides `compute_loss()` and `prediction_step()` to pop `decoder_inputs_embeds` from the batch before the forward call. No file patching or monkey-patching.

---

## Requirements

| Package | Minimum | Notes |
|---|---|---|
| `torch` | 2.1.0 | |
| `transformers` | 5.0.0 | |
| `peft` | 0.10.0 | |
| `datasets` | 2.18.0 | |
| `sacrebleu` | 2.3.1 | |
| `sentencepiece` | 0.1.99 | Required by M2M100Tokenizer |
| `accelerate` | 0.27.0 | Required by Trainer fp16 path |
