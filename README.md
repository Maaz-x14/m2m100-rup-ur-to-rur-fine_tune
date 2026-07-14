# M2M100 LoRA Fine-Tune — Urdu → Roman Urdu Transliteration

Fine-tunes [`Mavkif/m2m100_rup_ur_to_rur`](https://huggingface.co/Mavkif/m2m100_rup_ur_to_rur) with LoRA adapters (via `peft`) to improve transliteration of rare and domain-specific words (e.g. "exercise", "doctor", "computer") and iss/uss disambiguation.

---

## Project structure

```
m2m100-rup-ur-to-rur-fine_tune/
├── data/
│   ├── final_transliteration_dataset.csv   # original 2012-row dataset
│   └── transliteration_dataset.csv         # combined dataset (2501 rows, incl. iss/uss)
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

Always run this on the combined dataset before training:

```bash
python prepare_data.py \
    --csv data/roman_urdu_map.csv \
    --output_dir ./processed_dataset
```

Tokenises the CSV using `Mavkif/m2m100_rup_tokenizer_both`, performs a 95/5 train/validation split, and saves in HuggingFace Arrow format.

Expected output:
```
[prepare_data] Loaded 2501 rows from 'data/transliteration_dataset.csv'
[prepare_data] Split → train: 2376 | validation: 125
[prepare_data] Token IDs confirmed → __ur__: 128095 | __roman-ur__: 128105
[prepare_data] Dataset saved to './processed_dataset'
  first label token: 128105 (__roman-ur__ ✓)
```

If you see `train: 1911` instead of `~2376`, you ran prepare_data.py on the old dataset. Re-run with `--csv data/transliteration_dataset.csv`.

### 2 — Fine-tune (A5000 / full run)

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

Only the LoRA adapter weights are saved to `--final_model_dir` (a few MB, not the frozen base model).

### 2b — Fine-tune (Colab T4 / smoke test)

```python
# Cell 1 — pull latest code
%cd /content
!rm -rf m2m100-rup-ur-to-rur-fine_tune
!git clone https://github.com/Maaz-x14/m2m100-rup-ur-to-rur-fine_tune
%cd m2m100-rup-ur-to-rur-fine_tune
```

```python
# Cell 2 — install deps
!pip install -r requirements.txt -q
```

```python
# Cell 3 — verify GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

```python
# Cell 4 — upload combined dataset
import os, shutil
os.makedirs("data", exist_ok=True)
from google.colab import files
uploaded = files.upload()
shutil.move(list(uploaded.keys())[0], "data/transliteration_dataset.csv")
print("✓ dataset ready")
```

```python
# Cell 5 — prepare data
!python prepare_data.py \
    --csv data/transliteration_dataset.csv \
    --output_dir ./processed_dataset
```

```python
# Cell 6 — smoke test (2 epochs, T4-safe batch sizes)
!python train.py \
    --batch_size 4 \
    --grad_accum 16 \
    --eval_batch_size 2 \
    --num_epochs 2 \
    --early_stop_patience 5 \
    --dataloader_workers 0
```

T4 has 16 GB VRAM. Key differences from A5000 run: smaller batch, more grad accumulation to keep effective batch=64, `dataloader_workers=0` (Colab multiprocessing issues), `early_stop_patience=5` so early stopping doesn't fire before both epochs complete.

What to verify after epoch 1 eval:
- `train: 2376 | validation: 125` at startup (not 1911)
- PRED lines in probe output contain actual Roman Urdu text (not empty strings)
- `eval_loss` is in the range 2-5 (not 9+ which signals a broken training run)
- BLEU and chrF are non-zero

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

## Known bugs and fixes

### Training stability

| Bug | Symptom | Fix |
|---|---|---|
| #10 DataParallel + fp16 + custom loss | loss=103→125, catastrophic divergence | Single GPU enforced via `CUDA_VISIBLE_DEVICES=0` at startup |
| Stale/corrupted checkpoints | Loss explodes from step 1 if resuming from a crashed run | `rm -rf ./checkpoints/*` before any fresh run; startup now prints clearly whether it's resuming or starting fresh |

### BLEU/chrF always 0 during training eval

| Bug | Root cause | Fix |
|---|---|---|
| #11 `decoder_inputs_embeds` in generation path | `prediction_step` injected teacher-forced embeddings into `.generate()`, which treated them as already-decoded context → zero tokens generated → empty PRED strings | `decoder_inputs_embeds` injection removed from `prediction_step`; now only in `compute_loss` (training path) |

This did NOT affect inference quality — `inference.py` uses `.generate()` directly without the Trainer, so it was never affected. The model weights were always correct; only the eval metric computation was broken.

### transformers 5.x API changes

| Old (4.x) | New (5.x) | File |
|---|---|---|
| `Seq2SeqTrainer(tokenizer=...)` | `Seq2SeqTrainer(processing_class=...)` | `train.py` |
| `warmup_ratio=0.06` (float) | `warmup_steps=N` (integer, computed from ratio) | `train.py` |
| `logging_dir=...` | Set `TENSORBOARD_LOGGING_DIR` env var | `train.py` |
| `model.config.forced_bos_token_id` | `model.generation_config.forced_bos_token_id` | `train.py`, `inference.py` |

### M2M100 + transformers 5.x decoder conflict

`M2M100Model.forward()` in transformers 5.x pre-computes `decoder_inputs_embeds` from `decoder_input_ids` via `self.shared()`, then passes **both** to `M2M100Decoder`, which raises:
```
ValueError: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time
```

Two-part fix:
- `PatchedM2M100Model.forward()` drops `decoder_input_ids` when `decoder_inputs_embeds` is already present — covers the `.generate()` path
- `M2M100Seq2SeqTrainer.compute_loss()` injects `decoder_inputs_embeds` before the forward call — covers the training loss path

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
| `torchao` | 0.17.0 | peft dispatch requires ≥0.16.0 |