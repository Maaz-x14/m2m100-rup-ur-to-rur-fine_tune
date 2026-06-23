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

> **Python 3.10+ recommended.** The type hint `list[str]` in `inference.py` requires ≥3.9.

---

## Quick start

### 1 — Prepare the dataset

```bash
python prepare_data.py \
    --csv data/final_transliteration_dataset.csv \
    --output_dir ./processed_dataset
```

This tokenises the CSV using `Mavkif/m2m100_rup_tokenizer_both`, performs a 95/5 train/validation split, and saves the result in HuggingFace Arrow format.

Expected output:
```
[prepare_data] Loaded 1922 rows from 'data/final_transliteration_dataset.csv'
[prepare_data] Split → train: 1825 | validation: 97
[prepare_data] Token IDs confirmed → __ur__: 128095 | __roman-ur__: 128105
[prepare_data] Dataset saved to './processed_dataset'
```

### 2 — Fine-tune

```bash
python train.py \
    --dataset_dir ./processed_dataset \
    --output_dir ./checkpoints \
    --final_model_dir ./fine_tuned_model
```

Default hyperparameters target an **NVIDIA A5000 (24 GB)**. Adjust for your hardware:

| Flag | Default | Notes |
|---|---|---|
| `--batch_size` | 16 | Per-device train batch size |
| `--grad_accum` | 4 | Effective batch = `batch_size × grad_accum` = 64 |
| `--learning_rate` | 5e-4 | Higher than full FT is fine for LoRA adapters |
| `--lora_r` | 16 | LoRA rank — increase for more capacity |
| `--lora_alpha` | 32 | Scaling = `alpha / r`; keep at `2 × r` |
| `--num_epochs` | 30 | Early stopping (patience 3) will usually fire first |

For smaller GPUs (e.g. 8–12 GB), reduce `--batch_size 4 --grad_accum 16` to keep the effective batch the same.

The best checkpoint (by `eval_loss`) is saved to `--final_model_dir`. Only the LoRA adapter weights are saved (a few MB), not the frozen base model.

### 3 — Inference

```bash
# Default demo sentences
python inference.py --model_dir ./fine_tuned_model

# Custom sentences
python inference.py \
    --model_dir ./fine_tuned_model \
    --sentences "ڈاکٹر نے کہا" "کمپیوٹر بند ہے" "ورزش کرو"

# From a text file (one sentence per line)
python inference.py \
    --model_dir ./fine_tuned_model \
    --input_file urdu_sentences.txt
```

---

## How LoRA is applied

LoRA adapters are injected into all four attention projection layers across **both encoder and decoder** (self-attention and cross-attention):

```
q_proj, k_proj, v_proj, out_proj
```

`TaskType.SEQ_2_SEQ_LM` tells PEFT this is an encoder-decoder model, so it targets matching layers in both halves automatically. With default settings (`r=16`, `alpha=32`), trainable parameters are ~1.5–2 M out of ~418 M total — roughly 0.4%.

---

## Tokeniser notes

Always use **`Mavkif/m2m100_rup_tokenizer_both`**, never `facebook/m2m100_418M`. The custom tokeniser adds two language tokens absent from the base vocabulary:

| Token | ID |
|---|---|
| `__ur__` | 128095 |
| `__roman-ur__` | 128105 |

The scripts assert these IDs at startup (`prepare_data.py`) and hard-code `forced_bos_token_id=128105` at generation time so the decoder is steered to Roman Urdu output regardless of beam search.

In `prepare_data.py`, labels are tokenised by temporarily switching `tokenizer.src_lang = "roman-ur"` (the ≥5.x replacement for the removed `as_target_tokenizer()` context manager).

---

## Known issue — sacrebleu reference format

The `compute_metrics` function in `train.py` constructs `references` as a list of single-element lists and then transposes with `zip(*references)`:

```python
references = [[l] for l in decoded_labels]
bleu_score = sacrebleu.corpus_bleu(decoded_preds, list(zip(*references)))
```

For a single reference translation per example this works, but it's unnecessary transposition. A simpler and equally correct form is:

```python
bleu_score = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
```

Both produce identical results; the current form is just harder to read.

---

## Expanding the dataset

The model currently trains on ~1,900 rows. To reduce hallucination on domain-specific words, add more rows to the CSV before running `prepare_data.py`. The `validate.py` script (if present) checks for forbidden characters, token-range violations, and duplicates before authoring new content.

---

## Requirements

| Package | Minimum | Notes |
|---|---|---|
| `torch` | 2.1.0 | |
| `transformers` | **5.0.0** | `as_target_tokenizer()` removed in 5.x |
| `peft` | 0.10.0 | |
| `datasets` | 2.18.0 | |
| `sacrebleu` | 2.3.1 | |
| `sentencepiece` | 0.1.99 | Required by M2M100Tokenizer |
| `accelerate` | 0.27.0 | Required by Trainer fp16 path |
