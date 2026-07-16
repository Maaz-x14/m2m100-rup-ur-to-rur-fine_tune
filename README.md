
# M2M100 LoRA Fine-Tune — Urdu → Roman Urdu Transliteration

Fine-tunes [`Mavkif/m2m100_rup_ur_to_rur`](https://huggingface.co/Mavkif/m2m100_rup_ur_to_rur) with LoRA adapters (via `peft`).

---

## Project structure

```
m2m100-rup-ur-to-rur-fine_tune/
├── data/
│   ├── final_transliteration_dataset.csv   # original 2012-row dataset (run1)
│   ├── transliteration_dataset.csv         # run2 combined dataset (2501 rows)
│   ├── numbers_and_homographs.csv          # run3 corrective dataset (563 rows)
│   └── combined_run4.csv                   # run4 dataset: roman_urdu_map + numbers_and_homographs (~4862 rows)
├── fine_tuned_model/          # run2 adapter — best overall (BLEU 70.4)
├── fine_tuned_model_run3/     # run3 adapter — REGRESSED, do not use
├── fine_tuned_model_run4/     # run4 adapter — fresh train on full combined dataset
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

> **Python 3.10+ required.** Tested on 3.12.

---

## When to start fresh vs continue from adapter

This is the most important decision each run. Get it wrong and you either waste compute or cause catastrophic forgetting (run3's failure).

| Scenario                                                           | Decision                                                      | Reason                                                                      |
| ------------------------------------------------------------------ | ------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Large dataset (>3000 rows), full distribution coverage             | **Start fresh** — no `--init_adapter_dir`            | Enough data to train general capability from scratch; no risk of forgetting |
| Small targeted dataset (<1000 rows), fixing specific failure modes | **Continue from adapter** — use `--init_adapter_dir` | Preserves general capability; focused data corrects the gap                 |
| Previous adapter is degraded/regressed                             | **Start fresh** from base, not from bad adapter         | Initialising from a degraded adapter bakes in its errors                    |
| Adding domain data on top of a known-good adapter                  | **Continue from the good adapter**                      | Small domain data alone causes catastrophic forgetting if started fresh     |

**Run3 failure post-mortem:** Trained only 533 rows of numbers/homographs data continuing from run2. Dataset too small → adapter forgot run2's general capability → BLEU dropped from 70.4 to 47.97 across all categories. Fix: combine all data and train fresh (run4).

---

## Run history and benchmark results

| Model       | Dataset    | Strategy           | BLEU (ALL)      | Numbers BLEU | Loanwords BLEU  |
| ----------- | ---------- | ------------------ | --------------- | ------------ | --------------- |
| mavkif_base | —         | pretrained         | 36.71           | 37.90        | 30.60           |
| run1        | 2012 rows  | fresh              | 54.52           | 57.20        | 68.17           |
| run2        | 2501 rows  | fresh              | **70.40** | 50.34        | **87.76** |
| run3        | 563 rows   | continue from run2 | 47.97 ❌        | 59.23        | 51.03 ❌        |
| run4        | ~4862 rows | fresh from base    | TBD             | TBD          | TBD             |

Full per-category breakdown is in `benchmark/results/benchmark_results.csv`.

---

## Quick start

### 1 — Prepare the dataset

```bash
python prepare_data.py \
    --csv data/combined_run4.csv \
    --output_dir ./processed_dataset_run4
```

Tokenises the CSV using `Mavkif/m2m100_rup_tokenizer_both`, performs a 95/5 train/validation split, and saves in HuggingFace Arrow format.

### 2 — Fine-tune

#### Run4 — fresh train on full combined dataset (current)

```bash
python train.py \
    --dataset_dir ./processed_dataset_run4 \
    --output_dir ./checkpoints_run4 \
    --final_model_dir ./fine_tuned_model_run4 \
    --learning_rate 5e-4 \
    --batch_size 16 \
    --grad_accum 4 \
    --num_epochs 30
```

No `--init_adapter_dir` — starting fresh from base model. Early stopping (patience=3) will fire automatically.

#### Future corrective run (small targeted dataset on top of a good adapter)

Only use this pattern when the base adapter is known-good AND the new dataset is focused on specific failure modes:

```bash
python train.py \
    --dataset_dir ./processed_dataset_runN \
    --output_dir ./checkpoints_runN \
    --final_model_dir ./fine_tuned_model_runN \
    --init_adapter_dir ./fine_tuned_model_run4 \   # must be a GOOD adapter
    --learning_rate 1e-4 \                          # lower lr for corrective runs
    --batch_size 16 \
    --grad_accum 4 \
    --num_epochs 20
```

**WARNING:** If the corrective dataset is <1000 rows, expect catastrophic forgetting. Always run the full benchmark after and check every category — not just the target category.

#### All CLI flags

| Flag                   | Default | Notes                                                              |
| ---------------------- | ------- | ------------------------------------------------------------------ |
| `--batch_size`       | 16      | Per-device train batch size                                        |
| `--grad_accum`       | 4       | Effective batch =`batch_size × grad_accum` = 64                 |
| `--learning_rate`    | 5e-4    | Use 5e-4 for fresh runs, 1e-4 for corrective continues             |
| `--lora_r`           | 16      | LoRA rank                                                          |
| `--lora_alpha`       | 32      | Keep at`2 × lora_r`                                             |
| `--num_epochs`       | 30      | Early stopping (patience 3) fires first in practice                |
| `--init_adapter_dir` | None    | Path to existing adapter to continue from. If unset, starts fresh. |

Only the LoRA adapter weights are saved to `--final_model_dir` (a few MB, not the frozen base model).

**What to check after each run:**

- `numbers` category: was the weakest in run2 (BLEU 50.34) — should improve in run4
- `loanwords` category: was the strongest in run2 (BLEU 87.76) — must not regress
- If any category drops >3 BLEU vs run2, investigate before deploying

### 4 — Inference

```bash
python inference.py --model_dir ./fine_tuned_model_run4

python inference.py \
    --model_dir ./fine_tuned_model_run4 \
    --sentences "ڈاکٹر نے کہا" "کمپیوٹر بند ہے" "ورزش کرو"

python inference.py \
    --model_dir ./fine_tuned_model_run4 \
    --input_file urdu_sentences.txt
```

---

## How LoRA is applied

LoRA adapters are injected into all four attention projection layers across **both encoder and decoder**:

    q_proj, k_proj, v_proj, out_proj

`task_type` is intentionally NOT set in `LoraConfig` — see code comments in `train.py` for why (bug #4).

With default settings (`r=16`, `alpha=32`), ~4.7M of 488M parameters are trainable (~0.97%).

---

## Tokeniser notes

Always use **`Mavkif/m2m100_rup_tokenizer_both`**, never `facebook/m2m100_418M`.

| Token            | ID     |
| ---------------- | ------ |
| `__ur__`       | 128095 |
| `__roman-ur__` | 128105 |

---

## Requirements

| Package           | Minimum | Notes                         |
| ----------------- | ------- | ----------------------------- |
| `torch`         | 2.1.0   |                               |
| `transformers`  | 5.0.0   |                               |
| `peft`          | 0.10.0  |                               |
| `datasets`      | 2.18.0  |                               |
| `sacrebleu`     | 2.3.1   |                               |
| `sentencepiece` | 0.1.99  | Required by M2M100Tokenizer   |
| `accelerate`    | 0.27.0  | Required by Trainer fp16 path |
