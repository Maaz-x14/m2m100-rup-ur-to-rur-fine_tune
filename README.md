
# M2M100 LoRA Fine-Tune — Urdu → Roman Urdu Transliteration

Fine-tunes [`Mavkif/m2m100_rup_ur_to_rur`](https://huggingface.co/Mavkif/m2m100_rup_ur_to_rur) with LoRA adapters (via `peft`).

---

## Project structure

```
m2m100-rup-ur-to-rur-fine_tune/
├── data/
│   ├── final_transliteration_dataset.csv   # original 2012-row dataset
│   ├── transliteration_dataset.csv         # run2 combined dataset (2501 rows, incl. iss/uss)
│   └── numbers_and_homographs.csv          # run3 dataset (564 rows)
├── fine_tuned_model/          # run2 adapter
├── fine_tuned_model_run3/     # run3 adapter
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

### 2 — Fine-tune (A5000 / full run)

```bash
python train.py \
    --dataset_dir ./processed_dataset \
    --output_dir ./checkpoints \
    --final_model_dir ./fine_tuned_model
```

Default hyperparameters target an **NVIDIA A5000 (24 GB)**:

| Flag                   | Default  | Notes                                                                                                                                                                  |
| ---------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--batch_size`       | 16       | Per-device train batch size                                                                                                                                            |
| `--grad_accum`       | 4        | Effective batch =`batch_size × grad_accum` = 64                                                                                                                     |
| `--learning_rate`    | 5e-4     | Higher than full FT is fine for LoRA adapters                                                                                                                          |
| `--lora_r`           | 16       | LoRA rank                                                                                                                                                              |
| `--lora_alpha`       | 32       | Keep at`2 × lora_r`                                                                                                                                                 |
| `--num_epochs`       | 30       | Early stopping (patience 3) will usually fire first                                                                                                                    |
| `--init_adapter_dir` | `None` | Path to an existing trained adapter to continue from (e.g.`./fine_tuned_model` for run2). If unset, starts a fresh randomly-initialised adapter from the base model. |

Only the LoRA adapter weights are saved to `--final_model_dir` (a few MB, not the frozen base model).

### 2c — Continuing from a previous run (run3: numbers + homographs)

Checkpoint resume (built into `train.py`) only picks back up a crashed run of the same job. To continue training from a finished adapter (e.g. run2's `fine_tuned_model/`) on new data, use `--init_adapter_dir`:

```bash
python prepare_data.py \
    --csv data/numbers_and_homographs.csv \
    --output_dir ./processed_dataset_run3

python train.py \
    --dataset_dir ./processed_dataset_run3 \
    --output_dir ./checkpoints_run3 \
    --final_model_dir ./fine_tuned_model_run3 \
    --init_adapter_dir ./fine_tuned_model \
    --learning_rate 2e-4
```

At startup you should see:

```
[train] Loading EXISTING adapter from './fine_tuned_model' to continue training (NOT starting a fresh adapter).
```

If you instead see `[train] No --init_adapter_dir given — starting a FRESH LoRA adapter...`, run2's weights are not being used.

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

LoRA adapters are injected into all four attention projection layers across **both encoder and decoder** (self-attention and cross-attention):

    q_proj, k_proj, v_proj, out_proj

`task_type` is intentionally NOT set in `LoraConfig` — see code comments in `train.py` for why.

With default settings (`r=16`, `alpha=32`), ~4.7 M of 488 M parameters are trainable (~0.97%).

---

## Tokeniser notes

Always use **`Mavkif/m2m100_rup_tokenizer_both`**, never `facebook/m2m100_418M`.

| Token            | ID     |
| ---------------- | ------ |
| `__ur__`       | 128095 |
| `__roman-ur__` | 128105 |

---

## Requirements

| Package           | Minimum | Notes                           |
| ----------------- | ------- | ------------------------------- |
| `torch`         | 2.1.0   |                                 |
| `transformers`  | 5.0.0   |                                 |
| `peft`          | 0.10.0  |                                 |
| `datasets`      | 2.18.0  |                                 |
| `sacrebleu`     | 2.3.1   |                                 |
| `sentencepiece` | 0.1.99  | Required by M2M100Tokenizer     |
| `accelerate`    | 0.27.0  | Required by Trainer fp16 path   |
| `torchao`       | 0.17.0  | peft dispatch requires ≥0.16.0 |
