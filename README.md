# M2M100 Urdu → Roman Urdu Transliteration

A reproducible LoRA fine-tuning pipeline for adapting `Mavkif/m2m100_rup_ur_to_rur` to **modern Urdu-to-Roman Urdu transliteration**, with an emphasis on contemporary vocabulary, everyday language, English loanwords written in Urdu script, and practical user-oriented text.

The project includes:

* A custom Urdu–Roman Urdu parallel dataset
* Incremental dataset and experiment history
* LoRA-based fine-tuning of M2M100
* Reproducible data preparation
* Training scripts
* Standalone inference
* A held-out benchmark suite
* Per-category benchmark evaluation
* Comparison across the base model and multiple fine-tuning runs

The final Run 4 model is published separately on Hugging Face.

---

## Project Overview

The project starts from:

```text
Mavkif/m2m100_rup_ur_to_rur
```

and adapts it for a more contemporary Urdu distribution.

The motivation for additional fine-tuning was practical.

The base model was trained using the `Mavkif/Roman-Urdu-Parl-split` data. During development, we identified a gap between the language distribution represented in the available training resources and the language required by the intended application.

In particular, modern Urdu frequently contains:

* Everyday conversational vocabulary
* English loanwords written in Urdu script
* Technology-related terms
* Modern household vocabulary
* User-generated language
* Contemporary code-switched usage

Examples include words such as:

```text
فین       → fan
لائٹ      → light
بلب       → bulb
وائی فائی → wifi
ٹیبل      → table
```

The project therefore created an additional targeted Urdu–Roman Urdu corpus and used it for further adaptation.

The result is a model specifically optimized for the practical transliteration task:

```text
Urdu Nastaliq
      │
      ▼
M2M100 + LoRA Adapter
      │
      ▼
Roman Urdu
```

---

## Key Components

| Component                         | Purpose                                                      |
| --------------------------------- | ------------------------------------------------------------ |
| `data/rur_to_ur_data.csv`         | Canonical ordered Urdu–Roman Urdu corpus used by the project |
| `prepare_data.py`                 | Converts raw CSV data into tokenized Hugging Face datasets   |
| `train.py`                        | Performs LoRA fine-tuning                                    |
| `inference.py`                    | Runs transliteration using a trained adapter                 |
| `benchmark/run_benchmark.py`      | Generates predictions on the held-out benchmark              |
| `benchmark/score_benchmark.py`    | Calculates benchmark metrics and category-level results      |
| `benchmark/benchmark_dataset.csv` | Held-out benchmark dataset                                   |
| `benchmark/results/`              | Benchmark predictions and aggregated results                 |
| `DATASET_HISTORY.md`              | Documents dataset construction and lineage                   |
| `dataset_card.md`                 | Documents the dataset itself                                 |
| `model_card.md`                   | Documents the released model                                 |

---

## Repository Structure

```text
m2m100-rup-ur-to-rur-fine_tune/
│
├── benchmark/
│   ├── benchmark_dataset.csv
│   ├── run_benchmark.py
│   ├── score_benchmark.py
│   └── results/
│       ├── benchmark_results.csv
│       ├── predictions_mavkif_base.csv
│       ├── predictions_run1.csv
│       ├── predictions_run2.csv
│       ├── predictions_run3.csv
│       └── predictions_run4.csv
│
├── data/
│   └── rur_to_ur_data.csv
│
├── dataset_card.md
├── DATASET_HISTORY.md
├── dataset_run_summary.csv
│
├── inference.py
├── prepare_data.py
├── requirements.txt
├── train.py
└── README.md
```

The released model weights are not intended to be maintained as Git history in this repository. The production adapter is distributed through Hugging Face.

---

# Architecture

## Model Architecture

The project uses a sequence-to-sequence Transformer architecture based on M2M100.

The adaptation stack is:

```text
                    ┌────────────────────────────┐
                    │ Mavkif/m2m100_rup_ur_to_rur│
                    │                            │
                    │ M2M100 Seq2Seq Model       │
                    └─────────────┬──────────────┘
                                  │
                                  │ Frozen Base
                                  ▼
                    ┌───────────────────────────┐
                    │        LoRA Adapter       │
                    │                           │
                    │ Parameter-efficient       │
                    │ fine-tuning               │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
             ┌─────────────────────────────────────────┐
             │ Urdu → Roman Urdu Transliteration       │
             └─────────────────────────────────────────┘
```

The base model provides the pretrained multilingual sequence-to-sequence capability.

LoRA provides the task-specific adaptation without updating the full base model.

---

## Tokenization

The project uses the tokenizer associated with the M2M100 adaptation:

```text
Mavkif/m2m100_rup_tokenizer_both
```

The tokenizer configuration supports the project's Urdu and Roman Urdu language setup.

Data preparation converts the raw parallel CSV into tokenized examples suitable for sequence-to-sequence fine-tuning.

---

# Training Pipeline

The complete training pipeline is:

```text
Raw Urdu–Roman Urdu CSV
          │
          ▼
   prepare_data.py
          │
          ▼
Tokenization + Train/Validation Split
          │
          ▼
   Hugging Face Dataset
          │
          ▼
       train.py
          │
          ▼
M2M100 Base Model + LoRA
          │
          ▼
    LoRA Adapter
          │
          ▼
     Benchmarking
          │
          ▼
     Evaluation
```

The pipeline is intentionally divided into separate stages so that data preparation, training, inference, and evaluation can be reproduced independently.

---

# Dataset

The project's custom dataset contains:

```text
4,860 Urdu–Roman Urdu parallel pairs
```

The two columns are:

```text
urdu
roman_urdu
```

The dataset was assembled from multiple sources and collection strategies, including:

1. AI-assisted generation
2. Urdu user-oriented query data obtained from another project
3. Urdu text collected through search/web sources and converted to Roman Urdu

The dataset was manually reviewed by a native Urdu speaker and additionally checked using AI-assisted validation.

The dataset is intended to complement existing Urdu–Roman Urdu resources by providing additional coverage of contemporary Urdu usage.

For complete information about data collection, validation, statistics, and provenance, see:

* `dataset_card.md`
* `DATASET_HISTORY.md`

The canonical dataset is maintained separately as part of the project's dataset release.

---

# Data Preparation

`prepare_data.py` is responsible for converting the raw parallel corpus into the dataset consumed by the training pipeline.

Conceptually:

```text
data/rur_to_ur_data.csv
          │
          ▼
     CSV Loading
          │
          ▼
      Validation
          │
          ▼
      Tokenization
          │
          ▼
Train / Validation Split
          │
          ▼
Processed Hugging Face Dataset
```

Run:

```bash
python prepare_data.py \
    --csv data/rur_to_ur_data.csv \
    --output_dir ./processed_dataset
```

The resulting processed dataset is then passed to `train.py`.

### Important

The dataset contains duplicate and repeated content.

Therefore, the train/validation split should not be interpreted as a fully independent linguistic benchmark.

The project's separate benchmark dataset is used for final comparative evaluation.

---

# Training

Training is performed using LoRA-based parameter-efficient fine-tuning.

The general workflow is:

```bash
python prepare_data.py \
    --csv data/rur_to_ur_data.csv \
    --output_dir ./processed_dataset
```

Then:

```bash
python train.py \
    --dataset_dir ./processed_dataset \
    --output_dir ./checkpoints \
    --final_model_dir ./fine_tuned_model
```

The training script handles:

* Loading the base M2M100 model
* Loading the prepared dataset
* Configuring LoRA
* Fine-tuning the model
* Validation during training
* Saving the resulting adapter

The resulting directory contains the LoRA adapter rather than a complete copy of the base model.

---

# LoRA Fine-Tuning

The project uses parameter-efficient fine-tuning rather than updating every parameter in the M2M100 model.

Conceptually:

```text
Base M2M100
     │
     ├── Frozen pretrained parameters
     │
     └── Trainable LoRA parameters
                  │
                  ▼
         Task-specific adaptation
```

This approach provides several practical advantages:

* Lower memory requirements
* Smaller trained artifacts
* Faster fine-tuning
* Easier distribution of adapter weights
* Preservation of the pretrained base model

The adapter must therefore be loaded together with the compatible base model and tokenizer.

---

# Reproducing Run 4

Run 4 is the final consolidated training experiment documented by this project.

Its dataset lineage is:

```text
Run 1 + Run 2 combined dataset
                │
                ├───────────────┐
                │               │
                ▼               ▼
       Expanded Run 3      Additional data
                │
                └───────────────┐
                                ▼
                         Run 4 Dataset
                                │
                                ▼
                       Fresh fine-tuning
                         from base model
```

The important distinction is:

* **Run 3:** 563-row training run
* **Expanded Run 3:** later-expanded data collection
* **Run 4:** trained using the consolidated data resulting from the Run 1/Run 2 lineage combined with the Expanded Run 3 data

The exact dataset lineage is documented in:

```text
DATASET_HISTORY.md
```

The final raw dataset contains:

```text
4,860 rows
```

To reproduce Run 4:

### Step 1 — Obtain the dataset

Place the canonical dataset at:

```text
data/rur_to_ur_data.csv
```

### Step 2 — Prepare the dataset

```bash
python prepare_data.py \
    --csv data/rur_to_ur_data.csv \
    --output_dir ./processed_dataset
```

### Step 3 — Train the model

```bash
python train.py \
    --dataset_dir ./processed_dataset \
    --output_dir ./checkpoints \
    --final_model_dir ./fine_tuned_model
```

The exact hyperparameters used for the published Run 4 model should be taken from the finalized `train.py` configuration and the experiment history rather than inferred from earlier README versions.

This repository intentionally separates:

```text
Reproduction workflow
        │
        ├── Data preparation
        ├── Training
        ├── Inference
        └── Benchmarking
```

from:

```text
Historical experiment record
        │
        └── DATASET_HISTORY.md
```

---

# Experiment History

The project went through multiple fine-tuning iterations.

The high-level lineage is:

```text
Base Model
    │
    ▼
Run 1
    │
    ▼
Run 2
    │
    ▼
Run 3
    │
    ├── 563-row targeted training run
    │
    └── Observed regression
            │
            ▼
    Expanded data collection
            │
            ▼
          Run 4
```

Run 3 is particularly important because it demonstrated a failure mode of continuing training on a small targeted dataset.

The model's performance degraded when a relatively small corrective dataset was used as the primary training distribution.

Run 4 therefore used the consolidated dataset and restarted fine-tuning from the base model rather than continuing from the degraded Run 3 adapter.

The detailed experiment history, dataset lineage, and run-specific information should be maintained in:

```text
DATASET_HISTORY.md
```

and the project's experiment history documentation.

---

# Benchmarking

The project includes a separate benchmark pipeline.

```text
Benchmark Dataset
        │
        ▼
run_benchmark.py
        │
        ▼
Model Predictions
        │
        ▼
score_benchmark.py
        │
        ▼
Overall + Category Metrics
```

The benchmark is designed to compare:

* The original base model
* Run 1
* Run 2
* Run 3
* Run 4

Prediction files are stored under:

```text
benchmark/results/
```

---

## Run Benchmark

To generate predictions:

```bash
python benchmark/run_benchmark.py
```

The exact CLI arguments supported by the current script should be checked with:

```bash
python benchmark/run_benchmark.py --help
```

The script generates model predictions for the benchmark dataset.

---

## Score Predictions

Run:

```bash
python benchmark/score_benchmark.py
```

This evaluates the generated predictions and produces the benchmark results.

The aggregated results are stored in:

```text
benchmark/results/benchmark_results.csv
```

---

# Benchmark Dataset

The benchmark dataset is maintained separately from the training corpus:

```text
benchmark/benchmark_dataset.csv
```

It is intended to provide a consistent evaluation set for comparing different model versions.

The benchmark is categorized into linguistic groups used to analyze model behavior beyond a single aggregate score.

The categories include:

* Numbers
* Loanwords
* Code-switching
* Names and places
* Classical Urdu
* Other targeted transliteration cases

The benchmark should be treated as an internal project benchmark rather than a universally standardized Urdu transliteration benchmark.

---

# Benchmark Results

The benchmark results should be interpreted together with the per-category results stored in:

```text
benchmark/results/benchmark_results.csv
```

The project compares model versions rather than reporting only the final Run 4 score.

The comparison is intended to answer two questions:

1. Did additional fine-tuning improve overall transliteration?
2. Did it improve the specific linguistic failure modes targeted by the additional data?

This distinction is important because an aggregate BLEU score can hide regressions in individual categories.

For example, a model may improve on modern loanwords while simultaneously becoming worse on classical Urdu.

Therefore, the project reports both:

```text
Overall performance
```

and:

```text
Category-level performance
```

---

# Inference

The project provides `inference.py` for running transliteration with the trained adapter.

The inference architecture is:

```text
Urdu Input
    │
    ▼
Tokenizer
    │
    ▼
M2M100 Base Model
    +
LoRA Adapter
    │
    ▼
Generation
    │
    ▼
Roman Urdu Output
```

Run:

```bash
python inference.py
```

For the exact supported arguments:

```bash
python inference.py --help
```

The production model is distributed through Hugging Face.

The recommended production deployment should therefore load:

```text
Base Model
      +
Run 4 LoRA Adapter
      +
Compatible Tokenizer
```

rather than relying on the local `fine_tuned_model/` directory.

---

# Model Distribution

The GitHub repository contains the engineering and reproducibility code.

The Hugging Face model repository contains the released Run 4 adapter and its canonical model documentation.

The Hugging Face dataset repository contains the released dataset and its dataset card.

The intended project structure is therefore:

```text
GitHub
│
├── Code
├── Training Pipeline
├── Benchmarking
├── Reproducibility
└── Experiment Documentation
        │
        ├───────────────┐
        ▼               ▼
Hugging Face        Hugging Face
Model               Dataset
│                   │
├── Model Card      ├── Dataset Card
└── Run 4 Adapter   └── 4,860 Examples
```

---

# Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The exact package versions used for a reproducible experiment should be taken from the finalized `requirements.txt`.

---

# Reproducibility Checklist

To reproduce an experiment, record:

* Base model identifier
* Tokenizer identifier
* Dataset version
* Dataset row count
* Train/validation split
* Random seed
* LoRA configuration
* Learning rate
* Batch size
* Gradient accumulation
* Number of epochs
* Early stopping configuration
* Hardware
* Software versions
* Output adapter
* Benchmark version
* Evaluation results

A model result should not be considered fully reproducible from the final BLEU score alone.

The combination of:

```text
Code
+
Dataset
+
Configuration
+
Randomness
+
Evaluation Benchmark
```

defines the experiment.

---

# Known Engineering Lessons

## 1. Small corrective datasets can cause catastrophic forgetting

Run 3 demonstrated that continuing training on a small targeted dataset can damage previously learned capabilities.

A model that improves on the target failure case may simultaneously regress on the broader task.

Therefore, corrective fine-tuning should always be evaluated against the complete benchmark.

---

## 2. Aggregate metrics are insufficient

A single BLEU score does not explain where a transliteration model succeeds or fails.

The project therefore uses category-level benchmark analysis.

---

## 3. Data distribution matters

Increasing dataset size alone is not sufficient.

The additional data must also cover the linguistic patterns that matter for the intended application.

This project specifically targets modern Urdu usage and contemporary loanwords that were insufficiently represented in the available base training resources.

---

## 4. Benchmark before and after every major change

Every significant model change should be evaluated using the same benchmark whenever possible.

This allows regressions to be detected rather than hidden by qualitative examples.

---

# Limitations

This project has several limitations.

### Dataset limitations

The dataset contains AI-assisted data and does not have a precisely recorded source distribution.

### Roman Urdu variability

There is no universally standardized Roman Urdu orthography.

### Benchmark limitations

The benchmark is project-specific and should not be interpreted as a universal measure of Urdu transliteration quality.

### Evaluation limitations

BLEU provides a useful automated comparison but does not perfectly capture transliteration quality, phonetic correctness, or human preference.

### Model limitations

The model may still produce:

* Incorrect transliterations
* Inconsistent Roman Urdu spelling
* Errors on ambiguous Urdu words
* Errors on names and places
* Errors on unseen vocabulary
* Errors on unusual sentence structures

---

## Project Ecosystem

This project is part of a small set of repositories and Hugging Face resources:

| Resource | Purpose |
|---|---|
| [Fine-tuning Repository](...) | Training code, preprocessing, benchmarking, and experiment history |
| [Hugging Face Dataset](...) | Canonical 4,860-example Urdu → Roman Urdu dataset |
| [Hugging Face Model](...) | Current Run 4 production model |
| [Inference Repository](...) | FastAPI production inference service |

### Workflow

The project follows this pipeline:

Dataset
→ Data Preparation
→ M2M100 LoRA Fine-Tuning
→ Run 4 Model
→ Production Inference API
    
---

# Citation

If you use the code or methodology from this repository, please cite the project and associated model.

```bibtex
@software{maaz_m2m100_urdu_roman_urdu_2026,
  author       = {Maaz Ahmad},
  title        = {M2M100 Urdu to Roman Urdu Transliteration},
  year         = {2026},
  url          = {https://github.com/Maaz-x14/m2m100-rup-ur-to-rur-fine_tune}
}
```

For model and dataset citations, refer to their respective Hugging Face repositories.

---

# License

See the repository license file for the applicable terms.

Third-party models, datasets, and source materials may have their own licenses and terms.

Users are responsible for reviewing the applicable licenses before redistribution or commercial use.

---

# Acknowledgements

Developed by:

**Maaz Ahmad**

**National University of Sciences and Technology (NUST)**

Independent Open Source Project

The project builds upon the open-source M2M100 ecosystem and the work underlying the `Mavkif/m2m100_rup_ur_to_rur` model and associated Urdu–Roman Urdu resources.

---

## Project Status

The Run 4 model represents the current production-oriented release of this fine-tuning effort.

Future work may include:

* Larger and more diverse contemporary Urdu datasets
* More systematic human annotation
* Improved Roman Urdu normalization
* Larger and more rigorous benchmark sets
* Error-specific evaluation
* Production inference optimization
* Latency and throughput benchmarking
