 

# M2M100 Urdu → Roman Urdu Transliteration

A reproducible LoRA fine-tuning pipeline for adapting [`Mavkif/m2m100_rup_ur_to_rur`](https://huggingface.co/Mavkif/m2m100_rup_ur_to_rur) to **modern Urdu → Roman Urdu transliteration**.

The project focuses on contemporary Urdu usage, including:

* Everyday conversational vocabulary
* English loanwords written in Urdu script
* Technology-related terminology
* Modern household vocabulary
* User-oriented text
* Contemporary code-switched usage

The project provides the complete engineering workflow behind the released **Run 4** model:

```text
Canonical Dataset
      │
      ▼
Data Preparation
      │
      ▼
M2M100 + LoRA Fine-Tuning
      │
      ▼
Run 4 Model
      │
      ├──────────────► Benchmark Evaluation
      │
      ▼
Hugging Face Model
      │
      ▼
Production Inference API
```

---

# Project Resources

The project is split across GitHub and Hugging Face.

| Resource                         | Purpose                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------ |
| **Fine-Tuning Repository** | Training, preprocessing, benchmarking, reproducibility, and experiment history |
| **Hugging Face Dataset**   | Canonical 4,860-example Urdu → Roman Urdu dataset                             |
| **Hugging Face Model**     | Released Run 4 LoRA adapter and canonical model card                           |
| **Inference Repository**   | Production-oriented FastAPI inference service                                  |

### Canonical resources

* **Dataset:** [`Maaz-x14/urdu-to-roman-urdu`](https://huggingface.co/datasets/Maaz-x14/urdu-to-roman-urdu)
* **Run 4 Model:** [`Maaz-x14/m2m100-ur-to-roman-urdu`](https://huggingface.co/Maaz-x14/m2m100-ur-to-roman-urdu)
* **Production Inference:** [`Maaz-x14/m2m100-roman-deploy`](https://github.com/Maaz-x14/m2m100-roman-deploy)

### Repository ecosystem

```text
                    ┌──────────────────────────────┐
                    │  Hugging Face Dataset        │
                    │  Maaz-x14/urdu-to-roman-urdu │
                    │                              │
                    │  Canonical 4,860 examples    │
                    └──────────────┬───────────────┘
                                   │
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────┐
│                 Fine-Tuning Repository                      │
│                                                             │
│  prepare_data.py → train.py → benchmark → evaluation        │
│                                                             │
│  Experiment history + reproducibility documentation         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ Run 4 adapter
                               ▼
                    ┌──────────────────────────────┐
                    │  Hugging Face Model          │
                    │  Maaz-x14/m2m100-ur-to-      │
                    │  roman-urdu                  │
                    │                              │
                    │  Production Run 4 Adapter    │
                    └──────────────┬───────────────┘
                                   │
                                   │ Model loading
                                   ▼
                    ┌──────────────────────────────┐
                    │  Inference Repository        │
                    │  m2m100-roman-deploy         │
                    │                              │
                    │  FastAPI Production API      │
                    └──────────────────────────────┘
```

The responsibilities are intentionally separated:

* **This repository:** How the model was developed, trained, evaluated, and reproduced.
* **Hugging Face Dataset:** What data was used.
* **Hugging Face Model:** The released Run 4 model artifact.
* **Inference Repository:** How to deploy and serve the model.

---

# Architecture

The project starts from:

```text
Mavkif/m2m100_rup_ur_to_rur
```

and adapts it using LoRA.

```text
                  Mavkif/m2m100_rup_ur_to_rur
                              │
                              │ Frozen Base Model
                              ▼
                    M2M100 Seq2Seq Model
                              │
                              │ LoRA Fine-Tuning
                              ▼
                       LoRA Adapter
                              │
                              ▼
                  Urdu → Roman Urdu
                    Transliteration
```

At inference time:

```text
Urdu Nastaliq Input
        │
        ▼
     Tokenizer
        │
        ▼
 M2M100 Base Model
        +
   Run 4 LoRA Adapter
        │
        ▼
    Generation
        │
        ▼
Roman Urdu Output
```

The project uses:

```text
Base Model:
Mavkif/m2m100_rup_ur_to_rur

Tokenizer:
Mavkif/m2m100_rup_tokenizer_both
```

The target language token used by the training pipeline is:

```text
__roman-ur__
```

with target token ID:

```text
128105
```

---

# Repository Structure

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
│   └── ur_to_rur_data.csv  Place dataset here, get it from (Maaz-x14/urdu-to-roman-urdu)
│
├── DATASET_HISTORY.md
├── EXPERIMENT_HISTORY.md
├── dataset_run_summary.csv
│
├── fine_tuned_model/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── added_tokens.json
│   ├── README.md
│   ├── sentencepiece.bpe.model
│   ├── tokenizer_config.json
│   └── vocab.json
│
├── inference.py
├── prepare_data.py
├── README.md
├── requirements.txt
└── train.py
```

### Key files

| File                                | Purpose                                      |
| ----------------------------------- | -------------------------------------------- |
| `data/ur_to_rur_data.csv`         | Local copy of the canonical training corpus  |
| `prepare_data.py`                 | Prepares the raw CSV for training            |
| `train.py`                        | LoRA fine-tuning pipeline                    |
| `inference.py`                    | Local inference using a trained adapter      |
| `benchmark/benchmark_dataset.csv` | Held-out benchmark dataset                   |
| `benchmark/run_benchmark.py`      | Generates model predictions                  |
| `benchmark/score_benchmark.py`    | Scores benchmark predictions                 |
| `benchmark/results/`              | Predictions and aggregated benchmark results |
| `DATASET_HISTORY.md`              | Dataset construction and lineage             |
| `EXPERIMENT_HISTORY.md`           | Full experiment chronology                   |
| `dataset_run_summary.csv`         | Run-level dataset summary                    |

The production Run 4 adapter is distributed through Hugging Face rather than being treated as the primary model distribution mechanism of this GitHub repository.

---

# Quick Start

The following workflow is the recommended order for reproducing the project.

## 1. Clone Repository

```bash
git clone https://github.com/Maaz-x14/m2m100-rup-ur-to-rur-fine_tune.git
cd m2m100-rup-ur-to-rur-fine_tune
```

---

## 2. Create Virtual Environment

Create a dedicated Python virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Verify that the environment is active:

```bash
which python
```

On Windows:

```powershell
where python
```

---

## 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The training pipeline uses the M2M100, PEFT, PyTorch, Hugging Face Datasets, Accelerate, and SacreBLEU ecosystem.

For GPU training, ensure that your PyTorch installation is compatible with your CUDA environment.

---

## 4. Obtain Dataset

The canonical dataset is hosted on Hugging Face:

**[`Maaz-x14/urdu-to-roman-urdu`](https://huggingface.co/datasets/Maaz-x14/urdu-to-roman-urdu)**

The released dataset contains:

```text
4,860 Urdu–Roman Urdu parallel examples
```

For local Run 4 reproduction, place the canonical dataset at:

```text
data/ur_to_rur_data.csv
```

The expected columns are:

```text
urdu
roman_urdu
```

The local filename is not itself the source of truth; the canonical dataset release is the Hugging Face dataset repository above.

---

## 5. Prepare Dataset

Run:

```bash
python prepare_data.py \
    --csv data/ur_to_rur_data.csv \
    --output_dir ./processed_dataset
```

This produces the processed dataset consumed by the training pipeline.

The resulting directory is:

```text
processed_dataset/
```

---

## 6. Train Run 4

Run:

```bash
python train.py \
    --dataset_dir ./processed_dataset \
    --output_dir ./checkpoints \
    --final_model_dir ./fine_tuned_model
```

The training script:

1. Loads the M2M100 tokenizer.
2. Loads the M2M100 base model.
3. Configures the Roman Urdu target language.
4. Applies the M2M100 generation-path patch required by the training implementation.
5. Attaches a LoRA adapter.
6. Loads the prepared dataset.
7. Trains with sequence-to-sequence training.
8. Evaluates during training.
9. Saves the LoRA adapter and tokenizer.
10. Runs final evaluation.

The resulting model directory contains the trained adapter and tokenizer artifacts.

---

## 7. Run Benchmark

The benchmark uses a separate held-out dataset:

```text
benchmark/benchmark_dataset.csv
```

Run:

```bash
python benchmark/run_benchmark.py --help
```

Review the supported arguments, then execute the benchmark using the desired model configuration.

The benchmark pipeline is designed to generate predictions for model versions including:

```text
Mavkif base
Run 1
Run 2
Run 3
Run 4
```

Predictions are stored under:

```text
benchmark/results/
```

---

## 8. Score Benchmark

After generating predictions, run:

```bash
python benchmark/score_benchmark.py --help
```

Then execute the scoring command supported by the current script.

The aggregated results are written to:

```text
benchmark/results/benchmark_results.csv
```

The benchmark should be evaluated at both:

* Overall level
* Category level

This allows improvements and regressions to be identified for specific linguistic categories rather than relying only on a single aggregate metric.

---

## 9. Run Local Inference

To inspect the supported inference interface:

```bash
python inference.py --help
```

Run the inference script using the supported arguments.

The conceptual inference path is:

```text
Urdu Text
    │
    ▼
Tokenizer
    │
    ▼
M2M100 Base
    +
LoRA Adapter
    │
    ▼
Generation
    │
    ▼
Roman Urdu
```

For production deployment, use the dedicated inference repository rather than treating this local script as the production API.

---

# Reproducing Run 4

Run 4 is the current final training experiment.

The important dataset lineage is:

```text
Run 1
  │
  └──► Run 2
          │
          └──► Run 1 + Run 2 Combined Dataset
                         │
                         ▼
                  Expanded Run 3 Data
                         │
                         ▼
                    Run 4 Dataset
                         │
                         ▼
              Fresh Fine-Tuning from Base
                         │
                         ▼
                       Run 4
```

There is an important distinction between:

* **Run 3:** The 563-row training run.
* **Expanded Run 3:** Additional data collected after the Run 3 training run.
* **Run 4:** The final training run using the consolidated dataset.

Run 4 was trained from the base model using the consolidated dataset rather than continuing from the degraded Run 3 adapter.

For the complete chronology, see:

```text
EXPERIMENT_HISTORY.md
```

For dataset construction and lineage, see:

```text
DATASET_HISTORY.md
```

### Complete Run 4 reproduction sequence

```bash
# 1. Clone
git clone https://github.com/Maaz-x14/m2m100-rup-ur-to-rur-fine_tune.git
cd m2m100-rup-ur-to-rur-fine_tune

# 2. Create environment
python -m venv .venv

# 3. Activate environment
source .venv/bin/activate

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Place the canonical dataset at:
# data/ur_to_rur_data.csv

# 6. Prepare data
python prepare_data.py \
    --csv data/ur_to_rur_data.csv \
    --output_dir ./processed_dataset

# 7. Train Run 4
python train.py \
    --dataset_dir ./processed_dataset \
    --output_dir ./checkpoints \
    --final_model_dir ./fine_tuned_model

# 8. Inspect benchmark interface
python benchmark/run_benchmark.py --help

# 9. Generate benchmark predictions
python benchmark/run_benchmark.py

# 10. Inspect scoring interface
python benchmark/score_benchmark.py --help

# 11. Score benchmark
python benchmark/score_benchmark.py

# 12. Inspect local inference interface
python inference.py --help
```

> **Reproducibility note:** The commands above reproduce the project workflow. Exact numerical reproducibility also depends on the Python/package versions, hardware, CUDA environment, random seeds, and the exact dataset version used.

---

# Dataset

The canonical dataset contains:

```text
4,860 Urdu–Roman Urdu parallel examples
```

Columns:

```text
urdu
roman_urdu
```

The dataset was created because available Urdu → Roman Urdu resources did not sufficiently cover the modern language distribution targeted by this project.

The additional corpus focuses on patterns such as:

```text
فین       → fan
لائٹ      → light
بلب       → bulb
وائی فائی → wifi
ٹیبل      → table
```

The dataset was assembled using three broad approaches:

1. AI-assisted generation
2. Urdu user-query data obtained from another project and converted into Urdu → Roman Urdu pairs
3. Urdu text collected through search/web sources and converted into Roman Urdu

The resulting data was reviewed manually by a native Urdu speaker and additionally checked through AI-assisted validation.

The canonical public release is:

**[`Maaz-x14/urdu-to-roman-urdu`](https://huggingface.co/datasets/Maaz-x14/urdu-to-roman-urdu)**

For detailed dataset provenance and construction history, see:

```text
DATASET_HISTORY.md
```

---

# Training Pipeline

The training system is divided into four major stages:

```text
┌─────────────────────────┐
│ Canonical Dataset       │
│ 4,860 Urdu/Roman Urdu   │
│ parallel examples       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ prepare_data.py         │
│                         │
│ Dataset preparation     │
│ Tokenization            │
│ Train/validation split  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ train.py                │
│                         │
│ M2M100 base model       │
│ + LoRA adapter          │
│ + Seq2Seq training      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Trained LoRA Adapter    │
└────────────┬────────────┘
             │
             ├──────────────► Local Inference
             │
             ▼
┌─────────────────────────┐
│ Held-Out Benchmark      │
│                         │
│ Base vs Run 1–4         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Overall + Category      │
│ Evaluation              │
└─────────────────────────┘
```

---

## LoRA Configuration

The current training implementation targets:

```text
q_proj
k_proj
v_proj
out_proj
```

The default LoRA configuration in `train.py` is:

```text
r = 16
alpha = 32
dropout = 0.1
bias = none
```

Training defaults include:

```text
Epochs:                  30
Train batch size:        16
Evaluation batch size:   8
Gradient accumulation:   4
Learning rate:           5e-4
Warmup ratio:            0.06
Label smoothing:         0.1
Early stopping patience: 3
Max new tokens:          128
DataLoader workers:      2
Seed:                    42
```

The training implementation uses:

```text
FP16
Single-GPU execution
Greedy generation during evaluation
Cosine learning-rate scheduling
AdamW
```

These are the current defaults encoded in `train.py`.

---

# Benchmarking

The repository contains a separate benchmark suite so that model versions can be compared consistently.

```text
benchmark/benchmark_dataset.csv
          │
          ▼
  run_benchmark.py
          │
          ▼
     Predictions
          │
          ▼
 score_benchmark.py
          │
          ▼
benchmark_results.csv
```

The benchmark includes comparison data for:

```text
Mavkif Base
Run 1
Run 2
Run 3
Run 4
```

The prediction artifacts are:

```text
benchmark/results/
├── predictions_mavkif_base.csv
├── predictions_run1.csv
├── predictions_run2.csv
├── predictions_run3.csv
└── predictions_run4.csv
```

The aggregated results are:

```text
benchmark/results/benchmark_results.csv
```

The benchmark is intended to answer two questions:

1. Did fine-tuning improve overall Urdu → Roman Urdu transliteration?
2. Did fine-tuning improve the specific linguistic patterns targeted by the additional dataset?

The second question is critical.

A model can improve on modern loanwords while regressing on other categories.

Therefore, the project evaluates both:

```text
Overall Metrics
```

and:

```text
Category-Level Metrics
```

The benchmark is a project-specific evaluation suite and should not be interpreted as a universal standardized Urdu transliteration benchmark.

---

# Run History

The project's high-level experiment progression is:

```text
Base Model
    │
    ▼
Run 1
2,012 rows
    │
    ▼
Run 2
489 additional rows
    │
    ▼
Run 1 + Run 2
2,500 actual rows
    │
    ▼
Run 3
563-row targeted training run
    │
    ▼
Regression / training issues
    │
    ▼
Expanded Run 3 data collection
    │
    ▼
Run 4
Consolidated dataset
4,860 rows
    │
    ▼
Fresh fine-tuning from base model
```

Run 3 is an important part of the project's engineering history.

The experiment demonstrated that a small corrective dataset can cause regression when used as the dominant training distribution.

Run 4 therefore used the consolidated dataset and restarted from the base model.

For the full chronology, including bugs, failed approaches, recovery steps, and lessons learned:

```text
EXPERIMENT_HISTORY.md
```

For the dataset-specific lineage:

```text
DATASET_HISTORY.md
```

---

# Model Distribution

The current production-oriented model is:

**[`Maaz-x14/m2m100-ur-to-roman-urdu`](https://huggingface.co/Maaz-x14/m2m100-ur-to-roman-urdu)**

The model repository contains the released Run 4 LoRA adapter and its canonical model documentation.

The adapter is intended to be used with the compatible base model:

```text
Mavkif/m2m100_rup_ur_to_rur
```

and tokenizer:

```text
Mavkif/m2m100_rup_tokenizer_both
```

The recommended production artifact is therefore:

```text
M2M100 Base
      +
Run 4 LoRA Adapter
      +
Compatible Tokenizer
```

The production model is consumed by the separate inference service.

---

# Production Inference

For production deployment, use:

**[`Maaz-x14/m2m100-roman-deploy`](https://github.com/Maaz-x14/m2m100-roman-deploy)**

The production architecture is:

```text
Client
  │
  │ HTTP Request
  ▼
FastAPI Inference Service
  │
  ▼
M2M100 Base Model
  +
Run 4 LoRA Adapter
  │
  ▼
Roman Urdu
  │
  ▼
HTTP Response
```

This fine-tuning repository is responsible for:

```text
Research
Training
Evaluation
Reproducibility
```

The inference repository is responsible for:

```text
Serving
API
Deployment
Production inference
```

For production users, the recommended path is:

```text
Hugging Face Run 4 Model
           │
           ▼
Inference Repository
           │
           ▼
FastAPI Service
           │
           ▼
POST /transliterate
```

---

# Known Engineering Lessons

## 1. Small corrective datasets can cause regression

Run 3 demonstrated that continuing or adapting heavily toward a small targeted dataset can damage broader model behavior.

Targeted fine-tuning must therefore be evaluated against the full task distribution.

---

## 2. Dataset size is not enough

A larger dataset is not automatically a better dataset.

The data must represent the linguistic distribution of the intended application.

For this project, that includes contemporary Urdu and Urdu-script English loanwords.

---

## 3. Aggregate metrics can hide regressions

A single BLEU score is insufficient to understand model behavior.

Category-level evaluation is necessary to determine whether a change improves the intended failure modes without damaging other capabilities.

---

## 4. Training and evaluation must be separated

The training dataset and held-out benchmark serve different purposes.

```text
Training Dataset
    │
    └── Used to optimize model parameters

Held-Out Benchmark
    │
    └── Used to compare model versions
```

The benchmark should not be treated as another training resource.

---

## 5. Transformer library changes can affect generation

The training pipeline includes explicit handling for generation behavior in the M2M100 implementation used by this project.

The relevant fixes are documented directly in `train.py`.

The implementation addresses issues involving:

* Decoder input handling
* `decoder_inputs_embeds`
* Generation
* `forced_bos_token_id`
* Evaluation-time generation
* GPU memory behavior
* Checkpoint handling

The source code comments in `train.py` contain the detailed engineering rationale for these fixes.

---

# Limitations

## Dataset limitations

The dataset includes AI-assisted generation and conversion.

The exact numerical distribution of examples by collection source is not fully recorded.

---

## Roman Urdu variability

Roman Urdu does not have one universally standardized spelling system.

Multiple Roman Urdu spellings may be linguistically acceptable for the same Urdu input.

Therefore, exact string-match metrics may underestimate practical quality in some cases.

---

## Benchmark limitations

The benchmark is project-specific.

It is useful for comparing model versions within this project but should not be interpreted as a universal Urdu transliteration benchmark.

---

## Model limitations

The model may still produce:

* Incorrect transliterations
* Inconsistent Roman Urdu spelling
* Errors on ambiguous Urdu words
* Errors on names and places
* Errors on unseen vocabulary
* Errors on unusual sentence structures
* Inconsistent handling of code-switched text

---

## Evaluation limitations

Automated metrics such as BLEU and chrF do not fully capture:

* Phonetic correctness
* Orthographic preference
* Human readability
* Acceptable Roman Urdu spelling variation
* Contextual appropriateness

Human evaluation remains valuable for future work.

---

# Citation

If you use this repository, methodology, or training pipeline, please cite:

```bibtex
@software{maaz_m2m100_urdu_roman_urdu_2026,
  author       = {Maaz Ahmad},
  title        = {M2M100 Urdu to Roman Urdu Transliteration},
  year         = {2026},
  url          = {https://github.com/Maaz-x14/m2m100-rup-ur-to-rur-fine_tune}
}
```

For the released model and dataset, please refer to their respective Hugging Face repositories:

* [`Maaz-x14/m2m100-ur-to-roman-urdu`](https://huggingface.co/Maaz-x14/m2m100-ur-to-roman-urdu)
* [`Maaz-x14/urdu-to-roman-urdu`](https://huggingface.co/datasets/Maaz-x14/urdu-to-roman-urdu)

---

# License

See the repository license file for the applicable terms.

The base model, tokenizer, datasets, and other third-party resources may have separate licenses and terms.

Users are responsible for reviewing the applicable licenses before redistribution or commercial use.

---

# Acknowledgements

Developed by:

**Maaz Ahmad**

**National University of Sciences and Technology (NUST)**

**Independent Open Source Project**

This project builds upon the open-source M2M100 ecosystem and the work underlying:

```text
Mavkif/m2m100_rup_ur_to_rur
```

and its associated Urdu–Roman Urdu resources.

---

# Project Status

The **Run 4** model is the current production-oriented release of this fine-tuning effort.

The project is organized around four canonical artifacts:

```text
1. Fine-Tuning Repository
   Training + Evaluation + Reproducibility

2. Hugging Face Dataset
   Canonical 4,860-example Dataset

3. Hugging Face Model
   Run 4 LoRA Adapter

4. Inference Repository
   Production FastAPI Deployment
```

Future work may include:

* Larger and more diverse contemporary Urdu datasets
* More systematic human annotation
* Improved Roman Urdu normalization
* Larger and more rigorous benchmark sets
* Human evaluation
* Error-specific evaluation
* Improved handling of names and places
* Production inference optimization
* Latency and throughput benchmarking
