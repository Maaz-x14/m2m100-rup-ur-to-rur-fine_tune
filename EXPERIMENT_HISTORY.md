
# Experiment History

This document records the development history of the Urdu → Roman Urdu transliteration system, from the initial baseline through the current Run 4 production model.

The purpose of this document is to preserve the reasoning, dataset evolution, engineering failures, regressions, and lessons learned behind the project.

The experiment history is intentionally separate from the model card.

* The **Model Card** describes the current model.
* The **Dataset Card** describes the current dataset.
* This document explains  **how the project got here** .

The project evolved through multiple training runs. The most important progression was:

```text
Mavkif Base Model
        │
        ▼
     Run 1
  2,012 examples
        │
        ▼
     Run 2
  +489 examples
  2,500 total examples
        │
        ▼
     Run 3
   563-example
  experimental branch
        │
        ├──────► Training failure / regression
        │
        ▼
 Expanded Run 3
 Additional generated data
        │
        ▼
     Run 4
  Run 1 + Run 2
        +
  Expanded Run 3
        │
        ▼
  4,860-example
  production dataset
        │
        ▼
  Production Run 4
        │
        ▼
 Hugging Face Model
        │
        ▼
 Production Inference API
```

---

# 1. Project Background

## 1.1 The Original Problem

The goal of the project is to build a practical Urdu-script → Roman Urdu transliteration system.

The task is  **transliteration** , not translation.

The system should transform Urdu written in the native Urdu script into Roman Urdu while preserving the underlying linguistic content.

For example:

```text
Urdu:
آپ کیسے ہیں؟

Roman Urdu:
Aap kaise hain?
```

The project began from an existing M2M100-based Urdu → Roman Urdu model rather than training a large multilingual model from scratch.

The starting point was:

```text
Mavkif/m2m100_rup_ur_to_rur
```

The model itself is built on the M2M100 architecture and was already capable of Urdu → Roman Urdu transliteration.

However, the project identified a domain-coverage problem.

The existing training data was not sufficiently representative of the modern Urdu encountered in practical applications.

The target use case required coverage of:

* everyday conversational Urdu
* modern vocabulary
* English loanwords written in Urdu script
* technical terms
* common contemporary expressions
* code-switched language patterns

Examples of the type of vocabulary that motivated additional training included words such as:

```text
فین
لائٹ
بلب
وائی فائی
ٹیبل
کمپیوٹر
ڈاکٹر
```

The objective was therefore not to replace the original model, but to  **adapt and extend its domain coverage** .

---

# 2. Initial Baseline: Mavkif Model

The starting model was:

```text
Mavkif/m2m100_rup_ur_to_rur
```

The project treated this model as the initial baseline and foundation for further fine-tuning.

The motivation for additional fine-tuning was primarily domain adaptation.

The existing model was useful for Urdu → Roman Urdu transliteration, but the project required stronger coverage of modern and conversational Urdu.

The project therefore moved toward constructing a custom parallel dataset.

---

# 3. Dataset Creation

Finding a suitable Urdu → Roman Urdu dataset was one of the first major challenges.

The project investigated available resources but did not find a dataset that directly matched the intended production domain.

One important reference dataset was:

```text
Mavkif/Roman-Urdu-Parl-split
```

The project also considered the Dakshina dataset.

However, the project concluded that these existing resources did not fully solve the target domain problem.

The central issue was that the available corpora did not sufficiently represent the modern Urdu vocabulary and everyday usage required by the intended application.

As a result, the project moved toward creating its own parallel corpus.

---

# 4. Custom Data Collection Strategy

The custom dataset was created using three primary approaches.

## Approach 1 — AI-Generated Parallel Data

The first source consisted of Urdu → Roman Urdu examples generated with the assistance of AI agents.

The objective was to deliberately create examples covering the linguistic patterns and vocabulary that were missing or underrepresented in existing resources.

This allowed the dataset to target specific weaknesses rather than simply adding random Urdu sentences.

---

## Approach 2 — Existing User Query Data

The second source originated from another project.

This data consisted of actual Urdu user queries.

The original data contained Urdu text but did not contain corresponding Roman Urdu references.

Therefore, the Urdu sentences had to be converted into Roman Urdu.

AI agents were used to assist with this conversion.

The resulting pairs were then manually reviewed and validated.

This source was valuable because the Urdu text represented real user input rather than purely synthetic sentences.

---

## Approach 3 — Search-Engine-Sourced Urdu Data

The third source consisted of Urdu text collected from search-engine-related sources.

Again, the original material did not necessarily contain aligned Roman Urdu references.

The Urdu text was therefore converted into Roman Urdu with the assistance of AI agents.

The resulting pairs were then reviewed and validated.

---

# 5. Data Validation

Data generation alone was not considered sufficient.

The collected and generated data underwent validation.

Two complementary approaches were used.

### Manual Validation

The dataset was manually inspected by the project author, who is a native Urdu speaker.

This was particularly important because Roman Urdu does not have one universally standardized spelling system.

A model may produce a Romanization that is technically plausible but unnatural or inconsistent with common Pakistani usage.

Manual review was therefore used to identify:

* incorrect Urdu interpretation
* incorrect Roman Urdu mappings
* unnatural spellings
* malformed examples
* obvious AI-generation errors
* inconsistencies in transliteration

### AI-Assisted Validation

AI agents were also used to assist with validation.

The AI-assisted process was used as an additional quality-control layer rather than as the sole source of truth.

The final dataset therefore combined:

```text
Multiple Data Sources
        │
        ▼
Urdu → Roman Urdu Alignment
        │
        ▼
AI-Assisted Validation
        │
        ▼
Manual Native-Speaker Review
        │
        ▼
Training Dataset
```

The project did not maintain exact per-source counts for every data-generation approach.

Therefore, this history does **not** claim a numerical breakdown of the final dataset by source.

---

# 6. Run 1 — Initial Fine-Tuning

## Dataset Size

Run 1 used:

```text
2,012 examples
```

This is the confirmed actual number of data rows in the Run 1 dataset.

The Run 1 dataset represented the first substantial custom fine-tuning corpus created specifically for the project's target domain.

---

## Objective

The primary objective was to determine whether additional fine-tuning on modern, conversational, and domain-targeted Urdu data could improve the Mavkif baseline for the project's intended use case.

The model was fine-tuned using parameter-efficient LoRA adaptation.

The resulting model became the first major project-specific model version.

---

## Outcome

Run 1 established the first working project baseline.

It also revealed that the dataset-generation approach was viable and that the project could iteratively improve the model by expanding the training corpus.

The Run 1 model was subsequently published as the initial Hugging Face model version.

---

# 7. Run 2 — Dataset Expansion

Run 2 extended the Run 1 dataset.

The additional data generated for Run 2 consisted of:

```text
489 examples
```

The resulting combined dataset contained:

```text
Run 1:
2,012 rows

Run 2 additions:
489 rows

Total:
2,501 rows including the header
2,500 actual data rows
```

The distinction between **rows including the CSV header** and **actual training examples** is important.

The canonical Run 2 dataset contained:

```text
2,500 actual examples
+ 1 CSV header
= 2,501 CSV lines
```

---

## Objective

The purpose of Run 2 was to expand the training distribution beyond the original Run 1 corpus.

The project continued targeting modern Urdu, common conversational vocabulary, and the types of transliteration cases that motivated the original fine-tuning effort.

---

## Outcome

Run 2 produced the next stage of the fine-tuned model.

The Run 1 → Run 2 progression established the project's first iterative data-development cycle:

```text
Run 1
2,012 examples
       │
       │ +489
       ▼
Run 2
2,500 examples
```

The experiment demonstrated the project's core iterative methodology:

> Identify a coverage gap → add targeted data → retrain → evaluate → identify the next gap.

---

# 8. Run 3 — Experimental Expansion

Run 3 was a more problematic stage of the project.

The initial Run 3 training dataset contained:

```text
563 examples
```

These 563 examples represent the  **Run 3 dataset used for the Run 3 training experiment** .

This number must not be confused with the later expanded Run 3 dataset.

---

## Run 3 Objective

The objective was to continue improving the model through additional targeted data.

The project trained a model using the 563-example Run 3 dataset.

However, the training process encountered a serious error.

The exact technical details of every intermediate failure are not preserved in this experiment-history context, so this document does not invent a specific root cause.

What is confirmed is:

```text
Run 3 data
   │
   ▼
Model trained
   │
   ▼
Serious error / bad result
   │
   ▼
Training outcome considered unacceptable
```

The result was sufficiently problematic that the experiment was abandoned as a production candidate.

---

# 9. Run 3 Catastrophe

Run 3 became the major turning point in the project.

The problem was not simply that the model failed to improve.

The experiment exposed a broader issue with the project's iterative training process.

The project had to deal with:

* a bad training outcome
* frustration caused by the failure
* additional data generation performed during debugging and experimentation
* uncertainty around which data belonged to which experimental branch

At this point, it became important to distinguish between:

```text
Run 3
```

and:

```text
Expanded Run 3
```

These are  **not the same dataset** .

---

# 10. Expanded Run 3

After the problematic Run 3 experiment, additional data was generated.

The project did not train the model on the entire expanded dataset as a separate standalone Run 3 experiment.

Instead, the additional data became part of what is now referred to as:

```text
Expanded Run 3
```

The purpose of this distinction is historical accuracy.

The chronology is:

```text
Run 3
563 examples
    │
    ▼
Trained
    │
    ▼
Bad result / training problem
    │
    ▼
Additional data generated
    │
    ▼
Expanded Run 3 dataset
```

The expanded Run 3 data was subsequently combined with the earlier Run 3-related data and used as part of the Run 4 dataset lineage.

Therefore:

> **Run 3 is the 563-example experimental training run. Expanded Run 3 is the later expanded data branch that contributed to Run 4.**

This distinction should be preserved in future documentation.

---

# 11. Run 4 — Recovery and Consolidation

Run 4 was the recovery phase after the Run 3 failure.

Rather than treating the failed Run 3 experiment as the final direction, the project consolidated the data developed across the earlier experiments.

The Run 4 dataset was constructed from:

```text
Run 1 + Run 2 combined dataset
+
Expanded Run 3 dataset
```

The current canonical dataset contains:

```text
4,860 actual data rows
```

The CSV therefore contains:

```text
4,860 actual examples
+ 1 header
= 4,861 CSV lines
```

The final dataset statistics are:

```text
Total rows:
4,860

Unique Urdu sentences:
4,725

Unique Roman Urdu sentences:
4,726

Rows where both Urdu and Roman Urdu are unique:
4,594

Duplicate Urdu sentences:
135

Duplicate Roman Urdu sentences:
134

Duplicate Urdu/Roman Urdu pairs:
133
```

The important point is that the dataset was  **not created as four independent isolated datasets** .

The source file:

```text
data/rur_to_ur_data.csv
```

contains the data in chronological append order.

Therefore, the historical dataset boundaries can be reconstructed from row order.

This provides a reproducible way to derive the earlier experimental datasets without maintaining four permanently separate copies.

---

# 12. Run 4 Dataset Lineage

The final lineage is:

```text
Run 1
2,012 examples
        │
        ▼
Run 2 additions
489 examples
        │
        ▼
Run 1 + Run 2
2,500 examples
        │
        ├─────────────────────┐
        │                     │
        ▼                     ▼
Run 3 experimental       Expanded Run 3
563 examples             Additional generated data
        │                     │
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
              Run 4 Dataset
             4,860 examples
```

The exact row boundaries are preserved by the append order of the canonical dataset.

The dataset history should therefore be interpreted as a  **lineage** , not merely as four unrelated dataset snapshots.

---

# 13. Run 4 Training

Run 4 became the production-oriented training run.

The objective shifted from exploratory experimentation toward creating a coherent, documented, reproducible model release.

The Run 4 process included:

```text
Final Dataset
      │
      ▼
Data Preparation
      │
      ▼
Train / Validation Split
      │
      ▼
LoRA Fine-Tuning
      │
      ▼
Evaluation
      │
      ▼
Benchmarking
      │
      ▼
Hugging Face Model Release
      │
      ▼
Production Inference Service
```

The current training repository contains the code required for the training and data preparation workflow, including:

```text
prepare_data.py
train.py
benchmark/
inference.py
```

The final model was then documented as the canonical Hugging Face model:

```text
Maaz-x14/m2m100-ur-to-roman-urdu
```

The Hugging Face model is the public model artifact, while the GitHub training repository provides the engineering and reproducibility context.

---

# 14. Evaluation Evolution

Evaluation became increasingly important as the project progressed.

The early experiments were primarily focused on improving the model through additional data.

As the project matured, the evaluation process became more systematic.

The project eventually introduced a dedicated benchmark directory containing:

```text
benchmark/
├── benchmark_dataset.csv
├── run_benchmark.py
├── score_benchmark.py
└── results/
```

The benchmark results preserve predictions from:

```text
Mavkif Base
Run 1
Run 2
Run 3
Run 4
```

This makes it possible to compare the evolution of the system rather than evaluating Run 4 in isolation.

The benchmark process separates:

```text
Model Inference
        │
        ▼
Predictions
        │
        ▼
Scoring
        │
        ▼
Comparison
```

This separation is important because a model should not be judged solely by a single training loss or validation score.

---

# 15. Regression Lessons

The Run 3 failure was one of the most important lessons in the project.

The primary lesson was:

> Adding more data is not automatically equivalent to improving a model.

A dataset can grow while model quality decreases.

Potential causes include:

* distribution shifts
* noisy generated examples
* inconsistent annotation
* incorrect transliteration targets
* training instability
* insufficient evaluation coverage
* changes in preprocessing
* accidental contamination between training and evaluation data

The Run 3 experience reinforced the need for:

```text
Controlled Dataset Lineage
+
Reproducible Splits
+
Fixed Evaluation Sets
+
Benchmark Comparisons
+
Experiment Tracking
```

Future experiments should therefore preserve the ability to answer:

1. What exact data was used?
2. How many examples were used?
3. What changed from the previous run?
4. What preprocessing was applied?
5. What model configuration was used?
6. What evaluation data was used?
7. Did the model improve or regress?
8. Can the experiment be reproduced?

---

# 16. Dataset Lineage Lesson

Another major lesson was the importance of keeping dataset history explicit.

During development, several data-generation cycles occurred.

This created potential ambiguity between:

```text
Run 3
```

and:

```text
Expanded Run 3
```

The final project therefore adopted a chronological dataset model.

The canonical source:

```text
data/rur_to_ur_data.csv
```

contains the complete dataset in append order.

This makes the historical structure recoverable.

The project also introduced:

```text
DATASET_HISTORY.md
```

and:

```text
dataset_run_summary.csv
```

These files exist to prevent future confusion about dataset boundaries and lineage.

---

# 17. Productionization

After Run 4, the project moved beyond experimentation.

The project was organized into multiple artifacts, each with a distinct responsibility.

## Training Repository

Responsible for:

* dataset preparation
* training
* evaluation
* benchmark scripts
* experiment history
* reproducibility

Repository:

```text
m2m100-rup-ur-to-rur-fine_tune
```

---

## Hugging Face Model

Responsible for:

* public model distribution
* canonical model card
* model usage instructions
* model limitations
* evaluation summary

Model:

```text
Maaz-x14/m2m100-ur-to-roman-urdu
```

The current Hugging Face model is documented as a PEFT/LoRA adapter based on the Mavkif model lineage.

---

## Dataset

Responsible for:

* dataset distribution
* motivation
* collection methodology
* annotation methodology
* statistics
* examples
* dataset history

The dataset documentation is intentionally separated from the model documentation.

---

## Inference Repository

The production serving layer was separated into:

```text
m2m100-roman-deploy
```

The inference service provides:

* FastAPI HTTP API
* Urdu → Roman Urdu inference
* single-request inference
* explicit batch inference
* dynamic batching
* LoRA merging
* model warm-up
* health/readiness monitoring

This separation creates a clear architecture:

```text
Training Repository
        │
        │ produces
        ▼
Hugging Face Model
        │
        │ consumed by
        ▼
Inference Repository
        │
        ▼
Production API
```

---

# 18. Current Project State

The project has now progressed from an experimental fine-tuning effort into a complete open-source model project.

The current ecosystem consists of:

```text
                    ┌──────────────────────┐
                    │  Training Repository │
                    │                      │
                    │ prepare_data.py      │
                    │ train.py             │
                    │ benchmark/           │
                    │ inference.py         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Run 4 Model       │
                    │                      │
                    │ Hugging Face         │
                    │ Model Card           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Dataset Release    │
                    │                      │
                    │ 4,860 examples       │
                    │ Dataset Card         │
                    │ Dataset History      │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Inference Repository│
                    │                      │
                    │ FastAPI              │
                    │ Dynamic Batching     │
                    │ Production Serving   │
                    └──────────────────────┘
```

The project is no longer just a fine-tuned model.

It is now structured as:

```text
Dataset
+
Training Code
+
Experiment History
+
Evaluation
+
Model
+
Inference Service
```

This is the intended final open-source structure.

---

# 19. Complete Run Summary

| Stage                | Dataset / Model State                          | Purpose                                   | Outcome                                              |
| -------------------- | ---------------------------------------------- | ----------------------------------------- | ---------------------------------------------------- |
| Mavkif Baseline      | Existing M2M100-based Urdu → Roman Urdu model | Starting point                            | Useful baseline, insufficient target-domain coverage |
| Run 1                | 2,012 examples                                 | Initial custom domain adaptation          | First project-specific fine-tuned model              |
| Run 2                | +489 examples; 2,500 actual examples total     | Expand training coverage                  | Second iterative model                               |
| Run 3                | 563 examples                                   | Experimental additional fine-tuning       | Bad training outcome / regression                    |
| Expanded Run 3       | Additional data generated after Run 3 failure  | Recover and expand data                   | Became part of Run 4 lineage                         |
| Run 4                | 4,860 examples                                 | Consolidated production-oriented training | Current production model                             |
| Production Release   | Run 4 Hugging Face model                       | Public distribution                       | Canonical model artifact                             |
| Production Inference | FastAPI deployment repository                  | Application integration                   | Production-oriented serving layer                    |

---

# 20. What Went Wrong, and What Was Fixed

## Problem 1 — Existing Data Did Not Match the Target Domain

**Problem**

The existing model was not sufficiently adapted to modern, everyday Urdu and loanword-heavy usage.

**Response**

Create a targeted Urdu → Roman Urdu dataset.

---

## Problem 2 — Suitable Public Dataset Was Difficult to Find

**Problem**

Available datasets did not sufficiently match the project's desired domain.

**Response**

Build a custom dataset using multiple sources and generation methods.

---

## Problem 3 — AI-Generated Data Requires Validation

**Problem**

AI-generated parallel text can contain incorrect or unnatural transliterations.

**Response**

Combine:

```text
AI-assisted generation
+
AI-assisted validation
+
Native-speaker manual review
```

---

## Problem 4 — Run 3 Produced a Bad Result

**Problem**

The 563-example Run 3 experiment produced an unacceptable outcome.

**Response**

Do not promote the failed run to production.

Instead:

```text
Preserve Run 3 history
        │
        ▼
Generate additional data
        │
        ▼
Expand the dataset
        │
        ▼
Consolidate into Run 4
        │
        ▼
Evaluate again
```

---

## Problem 5 — Dataset History Became Confusing

**Problem**

Run 3 and the later expanded Run 3 data could easily be conflated.

**Response**

Explicitly distinguish:

```text
Run 3 = 563-example training experiment

Expanded Run 3 = later expanded data branch contributing to Run 4
```

---

## Problem 6 — Documentation Was Fragmented

**Problem**

The project contained information across model cards, README files, datasets, scripts, and experiment artifacts.

**Response**

Separate documentation by audience:

```text
Hugging Face Model Card
        │
        ├── What is the model?
        └── How good is it?

Dataset Card
        │
        ├── Where did the data come from?
        └── How was it constructed?

GitHub README
        │
        ├── How is it trained?
        └── How is it reproduced?

Inference README
        │
        ├── How do I deploy it?
        └── How do I call the API?

EXPERIMENT_HISTORY.md
        │
        └── How did the project evolve?
```

This separation is now the intended documentation architecture.

---

# 21. Future Work

The next stage of development should focus less on simply increasing dataset size and more on improving experimental rigor.

Potential directions include:

### Better Dataset Governance

* Record exact provenance for every example.
* Track source type per row.
* Track whether an example was AI-generated, user-derived, or externally sourced.
* Maintain versioned dataset releases.
* Record annotation and validation status.

### Stronger Evaluation

Future evaluations should consider metrics beyond BLEU where appropriate.

Potential metrics include:

* character-level metrics
* CHRF
* exact-match accuracy for controlled cases
* word-level accuracy
* human evaluation
* error-category analysis

The choice of metrics should reflect the fact that Roman Urdu has spelling variation.

### Better Error Analysis

Build dedicated evaluation sets for:

* modern loanwords
* English code-switching
* conversational Urdu
* ambiguous Urdu without diacritics
* names and proper nouns
* numbers
* punctuation
* long sentences
* rare vocabulary

### Controlled Experiments

Future experiments should change one major variable at a time whenever possible.

For example:

```text
Same Dataset
+
Different LoRA Configuration
```

or:

```text
Same Model Configuration
+
Different Dataset Version
```

This makes causal conclusions more reliable.

### Reproducible Environments

Future releases should pin exact versions of:

* PyTorch
* Transformers
* PEFT
* Datasets
* Accelerate
* CUDA-related dependencies

The current requirements files use minimum-version constraints in places, which are convenient for development but weaker for strict reproducibility.

### Production Deployment

The inference repository can eventually add:

* Docker support
* GPU-enabled container deployment
* authentication
* rate limiting
* observability
* structured metrics
* request tracing
* automated model download
* CI/CD
* deployment health checks

---

# 22. Final Perspective

The most important outcome of the project is not simply that the dataset grew from 2,012 examples to 4,860 examples.

The important progression was methodological.

The project moved from:

```text
Find a model
      │
      ▼
Fine-tune it
      │
      ▼
Generate more data
      │
      ▼
Train again
```

toward:

```text
Define the target domain
        │
        ▼
Identify dataset gaps
        │
        ▼
Construct targeted data
        │
        ▼
Validate data
        │
        ▼
Track dataset lineage
        │
        ▼
Run controlled experiments
        │
        ▼
Benchmark model variants
        │
        ▼
Document failures and regressions
        │
        ▼
Select production candidate
        │
        ▼
Publish model + dataset
        │
        ▼
Provide reproducible training code
        │
        ▼
Provide production inference service
```

The Run 3 failure was therefore not erased from the project's history.

It is part of the reason the project evolved toward better experiment tracking, explicit dataset lineage, benchmark comparison, and separation of research and production artifacts.

The current Run 4 release represents the culmination of that iterative process.

The project now has a coherent path from:

```text
Data
→ Training
→ Evaluation
→ Model
→ Deployment
```

with the experiment history preserved as the record of how each stage was reached.
