# Dataset History

## Purpose

This document records the exact dataset lineage for the Urdu → Roman Urdu transliteration fine-tuning experiments.

The canonical current dataset is:

```text
data/rur_to_ur_data.csv
```

It contains:

- **4,860 actual data rows**
- **4,861 CSV lines including the header**
- Columns: `urdu`, `roman_urdu`

The rows were appended in chronological order. There is no overlap between the historical dataset portions. Therefore, the historical datasets can be reconstructed from contiguous row ranges in the current CSV.

---

## Final dataset lineage

```text
Run 1
└── 2,012 rows
    │
    ├── + 488 subsequently appended rows
    │
    ▼
Run 2 dataset
└── 2,500 rows
    │
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     │
Run 3 initial dataset                     │
└── 563 rows                              │
    │                                     │
    ├── Model trained on these 563 rows   │
    │                                     │
    └── Poor benchmark result             │
                                          │
    + 1,797 additional rows               │
                                          │
    ▼                                     │
Expanded Run 3 dataset                    │
└── 2,360 rows                            │
    │                                     │
    └─────────────────────────────────────┘
                                          │
                                          ▼
Run 4 dataset
└── Run 2 dataset (2,500)
    + Expanded Run 3 dataset (2,360)
    = 4,860 rows
```

---

## Run 1

**Rows:** 2,012

This is the original dataset used for the first fine-tuning run.

Current CSV range:

```text
1–2,012
```

With the current `prepare_data.py`:

|   Raw | Train | Validation |
| ----: | ----: | ---------: |
| 2,012 | 1,911 |        101 |

Strategy:

```text
Fresh training from base model
```

Benchmark BLEU:

```text
54.52
```

---

## Run 2

Run 2 expanded the Run 1 dataset with newly generated data.

The final Run 2 dataset contains:

```text
2,500 actual rows
```

Current CSV range:

```text
1–2,500
```

The rows appended after the original Run 1 dataset are therefore:

```text
2,012 + 488 = 2,500
```

So the current canonical CSV contains **488 rows in the Run 2 expansion segment**.

> Historical notes previously described this expansion as 489 generated rows. The current canonical CSV, which is the source of truth for this dataset history, contains 2,500 rows total after the Run 1 expansion. Therefore, the actual appended segment present in the canonical file is 488 rows. This document uses the canonical CSV boundaries rather than the older remembered count.

With the current `prepare_data.py`:

|   Raw | Train | Validation |
| ----: | ----: | ---------: |
| 2,500 | 2,375 |        125 |

Strategy:

```text
Fresh training from base model
```

Benchmark BLEU:

```text
70.40
```

---

## Run 3 — Initial Dataset

Run 3 was a targeted corrective experiment.

The initial Run 3 dataset contains:

```text
563 rows
```

Because the canonical CSV is append-ordered, these are:

```text
Rows 2,501–3,063
```

This is the dataset that was **actually used to train Run 3**.

With the current `prepare_data.py`:

| Raw | Train | Validation |
| --: | ----: | ---------: |
| 563 |   534 |         29 |

Strategy:

```text
Continue training from Run 2 adapter
```

Benchmark BLEU:

```text
47.97
```

This was a major regression from Run 2:

```text
Run 2: 70.40 BLEU
Run 3: 47.97 BLEU
```

The failure demonstrated that continuing from a good adapter using only a small targeted dataset can cause severe loss of previously learned general capability.

---

## Run 3 — Expanded Dataset

After seeing the poor Run 3 results, additional data was generated.

The original 563 Run 3 rows were combined with the newly generated data.

The resulting expanded Run 3 dataset contains:

```text
4,860 - 2,500 = 2,360 rows
```

Current CSV range:

```text
Rows 2,501–4,860
```

Composition:

```text
563 original Run 3 rows
+ 1,797 additional rows
= 2,360 expanded Run 3 rows
```

**Important:** This expanded Run 3 dataset was **not used to retrain Run 3**.

It was instead used as part of the final Run 4 dataset.

---

## Run 4

Run 4 combined:

```text
Run 2 dataset
    2,500 rows

+

Expanded Run 3 dataset
    2,360 rows

=

Run 4 dataset
    4,860 rows
```

With the current `prepare_data.py`:

|   Raw | Train | Validation |
| ----: | ----: | ---------: |
| 4,860 | 4,617 |        243 |

Strategy:

```text
Fresh training from base model
```

Benchmark BLEU:

```text
72.14
```

This became the current best adapter.

---

## Exact Run Summary

| Run            | Dataset state                   | Raw rows | Train | Validation | Strategy                        |            BLEU |
| -------------- | ------------------------------- | -------: | ----: | ---------: | ------------------------------- | --------------: |
| Run 1          | Original dataset                |    2,012 | 1,911 |        101 | Fresh from base                 |           54.52 |
| Run 2          | Run 1 + appended expansion      |    2,500 | 2,375 |        125 | Fresh from base                 |           70.40 |
| Run 3          | Initial targeted dataset        |      563 |   534 |         29 | Continue from Run 2             |           47.97 |
| Run 3 Expanded | Initial Run 3 + additional data |    2,360 |    — |         — | Not trained as a standalone run |              — |
| Run 4          | Run 2 + Run 3 Expanded          |    4,860 | 4,617 |        243 | Fresh from base                 | **72.14** |

---

## Canonical CSV Row Ranges

Using 1-based row numbering for **data rows only**:

| Dataset segment            | Start |   End | Count |
| -------------------------- | ----: | ----: | ----: |
| Run 1                      |     1 | 2,012 | 2,012 |
| Run 2 appended expansion   | 2,013 | 2,500 |   488 |
| Initial Run 3              | 2,501 | 3,063 |   563 |
| Additional post-Run-3 data | 3,064 | 4,860 | 1,797 |
| Expanded Run 3             | 2,501 | 4,860 | 2,360 |
| Final Run 4                |     1 | 4,860 | 4,860 |

The CSV header is above these data rows and is not counted in the ranges.

---

## Exact Split Logic

`prepare_data.py` uses:

```python
TRAIN_SPLIT = 0.95
RANDOM_SEED = 42

full_dataset = Dataset.from_pandas(df).shuffle(seed=RANDOM_SEED)

split = full_dataset.train_test_split(
    test_size=(1.0 - TRAIN_SPLIT),
    seed=RANDOM_SEED,
)
```

Therefore, the historical raw dataset sizes map to:

```text
Run 1
2,012 raw → 1,911 train + 101 validation

Run 2
2,500 raw → 2,375 train + 125 validation

Run 3
563 raw → 534 train + 29 validation

Run 4
4,860 raw → 4,617 train + 243 validation
```

The expanded Run 3 dataset was not trained as its own standalone run, so no historical train/validation split is recorded for it.

---

## Important Historical Distinction

There are three concepts that should not be conflated:

1. **Run 3 initial dataset — 563 rows**

   - Actually used for Run 3 training.
   - Continued from the Run 2 adapter.
   - Produced the 47.97 BLEU regression.
2. **Expanded Run 3 dataset — 2,360 rows**

   - Created after the failed Run 3 result.
   - Contains the original 563 Run 3 rows plus 1,797 additional rows.
   - Was **not** used to retrain Run 3.
3. **Run 4 dataset — 4,860 rows**

   - Run 2's 2,500-row dataset plus the 2,360-row expanded Run 3 dataset.
   - Trained fresh from the base model.
   - Produced the current best BLEU of 72.14.

This distinction is the authoritative dataset history for the project.
