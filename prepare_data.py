"""
prepare_data.py
---------------
Loads the transliteration CSV, tokenizes using the custom M2M100 tokenizer
(Mavkif/m2m100_rup_tokenizer_both), splits 95/5 train/val, and saves the
processed dataset to disk in HuggingFace Arrow format.

Direction: urdu (Urdu script) → roman_urdu (Roman Urdu / Latin script)
"""

import os
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import M2M100Tokenizer

# ── Constants ────────────────────────────────────────────────────────────────

TOKENIZER_ID      = "Mavkif/m2m100_rup_tokenizer_both"
SRC_LANG          = "ur"                  # Urdu script is the input side
TGT_LANG_TOKEN_ID = 128105               # __roman-ur__ token ID (forced BOS)
MAX_SRC_LEN       = 128                  # max tokens for Urdu input
MAX_TGT_LEN       = 128                  # max tokens for Roman Urdu output
TRAIN_SPLIT       = 0.95                 # 95 % train, 5 % validation
RANDOM_SEED       = 42

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prepare transliteration dataset")
    p.add_argument(
        "--csv",
        default="data/final_transliteration_dataset_fixed_v2.csv",
        help="Path to the input CSV file"
    )
    p.add_argument(
        "--output_dir",
        default="./processed_dataset",
        help="Directory to save the HuggingFace DatasetDict"
    )
    return p.parse_args()

# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads the CSV and keeps only the two columns we need.
    The 'aerab_changes' annotation column is dropped silently.
    """
    df = pd.read_csv(csv_path)

    # Validate required columns exist
    required = {"urdu", "roman_urdu"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Keep only the two task columns
    df = df[["urdu", "roman_urdu"]].copy()

    # Drop rows where either column is empty / NaN
    before = len(df)
    df.dropna(subset=["urdu", "roman_urdu"], inplace=True)
    df = df[df["urdu"].str.strip().ne("") & df["roman_urdu"].str.strip().ne("")]
    after = len(df)
    if before != after:
        print(f"[prepare_data] Dropped {before - after} rows with empty values.")

    print(f"[prepare_data] Loaded {after} rows from '{csv_path}'.")
    return df.reset_index(drop=True)

# ── Tokenisation ──────────────────────────────────────────────────────────────

def build_tokenize_fn(tokenizer, max_src_len: int, max_tgt_len: int):
    """
    Returns a batch-map function that tokenizes (urdu → roman_urdu) pairs.

    Key details for M2M100:
      • tokenizer.src_lang = "ur"         prepends __ur__ to source inputs
      • tokenizer.src_lang = "roman-ur"   prepends __roman-ur__ to label inputs
        (as_target_tokenizer() was removed in transformers>=5.0; the correct
         replacement is to temporarily switch src_lang to the target language
         code, then restore it — M2M100Tokenizer uses src_lang to pick the
         language prefix token for both encoding directions)
    """
    def tokenize_batch(batch):
        # ── Tokenize source (Urdu script) ──────────────────────────────────
        tokenizer.src_lang = SRC_LANG  # "ur" → prepends __ur__ token
        model_inputs = tokenizer(
            batch["urdu"],
            max_length=max_src_len,
            truncation=True,
            padding="max_length",
        )

        # ── Tokenize target (Roman Urdu) ───────────────────────────────────
        # Temporarily switch src_lang to "roman-ur" so the tokenizer prepends
        # __roman-ur__ (id=128105) as the language prefix on the label sequence.
        # This is the transformers>=5.x replacement for as_target_tokenizer().
        tokenizer.src_lang = "roman-ur"
        labels = tokenizer(
            batch["roman_urdu"],
            max_length=max_tgt_len,
            truncation=True,
            padding="max_length",
        )
        # Restore src_lang for subsequent batches
        tokenizer.src_lang = SRC_LANG

        # Replace padding token ids in labels with -100 so CrossEntropyLoss
        # ignores them — standard seq2seq practice
        label_ids = labels["input_ids"]
        label_ids = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in ids]
            for ids in label_ids
        ]

        model_inputs["labels"] = label_ids
        return model_inputs

    return tokenize_batch

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. Load and clean the CSV
    df = load_csv(args.csv)

    # 2. Convert to HuggingFace Dataset and shuffle before splitting
    full_dataset = Dataset.from_pandas(df).shuffle(seed=RANDOM_SEED)

    # 3. 95 / 5 split
    split = full_dataset.train_test_split(
        test_size=(1.0 - TRAIN_SPLIT),
        seed=RANDOM_SEED,
    )
    dataset_dict = DatasetDict({
        "train":      split["train"],
        "validation": split["test"],
    })
    print(
        f"[prepare_data] Split → train: {len(dataset_dict['train'])} | "
        f"validation: {len(dataset_dict['validation'])}"
    )

    # 4. Load the custom tokenizer
    #    IMPORTANT: always use Mavkif/m2m100_rup_tokenizer_both, never the base
    #    facebook/m2m100_418M tokenizer — it lacks __ur__ and __roman-ur__ tokens
    print(f"[prepare_data] Loading tokenizer from '{TOKENIZER_ID}' ...")
    tokenizer = M2M100Tokenizer.from_pretrained(TOKENIZER_ID)

    # Confirm the custom tokens are present
    ur_id     = tokenizer.convert_tokens_to_ids("__ur__")
    roman_id  = tokenizer.convert_tokens_to_ids("__roman-ur__")
    assert ur_id    == 128095, f"__ur__ id mismatch: got {ur_id}"
    assert roman_id == TGT_LANG_TOKEN_ID, f"__roman-ur__ id mismatch: got {roman_id}"
    print(f"[prepare_data] Token IDs confirmed → __ur__: {ur_id} | __roman-ur__: {roman_id}")

    # 5. Tokenise both splits
    tokenize_fn = build_tokenize_fn(tokenizer, MAX_SRC_LEN, MAX_TGT_LEN)

    columns_to_remove = ["urdu", "roman_urdu"]

    tokenized = dataset_dict.map(
        tokenize_fn,
        batched=True,
        batch_size=64,
        remove_columns=columns_to_remove,
        desc="Tokenizing",
    )

    # 6. Save to disk
    os.makedirs(args.output_dir, exist_ok=True)
    tokenized.save_to_disk(args.output_dir)
    print(f"[prepare_data] Dataset saved to '{args.output_dir}'.")

    # 7. Sanity check — print one example
    sample = tokenized["train"][0]
    print("\n[prepare_data] Sample token counts:")
    print(f"  input_ids length : {len(sample['input_ids'])}")
    print(f"  labels length    : {len(sample['labels'])}")
    non_pad_labels = [t for t in sample["labels"] if t != -100]
    print(f"  non-masked label tokens: {len(non_pad_labels)}")


if __name__ == "__main__":
    main()