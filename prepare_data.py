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
UR_LANG_TOKEN_ID  = 128095               # __ur__ token ID (prepended by tokenizer)
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

    WHY we don't use tokenizer.src_lang = "roman-ur":
    ──────────────────────────────────────────────────
    M2M100Tokenizer.src_lang setter calls set_src_lang_special_tokens(), which
    does a lookup in self.lang_code_to_token. The token __roman-ur__ (id 128105)
    was added as a raw custom token but was NOT registered in lang_code_to_token,
    so setting src_lang = "roman-ur" raises KeyError: 'roman-ur'.

    as_target_tokenizer() was also removed in transformers >= 5.x.

    CORRECT APPROACH for label sequences:
    ──────────────────────────────────────
    1. Tokenize label strings normally with src_lang = "ur" — the tokenizer
       prepends __ur__ (id 128095) as the first token.
    2. Replace that first token id with __roman-ur__ (id 128105).

    This is equivalent to what as_target_tokenizer() used to do: the only
    difference between source and target encoding in M2M100 is which language
    token sits at position 0. Everything else (SentencePiece subword splitting,
    EOS, padding) is identical.
    """
    def tokenize_batch(batch):
        # ── Tokenize source (Urdu script) ──────────────────────────────────
        # src_lang = "ur" is already set on the tokenizer; leave it there.
        model_inputs = tokenizer(
            batch["urdu"],
            max_length=max_src_len,
            truncation=True,
            padding="max_length",
        )

        # ── Tokenize target (Roman Urdu) ───────────────────────────────────
        # Tokenize with src_lang still set to "ur" so the tokenizer prepends
        # __ur__ (id 128095) at position 0.  We then overwrite position 0 with
        # __roman-ur__ (id 128105) on every non-padding sequence.
        #
        # We must NOT change tokenizer.src_lang here — "roman-ur" is not in
        # lang_code_to_token and would raise a KeyError.
        label_encodings = tokenizer(
            batch["roman_urdu"],
            max_length=max_tgt_len,
            truncation=True,
            padding="max_length",
        )

        label_ids = []
        for ids in label_encodings["input_ids"]:
            ids = list(ids)  # make a mutable copy

            # Replace the language prefix token at position 0.
            # The tokenizer always puts a lang token first; swap __ur__ → __roman-ur__.
            # Guard: only replace if position 0 is actually __ur__ (sanity check).
            if ids[0] == UR_LANG_TOKEN_ID:
                ids[0] = TGT_LANG_TOKEN_ID

            # Replace padding token ids with -100 so CrossEntropyLoss ignores them.
            ids = [tok if tok != tokenizer.pad_token_id else -100 for tok in ids]

            label_ids.append(ids)

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
    tokenizer.src_lang = SRC_LANG  # set once; never changed during tokenization

    # Confirm the custom tokens are present
    ur_id     = tokenizer.convert_tokens_to_ids("__ur__")
    roman_id  = tokenizer.convert_tokens_to_ids("__roman-ur__")
    assert ur_id    == UR_LANG_TOKEN_ID,  f"__ur__ id mismatch: got {ur_id}"
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

    # Verify the first non-masked label token is __roman-ur__ (128105)
    first_real = next(t for t in sample["labels"] if t != -100)
    assert first_real == TGT_LANG_TOKEN_ID, (
        f"Label sequence should start with __roman-ur__ (128105), got {first_real}"
    )
    print(f"  first label token: {first_real} (__roman-ur__ ✓)")


if __name__ == "__main__":
    main()