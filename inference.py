"""
inference.py
------------
Loads the fine-tuned M2M100 model + LoRA adapters and runs transliteration
on a list of Urdu script sentences, printing the Roman Urdu output.

Usage examples
--------------
# Run on the default demo sentences:
python inference.py

# Point at a specific model directory:
python inference.py --model_dir ./fine_tuned_model

# Pass custom sentences via CLI:
python inference.py --sentences "ڈاکٹر نے کہا" "کمپیوٹر بند ہے" "ورزش کرو"

# Run on a plain-text file (one Urdu sentence per line):
python inference.py --input_file urdu_sentences.txt

python inference.py --model_dir ./fine_tuned_model --input_file hard_test_sentences.txt
"""

import argparse
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Model
from peft import PeftModel

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL_ID     = "Mavkif/m2m100_rup_ur_to_rur"
TOKENIZER_ID      = "Mavkif/m2m100_rup_tokenizer_both"
TGT_LANG_TOKEN_ID = 128105   # __roman-ur__
SRC_LANG          = "ur"

# Demo sentences that exercise the domain-specific words the base model
# struggled with before fine-tuning
DEFAULT_SENTENCES = [
    "میں نے اسے میسج کیا",
    "وہ روز ایکسرسائز کرتا ہے",
    "کل ایگزام ہے میرا",
    "وہ میتھ میں فیل ہو گیا",
    "اس نے پورا پروجیکٹ ڈیلیٹ کر دیا",
    "بیک اپ لینا ضروری ہے",
    "یہ فائل کرپٹ ہو گئی",
    "مجھے کل انٹرویو دینا ہے",
    "اس کا لیپ ٹاپ کریش ہو گیا",
    "میں نے سسٹم ری اسٹارٹ کیا",
    "وہ آن لائن کلاس میں تھا",
    "ایپ اپڈیٹ کر لو",
    "سرور ڈاؤن ہے ابھی",
    "پاس ورڈ چینج کرو اپنا",
    "اسکرین شاٹ لے لو اس کا",
    "وہ گھر پر نہیں ہے",
    "میں کل آؤں گا",
    "اس نے مجھے بلایا تھا",
    "کھانا تیار ہو گیا",
    "بارش ہو رہی ہے باہر",
    "مجھے نیند آ رہی ہے",
    "وہ بہت تھک گیا ہے",
    "یہ کام کل تک ہو جائے گا",
    "اس نے سچ نہیں بولا",
    "میں صبح اٹھ کر چلا گیا"
]

# ── Patch (bug #3) ────────────────────────────────────────────────────────────
# M2M100Model.forward() in transformers 5.x computes decoder_inputs_embeds
# from decoder_input_ids, then passes BOTH to M2M100Decoder → ValueError.
# Fix: drop decoder_input_ids when decoder_inputs_embeds is already present.
# Required for .generate() to work correctly.

class PatchedM2M100Model(M2M100Model):
    def forward(self, *args, **kwargs):
        if kwargs.get("decoder_inputs_embeds") is not None:
            kwargs.pop("decoder_input_ids", None)
        return super().forward(*args, **kwargs)


def patch_model(model: M2M100ForConditionalGeneration) -> M2M100ForConditionalGeneration:
    """Swaps model.model in-place with PatchedM2M100Model."""
    original = model.model
    patched  = PatchedM2M100Model(model.config)
    patched.load_state_dict(original.state_dict())
    patched.to(next(original.parameters()).dtype)
    patched.to(next(original.parameters()).device)
    del model.model        # free original before assigning to avoid double VRAM
    model.model = patched
    torch.cuda.empty_cache()
    return model

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Urdu → Roman Urdu transliteration inference")
    p.add_argument(
        "--model_dir",
        default="./fine_tuned_model",
        help="Directory containing the saved LoRA adapter weights + tokenizer"
    )
    p.add_argument(
        "--sentences",
        nargs="+",
        default=None,
        help="One or more Urdu sentences to transliterate (overrides default demo list)"
    )
    p.add_argument(
        "--input_file",
        default=None,
        help="Path to a text file with one Urdu sentence per line"
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of sentences to process per forward pass"
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per sentence"
    )
    p.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam search width. Higher = better quality, slower."
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device: 'cuda', 'cpu'. Defaults to cuda if available."
    )
    return p.parse_args()

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_dir: str, device: torch.device):
    """
    Loading order:
      1. Load frozen base model (M2M100ForConditionalGeneration) in fp16
      2. Apply PatchedM2M100Model (bug #3 — fixes decoder conflict in .generate())
      3. Wrap with PeftModel to attach LoRA adapter weights
      4. merge_and_unload() — bakes adapters into base weights, removes PEFT wrapper
      5. Load tokenizer from saved model dir (falls back to Hub if not found)

    After merge_and_unload() the model is a plain M2M100ForConditionalGeneration
    with no adapter overhead — fastest possible inference.
    """
    print(f"[inference] Loading base model '{BASE_MODEL_ID}' ...")
    base_model = M2M100ForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
    )

    # Must set these before patching and before .generate() is ever called.
    # transformers 5.x rejects forced_bos_token_id on model.config directly.
    base_model.config.forced_bos_token_id = None
    base_model.generation_config.forced_bos_token_id = TGT_LANG_TOKEN_ID

    print("[inference] Applying PatchedM2M100Model ...")
    base_model = patch_model(base_model)

    print(f"[inference] Loading LoRA adapters from '{model_dir}' ...")
    model = PeftModel.from_pretrained(
        base_model,
        model_dir,
        torch_dtype=torch.float16,
    )

    # Merge adapter weights into the base model for faster inference.
    # After merging, the model behaves like a standard (non-PEFT) model —
    # no adapter overhead per forward pass.
    print("[inference] Merging LoRA adapters into base weights ...")
    model = model.merge_and_unload()

    model.to(device)
    model.eval()

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"[inference] Loading tokenizer from '{model_dir}' ...")
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(model_dir)
    except Exception:
        print(f"  [warning] Tokenizer not found in '{model_dir}', "
              f"falling back to '{TOKENIZER_ID}' from Hub.")
        tokenizer = M2M100Tokenizer.from_pretrained(TOKENIZER_ID)

    tokenizer.src_lang = SRC_LANG

    return model, tokenizer

# ── Inference ─────────────────────────────────────────────────────────────────

def transliterate_batch(
    sentences: list[str],
    model: M2M100ForConditionalGeneration,
    tokenizer: M2M100Tokenizer,
    device: torch.device,
    max_new_tokens: int,
    num_beams: int,
) -> list[str]:
    """
    Tokenizes a batch of Urdu sentences and generates Roman Urdu output.

    Critical settings:
      • tokenizer.src_lang = "ur"          → prepends __ur__ to inputs
      • forced_bos_token_id = 128105       → forces __roman-ur__ as the first
                                             decoder token, steering generation
                                             to Roman Urdu output
    """
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=TGT_LANG_TOKEN_ID,  # __roman-ur__ (id 128105)
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [s.strip() for s in decoded]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Using device: {device}")

    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        print(f"[inference] Loaded {len(sentences)} sentences from '{args.input_file}'.")
    elif args.sentences:
        sentences = args.sentences
    else:
        sentences = DEFAULT_SENTENCES
        print("[inference] Using default demo sentences.")

    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)

    print(f"\n[inference] Transliterating {len(sentences)} sentence(s) "
          f"(batch_size={args.batch_size}, num_beams={args.num_beams}) ...\n")
    print("─" * 70)

    all_outputs = []
    for i in range(0, len(sentences), args.batch_size):
        batch = sentences[i : i + args.batch_size]
        outputs = transliterate_batch(
            batch, model, tokenizer, device,
            args.max_new_tokens, args.num_beams
        )
        all_outputs.extend(outputs)

    for urdu, roman in zip(sentences, all_outputs):
        print(f"Urdu  : {urdu}")
        print(f"Roman : {roman}")
        print("─" * 70)

    print(f"\n[inference] Done. {len(all_outputs)} sentence(s) transliterated.")


if __name__ == "__main__":
    main()