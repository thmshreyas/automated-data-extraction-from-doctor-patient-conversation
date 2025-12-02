#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run MedAlpaca on a doctor–patient conversation and extract structured JSON.

Requirements (install once):
  pip install "torch==2.2.2" "torchvision==0.17.2" "torchaudio==2.2.2" --index-url https://download.pytorch.org/whl/cu121
  pip install "transformers==4.44.0" "accelerate" "bitsandbytes" "peft" "sentencepiece" "numpy==1.26.4"
"""

import json
import os
import re
import sys
from typing import Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "medalpaca/medalpaca-7b"


def best_dtype() -> torch.dtype:
    # Use float16 on CUDA, else bfloat16 if available, otherwise float32
    if torch.cuda.is_available():
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def build_prompt(conversation: str) -> str:
    return f"""
Extract structured clinical information from the following doctor-patient conversation.

Return only JSON in this format:
{{
  "Chief Complaint": "",
  "Pain Details": "",
  "Duration": "",
  "Medications": "",
  "Medical History": "",
  "Family History": "",
  "Personal History": "",
  "Dental History": "",
  "Habits": ""
}}

Conversation:
{conversation}
""".strip()


def extract_json(text: str) -> Optional[str]:
    """
    Try to extract a JSON object from model output.
    - First, look for a fenced/indented JSON block.
    - Fallback: take the first balanced {...} block.
    """
    # Common fenced code block pattern
    fenced = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    # Greedy match from first '{' to last '}' and then try to JSON-parse
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
        # Sometimes models add trailing commas or weird unicode—try to clean lightly
        candidate = candidate.strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            # Try a more surgical approach: find the first top-level JSON object with nesting
            depth = 0
            start = None
            for i, ch in enumerate(text):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidate = text[start:i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except Exception:
                            start = None
            # Give up if none parse
            return None
    return None


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run MedAlpaca extraction on a conversation string or file.")
    ap.add_argument("--in", dest="in_file", required=False, help="Path to a text file containing the conversation. If omitted, a small example is used.")
    ap.add_argument("--out", dest="out_file", required=False, help="Optional path to write the extracted JSON result.")
    args = ap.parse_args()

    if args.in_file:
        in_path = Path(args.in_file)
        # If the provided path doesn't exist, try a common typo: 'segmentation_output' vs 'segementation_output'
        if not in_path.exists():
            # Suggest alternative folder name if it exists in the repo
            alt_parent_name = None
            if in_path.parent.name == "segmentation_output":
                alt_parent_name = "segementation_output"
            elif in_path.parent.name == "segementation_output":
                alt_parent_name = "segementation_output"

            if alt_parent_name:
                alt = in_path.parent.with_name(alt_parent_name) / in_path.name
                if alt.exists():
                    print(f"Input file not found at {in_path!s}.\nFound likely match at: {alt!s} — using that path.")
                    in_path = alt
                else:
                    print(f"Input file not found: {in_path!s}")
                    print(f"Current working directory: {Path.cwd()}")
                    print(f"Tip: check the directory name spelling. This repo uses 'segementation_output' (note the 'e').")
                    sys.exit(2)
            else:
                print(f"Input file not found: {in_path!s}")
                print(f"Current working directory: {Path.cwd()}")
                sys.exit(2)

        conversation = in_path.read_text(encoding="utf-8")
    else:
        conversation = """
[DOCTOR] Ma'am what is your chief complaint?
[DOCTOR] Why have you visited the clinic?
[PATIENT] I have a cavity problem where my tooth is itching for one week.
[DOCTOR] How is the pain ma'am?
[PATIENT] It hurts very badly, I can't even drink water.
[DOCTOR] From how many days?
[PATIENT] One week.
[DOCTOR] Have you taken any tablets?
[PATIENT] I have taken iMall tablet.
[DOCTOR] Does it go till your head?
[PATIENT] Yes.
[DOCTOR] Any medical history?
[DOCTOR] Any daily basis tablets?
[DOCTOR] Any allergies?
[DOCTOR] For dental history, have you visited dentist before?
[PATIENT] For root canal.
[DOCTOR] Family history — your mom or dad have BP or sugar?
[PATIENT] My father has sugar for past one year.
[DOCTOR] Personal history — marital status?
[PATIENT] Unmarried.
[DOCTOR] Diet?
[PATIENT] Mixed.
[DOCTOR] Sleep and menstruation?
[DOCTOR] It is not done.
[DOCTOR] Oral hygiene — how many times do you brush?
[PATIENT] One time in the morning.
[DOCTOR] Type of brush?
[PATIENT] Medium.
[DOCTOR] How do you brush?
[PATIENT] Horizontal.
[DOCTOR] How long do you brush?
[PATIENT] 15 minutes.
[DOCTOR] How often do you change brush?
[PATIENT] Once in three months.
[DOCTOR] Any smoking, alcohol or other habits?
[PATIENT] None.
""".strip()

    prompt = build_prompt(conversation)

    dtype = best_dtype()
    device_map = "auto"  # lets Accelerate place layers across available GPUs/CPU

    # Ask transformers to be slightly more verbose about downloads/loading (helps debugging)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")

    # Use modern API name `dtype` instead of deprecated `torch_dtype`.
    # Accelerate/Transformers accept `dtype` for newer releases.
    kwargs = {
        "dtype": dtype,
        "device_map": device_map,
    }

    # If you want to allow remote code for some models, set TRUST_REMOTE_CODE=1 env var.
    trust_remote = os.environ.get("TRUST_REMOTE_CODE", "0") == "1"
    if trust_remote:
        kwargs["trust_remote_code"] = True

    # Load model & tokenizer
    print("🔁 Loading tokenizer and model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)

    print("✅ MedAlpaca model loaded successfully!", file=sys.stderr, flush=True)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=700,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # The model may echo the prompt; strip it if present
    if full_text.startswith(prompt):
        generated = full_text[len(prompt):].lstrip()
    else:
        generated = full_text

    print(f"🧾 Generated text length: {len(generated)} chars", flush=True)
    if not generated.strip():
        print("⚠️ Warning: model generated empty text.", flush=True)

    # Try to extract and print pure JSON; fallback prints the raw generated text
   # === Try extracting JSON ===
    json_block = extract_json(generated)

    print("\n===============================")
    print("🩺 MODEL RAW OUTPUT (Preview):")
    print("===============================")
    print(generated[:1000])  # show first 1000 chars
    print("\n===============================")

    if json_block:
        print("✅ JSON block successfully extracted!\n")
        print(json_block)
        try:
            data = json.loads(json_block)
            print("\n=== ✅ Parsed JSON ===")
            print(json.dumps(data, indent=2))
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse failed: {e}")
    else:
        print("❌ No JSON detected in the model output. Showing raw output:\n")
        print(generated)

    # Always write output to a file. If user passed --out use it, otherwise write to outputs/<input_stem>.extracted.json
    out_path = None
    if args.out_file:
        out_path = Path(args.out_file)
    else:
        # derive output filename
        stem = "example"
        if args.in_file:
            try:
                stem = Path(args.in_file).stem
            except Exception:
                stem = "conversation"
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.extracted.json"

    # Prefer JSON block if present; otherwise write structured fallback JSON including raw output
    if json_block:
        to_write = json_block
    else:
        # create fallback JSON with keys and raw output
        fallback = {
            "Chief Complaint": "",
            "Pain Details": "",
            "Duration": "",
            "Medications": "",
            "Medical History": "",
            "Family History": "",
            "Personal History": "",
            "Dental History": "",
            "Habits": "",
            "raw_output": generated,
        }
        to_write = json.dumps(fallback, ensure_ascii=False, indent=2)

    out_path.write_text(to_write, encoding="utf-8")
    print(f"\n💾 Output written to: {out_path}", flush=True)

if __name__ == "__main__":
    main()
