#!/usr/bin/env python3
"""
Quick Start Script for CycleResearcher (CPU-friendly version)
Uses HuggingFace Transformers instead of vLLM for better CPU compatibility
"""

import os
from pathlib import Path
import torch
import json

print("🚀 Starting CycleResearcher Quick Start Demo (CPU Mode)...")
print("=" * 60)

# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from ai_researcher.utils import get_paper_from_generated_text

# Get the script directory and set up paths
script_dir = Path(__file__).parent
tutorial_dir = script_dir / "Tutorial"
bib_file = tutorial_dir / "cycleresearcher_references.bib"

print(f"\n📁 Working directory: {script_dir}")
print(f"📁 Tutorial directory: {tutorial_dir}")

print("\n📚 Step 1: Loading reference papers...")
# Load the BibTeX references
try:
    with open(bib_file, 'r') as f:
        references_content = f.read()
    
    print(f"   ✅ Loaded {len(references_content)} characters of reference data")
    print(f"   First 200 characters: {references_content[:200]}...")
except FileNotFoundError:
    print(f"   ❌ Error: Could not find {bib_file}")
    exit(1)

print("\n🤖 Step 2: Loading model (12B) with HuggingFace Transformers...")
print("   Note: This will download ~25GB on first run")
print("   This may take 10-30 minutes...")
print("   ⚠️  This is a large model - generation may be slow on CPU")

try:
    model_name = 'WestlakeNLP/CycleResearcher-ML-12B'
    
    print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("   Loading model (this is the slow part)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True,
    )
    
    print("   ✅ Model loaded successfully!")
    
    print("\n✍️  Step 3: Preparing prompt...")
    
    # Create the message in the format the model expects
    messages = [
        {
            "role": "system",
            "content": "You are a research assistant AI tasked with generating a scientific paper based on provided literature."
        },
        {
            "role": "user",
            "content": references_content
        }
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"   Prompt length: {len(prompt)} characters")
    
    print("\n✍️  Step 4: Generating paper...")
    print("   ⏱️  This may take 10-30 minutes on CPU...")
    print("   The model is analyzing references and generating content...")
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192  # Limit input to avoid memory issues
    )
    
    # Move to device
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
    
    print(f"   Input tokens: {inputs['input_ids'].shape[1]}")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,  # Reduced for CPU
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    print("\n✅ Generation complete!")
    print("=" * 60)
    
    # Process the output
    print("\n📝 Processing generated paper...")
    paper = get_paper_from_generated_text(generated_text)
    
    if paper is None:
        print("⚠️  Could not parse paper structure from generated text")
        print("Saving raw output instead...")
        paper = {"generated_text": generated_text}
    
    # Display results
    print("\n📄 GENERATED PAPER SUMMARY:")
    print("-" * 60)
    
    if "title" in paper:
        print(f"Title: {paper.get('title', 'N/A')[:100]}")
    if "abstract" in paper:
        print(f"\nAbstract: {paper.get('abstract', 'N/A')[:300]}...")
    if "motivation" in paper:
        print(f"\nMotivation: {paper.get('motivation', 'N/A')[:200]}...")
    
    # Save the full paper
    output_file = script_dir / 'generated_paper_output.json'
    with open(output_file, 'w') as f:
        json.dump(paper, f, indent=2)
    
    print(f"\n💾 Full paper saved to: {output_file}")
    print("\n🎉 Success! You've generated your first research paper!")
    print("\nNext steps:")
    print("  1. Open generated_paper_output.json to see the full output")
    print("  2. Check out Tutorial/tutorial_1_huggingface.ipynb for more examples")
    print("  3. Try Tutorial/tutorial_2.ipynb to learn about CycleReviewer")
    
except Exception as e:
    print(f"\n❌ Error occurred: {str(e)}")
    print("\nTroubleshooting tips:")
    print("  1. Make sure you have enough RAM (~32GB recommended for 12B model)")
    print("  2. Try closing other applications to free up memory")
    print("  3. If out of memory, you may need to use a smaller model or cloud GPU")
    print(f"\nFull error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc() 