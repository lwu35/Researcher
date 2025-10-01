#!/usr/bin/env python3
"""
Quick Start Script for CycleResearcher
This script demonstrates how to generate a research paper using CycleResearcher
"""

import os
from pathlib import Path

print("🚀 Starting CycleResearcher Quick Start Demo...")
print("=" * 60)

# Import necessary libraries
from ai_researcher import CycleResearcher
import json

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
    
    # Show a sample of the references
    print(f"   ✅ Loaded {len(references_content)} characters of reference data")
    print(f"   First 200 characters: {references_content[:200]}...")
except FileNotFoundError:
    print(f"   ❌ Error: Could not find {bib_file}")
    print(f"   Please make sure you're running this from the Researcher directory")
    exit(1)

print("\n🤖 Step 2: Initializing CycleResearcher (12B model)...")
print("   Note: This will download the model on first run (~25GB)")
print("   This may take 10-30 minutes depending on your internet speed...")

try:
    # Initialize CycleResearcher with the default 12B model
    researcher = CycleResearcher(
        model_size="12B",
        gpu_memory_utilization=0.9,  # Lower to reduce memory usage
        max_model_len=25000           # Lower to reduce memory requirements
    )
    print("   ✅ Model loaded successfully!")
    
    print("\n✍️  Step 3: Generating research paper...")
    print("   This may take 5-10 minutes...")
    print("   The model is analyzing references and generating content...")
    
    # Generate a paper
    generated_papers = researcher.generate_paper(
        topic="AI Researcher",
        references=references_content,
        n=1  # Generate a single paper
    )
    
    print("\n✅ Paper generation complete!")
    print("=" * 60)
    
    # Display results
    paper = generated_papers[0]
    
    print("\n📄 GENERATED PAPER SUMMARY:")
    print("-" * 60)
    print(f"Title: {paper.get('title', 'N/A')[:100]}")
    print(f"\nAbstract: {paper.get('abstract', 'N/A')[:300]}...")
    print(f"\nMotivation: {paper.get('motivation', 'N/A')[:200]}...")
    
    # Save the full paper to a file in the current directory
    output_file = script_dir / 'generated_paper_output.json'
    with open(output_file, 'w') as f:
        json.dump(paper, f, indent=2)
    
    print(f"\n💾 Full paper saved to: {output_file}")
    print("\n🎉 Success! You've generated your first research paper!")
    print("\nNext steps:")
    print("  1. Open generated_paper_output.json to see the full output")
    print("  2. Check out Tutorial/tutorial_1.ipynb for more detailed examples")
    print("  3. Try Tutorial/tutorial_2.ipynb to learn about CycleReviewer")
    
except Exception as e:
    print(f"\n❌ Error occurred: {str(e)}")
    print("\nTroubleshooting tips:")
    print("  1. Make sure you have enough disk space (~30GB for models)")
    print("  2. Check your internet connection for model download")
    print("  3. Try running with a smaller model or using API-based inference")
    print(f"\nFull error details: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc() 