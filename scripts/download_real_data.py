#!/usr/bin/env python3
"""
Download Real Datasets for ConformalDrift Experiments

Downloads from HuggingFace:
- TruthfulQA (smaller, faster than NQ)
- HaluEval QA samples
- SQuAD for cross-dataset evaluation

Usage:
    python scripts/download_real_data.py --n-samples 200
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

def install_datasets():
    """Install datasets library if needed."""
    try:
        import datasets
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets --quiet")

def download_truthfulqa(n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """
    Download TruthfulQA - high quality labeled hallucination data.
    """
    from datasets import load_dataset

    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")

    random.seed(seed)
    items = list(dataset)
    random.shuffle(items)

    examples = []
    for item in items[:n_samples]:
        question = item.get('question', '')
        best_answer = item.get('best_answer', '')
        incorrect_answers = item.get('incorrect_answers', [])

        if not question or not best_answer:
            continue

        # Create context from the question domain
        category = item.get('category', 'General')
        context = f"Topic: {category}. Question context for factual verification."

        # Faithful example
        examples.append({
            "query": question,
            "documents": [context],
            "response": best_answer,
            "label": 0
        })

        # Hallucinated example (use incorrect answer)
        if incorrect_answers:
            wrong_answer = incorrect_answers[0] if isinstance(incorrect_answers, list) else str(incorrect_answers)
            examples.append({
                "query": question,
                "documents": [context],
                "response": wrong_answer,
                "label": 1
            })

    print(f"  Loaded {len(examples)} TruthfulQA examples")
    return examples


def download_halueval(n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """
    Download HaluEval QA samples.
    """
    from datasets import load_dataset

    print("Loading HaluEval dataset...")
    try:
        dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    except Exception as e:
        print(f"  Error loading HaluEval: {e}")
        print("  Trying alternative source...")
        try:
            dataset = load_dataset("RUC-NLPIR/HaluEval", split="test")
        except:
            print("  Could not load HaluEval, skipping...")
            return []

    random.seed(seed)
    items = list(dataset)
    random.shuffle(items)

    examples = []
    for item in items[:n_samples]:
        question = item.get('question', item.get('query', ''))
        knowledge = item.get('knowledge', item.get('context', ''))
        answer = item.get('answer', item.get('response', ''))
        hallucination = item.get('hallucination', 'no')

        if not question or not answer:
            continue

        label = 1 if hallucination.lower() in ['yes', 'true', '1'] else 0

        examples.append({
            "query": question,
            "documents": [knowledge] if knowledge else ["No context provided."],
            "response": answer,
            "label": label
        })

    print(f"  Loaded {len(examples)} HaluEval examples")
    return examples


def download_squad(n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """
    Download SQuAD for cross-dataset evaluation.
    Creates hallucinated examples by swapping answers.
    """
    from datasets import load_dataset

    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="validation")

    random.seed(seed)
    items = list(dataset)
    random.shuffle(items)

    examples = []
    for i, item in enumerate(items[:n_samples]):
        question = item.get('question', '')
        context = item.get('context', '')
        answers = item.get('answers', {})

        if not question or not context:
            continue

        answer_text = answers.get('text', [''])[0] if answers.get('text') else ''
        if not answer_text:
            continue

        # Faithful example
        examples.append({
            "query": question,
            "documents": [context],
            "response": f"Based on the passage, {answer_text}.",
            "label": 0
        })

        # Hallucinated example (use answer from different item)
        other_idx = (i + 37) % len(items)
        other_answers = items[other_idx].get('answers', {})
        wrong_answer = other_answers.get('text', ['incorrect'])[0] if other_answers.get('text') else 'unknown'

        examples.append({
            "query": question,
            "documents": [context],
            "response": f"The answer is {wrong_answer}.",
            "label": 1
        })

    print(f"  Loaded {len(examples)} SQuAD examples")
    return examples


def download_fever(n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """
    Download FEVER fact verification dataset.
    """
    from datasets import load_dataset

    print("Loading FEVER dataset...")
    try:
        dataset = load_dataset("fever", "v1.0", split="labelled_dev")
    except:
        try:
            dataset = load_dataset("fever", split="train[:2000]")
        except Exception as e:
            print(f"  Could not load FEVER: {e}")
            return []

    random.seed(seed)
    items = list(dataset)
    random.shuffle(items)

    examples = []
    for item in items[:n_samples]:
        claim = item.get('claim', '')
        evidence = item.get('evidence', item.get('evidence_sentence', ''))
        label_str = item.get('label', '')

        if not claim:
            continue

        # FEVER labels: SUPPORTS=0, REFUTES=1, NOT ENOUGH INFO=skip
        if label_str == 'SUPPORTS':
            label = 0
            response = f"This claim is supported: {claim}"
        elif label_str == 'REFUTES':
            label = 1
            response = f"This claim is true: {claim}"  # Hallucinating that refuted claim is true
        else:
            continue

        examples.append({
            "query": f"Is this claim true? {claim}",
            "documents": [evidence] if evidence else [claim],
            "response": response,
            "label": label
        })

    print(f"  Loaded {len(examples)} FEVER examples")
    return examples


def save_dataset(examples: List[Dict], output_path: Path):
    """Save dataset to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)

    n_faithful = sum(1 for e in examples if e['label'] == 0)
    n_halluc = sum(1 for e in examples if e['label'] == 1)
    print(f"  Saved {len(examples)} examples ({n_faithful} faithful, {n_halluc} hallucinated) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download real datasets")
    parser.add_argument("--n-samples", type=int, default=200, help="Samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    install_datasets()

    print("=" * 60)
    print("Downloading Real Datasets for ConformalDrift")
    print("=" * 60)

    # TruthfulQA for calibration (high-quality hallucination labels)
    print("\n--- TruthfulQA (Calibration) ---")
    truthfulqa = download_truthfulqa(args.n_samples, args.seed)
    if truthfulqa:
        n_cal = len(truthfulqa) // 2
        save_dataset(truthfulqa[:n_cal], output_dir / "nq_calibration.json")
        save_dataset(truthfulqa[n_cal:], output_dir / "nq_test.json")

    # SQuAD for cross-dataset shift
    print("\n--- SQuAD (Cross-Dataset Test) ---")
    squad = download_squad(args.n_samples, args.seed)
    if squad:
        save_dataset(squad, output_dir / "ragtruth_test.json")

    # HaluEval for RLHF shift
    print("\n--- HaluEval (RLHF Shift) ---")
    halueval = download_halueval(args.n_samples, args.seed)
    if halueval:
        save_dataset(halueval, output_dir / "halueval_test.json")

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
