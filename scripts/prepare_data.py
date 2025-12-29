#!/usr/bin/env python3
"""
Data Preparation Script for ConformalDrift Experiments

Downloads and formats datasets for hallucination detection experiments:
- Natural Questions (NQ) - creates synthetic hallucinations
- RAGTruth - pre-labeled hallucination dataset
- HaluEval - RLHF-style hallucinations

Usage:
    python scripts/prepare_data.py --dataset nq --output data/nq_calibration.json
    python scripts/prepare_data.py --dataset ragtruth --output data/ragtruth_test.json
    python scripts/prepare_data.py --dataset all
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RAGExample:
    query: str
    documents: List[str]
    response: str
    label: int  # 0=faithful, 1=hallucinated


def download_nq_sample(n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """
    Download Natural Questions sample from HuggingFace.
    Creates synthetic hallucinations by swapping answers.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets --quiet")
        from datasets import load_dataset

    print(f"Loading Natural Questions dataset (sampling {n_samples} examples)...")

    # Load NQ dataset
    dataset = load_dataset("google-research-datasets/natural_questions",
                          "default",
                          split=f"train[:{n_samples*2}]",
                          trust_remote_code=True)

    random.seed(seed)
    examples = []

    # Process examples
    items = list(dataset)
    random.shuffle(items)

    for i, item in enumerate(items[:n_samples]):
        question = item.get('question', {}).get('text', '') if isinstance(item.get('question'), dict) else str(item.get('question', ''))

        # Get document/context
        doc = item.get('document', {}).get('tokens', {}).get('token', [])
        if isinstance(doc, list):
            context = ' '.join(doc[:500])  # First 500 tokens
        else:
            context = str(doc)[:2000]

        # Get answer
        annotations = item.get('annotations', [{}])
        if annotations and len(annotations) > 0:
            short_answers = annotations[0].get('short_answers', [])
            if short_answers:
                answer_tokens = short_answers[0].get('text', '')
                if not answer_tokens:
                    start = short_answers[0].get('start_token', 0)
                    end = short_answers[0].get('end_token', start + 5)
                    if isinstance(doc, list):
                        answer_tokens = ' '.join(doc[start:end])
                    else:
                        answer_tokens = "Unknown"
            else:
                answer_tokens = "No short answer available"
        else:
            answer_tokens = "No annotation"

        if not question or not context or len(context) < 50:
            continue

        # Create faithful example
        examples.append({
            'query': question,
            'documents': [context],
            'response': f"Based on the context, {answer_tokens}",
            'label': 0
        })

        # Create hallucinated example (swap with random answer)
        if i + 1 < len(items):
            other_item = items[(i + 17) % len(items)]  # Pick different item
            other_annotations = other_item.get('annotations', [{}])
            if other_annotations:
                other_answers = other_annotations[0].get('short_answers', [])
                if other_answers:
                    wrong_answer = other_answers[0].get('text', 'incorrect information')
                else:
                    wrong_answer = "This information cannot be determined"
            else:
                wrong_answer = "I believe the answer is something else entirely"

            examples.append({
                'query': question,
                'documents': [context],
                'response': f"The answer is {wrong_answer}",
                'label': 1
            })

    print(f"  Created {len(examples)} NQ examples ({len([e for e in examples if e['label']==0])} faithful, {len([e for e in examples if e['label']==1])} hallucinated)")
    return examples


def download_ragtruth(n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """
    Download RAGTruth dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets --quiet")
        from datasets import load_dataset

    print(f"Loading RAGTruth dataset...")

    try:
        # Try loading from HuggingFace
        dataset = load_dataset("RAGTruth/ragtruth", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  Could not load RAGTruth from HuggingFace: {e}")
        print("  Creating synthetic RAGTruth-style data...")
        return create_synthetic_ragtruth(n_samples, seed)

    random.seed(seed)
    examples = []

    items = list(dataset)
    random.shuffle(items)

    for item in items[:n_samples]:
        examples.append({
            'query': item.get('question', item.get('query', '')),
            'documents': [item.get('context', item.get('passage', ''))],
            'response': item.get('response', item.get('answer', '')),
            'label': item.get('label', 0)
        })

    print(f"  Loaded {len(examples)} RAGTruth examples")
    return examples


def create_synthetic_ragtruth(n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """
    Create synthetic RAGTruth-style data for testing.
    Uses different style than NQ to simulate cross-dataset shift.
    """
    random.seed(seed)

    # Technical/scientific topics (different from NQ's factoid style)
    topics = [
        {
            'query': "Explain how neural networks learn through backpropagation",
            'context': "Backpropagation is an algorithm for training neural networks. It computes gradients of the loss function with respect to weights by applying the chain rule, propagating errors backward from output to input layers.",
            'faithful': "Backpropagation trains neural networks by computing loss gradients through the chain rule, propagating errors backward through layers.",
            'hallucinated': "Backpropagation works by randomly adjusting weights until the network produces correct outputs."
        },
        {
            'query': "What is the difference between supervised and unsupervised learning?",
            'context': "Supervised learning uses labeled training data where inputs are paired with correct outputs. Unsupervised learning finds patterns in data without labels, discovering hidden structures.",
            'faithful': "Supervised learning uses labeled data with input-output pairs, while unsupervised learning discovers patterns without labels.",
            'hallucinated': "Supervised learning requires human supervision during training, while unsupervised learning runs automatically without any data."
        },
        {
            'query': "How does attention mechanism work in transformers?",
            'context': "Attention in transformers computes weighted sums of values based on query-key similarity scores. Self-attention allows each position to attend to all positions in the sequence.",
            'faithful': "Attention computes weighted value sums based on query-key similarities, enabling each position to attend to all sequence positions.",
            'hallucinated': "Attention in transformers is similar to human attention, focusing on one word at a time sequentially."
        },
        {
            'query': "What is gradient descent optimization?",
            'context': "Gradient descent is an optimization algorithm that iteratively adjusts parameters in the direction of steepest descent of the loss function, using the negative gradient.",
            'faithful': "Gradient descent iteratively updates parameters following the negative gradient direction to minimize the loss function.",
            'hallucinated': "Gradient descent randomly samples gradients to find the global minimum of any function."
        },
        {
            'query': "Explain the concept of overfitting in machine learning",
            'context': "Overfitting occurs when a model learns noise in training data rather than general patterns, resulting in poor generalization to new data. Regularization techniques help prevent it.",
            'faithful': "Overfitting happens when models learn training noise instead of patterns, causing poor generalization. Regularization helps prevent this.",
            'hallucinated': "Overfitting means the model is too small to fit the training data properly."
        },
        {
            'query': "What is the purpose of batch normalization?",
            'context': "Batch normalization normalizes layer inputs by subtracting batch mean and dividing by batch standard deviation. This stabilizes training and allows higher learning rates.",
            'faithful': "Batch normalization normalizes inputs using batch statistics, stabilizing training and enabling higher learning rates.",
            'hallucinated': "Batch normalization increases the batch size to improve model accuracy."
        },
        {
            'query': "How do convolutional neural networks detect features?",
            'context': "CNNs use learnable filters that slide across input images, computing dot products to create feature maps. Hierarchical layers detect increasingly complex patterns.",
            'faithful': "CNNs apply learnable filters across images to create feature maps, with hierarchical layers detecting progressively complex patterns.",
            'hallucinated': "CNNs use pre-defined edge detection filters that cannot be modified during training."
        },
        {
            'query': "What is the vanishing gradient problem?",
            'context': "The vanishing gradient problem occurs in deep networks when gradients become exponentially small during backpropagation, preventing effective learning in early layers.",
            'faithful': "Vanishing gradients occur when gradients diminish exponentially in deep networks, hindering learning in early layers.",
            'hallucinated': "The vanishing gradient problem means gradients disappear completely after the first epoch of training."
        },
        {
            'query': "Explain transfer learning in deep learning",
            'context': "Transfer learning leverages knowledge from pre-trained models on large datasets, fine-tuning them for new tasks with less data. This reduces training time and improves performance.",
            'faithful': "Transfer learning uses pre-trained model knowledge, fine-tuning for new tasks with less data to reduce training time.",
            'hallucinated': "Transfer learning copies data from one dataset to another to increase training samples."
        },
        {
            'query': "What is the role of activation functions?",
            'context': "Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Common functions include ReLU, sigmoid, and tanh.",
            'faithful': "Activation functions add non-linearity, allowing networks to learn complex patterns. ReLU, sigmoid, and tanh are common choices.",
            'hallucinated': "Activation functions only activate certain neurons randomly to prevent overfitting."
        }
    ]

    examples = []

    # Repeat topics to reach n_samples
    while len(examples) < n_samples:
        for topic in topics:
            if len(examples) >= n_samples:
                break

            # Add faithful example
            examples.append({
                'query': topic['query'],
                'documents': [topic['context']],
                'response': topic['faithful'],
                'label': 0
            })

            if len(examples) >= n_samples:
                break

            # Add hallucinated example
            examples.append({
                'query': topic['query'],
                'documents': [topic['context']],
                'response': topic['hallucinated'],
                'label': 1
            })

    random.shuffle(examples)
    print(f"  Created {len(examples)} synthetic RAGTruth examples")
    return examples


def download_halueval(n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """
    Download HaluEval dataset for RLHF-style hallucinations.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        os.system(f"{sys.executable} -m pip install datasets --quiet")
        from datasets import load_dataset

    print(f"Loading HaluEval dataset...")

    try:
        dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data", trust_remote_code=True)
    except Exception as e:
        print(f"  Could not load HaluEval: {e}")
        print("  Creating synthetic HaluEval-style data...")
        return create_synthetic_halueval(n_samples, seed)

    random.seed(seed)
    examples = []

    items = list(dataset)
    random.shuffle(items)

    for item in items[:n_samples]:
        # HaluEval format varies, adapt as needed
        examples.append({
            'query': item.get('question', ''),
            'documents': [item.get('knowledge', item.get('context', ''))],
            'response': item.get('answer', item.get('response', '')),
            'label': 1 if item.get('hallucination', 'no') == 'yes' else 0
        })

    print(f"  Loaded {len(examples)} HaluEval examples")
    return examples


def create_synthetic_halueval(n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """
    Create synthetic HaluEval-style data (RLHF hallucinations).
    These are plausible-sounding but incorrect responses.
    """
    random.seed(seed)

    # RLHF-style: confident, helpful-sounding but wrong
    topics = [
        {
            'query': "What year was the first iPhone released?",
            'context': "The first iPhone was announced by Steve Jobs on January 9, 2007 and released on June 29, 2007.",
            'faithful': "The first iPhone was released on June 29, 2007.",
            'hallucinated': "The first iPhone was released in 2005, revolutionizing the smartphone industry two years before most people realize."
        },
        {
            'query': "Who wrote the Harry Potter series?",
            'context': "The Harry Potter series was written by British author J.K. Rowling, with the first book published in 1997.",
            'faithful': "J.K. Rowling wrote the Harry Potter series.",
            'hallucinated': "The Harry Potter series was collaboratively written by J.K. Rowling and Stephen King, combining their unique storytelling styles."
        },
        {
            'query': "What is the capital of Australia?",
            'context': "Canberra is the capital city of Australia, chosen as a compromise between Sydney and Melbourne.",
            'faithful': "Canberra is the capital of Australia.",
            'hallucinated': "Sydney is the capital of Australia, being the largest and most internationally recognized city in the country."
        },
        {
            'query': "How many planets are in our solar system?",
            'context': "Our solar system has eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
            'faithful': "There are eight planets in our solar system.",
            'hallucinated': "There are nine planets in our solar system, including Pluto which remains classified as a full planet by most astronomers."
        },
        {
            'query': "What element has the chemical symbol 'Au'?",
            'context': "Gold has the chemical symbol Au, derived from the Latin word 'aurum'.",
            'faithful': "Au is the chemical symbol for gold.",
            'hallucinated': "Au is the chemical symbol for silver, derived from the Latin 'argentum' which relates to its lustrous appearance."
        },
    ]

    examples = []

    while len(examples) < n_samples:
        for topic in topics:
            if len(examples) >= n_samples:
                break
            examples.append({
                'query': topic['query'],
                'documents': [topic['context']],
                'response': topic['faithful'],
                'label': 0
            })
            if len(examples) >= n_samples:
                break
            examples.append({
                'query': topic['query'],
                'documents': [topic['context']],
                'response': topic['hallucinated'],
                'label': 1
            })

    random.shuffle(examples)
    print(f"  Created {len(examples)} synthetic HaluEval examples")
    return examples


def save_dataset(examples: List[Dict], output_path: str):
    """Save dataset to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)

    print(f"Saved {len(examples)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for ConformalDrift experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nq", "ragtruth", "halueval", "all"],
        default="all",
        help="Dataset to prepare"
    )
    parser.add_argument("--n-samples", type=int, default=200, help="Number of samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("ConformalDrift Data Preparation")
    print("=" * 60)

    if args.dataset in ["nq", "all"]:
        print("\n--- Natural Questions ---")
        try:
            nq_examples = download_nq_sample(args.n_samples, args.seed)
            # Split into calibration and test
            n_cal = len(nq_examples) // 2
            save_dataset(nq_examples[:n_cal], output_dir / "nq_calibration.json")
            save_dataset(nq_examples[n_cal:], output_dir / "nq_test.json")
        except Exception as e:
            print(f"  Error preparing NQ data: {e}")
            print("  Using synthetic data instead...")
            # Create minimal synthetic NQ-style data
            synthetic = create_synthetic_ragtruth(args.n_samples, args.seed)
            n_cal = len(synthetic) // 2
            save_dataset(synthetic[:n_cal], output_dir / "nq_calibration.json")
            save_dataset(synthetic[n_cal:], output_dir / "nq_test.json")

    if args.dataset in ["ragtruth", "all"]:
        print("\n--- RAGTruth ---")
        ragtruth_examples = download_ragtruth(args.n_samples, args.seed)
        save_dataset(ragtruth_examples, output_dir / "ragtruth_test.json")

    if args.dataset in ["halueval", "all"]:
        print("\n--- HaluEval ---")
        halueval_examples = download_halueval(args.n_samples, args.seed)
        save_dataset(halueval_examples, output_dir / "halueval_test.json")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
