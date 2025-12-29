#!/usr/bin/env python3
"""
Generate Synthetic Data for ConformalDrift Experiments

Creates synthetic datasets mimicking:
- Natural Questions (NQ) - factoid QA with Wikipedia-style context
- RAGTruth - technical/scientific explanations (different style = cross-dataset shift)
- HaluEval - RLHF-style confident but wrong responses

Usage:
    python scripts/generate_synthetic_data.py --n-samples 100
"""
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict


def generate_nq_style(n_samples: int, seed: int = 42) -> List[Dict]:
    """
    Generate NQ-style factoid QA examples.
    Style: Wikipedia-like context, short factual answers.
    """
    random.seed(seed)

    topics = [
        # Geography
        {"q": "What is the capital of France?", "ctx": "France is a country in Western Europe. Paris is the capital and largest city of France, situated on the River Seine.", "a": "Paris", "wrong": "Lyon"},
        {"q": "What is the longest river in the world?", "ctx": "The Nile is a major north-flowing river in northeastern Africa. It is generally regarded as the longest river in the world at about 6,650 km.", "a": "The Nile", "wrong": "The Amazon"},
        {"q": "What is the largest country by area?", "ctx": "Russia is the largest country in the world by area, covering over 17 million square kilometers.", "a": "Russia", "wrong": "Canada"},
        {"q": "What is the deepest ocean?", "ctx": "The Pacific Ocean is the largest and deepest ocean on Earth. The Mariana Trench in the Pacific reaches depths of nearly 11,000 meters.", "a": "The Pacific Ocean", "wrong": "The Atlantic Ocean"},
        {"q": "What is the highest mountain in the world?", "ctx": "Mount Everest, located in the Himalayas, is Earth's highest mountain above sea level at 8,848.86 meters.", "a": "Mount Everest", "wrong": "K2"},

        # History
        {"q": "When did World War II end?", "ctx": "World War II ended in 1945. Germany surrendered in May 1945, and Japan surrendered in September 1945 after the atomic bombings.", "a": "1945", "wrong": "1944"},
        {"q": "Who was the first President of the United States?", "ctx": "George Washington served as the first President of the United States from 1789 to 1797.", "a": "George Washington", "wrong": "Thomas Jefferson"},
        {"q": "When was the Declaration of Independence signed?", "ctx": "The Declaration of Independence was adopted by the Continental Congress on July 4, 1776.", "a": "July 4, 1776", "wrong": "1783"},
        {"q": "Who invented the telephone?", "ctx": "Alexander Graham Bell is credited with inventing the first practical telephone in 1876.", "a": "Alexander Graham Bell", "wrong": "Thomas Edison"},
        {"q": "When did the Berlin Wall fall?", "ctx": "The Berlin Wall fell on November 9, 1989, leading to German reunification.", "a": "November 9, 1989", "wrong": "1991"},

        # Science
        {"q": "What is the chemical symbol for gold?", "ctx": "Gold is a chemical element with symbol Au (from Latin aurum) and atomic number 79.", "a": "Au", "wrong": "Go"},
        {"q": "What is the speed of light?", "ctx": "The speed of light in a vacuum is approximately 299,792 kilometers per second.", "a": "299,792 km/s", "wrong": "150,000 km/s"},
        {"q": "What is the atomic number of carbon?", "ctx": "Carbon is a chemical element with atomic number 6. It is the basis of all known life on Earth.", "a": "6", "wrong": "12"},
        {"q": "What is the boiling point of water?", "ctx": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.", "a": "100 degrees Celsius", "wrong": "90 degrees Celsius"},
        {"q": "What planet is known as the Red Planet?", "ctx": "Mars is often called the Red Planet due to its reddish appearance caused by iron oxide on its surface.", "a": "Mars", "wrong": "Jupiter"},

        # Literature/Culture
        {"q": "Who wrote Romeo and Juliet?", "ctx": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.", "a": "William Shakespeare", "wrong": "Charles Dickens"},
        {"q": "Who painted the Mona Lisa?", "ctx": "The Mona Lisa is a half-length portrait painting by Italian Renaissance artist Leonardo da Vinci.", "a": "Leonardo da Vinci", "wrong": "Michelangelo"},
        {"q": "What year was the first Harry Potter book published?", "ctx": "Harry Potter and the Philosopher's Stone was first published in 1997 by Bloomsbury.", "a": "1997", "wrong": "2001"},
        {"q": "Who wrote 1984?", "ctx": "1984 is a dystopian novel published in 1949 by English author George Orwell.", "a": "George Orwell", "wrong": "Aldous Huxley"},
        {"q": "What is the national sport of Japan?", "ctx": "Sumo wrestling is considered the national sport of Japan, with a history spanning centuries.", "a": "Sumo wrestling", "wrong": "Baseball"},
    ]

    examples = []
    while len(examples) < n_samples:
        for topic in topics:
            if len(examples) >= n_samples:
                break
            # Faithful
            examples.append({
                "query": topic["q"],
                "documents": [topic["ctx"]],
                "response": f"Based on the context, {topic['a']}.",
                "label": 0
            })
            if len(examples) >= n_samples:
                break
            # Hallucinated
            examples.append({
                "query": topic["q"],
                "documents": [topic["ctx"]],
                "response": f"The answer is {topic['wrong']}.",
                "label": 1
            })

    random.shuffle(examples)
    return examples[:n_samples]


def generate_ragtruth_style(n_samples: int, seed: int = 42) -> List[Dict]:
    """
    Generate RAGTruth-style technical explanations.
    Style: Longer, more technical, explanatory responses.
    This creates a CROSS-DATASET SHIFT from NQ style.
    """
    random.seed(seed)

    topics = [
        {
            "q": "Explain how neural networks learn",
            "ctx": "Neural networks learn through backpropagation, computing gradients of the loss function with respect to weights using the chain rule, then updating weights via gradient descent.",
            "faithful": "Neural networks learn by computing loss gradients through backpropagation and updating weights via gradient descent. The chain rule enables gradient computation across layers.",
            "hallucinated": "Neural networks learn by randomly adjusting weights until outputs match expected values. No mathematical optimization is involved."
        },
        {
            "q": "What is the difference between TCP and UDP?",
            "ctx": "TCP (Transmission Control Protocol) provides reliable, ordered delivery with error checking and acknowledgments. UDP (User Datagram Protocol) provides faster but unreliable connectionless transmission.",
            "faithful": "TCP ensures reliable, ordered data delivery with acknowledgments and error correction, while UDP offers faster connectionless transmission without delivery guarantees.",
            "hallucinated": "TCP and UDP are essentially identical protocols. The only difference is that TCP uses ports while UDP does not."
        },
        {
            "q": "How does photosynthesis work?",
            "ctx": "Photosynthesis converts light energy to chemical energy. In the light reactions, chlorophyll absorbs light to split water, releasing oxygen. The Calvin cycle then uses ATP and NADPH to fix CO2 into glucose.",
            "faithful": "Photosynthesis captures light via chlorophyll, splitting water and releasing oxygen. ATP and NADPH from light reactions power the Calvin cycle, which converts CO2 into glucose.",
            "hallucinated": "Photosynthesis is a process where plants absorb oxygen from the air and release carbon dioxide. Light is optional for this process."
        },
        {
            "q": "Explain how vaccines work",
            "ctx": "Vaccines train the immune system by introducing antigens (weakened, inactivated, or partial pathogens). This triggers antibody production and memory cell formation without causing disease.",
            "faithful": "Vaccines introduce antigens to trigger immune response without disease. The body produces antibodies and memory cells, enabling rapid response to future infections.",
            "hallucinated": "Vaccines contain live diseases that make you sick temporarily to build immunity. They suppress the immune system initially."
        },
        {
            "q": "What causes inflation?",
            "ctx": "Inflation occurs when money supply grows faster than economic output (monetary inflation), when production costs rise (cost-push), or when demand exceeds supply (demand-pull).",
            "faithful": "Inflation results from money supply outpacing output, rising production costs, or excess demand. These factors reduce currency purchasing power over time.",
            "hallucinated": "Inflation is caused solely by government printing money. Supply and demand have no effect on price levels."
        },
        {
            "q": "How does blockchain work?",
            "ctx": "Blockchain is a distributed ledger where transactions are grouped into blocks, cryptographically hashed, and linked chronologically. Consensus mechanisms validate new blocks across the network.",
            "faithful": "Blockchain links cryptographically hashed transaction blocks chronologically across a distributed network. Consensus protocols ensure all nodes agree on the valid chain state.",
            "hallucinated": "Blockchain stores data in a central database controlled by blockchain companies. Each company has its own separate blockchain."
        },
        {
            "q": "Explain quantum entanglement",
            "ctx": "Quantum entanglement occurs when particles become correlated such that the quantum state of one instantly affects the other, regardless of distance. Measurement of one particle determines the state of its pair.",
            "faithful": "Entangled particles share quantum states where measuring one instantaneously determines the other's state, regardless of separation distance, though no information travels faster than light.",
            "hallucinated": "Quantum entanglement allows instant communication across any distance. Scientists use it for faster-than-light messaging."
        },
        {
            "q": "How does CRISPR gene editing work?",
            "ctx": "CRISPR-Cas9 uses guide RNA to locate specific DNA sequences. The Cas9 enzyme then cuts the DNA at that location, allowing genes to be deleted, modified, or inserted.",
            "faithful": "CRISPR uses guide RNA to target specific DNA sequences. Cas9 enzyme cuts at these locations, enabling precise gene deletion, modification, or insertion.",
            "hallucinated": "CRISPR randomly modifies genes throughout the genome. Scientists cannot control which genes are affected."
        },
        {
            "q": "What is machine learning?",
            "ctx": "Machine learning enables computers to learn patterns from data without explicit programming. Algorithms improve performance through experience, using statistical methods to find patterns.",
            "faithful": "Machine learning allows systems to learn patterns from data automatically, improving through experience. Statistical algorithms identify relationships without explicit programming.",
            "hallucinated": "Machine learning requires programmers to manually code every decision rule. It cannot learn from data automatically."
        },
        {
            "q": "How does the Internet work?",
            "ctx": "The Internet is a global network using TCP/IP protocols. Data is broken into packets, routed through interconnected networks via routers, and reassembled at the destination.",
            "faithful": "The Internet connects global networks using TCP/IP. Data travels as packets through routers across interconnected networks, reassembling at destinations.",
            "hallucinated": "The Internet is a single cable connecting all computers directly. Data travels as continuous streams without packets."
        },
    ]

    examples = []
    while len(examples) < n_samples:
        for topic in topics:
            if len(examples) >= n_samples:
                break
            examples.append({
                "query": topic["q"],
                "documents": [topic["ctx"]],
                "response": topic["faithful"],
                "label": 0
            })
            if len(examples) >= n_samples:
                break
            examples.append({
                "query": topic["q"],
                "documents": [topic["ctx"]],
                "response": topic["hallucinated"],
                "label": 1
            })

    random.shuffle(examples)
    return examples[:n_samples]


def generate_halueval_style(n_samples: int, seed: int = 42) -> List[Dict]:
    """
    Generate HaluEval-style RLHF hallucinations.
    Style: Confident, helpful-sounding but subtly incorrect.
    This simulates RLHF-induced hallucinations.
    """
    random.seed(seed)

    topics = [
        {
            "q": "What year was the first iPhone released?",
            "ctx": "The first iPhone was announced on January 9, 2007 and released on June 29, 2007.",
            "faithful": "The first iPhone was released on June 29, 2007.",
            "hallucinated": "The first iPhone was released in 2005, making it the pioneer of the smartphone revolution."
        },
        {
            "q": "How many bones are in the adult human body?",
            "ctx": "An adult human body contains 206 bones. Babies are born with about 270 bones, many of which fuse together.",
            "faithful": "An adult human body has 206 bones.",
            "hallucinated": "The human body contains 312 bones, making it one of the most complex skeletal structures in nature."
        },
        {
            "q": "What is the population of Tokyo?",
            "ctx": "Tokyo has a population of approximately 14 million in the city proper and over 37 million in the greater metropolitan area.",
            "faithful": "Tokyo has about 14 million people in the city proper, 37 million in the greater metro area.",
            "hallucinated": "Tokyo's population is 50 million, making it by far the largest city in the world."
        },
        {
            "q": "Who discovered penicillin?",
            "ctx": "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London.",
            "faithful": "Alexander Fleming discovered penicillin in 1928.",
            "hallucinated": "Penicillin was discovered by Louis Pasteur in 1895 during his research on fermentation."
        },
        {
            "q": "What is the distance from Earth to the Sun?",
            "ctx": "The average distance from Earth to the Sun is about 150 million kilometers (93 million miles), defined as one astronomical unit.",
            "faithful": "Earth is about 150 million km (93 million miles) from the Sun.",
            "hallucinated": "Earth is 50 million kilometers from the Sun, which is why we experience moderate temperatures."
        },
        {
            "q": "How fast can a cheetah run?",
            "ctx": "Cheetahs can reach speeds of up to 120 km/h (75 mph), making them the fastest land animals.",
            "faithful": "Cheetahs can run up to 120 km/h (75 mph), the fastest of any land animal.",
            "hallucinated": "Cheetahs can reach speeds of 200 km/h, faster than most cars on highways."
        },
        {
            "q": "What causes the seasons on Earth?",
            "ctx": "Earth's seasons are caused by its axial tilt of 23.5 degrees relative to its orbital plane, not by distance from the Sun.",
            "faithful": "Seasons result from Earth's 23.5-degree axial tilt, not from varying distance to the Sun.",
            "hallucinated": "Seasons are caused by Earth's changing distance from the Sun throughout the year."
        },
        {
            "q": "How many chromosomes do humans have?",
            "ctx": "Humans have 46 chromosomes in total, arranged in 23 pairs. One set comes from each parent.",
            "faithful": "Humans have 46 chromosomes (23 pairs), with one set from each parent.",
            "hallucinated": "Humans have 52 chromosomes, more than any other primate species."
        },
        {
            "q": "What is the Great Wall of China made of?",
            "ctx": "The Great Wall was built using various materials including rammed earth, wood, stone, and bricks depending on the section and era.",
            "faithful": "The Great Wall uses various materials: rammed earth, wood, stone, and bricks, varying by section.",
            "hallucinated": "The Great Wall is made entirely of marble blocks imported from Italy during the Ming dynasty."
        },
        {
            "q": "How long is a light year?",
            "ctx": "A light year is the distance light travels in one year, approximately 9.46 trillion kilometers.",
            "faithful": "A light year is about 9.46 trillion km, the distance light travels in one year.",
            "hallucinated": "A light year is approximately 1 billion kilometers, roughly the distance from Earth to Jupiter."
        },
    ]

    examples = []
    while len(examples) < n_samples:
        for topic in topics:
            if len(examples) >= n_samples:
                break
            examples.append({
                "query": topic["q"],
                "documents": [topic["ctx"]],
                "response": topic["faithful"],
                "label": 0
            })
            if len(examples) >= n_samples:
                break
            examples.append({
                "query": topic["q"],
                "documents": [topic["ctx"]],
                "response": topic["hallucinated"],
                "label": 1
            })

    random.shuffle(examples)
    return examples[:n_samples]


def save_dataset(examples: List[Dict], output_path: Path):
    """Save dataset to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"  Saved {len(examples)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for experiments")
    parser.add_argument("--n-samples", type=int, default=100, help="Samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Generating Synthetic Data for ConformalDrift")
    print("=" * 60)

    # NQ-style data (split into calibration and test)
    print("\n--- NQ-Style (Factoid QA) ---")
    nq_data = generate_nq_style(args.n_samples * 2, args.seed)
    n_cal = len(nq_data) // 2
    save_dataset(nq_data[:n_cal], output_dir / "nq_calibration.json")
    save_dataset(nq_data[n_cal:], output_dir / "nq_test.json")

    # RAGTruth-style data (creates cross-dataset shift)
    print("\n--- RAGTruth-Style (Technical Explanations) ---")
    ragtruth_data = generate_ragtruth_style(args.n_samples, args.seed)
    save_dataset(ragtruth_data, output_dir / "ragtruth_test.json")

    # HaluEval-style data (RLHF shift)
    print("\n--- HaluEval-Style (RLHF Hallucinations) ---")
    halueval_data = generate_halueval_style(args.n_samples, args.seed)
    save_dataset(halueval_data, output_dir / "halueval_test.json")

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
