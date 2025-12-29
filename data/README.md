# Data Directory

This directory should contain the evaluation datasets for ConformalDrift experiments.

## Required Files

### nq_calibration.json
Natural Questions calibration set with labeled hallucinations.

```json
[
  {
    "query": "What is the capital of France?",
    "documents": ["France is a country in Western Europe. Its capital is Paris."],
    "response": "The capital of France is Paris.",
    "label": 0
  },
  {
    "query": "When was Python released?",
    "documents": ["Python was created by Guido van Rossum in 1991."],
    "response": "Python was released in 2010.",
    "label": 1
  }
]
```

- `label`: 0 = faithful, 1 = hallucinated

### nq_test.json
Natural Questions test set (same format as calibration).

### ragtruth_test.json
RAGTruth cross-dataset evaluation set (same format).

### halueval_test.json (optional)
HaluEval RLHF-style hallucination evaluation set (same format).

## Data Sources

- **Natural Questions**: https://ai.google.com/research/NaturalQuestions
- **RAGTruth**: https://github.com/RAGTruth/RAGTruth
- **HaluEval**: https://github.com/RUCAIBox/HaluEval
