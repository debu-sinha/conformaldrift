# Examples

Practical examples demonstrating Conformal-Drift usage.

## RAG Hallucination Detection

Audit a conformal predictor used for RAG hallucination detection:

```python
import numpy as np
from conformal_drift import ConformalDriftAuditor

# Load calibration nonconformity scores (e.g., 1 - cosine similarity)
cal_scores = np.load("rag_calibration_scores.npy")

# Initialize auditor
auditor = ConformalDriftAuditor(
    calibration_scores=cal_scores,
    alpha=0.1  # 90% coverage target
)

# Prepare test data
test_data = {
    'queries': test_queries,
    'responses': test_responses,
    'labels': is_hallucination,  # Ground truth
    'scores': 1 - cosine_similarity(query_embeddings, response_embeddings),
    'timestamps': query_timestamps
}

# Audit under temporal shift (news/topics change over time)
results = auditor.audit(
    test_data=test_data,
    shift_type="temporal",
    shift_intensity=np.linspace(0, 1, 11)
)

# Analyze
print(f"Baseline coverage: {results.coverage[0]:.3f}")
print(f"Coverage at max shift: {results.coverage[-1]:.3f}")
print(f"Coverage gap: {results.max_coverage_gap:.3f}")
```

## LangChain Integration

Use Conformal-Drift with LangChain RAG pipelines:

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from conformal_drift import ConformalDriftAuditor
import numpy as np

# Your existing LangChain setup
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

# Compute nonconformity scores for calibration set
def compute_rag_scores(queries, documents, responses):
    """Compute nonconformity scores for RAG outputs."""
    scores = []
    for q, d, r in zip(queries, documents, responses):
        # Score based on document-response alignment
        doc_emb = embeddings.embed_documents([d])[0]
        resp_emb = embeddings.embed_documents([r])[0]
        similarity = np.dot(doc_emb, resp_emb) / (
            np.linalg.norm(doc_emb) * np.linalg.norm(resp_emb)
        )
        scores.append(1 - similarity)  # Nonconformity = 1 - similarity
    return np.array(scores)

# Calibrate
cal_scores = compute_rag_scores(cal_queries, cal_docs, cal_responses)

# Initialize auditor
auditor = ConformalDriftAuditor(
    calibration_scores=cal_scores,
    alpha=0.1
)

# Audit with semantic shift
test_scores = compute_rag_scores(test_queries, test_docs, test_responses)
test_data = {
    'scores': test_scores,
    'labels': test_labels
}

results = auditor.audit(test_data, shift_type="semantic")
```

## LlamaIndex Integration

Use with LlamaIndex for document QA:

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding
from conformal_drift import ConformalDriftAuditor
import numpy as np

# Load documents and create index
documents = SimpleDirectoryReader("data").load_data()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Query engine
query_engine = index.as_query_engine()

# Compute calibration scores
def get_nonconformity_score(query, response, source_nodes):
    """Score based on source relevance."""
    if not source_nodes:
        return 1.0  # Maximum nonconformity if no sources

    # Average similarity to retrieved sources
    similarities = [node.score for node in source_nodes if node.score]
    if similarities:
        return 1 - np.mean(similarities)
    return 0.5

# Run calibration queries
cal_scores = []
for query, label in zip(cal_queries, cal_labels):
    response = query_engine.query(query)
    score = get_nonconformity_score(query, response, response.source_nodes)
    cal_scores.append(score)

# Audit
auditor = ConformalDriftAuditor(
    calibration_scores=np.array(cal_scores),
    alpha=0.1
)

# Test under domain shift
results = auditor.audit(test_data, shift_type="semantic")
```

## MLflow Integration

Track audits with MLflow for experiment management:

```python
import mlflow
import numpy as np
from conformal_drift import ConformalDriftAuditor

# Start MLflow run
mlflow.set_experiment("conformal-drift-audit")

with mlflow.start_run(run_name="temporal_shift_audit"):
    # Log parameters
    mlflow.log_param("alpha", 0.1)
    mlflow.log_param("shift_type", "temporal")
    mlflow.log_param("n_calibration", len(cal_scores))
    mlflow.log_param("n_test", len(test_data['scores']))

    # Initialize auditor
    auditor = ConformalDriftAuditor(
        calibration_scores=cal_scores,
        alpha=0.1
    )

    # Run audit
    shift_intensities = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = auditor.audit(
        test_data=test_data,
        shift_type="temporal",
        shift_intensity=shift_intensities
    )

    # Log metrics at each shift level
    for intensity, coverage, set_size in zip(
        results.shift_intensities,
        results.coverage,
        results.set_sizes
    ):
        mlflow.log_metric(f"coverage_shift_{intensity:.2f}", coverage)
        mlflow.log_metric(f"set_size_shift_{intensity:.2f}", set_size)

    # Log summary metrics
    mlflow.log_metric("max_coverage_gap", results.max_coverage_gap)
    mlflow.log_metric("critical_intensity", results.critical_intensity)
    mlflow.log_metric("baseline_coverage", results.coverage[0])
    mlflow.log_metric("final_coverage", results.coverage[-1])

    # Save and log coverage curve
    import matplotlib.pyplot as plt
    from conformal_drift.viz import plot_coverage_curve

    fig = plot_coverage_curve(results, nominal_coverage=0.9)
    fig.savefig("coverage_curve.png", dpi=150)
    mlflow.log_artifact("coverage_curve.png")

    # Log results as artifact
    results.save("audit_results.json")
    mlflow.log_artifact("audit_results.json")

    # Tag run with status
    if results.max_coverage_gap > 0.1:
        mlflow.set_tag("status", "CRITICAL_FAILURE")
    elif results.max_coverage_gap > 0.05:
        mlflow.set_tag("status", "WARNING")
    else:
        mlflow.set_tag("status", "PASS")

print(f"Run logged: {mlflow.active_run().info.run_id}")
```

### MLflow Multi-Shift Comparison

```python
import mlflow
import numpy as np
from conformal_drift import ConformalDriftAuditor

mlflow.set_experiment("conformal-drift-comprehensive")

# Parent run for all shifts
with mlflow.start_run(run_name="comprehensive_audit"):
    auditor = ConformalDriftAuditor(calibration_scores=cal_scores, alpha=0.1)

    shift_types = ["temporal", "semantic", "lexical"]
    all_results = {}

    for shift_type in shift_types:
        # Nested run for each shift type
        with mlflow.start_run(run_name=f"{shift_type}_shift", nested=True):
            mlflow.log_param("shift_type", shift_type)

            results = auditor.audit(
                test_data=test_data,
                shift_type=shift_type,
                shift_intensity=np.linspace(0, 1, 11)
            )
            all_results[shift_type] = results

            # Log metrics
            mlflow.log_metric("max_coverage_gap", results.max_coverage_gap)
            mlflow.log_metric("critical_intensity", results.critical_intensity)

            # Log curve
            from conformal_drift.viz import plot_coverage_curve
            fig = plot_coverage_curve(results)
            fig.savefig(f"{shift_type}_curve.png")
            mlflow.log_artifact(f"{shift_type}_curve.png")

    # Log comparison chart in parent run
    from conformal_drift.viz import plot_coverage_comparison
    fig = plot_coverage_comparison(all_results)
    fig.savefig("comparison.png")
    mlflow.log_artifact("comparison.png")

    # Summary table
    summary = {
        shift_type: {
            "max_gap": r.max_coverage_gap,
            "critical": r.critical_intensity
        }
        for shift_type, r in all_results.items()
    }
    mlflow.log_dict(summary, "summary.json")
```

## Classification Model Audit

Audit a conformal classifier under covariate shift:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from conformal_drift import ConformalDriftAuditor

# Create dataset
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute calibration nonconformity scores
cal_probs = model.predict_proba(X_cal)
cal_scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]

# Initialize auditor
auditor = ConformalDriftAuditor(
    calibration_scores=cal_scores,
    alpha=0.1
)

# Compute test scores
test_probs = model.predict_proba(X_test)
test_scores = 1 - test_probs[np.arange(len(y_test)), y_test]

test_data = {
    'inputs': X_test,
    'labels': y_test,
    'scores': test_scores
}

# Audit under semantic shift (feature distribution change)
results = auditor.audit(
    test_data=test_data,
    shift_type="semantic",
    shift_intensity=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

# Report
print("\nClassification Model Audit Results:")
print("=" * 40)
for i, cov in zip(results.shift_intensities, results.coverage):
    status = "OK" if cov >= 0.88 else "WARNING" if cov >= 0.8 else "FAIL"
    print(f"Shift {i:.0%}: Coverage = {cov:.3f} [{status}]")
```

## Continuous Monitoring

Set up continuous monitoring of coverage:

```python
import time
from datetime import datetime
from conformal_drift import ConformalDriftAuditor
import numpy as np

class CoverageMonitor:
    def __init__(self, cal_scores, alpha=0.1, window_size=1000):
        self.auditor = ConformalDriftAuditor(
            calibration_scores=cal_scores,
            alpha=alpha
        )
        self.window_size = window_size
        self.score_buffer = []
        self.label_buffer = []
        self.coverage_history = []

    def add_sample(self, score, label):
        self.score_buffer.append(score)
        self.label_buffer.append(label)

        # Check coverage when window is full
        if len(self.score_buffer) >= self.window_size:
            coverage = self._compute_window_coverage()
            self.coverage_history.append({
                'timestamp': datetime.now(),
                'coverage': coverage,
                'n_samples': len(self.score_buffer)
            })

            # Alert if coverage drops
            if coverage < 0.85:
                self._alert(f"Coverage dropped to {coverage:.3f}")

            # Slide window
            self.score_buffer = self.score_buffer[100:]
            self.label_buffer = self.label_buffer[100:]

    def _compute_window_coverage(self):
        quantile = np.percentile(self.auditor.calibration_scores, 90)
        in_set = np.array(self.score_buffer) <= quantile
        correct = np.array(self.label_buffer)
        return np.mean(in_set == correct)

    def _alert(self, message):
        print(f"ALERT [{datetime.now()}]: {message}")

# Usage
monitor = CoverageMonitor(cal_scores, alpha=0.1)

# Simulate streaming data
for score, label in zip(streaming_scores, streaming_labels):
    monitor.add_sample(score, label)
    time.sleep(0.01)  # Simulate delay

# Check history
for entry in monitor.coverage_history[-5:]:
    print(f"{entry['timestamp']}: {entry['coverage']:.3f}")
```
