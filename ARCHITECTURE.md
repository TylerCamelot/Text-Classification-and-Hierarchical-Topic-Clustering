# 🏗 Project Architecture: Hierarchical NLP Pipeline

This document details the architectural design and technical flow of the NLP Topic Tree project. The system is built on a modular, multi-stage architecture that transitions from supervised classification to unsupervised hierarchical discovery.

---

## 🛰 System Overview

The project is structured into three distinct functional layers: **Data Ingestion**, **Model Benchmarking**, and **Hierarchical Synthesis**.

```mermaid
graph TD
    subgraph "1. Data Layer"
    A[20 Newsgroups Dataset] --> B(Data Preprocessor)
    B -->|Metadata Stripping| C[Cleaned 10k Corpus]
    end

    subgraph "2. Transformation Layer"
    C --> D{Feature Extractors}
    D -->|Sparse| E[TF-IDF / BoW]
    D -->|Dense| F[Sentence Transformers]
    end

    subgraph "3. Execution Layer"
    E --> G[Part 1: Classic Classifiers]
    F --> H[Part 2: Neural Classifiers]
    F --> I[Part 3: Hierarchical Clustering]
    end

    subgraph "4. Output Layer"
    G --> J[Accuracy Tables]
    H --> K[Confusion Matrices]
    I --> L[Topic Tree Viz & PDF Summary]
    end
```

---

## 🔄 Document Processing Sequence (Part 3)

The following sequence diagram illustrates the lifecycle of a document as it moves through the unsupervised discovery and labeling phase.

```mermaid
sequenceDiagram
    participant D as Dataset (10k Rows)
    participant E as Embedding Engine (all-MiniLM-L6-v2)
    participant K as K-Means Controller
    participant L as LLM Labeler (GPT-4o-mini)
    participant V as Visualizer / PDF Generator

    D->>E: Batch input text
    E->>E: Generate 384-dim dense vectors
    E->>K: Provide semantic embeddings
    K->>K: Run Elbow Method (k=2..10)
    K->>K: Assign documents to optimal clusters
    K->>K: Sub-cluster top 2 largest groups
    K->>D: Retrieve original Row IDs for centroids
    D->>L: Send top representative snippets
    L->>L: Check for refusal/noise
    alt LLM Success
        L->>V: Return 2-4 word professional label
    else LLM Refusal/Noise
        L->>V: Fallback to TF-IDF keywords
    end
    V->>V: Map Global Row IDs to snippets
    V->>V: Generate PNG Tree & Summary PDF
```

---

## 🛠 Component Specifications

### 1. Data Preprocessing (`src/data.py`)

- **Sampling**: Restricts processing to a 10,000-document subset to maintain computational efficiency while ensuring statistical significance.
- **Sanitization**: Specifically removes headers, footers, and quotes to ensure models learn semantic content rather than metadata patterns.
- **Splitting**: Implements stratified sampling to maintain class balance across training and testing sets.

### 2. Classification Engines

- **Classic ML** (`src/part1_classic.py`): Utilizes scikit-learn pipelines to compare four distinct mathematical approaches to text classification.
- **Neural ML** (`src/part2_embeddings.py`): Uses the **all-MiniLM-L6-v2** model to transform text into 384-dimensional semantic space.
- **Scaling Logic**: Implements **MinMaxScaler** for neural embeddings to support the mathematical requirements of the Multinomial Naive Bayes classifier.

### 3. Hierarchical Discovery (`src/part3_clustering.py`)

- **Unsupervised Learning**: Employs K-Means clustering where the optimal number of clusters is determined dynamically via the Elbow Method (Inertia vs. K).
- **Recursive Depth**: Identifies the two largest clusters and performs secondary sub-clustering to uncover granular sub-topics.
- **Label Synthesis**: Integrates a "Refusal-Aware" OpenAI labeler that translates document snippets into professional topic labels with a TF-IDF keyword fallback for noisy data.

---

## 🔄 Data Integrity & Mapping

A core architectural requirement is the **Global Row ID Mapping**.

- **Traceability**: Every document maintains its original index (0–9999) throughout the pipeline.
- **Verification**: The final `cluster_summary.pdf` maps clusters back to these global row numbers, allowing for 1:1 data audits.
- **Serialization**: All numerical outputs are cast from `numpy.int64` to standard Python integers to ensure 100% JSON compatibility.

---

## 📁 Directory Structure

```
├── src/
│   ├── data.py           # Data loading & metadata stripping
│   ├── llm_labeler.py    # OpenAI integration & fallback logic
│   ├── part3_clustering.py # K-Means & Tree visualization
│   └── eval.py           # Metric calculation & plotting
├── scripts/
│   ├── run_part1.py      # Classic ML runner
│   ├── run_part2.py      # Embedding ML runner
│   └── run_part3.py      # Clustering & PDF runner
└── outputs/              # PNGs, PDFs, and JSON results
```****
