# Domain Adaptation of Performance Modeling in Edge FaaS Systems

## Overview

This project investigates **domain adaptation in decentralized Function-as-a-Service (DFaaS) systems**, focusing on robust performance prediction under **domain shift** across heterogeneous edge environments.

In DFaaS, multiple edge nodes collaborate to execute serverless functions. Each node dynamically decides whether to execute requests locally or offload them to neighboring nodes. These decisions rely heavily on accurate performance prediction models.

However, due to differences in hardware configurations and workload patterns, models trained on one node often fail when deployed on another. This project addresses this challenge using **deep learning and adapter-based domain adaptation**.

---

## Problem Statement

Machine learning models trained on a **source domain** (specific node configuration) often fail to generalize to a **target domain** due to:

* Hardware variability (Light / Mid / Heavy nodes)
* Differences in workload distributions
* Environmental heterogeneity

This leads to:

* Reduced prediction accuracy
* Unreliable overload detection
* Degraded workload routing decisions

This phenomenon is known as **domain shift**, including:

* **Covariate Shift** → Changes in input distribution
* **Concept Shift** → Changes in input-output relationships
* **Prior Shift** → Changes in label distribution

---

## Proposed Approach

This project proposes a unified deep learning framework combining:

---

### Multi-Task Learning (MTL)

A single model **jointly predicts both function-level and node-level metrics**:

* Function-level:

  * CPU usage
  * RAM usage
  * Latency

* Node-level:

  * CPU usage
  * RAM usage
  * Overload status

This joint learning setup captures interdependencies between individual functions and overall system behavior.

---

### Permutation-Invariant Modeling

In DFaaS, active functions form a **set of variable length**, not an ordered sequence.

To handle this:

* The model is designed to be **permutation-invariant**
* Uses **Janossy pooling** to:

  * Handle **variable-length inputs**
  * Ensure order-invariant predictions
  * Capture interactions between functions

---

### Adapter-Based Domain Adaptation

To address domain shift:

* A **source-trained model is frozen**
* Lightweight **adapter modules** are inserted into the network
* Only adapter parameters are trained using target-domain data

Additionally:

* Multiple **adapter architectures** are explored
* Adapters are inserted at **different positions in the model**
* This enables identification of the **most effective adaptation strategy**

---

## Adapter Architectures

The following adapter types are evaluated:

* Linear Projection
* Non-linear Projection (ReLU / Tanh)
* Mask Adapter
* FiLM (Feature-wise Linear Modulation)
* Residual Linear Adapter
* Bottleneck Adapter
* Residual Bottleneck Adapter

---

## Adapter Placement Strategies

Adapters are inserted at different locations:

* Post-Embedding
* Post-RNN
* Post-Janossy Pooling
* Post-Shared Representation
* Post-Shared Representation (Task-Specific)
* Pre-Output (Task-Specific)

Each placement is evaluated across multiple configurations.

---

## Experimental Design

This project performs **systematic experimentation** across:

* Multiple adapter architectures
* Multiple insertion positions
* Multiple hyperparameter configurations

Each configuration is implemented as a separate notebook.

---

## Repository Structure

```text
notebooks/
│
├── experiments/
│   ├── analysis/
│   │   ├── shift/        # Covariate and concept shift analysis
│   │   └── overview/     # Dataset exploration
│   │
│   ├── baseline_training/
│   │   └── cross_validation/
│   │        ├── system_forecaster_functions_all_baselines_V1.ipynb
│   │        │   # Train base model on Source Domain
│   │        │
│   │        └── system_forecaster_functions_all_baselines_V2.ipynb
│   │            # Train model from scratch on Target Domain (reference performance)
│   │
│   └── domain_adaptation/
│        ├── adapter_post_embeddings/
│        ├── adapter_pre_output/
│        ├── adapter_post_shared/
│        ├── adapter_post_rnn/
│        ├── adapter_post_janossy/
│        ├── adapter_post_shared_task_specific/
│        ├── data_reduction_experiments/
│        └── thesis_results/
│
└── utils/
    └── domain_projection.py   # Core adapter implementation utilities
```

---

## How to Navigate the Project

### Domain Shift Analysis

```text
notebooks/experiments/analysis/
```

* Understand distribution differences between domains
* Visualize covariate and concept shift

---

### Baseline Models

```text
baseline_training/cross_validation/
```

* `V1` → Train model on **Source Domain**
* `V2` → Train model from scratch on **Target Domain (upper-bound reference)**

---

### Domain Adaptation Experiments

```text
domain_adaptation/
```

Each folder corresponds to a **specific adapter insertion position**.

Each notebook represents a **specific adapter architecture/configuration**.

---

## Naming Convention

Adapter versions follow a structured naming scheme:

* **Vx** → Linear adapter
* **Vx_1** → Non-linear (ReLU)
* **Vx_2** → Mask adapter
* **Vx_3** → FiLM adapter
* **Vx_4** → Residual linear
* **Vx_5** → Bottleneck adapter
* **Vx_6** → Residual bottleneck
* **Vx_7** → Non-linear (Tanh)

---

## Key Results

### 1. Target Domain (From Scratch – Reference)

* Model trained directly on target domain
* Establishes **upper-bound performance**

---

### 2. Source Model on Target Domain (No Adaptation)

* Severe performance degradation due to domain shift
* Demonstrates lack of generalization

---

### 3. Adapter-Based Domain Adaptation

* Recovers most of the lost performance
* Best configuration:

  * **Bottleneck Adapter + Pre-Output (Task-Specific)**

---

### Performance Summary

* **R² ≈ 0.83**
* **Accuracy ≈ 0.96**

---

### Efficiency Gains

* Only **adapter parameters are trained**
* Significant reduction in:

  * Number of trainable parameters
  * Training time
* No need to retrain full model

---

### Data Reduction Experiments

* With ~30% data:

  * Strong regression performance maintained (R² ≈ 0.73)
* Performance drops significantly below ~10%
* Classification (overload detection) remains more robust

---

## Dataset

### Inputs

* Request rates of 6 serverless functions
* Node type (Light / Mid / Heavy)

### Outputs

* Function-level metrics:

  * CPU, RAM, latency
* Node-level metrics:

  * CPU, RAM
  * Overload status

> Only a small sample dataset is included. Full dataset is not provided due to size constraints.

---

## Tech Stack

* Python
* Deep Learning (PyTorch / TensorFlow)
* NumPy, Pandas
* Scikit-learn
* Matplotlib

---

## Current Status

* Notebook-based experimental implementation
* Full research pipeline available

Planned:

* Refactoring into modular Python code
* Training scripts (`train.py`)
* Model abstraction

---

## Author

**Mohammad Noaman Khan**
MSc Data Science
University of Milano-Bicocca

---

## Acknowledgements

Supervisor: Prof. Michele Ciavotta
Co-Supervisor: Dr.ssa Federica Filippini

