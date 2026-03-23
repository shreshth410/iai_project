# Credit Risk as a Zero-Sum Game
### Adversarial Search + Bayesian Reasoning on the Kaggle *Give Me Some Credit* Dataset

---

## Overview

This project frames credit lending as a **two-player zero-sum game** between a lender and an adversary (representing borrower stress/market conditions). Rather than using a standard ML classifier, it combines:

- A **Naive Bayes risk scorer** to estimate the probability of borrower default, and
- A **Minimax game engine with Alpha-Beta pruning** to determine the optimal lending action (approve, reject, or modify the interest rate) that maximises expected portfolio value.

The pipeline is evaluated on the [Kaggle Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset and benchmarks the efficiency gain of Alpha-Beta pruning over vanilla Minimax.

---

## Architecture

The codebase is split into 6 self-contained modules:

```
Module 1 — Environment & State     BorrowerState, LenderAction, state transitions
Module 2 — Bayesian Risk Scorer    Naive Bayes P(default | evidence)
Module 3 — Minimax Game Engine     Minimax + optional Alpha-Beta pruning
Module 4 — Data Loading            Preprocessing the Kaggle CSV
Module 5 — Evaluation              Benchmarking both engines on test set
Module 6 — Visualization           6-panel results dashboard (matplotlib)
```

---

## How It Works

### 1. State Representation
Each borrower is encoded as a `BorrowerState` dataclass containing their financial attributes (credit utilization, debt ratio, income, delinquency history, etc.) alongside the current loan offer (interest rate, amount, tenure).

### 2. Bayesian Risk Scorer
A `BayesianRiskScorer` is trained on the dataset to learn likelihoods for 10 binary evidence features (e.g. `revolving_high`, `times_90_late`, `low_income`). At inference time, it starts from the dataset's base default rate (~6.7%) and performs sequential Bayesian updates — one per triggered evidence condition — using the Naive Bayes independence assumption. Laplace smoothing prevents zero-probability issues.

### 3. Game Engine (Minimax + Alpha-Beta)
The `CreditRiskGameEngine` models lending as a two-player game:
- **Maximising player (Lender):** Chooses from 5 legal actions — reject, approve at base rate, or modify rate by −2%, +2%, or +5%.
- **Minimising player (Adversary):** Deterministically stresses the borrower in response to higher rates (increases revolving utilization proportionally), simulating how aggressive pricing raises default risk.

At each node, the Bayesian scorer re-evaluates P(default) given the updated state. The leaf node value is the **expected monetary value** of the loan: `(1 - P(default)) × interest earned − P(default) × expected loss`, with a penalty applied for rates above 25%.

Alpha-Beta pruning is toggled via a flag, enabling a direct benchmark comparison.

### 4. Evaluation
Both engines (with and without Alpha-Beta) are run on up to 200 test borrowers. The pipeline records per-borrower: decision taken, expected value, runtime (ms), and nodes explored. AUC is computed using the Bayesian P(default) scores against ground-truth labels.

### 5. Visualization
A 6-panel dashboard is saved as a PNG, covering:
- P(default) distribution by true class
- Nodes explored: with vs. without Alpha-Beta
- Runtime comparison
- Lender action distribution (pie chart)
- Risk vs. expected return scatter plot
- Portfolio value by decision threshold

---

## Setup & Usage

### Requirements

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Dataset

Download `cs-training.csv` from the [Kaggle Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) competition and update the path in the script.

### Running

```python
from credit_risk_pipeline import run_pipeline

summary = run_pipeline(
    csv_path="path/to/cs-training.csv",
    n_eval_samples=200,   # number of test borrowers to evaluate
    depth=4               # minimax search depth
)
```

Or directly:

```bash
python credit_risk_pipeline.py
```

> **Note:** Update the hardcoded paths for the CSV (`cs-training.csv`) and output PNG in `load_and_preprocess()` and `plot_results()` before running.

---

## Output

```
══════════════════════════════════════════════════
  RESULTS SUMMARY
══════════════════════════════════════════════════
  Bayesian AUC Score:       0.XXXX
  Avg time WITH  α-β:       X.XX ms
  Avg time WITHOUT α-β:     X.XX ms
  Speedup from pruning:     X.XXx
  Avg nodes WITH  α-β:      XX.X
  Avg nodes WITHOUT α-β:    XX.X
  Node reduction:           XX.X%
  Approval rate (α-β):      XX.X%
══════════════════════════════════════════════════
```

A results dashboard PNG is also saved to the configured output path.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Naive Bayes (not logistic regression) | Interpretable, sequential update, no matrix inversion |
| Minimax over a classifier | Models adversarial dynamics; rate changes affect future default risk |
| Deterministic adversary | Keeps the game tree tractable; adversary models market/behavioural stress |
| Alpha-Beta pruning | Reduces nodes explored significantly without changing the optimal decision |
| Expected value leaf scoring | Directly optimises for portfolio profitability, not just accuracy |

---

## File Structure

```
credit_risk_pipeline.py   # Full pipeline (single file)
README.md                 # This file
cs-training.csv           # Kaggle dataset (not included)
credit_risk_results.png   # Output dashboard (generated on run)
```

---

## Concepts Demonstrated

- Bayesian inference / Naive Bayes classification
- Minimax adversarial search
- Alpha-Beta pruning and search efficiency benchmarking
- Expected value modelling for financial decision-making
- Data preprocessing (imputation, outlier clipping, stratified split)
