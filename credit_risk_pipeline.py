"""
Credit Risk as a Zero-Sum Game
Adversarial Search + Bayesian Reasoning
Wired to: Kaggle - Give Me Some Credit Dataset
"""

import numpy as np
import pandas as pd
import math
import time
import copy
import warnings
from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("Done with imports.")

# ─────────────────────────────────────────────
# MODULE 1: ENVIRONMENT AND STATE DEFINITION
# ─────────────────────────────────────────────

@dataclass
class BorrowerState:
    borrower_id: str
    revolving_utilization: float   # RevolvingUtilizationOfUnsecuredLines
    age: int
    past_due_30_59: int            # NumberOfTime30-59DaysPastDueNotWorse
    debt_ratio: float              # DebtRatio
    monthly_income: float          # MonthlyIncome
    open_credit_lines: int         # NumberOfOpenCreditLinesAndLoans
    times_90_days_late: int        # NumberOfTimes90DaysLate
    real_estate_loans: int         # NumberRealEstateLoansOrLines
    past_due_60_89: int            # NumberOfTime60-89DaysPastDueNotWorse
    dependents: int                # NumberOfDependents
    # Game state
    interest_rate: float = 12.0   # % annual, starting offer
    loan_amount: float = 10000.0
    loan_tenure: int = 36          # months
    is_terminal: bool = False
    outcome: Optional[str] = None  # 'approved' | 'rejected' | 'default' | 'repaid'
    true_label: Optional[int] = None  # ground truth for evaluation

@dataclass
class LenderAction:
    action_type: str               # 'approve' | 'reject' | 'modify_rate'
    modified_rate: Optional[float] = None
    label: str = ""

    def __post_init__(self):
        if not self.label:
            if self.action_type == 'reject':
                self.label = "Reject"
            elif self.action_type == 'approve':
                self.label = f"Approve @ {self.interest_rate_display()}%"
            elif self.action_type == 'modify_rate':
                self.label = f"Offer {self.modified_rate:.1f}%"

    def interest_rate_display(self):
        return "base"


def get_legal_actions(state: BorrowerState) -> list:
    actions = [LenderAction('reject', label="Reject")]
    # Always can approve at base rate
    actions.append(LenderAction('approve', label=f"Approve @ {state.interest_rate:.1f}%"))
    # Can offer modified rates
    for delta in [-2.0, 2.0, 5.0]:
        new_rate = max(5.0, min(30.0, state.interest_rate + delta))
        actions.append(LenderAction(
            'modify_rate',
            modified_rate=new_rate,
            label=f"Modify rate → {new_rate:.1f}%"
        ))
    return actions


def state_from_row(row: pd.Series, idx: int) -> BorrowerState:
    """Convert a dataset row into a BorrowerState."""
    return BorrowerState(
        borrower_id=f"B{idx:06d}",
        revolving_utilization=float(row.get('RevolvingUtilizationOfUnsecuredLines', 0.5)),
        age=int(row.get('age', 40)),
        past_due_30_59=int(row.get('NumberOfTime30-59DaysPastDueNotWorse', 0)),
        debt_ratio=float(row.get('DebtRatio', 0.3)),
        monthly_income=float(row.get('MonthlyIncome', 5000)),
        open_credit_lines=int(row.get('NumberOfOpenCreditLinesAndLoans', 5)),
        times_90_days_late=int(row.get('NumberOfTimes90DaysLate', 0)),
        real_estate_loans=int(row.get('NumberRealEstateLoansOrLines', 1)),
        past_due_60_89=int(row.get('NumberOfTime60-89DaysPastDueNotWorse', 0)),
        dependents=int(row.get('NumberOfDependents', 0)),
        loan_amount=float(row.get('MonthlyIncome', 5000)) * 3,  # 3x monthly income
        true_label=int(row.get('SeriousDlqin2yrs', 0))
    )

# ─────────────────────────────────────────────
# MODULE 2: BAYESIAN RISK SCORER
# ─────────────────────────────────────────────

class BayesianRiskScorer:
    """
    Learns P(evidence | default) and P(evidence | repay) from training data,
    then uses Bayes' Theorem to compute and update P(default | evidence).
    """

    def __init__(self):
        self.likelihood_table: dict = {}
        self.base_rate: float = 0.067  # ~6.7% default rate in Give Me Some Credit

    def fit(self, df: pd.DataFrame):
        """
        Learn likelihoods from training data.
        df must contain 'SeriousDlqin2yrs' as target.
        """
        target = 'SeriousDlqin2yrs'
        defaults = df[df[target] == 1]
        repays   = df[df[target] == 0]
        self.base_rate = df[target].mean()

        def p_evidence(subset, condition_fn):
            return condition_fn(subset).mean()

        conditions = {
            'revolving_high':    lambda d: (d['RevolvingUtilizationOfUnsecuredLines'] > 0.75),
            'revolving_extreme': lambda d: (d['RevolvingUtilizationOfUnsecuredLines'] > 1.0),
            'past_due_30_59':    lambda d: (d['NumberOfTime30-59DaysPastDueNotWorse'] > 0),
            'past_due_60_89':    lambda d: (d['NumberOfTime60-89DaysPastDueNotWorse'] > 0),
            'times_90_late':     lambda d: (d['NumberOfTimes90DaysLate'] > 0),
            'debt_ratio_high':   lambda d: (d['DebtRatio'] > 0.4),
            'debt_ratio_extreme':lambda d: (d['DebtRatio'] > 1.0),
            'low_income':        lambda d: (d['MonthlyIncome'] < 3000),
            'age_young':         lambda d: (d['age'] < 30),
            'many_dependents':   lambda d: (d['NumberOfDependents'] > 2),
        }

        for key, fn in conditions.items():
            p_e_d = float(p_evidence(defaults, fn))
            p_e_r = float(p_evidence(repays, fn))
            # Laplace smoothing to avoid zero probabilities
            self.likelihood_table[key] = {
                'default': max(p_e_d, 1e-6),
                'repay':   max(p_e_r, 1e-6),
            }

        print(f"[Bayesian Scorer] Trained on {len(df):,} samples | "
              f"Base default rate: {self.base_rate:.3f}")
        print(f"[Bayesian Scorer] Learned {len(self.likelihood_table)} evidence features")

    def get_evidence_keys(self, state: BorrowerState) -> list:
        keys = []
        if state.revolving_utilization > 0.75:   keys.append('revolving_high')
        if state.revolving_utilization > 1.0:    keys.append('revolving_extreme')
        if state.past_due_30_59 > 0:             keys.append('past_due_30_59')
        if state.past_due_60_89 > 0:             keys.append('past_due_60_89')
        if state.times_90_days_late > 0:         keys.append('times_90_late')
        if state.debt_ratio > 0.4:               keys.append('debt_ratio_high')
        if state.debt_ratio > 1.0:               keys.append('debt_ratio_extreme')
        if state.monthly_income < 3000:          keys.append('low_income')
        if state.age < 30:                       keys.append('age_young')
        if state.dependents > 2:                 keys.append('many_dependents')
        return keys

    def update(self, prior: float, evidence_keys: list) -> float:
        """
        Sequential Bayesian update.
        P(D|e1,e2,...) updated one piece of evidence at a time.
        Assumes conditional independence of evidence given class (Naive Bayes).
        """
        p_default = prior
        for key in evidence_keys:
            if key not in self.likelihood_table:
                continue
            lh = self.likelihood_table[key]
            p_e_d = lh['default']
            p_e_r = lh['repay']
            numerator   = p_e_d * p_default
            denominator = numerator + p_e_r * (1 - p_default)
            if denominator > 0:
                p_default = numerator / denominator
        return float(np.clip(p_default, 0.001, 0.999))

    def score(self, state: BorrowerState) -> float:
        """Full pipeline: prior → update with all evidence → return P(default)."""
        evidence = self.get_evidence_keys(state)
        return self.update(self.base_rate, evidence)


# ─────────────────────────────────────────────
# MODULE 3: MIN-MAX + ALPHA-BETA ENGINE
# ─────────────────────────────────────────────

class CreditRiskGameEngine:
    def __init__(self, scorer: BayesianRiskScorer, max_depth: int = 4,
                 use_alpha_beta: bool = True):
        self.scorer = scorer
        self.max_depth = max_depth
        self.use_alpha_beta = use_alpha_beta
        self.nodes_explored = 0
        self.pruned_branches = 0

    def evaluate(self, state: BorrowerState, p_default: float) -> float:
        if state.outcome == 'rejected':
            return 0.0
        annual_rate = state.interest_rate / 100
        total_interest = state.loan_amount * annual_rate * (state.loan_tenure / 12)
        recovery_rate = 0.40
        expected_loss = state.loan_amount * (1 - recovery_rate)
        ev = (1 - p_default) * total_interest - p_default * expected_loss
        if state.interest_rate > 25.0:
            ev *= 0.85
        return ev

    def lender_transition(self, state: BorrowerState,
                          action: LenderAction) -> BorrowerState:
        """Lender picks a rate. Reject is terminal. Approve/modify is not — adversary responds next."""
        new_state = copy.deepcopy(state)
        if action.action_type == 'reject':
            new_state.is_terminal = True
            new_state.outcome = 'rejected'
            return new_state
        if action.action_type == 'modify_rate' and action.modified_rate is not None:
            new_state.interest_rate = action.modified_rate
        # approve keeps current rate, no change
        new_state.is_terminal = False
        new_state.outcome = None
        return new_state

    def adversary_transition(self, state: BorrowerState) -> BorrowerState:
        """Adversary deterministically stresses the borrower based on the rate. Terminal."""
        new_state = copy.deepcopy(state)
        stress = max(0, (new_state.interest_rate - 12) / 18)
        new_state.revolving_utilization = min(
            1.5, new_state.revolving_utilization + stress * 0.3
        )
        new_state.is_terminal = True
        new_state.outcome = 'approved'
        return new_state

    def minimax(self, state: BorrowerState, depth: int,
                alpha: float, beta: float,
                is_maximizing: bool, p_default: float) -> float:
        self.nodes_explored += 1

        if state.is_terminal or depth == 0:
            return self.evaluate(state, p_default)

        if is_maximizing:
            # ── LENDER'S TURN: pick best action from legal moves ──
            max_eval = -math.inf
            for action in get_legal_actions(state):
                next_state = self.lender_transition(state, action)
                new_evidence = self.scorer.get_evidence_keys(next_state)
                new_p = self.scorer.update(p_default, new_evidence)

                val = self.minimax(next_state, depth - 1,
                                   alpha, beta, False, new_p)
                max_eval = max(max_eval, val)
                if self.use_alpha_beta:
                    alpha = max(alpha, val)
                    if beta <= alpha:
                        self.pruned_branches += 1
                        break
            return max_eval

        else:
            # ── ADVERSARY'S TURN: single deterministic stress response ──
            # No action loop — adversary just reacts, no branching here
            next_state = self.adversary_transition(state)
            new_evidence = self.scorer.get_evidence_keys(next_state)
            new_p = self.scorer.update(p_default, new_evidence)

            val = self.minimax(next_state, depth - 1,
                               alpha, beta, True, new_p)
            return val

    def best_action(self, state: BorrowerState) -> tuple:
        p_default = self.scorer.score(state)
        self.nodes_explored = 0
        self.pruned_branches = 0

        best_val = -math.inf
        best_act = None
        action_values = []

        for action in get_legal_actions(state):
            next_state = self.lender_transition(state, action)
            new_evidence = self.scorer.get_evidence_keys(next_state)
            new_p = self.scorer.update(p_default, new_evidence)

            val = self.minimax(next_state, self.max_depth - 1,
                               -math.inf, math.inf,
                               False, new_p)
            action_values.append((action, val))

            if val > best_val:
                best_val = val
                best_act = action

        return best_act, best_val, p_default, self.nodes_explored, action_values


# ─────────────────────────────────────────────
# MODULE 4: LOADING AND PREPROCESSING DATA
# ─────────────────────────────────────────────

def load_and_preprocess(filepath: str) -> tuple:
    """
    Load Give Me Some Credit CSV and return train/test DataFrames.
    Expected columns match the Kaggle competition dataset.
    """
    filepath = r"D:\codes\Datasets\cs-training.csv"
    print(f"\n[Data] Loading from: {filepath}")
    df = pd.read_csv(filepath, index_col=0)
    print(f"[Data] Raw shape: {df.shape}")
    print(f"[Data] Columns: {list(df.columns)}")
    print(f"[Data] Default rate: {df['SeriousDlqin2yrs'].mean():.3f}")
    print(f"[Data] Missing values:\n{df.isnull().sum()}")

    # ── Imputation ──────────────────────────────────
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(0, inplace=True)

    # ── Clip extreme outliers ────────────────────────
    df['RevolvingUtilizationOfUnsecuredLines'] = df[
        'RevolvingUtilizationOfUnsecuredLines'].clip(0, 5)
    df['DebtRatio'] = df['DebtRatio'].clip(0, 5)
    df['MonthlyIncome'] = df['MonthlyIncome'].clip(0, 100000)

    # ── Cap late payment counters ────────────────────
    for col in ['NumberOfTime30-59DaysPastDueNotWorse',
                'NumberOfTime60-89DaysPastDueNotWorse',
                'NumberOfTimes90DaysLate']:
        df[col] = df[col].clip(0, 20)

    df.dropna(inplace=True)
    print(f"[Data] Clean shape: {df.shape}")

    train_df, test_df = train_test_split(df, test_size=0.2,
                                         stratify=df['SeriousDlqin2yrs'],
                                         random_state=42)
    print(f"[Data] Train: {len(train_df):,} | Test: {len(test_df):,}")
    return train_df, test_df


# ─────────────────────────────────────────────
# MODULE 5: EVALUATION & BENCHMARKING
# ─────────────────────────────────────────────

def evaluate_on_dataset(test_df: pd.DataFrame,
                         engine_ab: CreditRiskGameEngine,
                         engine_no_ab: CreditRiskGameEngine,
                         n_samples: int = 200) -> dict:
    """
    Run both engines on n_samples borrowers.
    Compare: decision quality, runtime, nodes explored.
    """
    sample = test_df.sample(n=min(n_samples, len(test_df)), random_state=42)
    results = []

    print(f"\n[Eval] Running on {len(sample)} borrowers...")
    for i, (idx, row) in enumerate(sample.iterrows()):
        state = state_from_row(row, i)

        # ── With Alpha-Beta ──
        t0 = time.perf_counter()
        act_ab, val_ab, p_def, nodes_ab, _ = engine_ab.best_action(state)
        time_ab = (time.perf_counter() - t0) * 1000

        # ── Without Alpha-Beta ──
        t0 = time.perf_counter()
        act_no, val_no, _, nodes_no, _ = engine_no_ab.best_action(state)
        time_no = (time.perf_counter() - t0) * 1000

        results.append({
            'borrower_id': state.borrower_id,
            'true_label': state.true_label,
            'p_default': p_def,
            'action_ab': act_ab.action_type,
            'action_no': act_no.action_type,
            'value_ab': val_ab,
            'value_no': val_no,
            'time_ab_ms': time_ab,
            'time_no_ms': time_no,
            'nodes_ab': nodes_ab,
            'nodes_no': nodes_no,
            'pruned': engine_ab.pruned_branches,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)} processed...")

    results_df = pd.DataFrame(results)

    # ── Derive predictions ───────────────────────────
    # If action is 'reject' → predict default=1, else predict based on p_default
    results_df['pred_ab'] = (
        (results_df['action_ab'] == 'reject') |
        (results_df['p_default'] > 0.3)
    ).astype(int)

    results_df['pred_no'] = (
        (results_df['action_no'] == 'reject') |
        (results_df['p_default'] > 0.3)
    ).astype(int)

    # ── Metrics ─────────────────────────────────────
    auc = roc_auc_score(results_df['true_label'], results_df['p_default'])

    summary = {
        'n_samples': len(results_df),
        'auc': auc,
        'avg_time_ab_ms': results_df['time_ab_ms'].mean(),
        'avg_time_no_ms': results_df['time_no_ms'].mean(),
        'speedup': results_df['time_no_ms'].mean() / results_df['time_ab_ms'].mean(),
        'avg_nodes_ab': results_df['nodes_ab'].mean(),
        'avg_nodes_no': results_df['nodes_no'].mean(),
        'node_reduction_pct': (1 - results_df['nodes_ab'].mean() /
                               results_df['nodes_no'].mean()) * 100,
        'avg_pruned': results_df['pruned'].mean(),
        'approve_rate_ab': (results_df['action_ab'] != 'reject').mean(),
        'approve_rate_no': (results_df['action_no'] != 'reject').mean(),
        'results_df': results_df,
    }

    print(f"\n{'═'*50}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═'*50}")
    print(f"  Bayesian AUC Score:       {auc:.4f}")
    print(f"  Avg time WITH  α-β:       {summary['avg_time_ab_ms']:.2f} ms")
    print(f"  Avg time WITHOUT α-β:     {summary['avg_time_no_ms']:.2f} ms")
    print(f"  Speedup from pruning:     {summary['speedup']:.2f}x")
    print(f"  Avg nodes WITH  α-β:      {summary['avg_nodes_ab']:.1f}")
    print(f"  Avg nodes WITHOUT α-β:    {summary['avg_nodes_no']:.1f}")
    print(f"  Node reduction:           {summary['node_reduction_pct']:.1f}%")
    print(f"  Approval rate (α-β):      {summary['approve_rate_ab']:.1%}")
    print(f"{'═'*50}")

    return summary



# ─────────────────────────────────────────────
# MODULE 6: VISUALIZATION
# ─────────────────────────────────────────────

def plot_results(summary: dict, save_path: str = r"D:\codes\Python\credit_risk_results.png"):
    results_df = summary['results_df']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#0f0f1a')
    for ax in axes.flat:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#aaaacc')
        ax.spines['bottom'].set_color('#333355')
        ax.spines['left'].set_color('#333355')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    COLOR_AB  = '#00d4ff'
    COLOR_NO  = '#ff6b6b'
    COLOR_BAY = '#ffd700'

    # ── 1. P(default) distribution ────────────────────
    ax = axes[0, 0]
    defaults = results_df[results_df['true_label'] == 1]['p_default']
    repays   = results_df[results_df['true_label'] == 0]['p_default']
    ax.hist(repays,   bins=25, alpha=0.7, color='#44ff88', label='Repaid',  density=True)
    ax.hist(defaults, bins=25, alpha=0.7, color='#ff4444', label='Default', density=True)
    ax.set_title('Bayesian P(default) Distribution', color='white', fontsize=11)
    ax.set_xlabel('P(default)', color='#aaaacc')
    ax.set_ylabel('Density', color='#aaaacc')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')

    # ── 2. Nodes explored: with vs without α-β ─────────
    ax = axes[0, 1]
    x = np.arange(2)
    vals = [summary['avg_nodes_ab'], summary['avg_nodes_no']]
    bars = ax.bar(['With α-β\nPruning', 'Without α-β\nPruning'],
                  vals, color=[COLOR_AB, COLOR_NO], width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.0f}', ha='center', va='bottom', color='white', fontsize=11)
    ax.set_title('Avg Nodes Explored', color='white', fontsize=11)
    ax.set_ylabel('Nodes', color='#aaaacc')

    # ── 3. Runtime comparison ───────────────────────────
    ax = axes[0, 2]
    vals = [summary['avg_time_ab_ms'], summary['avg_time_no_ms']]
    bars = ax.bar(['With α-β\nPruning', 'Without α-β\nPruning'],
                  vals, color=[COLOR_AB, COLOR_NO], width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.2f}ms', ha='center', va='bottom', color='white', fontsize=11)
    ax.set_title('Avg Execution Time', color='white', fontsize=11)
    ax.set_ylabel('Time (ms)', color='#aaaacc')

    # ── 4. Action distribution (α-β) ───────────────────
    ax = axes[1, 0]
    action_counts = results_df['action_ab'].value_counts()
    wedge_colors = ['#00d4ff', '#ff6b6b', '#ffd700', '#44ff88']
    wedges, texts, autotexts = ax.pie(
        action_counts.values,
        labels=action_counts.index,
        autopct='%1.1f%%',
        colors=wedge_colors[:len(action_counts)],
        textprops={'color': 'white'}
    )
    ax.set_title('Lender Action Distribution (α-β)', color='white', fontsize=11)

    # ── 5. P(default) vs Expected Value scatter ─────────
    ax = axes[1, 1]
    sc = ax.scatter(results_df['p_default'], results_df['value_ab'],
                    c=results_df['true_label'], cmap='RdYlGn_r',
                    alpha=0.6, s=20)
    ax.axvline(0.3, color='white', linestyle='--', alpha=0.5, label='Decision threshold')
    ax.set_xlabel('P(default)', color='#aaaacc')
    ax.set_ylabel('Expected Value (£)', color='#aaaacc')
    ax.set_title('Risk vs Expected Return', color='white', fontsize=11)
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    plt.colorbar(sc, ax=ax, label='True Label')

    # ── 6. Cumulative value by threshold ───────────────
    ax = axes[1, 2]
    thresholds = np.linspace(0, 1, 50)
    cumulative_vals = []
    for thresh in thresholds:
        approved = results_df[results_df['p_default'] < thresh]
        cumulative_vals.append(approved['value_ab'].sum())
    ax.plot(thresholds, cumulative_vals, color=COLOR_BAY, linewidth=2)
    best_thresh = thresholds[np.argmax(cumulative_vals)]
    ax.axvline(best_thresh, color=COLOR_AB, linestyle='--',
               label=f'Optimal threshold: {best_thresh:.2f}')
    ax.set_xlabel('Default Threshold', color='#aaaacc')
    ax.set_ylabel('Total Portfolio Value (£)', color='#aaaacc')
    ax.set_title('Portfolio Value by Decision Threshold', color='white', fontsize=11)
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)

    fig.suptitle('Credit Risk as a Zero-Sum Game — Results Dashboard',
                 color='white', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    print(f"\n[Plot] Saved to: {save_path}")
    plt.close()

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(csv_path: str, n_eval_samples: int = 200, depth: int = 4):
    print("=" * 60)
    print(" CREDIT RISK AS A ZERO-SUM GAME")
    print(" Adversarial Search + Bayesian Reasoning")
    print("=" * 60)

    # 1. Load data
    train_df, test_df = load_and_preprocess(csv_path)

    # 2. Train Bayesian scorer
    scorer = BayesianRiskScorer()
    scorer.fit(train_df)

    # 3. Build engines
    engine_with_ab = CreditRiskGameEngine(scorer, max_depth=depth, use_alpha_beta=True)
    engine_no_ab   = CreditRiskGameEngine(scorer, max_depth=depth, use_alpha_beta=False)

    # 4. Demo: single borrower walkthrough
    demo_row = test_df.sample(1, random_state=7).iloc[0]
    demo_state = state_from_row(demo_row, 0)

    print(f"\n[Demo] Borrower profile:")
    print(f"  Credit utilization:  {demo_state.revolving_utilization:.2f}")
    print(f"  Age:                 {demo_state.age}")
    print(f"  Debt ratio:          {demo_state.debt_ratio:.2f}")
    print(f"  Monthly income:      £{demo_state.monthly_income:,.0f}")
    print(f"  Times 90 days late:  {demo_state.times_90_days_late}")
    print(f"  True label:          {'DEFAULT' if demo_state.true_label == 1 else 'REPAID'}")

    best_act, best_val, p_def, nodes, action_vals = engine_with_ab.best_action(demo_state)
    print(f"\n[Demo] Bayesian P(default): {p_def:.4f}")
    print(f"[Demo] All action values:")
    for act, val in sorted(action_vals, key=lambda x: -x[1]):
        print(f"  {act.label:<30} → £{val:>10,.2f}")
    print(f"\n[Demo] BEST ACTION: {best_act.label}")
    print(f"[Demo] Expected value: £{best_val:,.2f}")
    print(f"[Demo] Nodes explored: {nodes}")

    # 5. Full evaluation
    summary = evaluate_on_dataset(test_df, engine_with_ab, engine_no_ab,
                                   n_samples=n_eval_samples)

    # 6. Plot
    plot_results(summary, save_path=r"D:\codes\Python\credit_risk_results.png")

    return summary


if __name__ == "__main__":
    csv_path = r"D:\codes\Datasets\cs-training.csv"
    run_pipeline(csv_path, n_eval_samples=200, depth=4)