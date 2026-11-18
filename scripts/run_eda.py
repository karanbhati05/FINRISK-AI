#!/usr/bin/env python
"""
FinRisk AI — Exploratory Data Analysis Script
==============================================
Runs the full EDA on the UCI Default of Credit Card Clients dataset.
Generates all plots (saved to outputs/eda_plots/), prints statistical test
results, and writes a summary report to outputs/eda_report.txt.
"""

import os
import sys
import warnings
import textwrap
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "eda_plots"
REPORT_PATH = PROJECT_ROOT / "outputs" / "eda_report.txt"
RAW_CSV = DATA_RAW / "credit_default.csv"
RAW_XLS = DATA_RAW / "credit_default.xls"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW.mkdir(parents=True, exist_ok=True)

# ── styling ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

# report buffer
report_lines: list[str] = []


def report(text: str = "", end: str = "\n") -> None:
    """Print and buffer a line for the report file."""
    print(text, end=end)
    report_lines.append(text + end)


def save_fig(name: str) -> None:
    """Save the current figure and close."""
    path = PLOTS_DIR / f"{name}.png"
    plt.savefig(path)
    plt.close()
    report(f"  [plot saved → {path.relative_to(PROJECT_ROOT)}]")


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def download_data() -> None:
    """Download the UCI xls file and convert to CSV."""
    if RAW_CSV.exists():
        report(f"Dataset already exists at {RAW_CSV.relative_to(PROJECT_ROOT)}")
        return

    report("Downloading dataset …")
    if not RAW_XLS.exists():
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, str(RAW_XLS))
        report(f"  Downloaded → {RAW_XLS.relative_to(PROJECT_ROOT)}")

    # The file has a header row at row 0 with real column names and
    # an extra first row (row 1 in Excel is the header row for the actual data).
    df = pd.read_excel(RAW_XLS, header=1, engine="xlrd")
    df.to_csv(RAW_CSV, index=False)
    report(f"  Converted to CSV → {RAW_CSV.relative_to(PROJECT_ROOT)}")


def load_data() -> pd.DataFrame:
    download_data()
    df = pd.read_csv(RAW_CSV)
    # Rename target column for convenience
    if "default.payment" in " ".join(df.columns):
        # find the actual target column name (may vary)
        target_cols = [c for c in df.columns if "default" in c.lower() and c != "ID"]
        if target_cols:
            df = df.rename(columns={target_cols[0]: "default"})
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})
    # Drop ID if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

def section_data_overview(df: pd.DataFrame) -> None:
    report("\n" + "=" * 72)
    report("SECTION 1: DATA OVERVIEW")
    report("=" * 72)

    report(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    report(f"\nData types:\n{df.dtypes.to_string()}")
    report(f"\nFirst 5 rows:\n{df.head().to_string()}")

    buf = StringIO()
    df.info(buf=buf, memory_usage="deep")
    report(f"\nInfo & memory usage:\n{buf.getvalue()}")

    report(f"\nDescriptive statistics:\n{df.describe().T.to_string()}")

    # Target distribution
    target_counts = df["default"].value_counts()
    target_pct = df["default"].value_counts(normalize=True) * 100
    report("\nTarget distribution (default):")
    for val in sorted(target_counts.index):
        label = "Default" if val == 1 else "No Default"
        report(f"  {label} ({val}): {target_counts[val]:,} ({target_pct[val]:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(["No Default (0)", "Default (1)"],
                  [target_counts[0], target_counts[1]], color=colors, edgecolor="black")
    for bar, cnt, pct in zip(bars, [target_counts[0], target_counts[1]],
                              [target_pct[0], target_pct[1]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Target Variable Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    save_fig("01_target_distribution")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MISSING VALUES & DATA QUALITY
# ════════════════════════════════════════════════════════════════════════════

def section_data_quality(df: pd.DataFrame) -> None:
    report("\n" + "=" * 72)
    report("SECTION 2: MISSING VALUES & DATA QUALITY")
    report("=" * 72)

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    report("\nMissing values per column:")
    if missing.sum() == 0:
        report("  ✓ No missing values found!")
    else:
        for col in missing[missing > 0].index:
            report(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")

    # Duplicates
    n_dup = df.duplicated().sum()
    report(f"\nDuplicate rows: {n_dup}")

    # Data type validation
    report("\nData type validation:")
    expected_cats = {"SEX": {1, 2}, "EDUCATION": {1, 2, 3, 4},
                     "MARRIAGE": {1, 2, 3}}
    for col, expected in expected_cats.items():
        actual = set(df[col].unique())
        unexpected = actual - expected
        if unexpected:
            report(f"  ⚠ {col}: unexpected categories {unexpected} "
                   f"(expected {sorted(expected)})")
        else:
            report(f"  ✓ {col}: categories OK — {sorted(actual)}")

    # Analyze undocumented categories
    report("\nUndocumented category analysis:")
    for col in ["EDUCATION", "MARRIAGE"]:
        report(f"\n  {col} value counts:")
        vc = df[col].value_counts().sort_index()
        for val, cnt in vc.items():
            pct = cnt / len(df) * 100
            flag = " ← undocumented" if val not in expected_cats[col] else ""
            report(f"    {val}: {cnt:>6,} ({pct:>5.1f}%){flag}")

    # Undocumented categories default rate
    report("\n  Default rate by EDUCATION (including undocumented):")
    ed_default = df.groupby("EDUCATION")["default"].mean() * 100
    for val, rate in ed_default.items():
        flag = " ← undocumented" if val not in {1, 2, 3, 4} else ""
        report(f"    EDUCATION={val}: {rate:.1f}% default rate{flag}")

    report("\n  Default rate by MARRIAGE (including undocumented):")
    mar_default = df.groupby("MARRIAGE")["default"].mean() * 100
    for val, rate in mar_default.items():
        flag = " ← undocumented" if val not in {1, 2, 3} else ""
        report(f"    MARRIAGE={val}: {rate:.1f}% default rate{flag}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — UNIVARIATE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def section_univariate(df: pd.DataFrame) -> None:
    report("\n" + "=" * 72)
    report("SECTION 3: UNIVARIATE ANALYSIS")
    report("=" * 72)

    numeric_cols = ["LIMIT_BAL", "AGE"] + \
                   [f"BILL_AMT{i}" for i in range(1, 7)] + \
                   [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_cols = [f"PAY_{i}" for i in range(0, 7)]
    # check column naming (PAY_0 vs PAY_1 etc.)
    if "PAY_0" not in df.columns and "PAY_1" in df.columns:
        pay_cols = [f"PAY_{i}" for i in range(1, 7)]
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]

    # ── histograms with KDE ──
    n = len(numeric_cols)
    ncols_plot = 4
    nrows_plot = (n + ncols_plot - 1) // ncols_plot
    fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(20, 4 * nrows_plot))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue", bins=50)
            axes[i].set_title(col, fontsize=11)
            axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distributions of Numeric Features", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("03_numeric_distributions")

    # ── categorical bar plots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels_map = {
        "SEX": {1: "Male", 2: "Female"},
        "EDUCATION": {0: "Unknown", 1: "Grad School", 2: "University",
                      3: "High School", 4: "Other", 5: "Unknown5", 6: "Unknown6"},
        "MARRIAGE": {0: "Unknown", 1: "Married", 2: "Single", 3: "Other"},
    }
    for idx, col in enumerate(categorical_cols):
        vc = df[col].value_counts().sort_index()
        labels = [labels_map[col].get(v, str(v)) for v in vc.index]
        bars = axes[idx].bar(labels, vc.values, color=sns.color_palette("Set2", len(vc)),
                             edgecolor="black")
        axes[idx].set_title(col, fontsize=13, fontweight="bold")
        axes[idx].set_ylabel("Count")
        for bar, cnt in zip(bars, vc.values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           f"{cnt:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save_fig("03_categorical_distributions")

    # ── box plots ──
    fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(20, 4 * nrows_plot))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        if col in df.columns:
            sns.boxplot(x=df[col], ax=axes[i], color="lightcoral")
            axes[i].set_title(col, fontsize=11)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Box Plots — Outlier Identification", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("03_boxplots")

    # ── skewness & kurtosis ──
    report("\nSkewness & Kurtosis:")
    report(f"  {'Feature':<20} {'Skewness':>10} {'Kurtosis':>10}")
    report(f"  {'-'*20} {'-'*10} {'-'*10}")
    all_numeric = numeric_cols + [c for c in pay_cols if c in df.columns]
    for col in all_numeric:
        if col in df.columns:
            sk = df[col].skew()
            ku = df[col].kurtosis()
            report(f"  {col:<20} {sk:>10.3f} {ku:>10.3f}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BIVARIATE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def section_bivariate(df: pd.DataFrame) -> None:
    report("\n" + "=" * 72)
    report("SECTION 4: BIVARIATE ANALYSIS")
    report("=" * 72)

    labels_map = {
        "SEX": {1: "Male", 2: "Female"},
        "EDUCATION": {0: "Unk-0", 1: "Grad", 2: "Uni", 3: "HS", 4: "Other",
                      5: "Unk-5", 6: "Unk-6"},
        "MARRIAGE": {0: "Unk-0", 1: "Married", 2: "Single", 3: "Other"},
    }

    # ── default rate by categorical features ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, col in enumerate(["SEX", "EDUCATION", "MARRIAGE"]):
        grouped = df.groupby(col)["default"].mean() * 100
        labels = [labels_map[col].get(v, str(v)) for v in grouped.index]
        bars = axes[idx].bar(labels, grouped.values,
                             color=sns.color_palette("RdYlGn_r", len(grouped)),
                             edgecolor="black")
        for bar, rate in zip(bars, grouped.values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                           f"{rate:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        axes[idx].set_title(f"Default Rate by {col}", fontsize=13, fontweight="bold")
        axes[idx].set_ylabel("Default Rate (%)")
    plt.tight_layout()
    save_fig("04_default_rate_by_categorical")

    report("\nDefault rate by categorical features:")
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        report(f"\n  {col}:")
        grouped = df.groupby(col)["default"].agg(["mean", "count"])
        for val, row in grouped.iterrows():
            lbl = labels_map[col].get(val, str(val))
            report(f"    {lbl} ({val}): {row['mean']*100:.1f}% default  (n={row['count']:,})")

    # ── default rate by AGE groups ──
    df["AGE_GROUP"] = pd.cut(df["AGE"], bins=[20, 30, 40, 50, 60, 80],
                             labels=["21-30", "31-40", "41-50", "51-60", "61-80"],
                             right=True)
    age_default = df.groupby("AGE_GROUP", observed=True)["default"].agg(["mean", "count"])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(age_default.index.astype(str), age_default["mean"] * 100,
                  color=sns.color_palette("viridis", len(age_default)), edgecolor="black")
    for bar, (_, row) in zip(bars, age_default.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{row['mean']*100:.1f}%\n(n={row['count']:,})",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Default Rate by Age Group", fontsize=14, fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Age Group")
    save_fig("04_default_rate_by_age")

    report("\nDefault rate by AGE group:")
    for grp, row in age_default.iterrows():
        report(f"  {grp}: {row['mean']*100:.1f}%  (n={row['count']:,})")

    # ── default rate by PAY_0 to PAY_6 ──
    pay_cols = [c for c in df.columns if c.startswith("PAY_") and c.replace("PAY_", "").isdigit()]
    pay_cols = sorted(pay_cols, key=lambda x: int(x.split("_")[1]))

    fig, ax = plt.subplots(figsize=(12, 6))
    pay_default_rates = {}
    for col in pay_cols:
        grouped = df.groupby(col)["default"].mean() * 100
        pay_default_rates[col] = grouped
    # plot mean default rate for each delay value
    delay_vals = sorted(set().union(*[set(v.index) for v in pay_default_rates.values()]))
    for col in pay_cols:
        rates = [pay_default_rates[col].get(d, np.nan) for d in delay_vals]
        ax.plot(delay_vals, rates, marker="o", label=col, linewidth=2)
    ax.set_title("Default Rate by Payment Delay Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Repayment Status (delay months)")
    ax.set_ylabel("Default Rate (%)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlim(-3, 9)
    plt.tight_layout()
    save_fig("04_default_rate_by_payment_delay")

    report(f"\nDefault rate by PAY_0 (most recent month):")
    if "PAY_0" in df.columns:
        g = df.groupby("PAY_0")["default"].agg(["mean", "count"])
        for val, row in g.iterrows():
            report(f"  PAY_0={val}: {row['mean']*100:.1f}%  (n={row['count']:,})")

    # ── correlation matrix ──
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["AGE_GROUP"], errors="ignore")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7},
                vmin=-1, vmax=1)
    ax.set_title("Pearson Correlation Matrix", fontsize=15, fontweight="bold")
    save_fig("04_correlation_matrix")

    # top correlations with target
    target_corr = corr["default"].drop("default").abs().sort_values(ascending=False)
    report("\nTop 10 features correlated with target (|r|):")
    for col, r in target_corr.head(10).items():
        direction = corr.loc[col, "default"]
        report(f"  {col:<20} r = {direction:+.4f} (|r| = {r:.4f})")

    df.drop(columns=["AGE_GROUP"], inplace=True, errors="ignore")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — OUTLIER DETECTION
# ════════════════════════════════════════════════════════════════════════════

def section_outliers(df: pd.DataFrame) -> dict:
    report("\n" + "=" * 72)
    report("SECTION 5: OUTLIER DETECTION")
    report("=" * 72)

    outlier_cols = (["LIMIT_BAL"] +
                    [f"BILL_AMT{i}" for i in range(1, 7)] +
                    [f"PAY_AMT{i}" for i in range(1, 7)])
    outlier_cols = [c for c in outlier_cols if c in df.columns]

    # IQR method
    report("\nIQR-based outlier detection:")
    report(f"  {'Feature':<18} {'Q1':>12} {'Q3':>12} {'IQR':>12} {'Outliers':>10} {'%':>7}")
    report(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*7}")
    iqr_results = {}
    for col in outlier_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_out / len(df) * 100
        iqr_results[col] = {"n": n_out, "pct": pct}
        report(f"  {col:<18} {q1:>12,.0f} {q3:>12,.0f} {iqr:>12,.0f} {n_out:>10,} {pct:>6.1f}%")

    # Z-score method
    report("\nZ-score outlier detection (|z| > 3):")
    report(f"  {'Feature':<18} {'Mean':>14} {'Std':>14} {'Outliers':>10} {'%':>7}")
    report(f"  {'-'*18} {'-'*14} {'-'*14} {'-'*10} {'-'*7}")
    zscore_results = {}
    for col in outlier_cols:
        z = np.abs(stats.zscore(df[col].dropna()))
        n_out = (z > 3).sum()
        pct = n_out / len(df) * 100
        zscore_results[col] = {"n": n_out, "pct": pct}
        report(f"  {col:<18} {df[col].mean():>14,.0f} {df[col].std():>14,.0f} "
               f"{n_out:>10,} {pct:>6.1f}%")

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    # IQR
    iqr_pcts = [iqr_results[c]["pct"] for c in outlier_cols]
    axes[0].barh(outlier_cols, iqr_pcts, color="salmon", edgecolor="black")
    axes[0].set_title("Outlier Percentage (IQR Method)", fontweight="bold")
    axes[0].set_xlabel("% Outliers")
    # Z-score
    z_pcts = [zscore_results[c]["pct"] for c in outlier_cols]
    axes[1].barh(outlier_cols, z_pcts, color="cornflowerblue", edgecolor="black")
    axes[1].set_title("Outlier Percentage (Z-score > 3)", fontweight="bold")
    axes[1].set_xlabel("% Outliers")
    plt.tight_layout()
    save_fig("05_outlier_percentages")

    return iqr_results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — STATISTICAL HYPOTHESIS TESTING
# ════════════════════════════════════════════════════════════════════════════

def section_hypothesis_tests(df: pd.DataFrame) -> dict:
    report("\n" + "=" * 72)
    report("SECTION 6: STATISTICAL HYPOTHESIS TESTING")
    report("=" * 72)
    results = {}

    alpha = 0.05

    # ── Chi-squared tests ──
    report("\n── Chi-Squared Tests of Independence ──")
    for col in ["EDUCATION", "SEX", "MARRIAGE"]:
        ct = pd.crosstab(df[col], df["default"])
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        n = ct.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        sig = "SIGNIFICANT" if p < alpha else "NOT significant"
        results[f"chi2_{col}"] = {"chi2": chi2, "p": p, "cramers_v": cramers_v, "sig": sig}
        report(f"\n  H0: Default is independent of {col}")
        report(f"  χ²({dof}) = {chi2:.2f}, p = {p:.2e}")
        report(f"  Cramér's V = {cramers_v:.4f} (effect size)")
        report(f"  → {sig} at α={alpha}")

    # ── Mann-Whitney U test: LIMIT_BAL ──
    report("\n── Mann-Whitney U Test: LIMIT_BAL by Default Status ──")
    group0 = df.loc[df["default"] == 0, "LIMIT_BAL"]
    group1 = df.loc[df["default"] == 1, "LIMIT_BAL"]
    u_stat, p_mw = stats.mannwhitneyu(group0, group1, alternative="two-sided")
    # rank-biserial correlation as effect size
    n1, n2 = len(group0), len(group1)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)
    sig = "SIGNIFICANT" if p_mw < alpha else "NOT significant"
    results["mw_limit_bal"] = {"U": u_stat, "p": p_mw, "r_rb": r_rb, "sig": sig}
    report(f"  H0: LIMIT_BAL distribution is the same for defaulters and non-defaulters")
    report(f"  U = {u_stat:,.0f}, p = {p_mw:.2e}")
    report(f"  Rank-biserial r = {r_rb:.4f} (effect size)")
    report(f"  Non-defaulter mean: {group0.mean():,.0f}  |  Defaulter mean: {group1.mean():,.0f}")
    report(f"  → {sig} at α={alpha}")

    # ── T-test: LIMIT_BAL ──
    report("\n── Welch's T-test: LIMIT_BAL by Default Status ──")
    t_stat, p_t = stats.ttest_ind(group0, group1, equal_var=False)
    # Cohen's d
    pooled_std = np.sqrt(((n1 - 1) * group0.std()**2 + (n2 - 1) * group1.std()**2) / (n1 + n2 - 2))
    cohens_d = (group0.mean() - group1.mean()) / pooled_std
    sig = "SIGNIFICANT" if p_t < alpha else "NOT significant"
    results["ttest_limit_bal"] = {"t": t_stat, "p": p_t, "cohens_d": cohens_d, "sig": sig}
    report(f"  t = {t_stat:.4f}, p = {p_t:.2e}")
    report(f"  Cohen's d = {cohens_d:.4f} (effect size)")
    report(f"  → {sig} at α={alpha}")

    # ── ANOVA: Default rate across AGE groups ──
    report("\n── Kruskal-Wallis (non-parametric ANOVA): Default by AGE Group ──")
    df["AGE_GROUP"] = pd.cut(df["AGE"], bins=[20, 30, 40, 50, 60, 80],
                             labels=["21-30", "31-40", "41-50", "51-60", "61-80"])
    groups = [grp["default"].values for _, grp in df.groupby("AGE_GROUP", observed=True)]
    h_stat, p_kw = stats.kruskal(*groups)
    # eta-squared approximation
    n_total = len(df)
    k = len(groups)
    eta_sq = (h_stat - k + 1) / (n_total - k)
    sig = "SIGNIFICANT" if p_kw < alpha else "NOT significant"
    results["kruskal_age"] = {"H": h_stat, "p": p_kw, "eta_sq": eta_sq, "sig": sig}
    report(f"  H0: Default rate is the same across all AGE groups")
    report(f"  H = {h_stat:.4f}, p = {p_kw:.2e}")
    report(f"  η² (approx) = {eta_sq:.4f}")
    report(f"  → {sig} at α={alpha}")

    df.drop(columns=["AGE_GROUP"], inplace=True, errors="ignore")
    return results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FEATURE RELATIONSHIPS
# ════════════════════════════════════════════════════════════════════════════

def section_feature_relationships(df: pd.DataFrame) -> None:
    report("\n" + "=" * 72)
    report("SECTION 7: FEATURE RELATIONSHIPS")
    report("=" * 72)

    pay_cols = sorted([c for c in df.columns if c.startswith("PAY_") and
                       c.replace("PAY_", "").isdigit()],
                      key=lambda x: int(x.split("_")[1]))

    # ── Payment delay trends ──
    report("\nMean payment delay by month:")
    delay_means = df[pay_cols].mean()
    for col, m in delay_means.items():
        report(f"  {col}: {m:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # By default status
    for label, grp in df.groupby("default"):
        means = grp[pay_cols].mean()
        lbl = "Default" if label == 1 else "No Default"
        color = "#e74c3c" if label == 1 else "#2ecc71"
        axes[0].plot(range(len(pay_cols)), means.values, marker="o",
                     label=lbl, color=color, linewidth=2.5)
    axes[0].set_xticks(range(len(pay_cols)))
    axes[0].set_xticklabels(pay_cols, rotation=45)
    axes[0].set_title("Mean Payment Delay by Month & Default Status",
                       fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Mean Repayment Status")
    axes[0].legend()
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Overall trend
    axes[1].bar(range(len(pay_cols)), delay_means.values,
                color=sns.color_palette("coolwarm", len(pay_cols)), edgecolor="black")
    axes[1].set_xticks(range(len(pay_cols)))
    axes[1].set_xticklabels(pay_cols, rotation=45)
    axes[1].set_title("Average Payment Delay Trend", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Mean Repayment Status")
    plt.tight_layout()
    save_fig("07_payment_delay_trends")

    # ── Bill amount vs payment amount ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i in range(1, 7):
        bill_col = f"BILL_AMT{i}"
        pay_col = f"PAY_AMT{i}"
        if bill_col in df.columns and pay_col in df.columns:
            # sample for speed
            sample = df.sample(min(5000, len(df)), random_state=42)
            axes[i-1].scatter(sample[bill_col], sample[pay_col],
                              c=sample["default"], cmap="RdYlGn_r",
                              alpha=0.3, s=8)
            axes[i-1].set_xlabel(bill_col)
            axes[i-1].set_ylabel(pay_col)
            axes[i-1].set_title(f"{bill_col} vs {pay_col}", fontsize=11)
    fig.suptitle("Bill Amount vs Payment Amount (colored by default)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("07_bill_vs_payment_scatter")

    # ── Utilization ratio ──
    report("\nUtilization Ratio Analysis (BILL_AMT1 / LIMIT_BAL):")
    df["util_ratio"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)
    report(f"  Mean utilization: {df['util_ratio'].mean():.4f}")
    report(f"  Median utilization: {df['util_ratio'].median():.4f}")

    for label in [0, 1]:
        lbl = "Default" if label == 1 else "No Default"
        mean_u = df.loc[df["default"] == label, "util_ratio"].mean()
        report(f"  {lbl} mean utilization: {mean_u:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, color, lbl in [(0, "#2ecc71", "No Default"), (1, "#e74c3c", "Default")]:
        subset = df.loc[df["default"] == label, "util_ratio"].clip(-1, 3)
        sns.kdeplot(subset, ax=axes[0], color=color, label=lbl, fill=True, alpha=0.3)
    axes[0].set_title("Utilization Ratio Distribution by Default", fontweight="bold")
    axes[0].set_xlabel("Utilization Ratio (BILL_AMT1 / LIMIT_BAL)")
    axes[0].legend()

    # Binned utilization
    df["util_bin"] = pd.cut(df["util_ratio"], bins=[-np.inf, 0, 0.3, 0.6, 1.0, np.inf],
                            labels=["<0", "0-0.3", "0.3-0.6", "0.6-1.0", ">1.0"])
    util_default = df.groupby("util_bin", observed=True)["default"].agg(["mean", "count"])
    bars = axes[1].bar(util_default.index.astype(str), util_default["mean"] * 100,
                       color=sns.color_palette("YlOrRd", len(util_default)), edgecolor="black")
    for bar, (_, row) in zip(bars, util_default.iterrows()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{row['mean']*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_title("Default Rate by Utilization Ratio", fontweight="bold")
    axes[1].set_ylabel("Default Rate (%)")
    axes[1].set_xlabel("Utilization Ratio Bin")
    plt.tight_layout()
    save_fig("07_utilization_ratio")

    report("\n  Default rate by utilization bin:")
    for grp, row in util_default.iterrows():
        report(f"    {grp}: {row['mean']*100:.1f}% (n={row['count']:,})")

    df.drop(columns=["util_ratio", "util_bin"], inplace=True, errors="ignore")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — KEY FINDINGS SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def section_summary(df: pd.DataFrame, test_results: dict) -> None:
    report("\n" + "=" * 72)
    report("SECTION 8: KEY FINDINGS SUMMARY")
    report("=" * 72)

    report("""
KEY FINDINGS
============

1. CLASS IMBALANCE
   - Target variable is imbalanced: ~22% defaults vs ~78% non-defaults.
   - Models should use SMOTE, class weights, or stratified sampling.

2. PAYMENT DELAY (PAY_0) IS THE STRONGEST PREDICTOR
   - PAY_0 (most recent repayment status) has the highest correlation with default.
   - Customers with PAY_0 >= 2 (2+ months delay) have dramatically higher default rates.
   - Defaulters have consistently higher mean payment delays across all months.

3. CREDIT LIMIT MATTERS
   - LIMIT_BAL is significantly lower for defaulters (Mann-Whitney p ≈ 0, Welch's t-test p ≈ 0).
   - Lower credit limits associate with higher default risk.

4. EDUCATION AND MARRIAGE HAVE UNDOCUMENTED CATEGORIES
   - EDUCATION has values 0, 5, 6 not documented; MARRIAGE has value 0.
   - These should be merged into 'Other' category during preprocessing.

5. BILL_AMT FEATURES ARE HIGHLY CORRELATED
   - BILL_AMT1–6 are strongly inter-correlated (r > 0.8).
   - Dimensionality reduction or selective feature inclusion recommended.

6. UTILIZATION RATIO IS INFORMATIVE
   - Higher credit utilization (BILL_AMT / LIMIT_BAL) correlates with higher default.
   - Recommend engineering utilization ratio as a feature.

7. PAY_AMT FEATURES HAVE HEAVY-TAILED DISTRIBUTIONS
   - Most payment amounts are small, with extreme outliers.
   - Log-transformation recommended for PAY_AMT features.

FEATURE ENGINEERING RECOMMENDATIONS
====================================
- Create utilization ratio: BILL_AMT1 / LIMIT_BAL
- Create average and max payment delay across PAY_0 to PAY_6
- Create payment-to-bill ratio: PAY_AMT1 / BILL_AMT1
- Create total_bill and total_payment across months
- Create has_delayed_payment binary flag
- Create num_delayed_months count
- Bin AGE into groups (risk varies by age)
- Log-transform PAY_AMT and BILL_AMT to handle skew

MODELING RECOMMENDATIONS
========================
- Use stratified k-fold cross-validation
- Handle class imbalance with SMOTE or class_weight='balanced'
- Primary metric: AUC-ROC (robust to class imbalance)
- Secondary metrics: F1-score, precision-recall AUC
- Try ensemble methods: Random Forest, XGBoost, LightGBM
- Logistic Regression as baseline
- Watch for multicollinearity in BILL_AMT features
""")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    report("=" * 72)
    report("FinRisk AI — Exploratory Data Analysis Report")
    report(f"Generated: {pd.Timestamp.now()}")
    report("=" * 72)

    df = load_data()
    report(f"\nDataset loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    section_data_overview(df)
    section_data_quality(df)
    section_univariate(df)
    section_bivariate(df)
    section_outliers(df)
    test_results = section_hypothesis_tests(df)
    section_feature_relationships(df)
    section_summary(df, test_results)

    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"\n✓ Report saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"✓ Plots saved to {PLOTS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
