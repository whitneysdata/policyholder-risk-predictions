import numpy as np
import pandas as pd


def assign_risk_category(score: float) -> str:
    """
    Map a numeric risk score (0-100) to a risk category label.

    Thresholds:
        Low Risk    : 0  - 30
        Medium Risk : 30 - 60
        High Risk   : 60 - 100

    Parameters
    ----------
    score : float — risk score between 0 and 100

    Returns
    -------
    str — risk category label
    """
    if score < 30:
        return 'Low Risk'
    elif score < 60:
        return 'Medium Risk'
    else:
        return 'High Risk'


def build_risk_df(proba: np.ndarray,
                  y_true: pd.Series) -> pd.DataFrame:
    """
    Build a risk score dataframe from predicted probabilities.

    Converts raw model probabilities to 0-100 risk scores,
    assigns category labels and includes actual target for
    validation purposes.

    Parameters
    ----------
    proba  : np.ndarray — predicted probabilities (claim class)
    y_true : pd.Series  — actual target labels

    Returns
    -------
    pd.DataFrame with columns:
        Risk_Score, Risk_Category, Actual_Target
    """
    risk_scores = proba * 100

    return pd.DataFrame({
        'Risk_Score'    : risk_scores.round(2),
        'Risk_Category' : [assign_risk_category(s) for s in risk_scores],
        'Actual_Target' : y_true.values
    })


def validate_risk_scores(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate risk scoring system by computing actual claim rate
    per risk category.

    A valid scoring system should show monotonically increasing
    claim rates from Low to Medium to High Risk.

    Parameters
    ----------
    risk_df : pd.DataFrame — output of build_risk_df()

    Returns
    -------
    pd.DataFrame — claim rate per category with pass/fail flag
    """
    category_order = ['Low Risk', 'Medium Risk', 'High Risk']

    validation = (
        risk_df.groupby('Risk_Category')['Actual_Target']
        .agg(['mean', 'count', 'sum'])
        .reindex(category_order)
        .reset_index()
    )
    validation.columns = ['Risk Category', 'Claim Rate',
                          'Total', 'Claims']
    validation['Claim Rate %'] = (
        validation['Claim Rate'] * 100
    ).round(2)

    # Check monotonic increase from Low to Medium to High Risk
    # Pass = claim rate is higher than previous category
    # Review = claim rate is not higher — thresholds may need adjustment
    rates = validation['Claim Rate'].values
    validation['Monotonic'] = [
        'Pass' if i == 0 or rates[i] > rates[i-1] else 'Review'
        for i in range(len(rates))
    ]

    return validation[['Risk Category', 'Total',
                        'Claims', 'Claim Rate %', 'Monotonic']]


def risk_score_summary(risk_df: pd.DataFrame) -> None:
    """
    Print a formatted summary of risk score distribution.

    Parameters
    ----------
    risk_df : pd.DataFrame — output of build_risk_df()
    """
    category_order = ['Low Risk', 'Medium Risk', 'High Risk']
    counts = risk_df['Risk_Category'].value_counts().reindex(
        category_order
    )
    total = len(risk_df)

    print("Risk Score Distribution:")
    print(f"   {'Category':<15} {'Count':>8} {'Percentage':>12}")
    print("   " + "-" * 37)
    for cat, count in counts.items():
        pct = count / total * 100
        print(f"   {cat:<15} {count:>8,} {pct:>11.1f}%")
    print(f"   {'TOTAL':<15} {total:>8,} {'100.0%':>12}")
    print(f"\n   Mean score   : {risk_df['Risk_Score'].mean():.2f}")
    print(f"   Median score : {risk_df['Risk_Score'].median():.2f}")
    print(f"   Std deviation: {risk_df['Risk_Score'].std():.2f}")
    