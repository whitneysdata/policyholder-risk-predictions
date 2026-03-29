import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summarise_imbalance(y: pd.Series) -> pd.DataFrame:
    """
    Print and return a summary of class imbalance in target variable.

    Parameters
    ----------
    y : pd.Series — target variable (binary 0/1)

    Returns
    -------
    pd.DataFrame — counts and percentages per class
    """
    counts = y.value_counts()
    pcts   = (y.value_counts(normalize=True) * 100).round(2)

    summary = pd.DataFrame({
        'Class'    : ['No Claim (0)', 'Claim (1)'],
        'Count'    : counts.values,
        'Percentage': pcts.values
    })

    rio = counts[0] / counts[1]
    print("  Class Imbalance Summary:")
    print(summary.to_string(index=False))
    print(f"\n   Imbalance ratio : {ratio:.1f}:1")
    print(f"   Strategy        : class_weight='balanced'")

    return summary


def get_top_corr_features(df: pd.DataFrame,
                           target: str = 'target',
                           n: int = 20) -> list:
    """
    Return the top N features most correlated with the target.

    Parameters
    ----------
    df     : pd.DataFrame — full dataset including target
    target : str          — name of target column
    n      : int          — number of top features to return

    Returns
    -------
    list of column names including target
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    top_cols     = (
        df[numeric_cols]
        .corr()[target]
        .abs()
        .sort_values(ascending=False)
        .head(n + 1)      # +1 to include target itself
        .index
        .tolist()
    )
    return top_cols


def flag_high_correlations(df: pd.DataFrame,
                            cols: list,
                            threshold: float = 0.7) -> pd.DataFrame:
    """
    Identify feature pairs with correlation above a threshold.

    Useful for detecting multicollinearity before Logistic
    Regression, highly correlated features can destabilise
    coefficient estimates.

    Parameters
    ----------
    df        : pd.DataFrame — dataset
    cols      : list         — columns to check
    threshold : float        — correlation threshold (default 0.7)

    Returns
    -------
    pd.DataFrame — pairs exceeding the threshold
    """
    # Exclude target if present
    check_cols  = [c for c in cols if c != 'target']
    corr_matrix = df[check_cols].corr().abs()
    upper       = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    pairs = [
        {'Feature 1': col, 'Feature 2': row,
         'Correlation': round(upper.loc[row, col], 4)}
        for col in upper.columns
        for row in upper.index
        if upper.loc[row, col] > threshold
    ]

    result = pd.DataFrame(pairs).sort_valu(
        'Correlation', ascending=False
    ).reset_index(drop=True)

    if len(result) > 0:
        print(f"  {len(reult)} feature pairs exceed r = {threshold}:")
        print(result.to_string(index=False))
    else:
        print(f" No feature pairs exceed correlation threshold of {threshold}.")

    return result


def count_outliers_iqr(df: pd.DataFrame,
                        cols: list) -> pd.DataFrame:
    """
    Count outliers in each column using the Tukey IQR fence method.

    An outlier is any value below Q1 - 1.5*IQR or
    above Q3 + 1.5*IQR.

    Parameters
    ----------
    df   : pd.DataFrame — dataset
    cols : list         — columns to check

    Returns
    -------
    pd.DataFrame — outlier counts and percentages per column
    """
    results = []
    for col in cols:
        q1    = df[col].quantile(0.25)
        q3    = df[col].quantile(0.75)
        iqr   = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        results.append({
            'Feature'       : col,
            'Outlier Count' : n_out,
            'Outlier %'     : round(n_out / len(df) * 100, 2),
            'Lower Fence'   : round(lower, 4),
            '
    Upper Fence'   : round(upper, 4)
        })

    return pd.DataFrame(results).sort_values(
        'Outlier %', ascending=False
    ).reset_index(drop=True)
        