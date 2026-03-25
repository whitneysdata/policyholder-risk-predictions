# ============================================================
# src/data_loader.py
# ============================================================
# Reusable data loading and preparation functions for the
# Policyholder Risk Scoring project.
#
# Import into any notebook with:
#   import sys
#   sys.path.append('../src')
#   from data_loader import run_full_pipeline
# ============================================================

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw_data(data_folder: str = '../data') -> pd.DataFrame:
    """
    Load raw Porto Seguro training data from the data folder.

    The raw file is a zip archive (train.csv.zip) containing
    a single CSV file. pandas reads directly from the zip
    without requiring manual extraction.

    Parameters
    ----------
    data_folder : str
        Relative path to the data folder from the calling notebook.
        Default is '../data' which works when calling from notebooks/

    Returns
    -------
    pd.DataFrame
        Raw dataset as loaded from zip.
    """
    # Look for zip first, fall back to plain CSV if not found
    zip_path = os.path.join(data_folder, 'train.csv.zip')
    csv_path = os.path.join(data_folder, 'train.csv')

    if os.path.exists(zip_path):
        df = pd.read_csv(zip_path, compression='zip')
        print(f"Loaded from zip : {zip_path}")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded from CSV : {csv_path}")
    else:
        raise FileNotFoundError(
            f" No data file found in '{data_folder}'.\n"
            f"   Expected: train.csv.zip or train.csv\n"
            f"   Check your data folder path."
        )

    print(f" Raw data shape  : {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Porto Seguro dataset.

    Steps:
    1. Drop the 'id' column (row identifier, not a feature)
    2. Replace -1 with NaN (Porto Seguro's missing value encoding)
    3. Impute missing values:
       - Numeric     → median (robust to outliers)
       - Categorical → mode   (most frequent value)
    4. Remove duplicate rows

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # Drop ID — row identifier with no predictive value
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    # Replace Porto Seguro's -1 missing encoding with NaN
    df.replace(-1, np.nan, inplace=True)

    # Impute missing values
    # Numeric  → median (robust to outliers in insurance data)
    # Object   → mode   (most frequent category)
    imputed = []
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
            imputed.append(col)

    # Remove exact duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)

    print(f" Data cleaned.")
    print(f"   Imputed {len(imputed)} columns | Removed {removed} duplicates")
    print(f"   Clean shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the cleaned dataset.

    Steps:
    1. One-hot encode all '_cat' categorical columns
       drop_first=True avoids the dummy variable trap
    2. Create 'risk_flag_sum' by summing all '_bin' binary
       columns per row — higher value means more risk
       indicators are active for that policyholder

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe ready for modelling.
    """
    df = df.copy()

    # One-hot encode categorical columns
    cat_cols = [col for col in df.columns if col.endswith('_cat')]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Aggregate binary risk flags into a single count feature
    bin_cols = [col for col in df.columns if col.endswith('_bin')]
    df['risk_flag_sum'] = df[bin_cols].sum(axis=1)

    print(f" Feature engineering complete.")
    print(f"   Encoded {len(cat_cols)} categorical columns")
    print(f"   Created risk_flag_sum from {len(bin_cols)} binary columns")
    print(f"   Final shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame,
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Split into stratified train and test sets.

    Stratification preserves the ~3.6% claim rate in both
    the training and test sets — critical for imbalanced data.

    Parameters
    ----------
    df           : pd.DataFrame — feature-engineered dataframe
    test_size    : float        — test proportion (default 0.2)
    random_state : int          — reproducibility seed

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y                  # Preserves class ratio
    )

    print(f" Train-test split complete.")
    print(f"   Train : {X_train.shape[0]:,} rows")
    print(f"   Test  : {X_test.shape[0]:,} rows")
    print(f"   Class imbalance strategy: class_weight='balanced' at model stage")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame):
    """
    Apply StandardScaler to features.

    Fit on training data only — applying to test data separately
    prevents data leakage. Required for Logistic Regression.
    Not required for Random Forest or XGBoost (tree-based).

    Parameters
    ----------
    X_train : training features (pd.DataFrame or np.ndarray)
    X_test  : test features     (pd.DataFrame or np.ndarray)

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform
    X_test_scaled  = scaler.transform(X_test)        # Transform only

    print(f" Scaling complete.")
    print(f"   Fit on training data only (no data leakage)")
    return X_train_scaled, X_test_scaled, scaler


def run_full_pipeline(data_folder: str = '../data'):
    """
    Run the complete data preparation pipeline in one call.

    Chains: load → clean → engineer → split → scale

    Used at the top of Chapters 2 through 5 to reproduce the
    full pipeline without repeating individual steps.

    Parameters
    ----------
    data_folder : str
        Path to data folder. Default '../data' works when
        calling from the notebooks/ folder.

    Returns
    -------
    dict with keys:
        df              — full cleaned + engineered dataframe
        X_train         — unscaled training features
        X_test          — unscaled test features
        y_train         — training labels
        y_test          — test labels
        X_train_scaled  — scaled training features (for LR)
        X_test_scaled   — scaled test features     (for LR)
        scaler          — fitted StandardScaler object
    """
    df = load_raw_data(data_folder)
    df = clean_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test
    )

    print(f"\n Full pipeline complete. Ready for modelling.")

    return {
        'df'            : df,
        'X_train'       : X_train,
        'X_test'        : X_test,
        'y_train'       : y_train,
        'y_test'        : y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled' : X_test_scaled,
        'scaler'        : scaler
    }


def load_cleaned_data(data_folder: str = '../data'):
    """
    Load the saved cleaned and split data from Chapter 1.

    Used by Chapters 2-5 to skip rerunning the full pipeline.
    Reads compressed .csv.gz files — pandas handles these
    automatically, no manual decompression needed.

    Parameters
    ----------
    data_folder : str
        Path to data folder. Default '../data' works when
        calling from the notebooks/ folder.

    Returns
    -------
    dict with keys:
        df, X_train, X_test, y_train, y_test
    """
    print(" Loading cleaned data from data folder...")

    df = pd.read_csv(
        os.path.join(data_folder, 'train_cleaned.csv.gz'),
        compression='gzip'
    )
    X_train = pd.read_csv(
        os.path.join(data_folder, 'X_train.csv.gz'),
        compression='gzip'
    )
    X_test = pd.read_csv(
        os.path.join(data_folder, 'X_test.csv.gz'),
        compression='gzip'
    )
    y_train = pd.read_csv(
        os.path.join(data_folder, 'y_train.csv.gz'),
        compression='gzip'
    ).squeeze()   # Convert single column DataFrame to Series
    y_test = pd.read_csv(
        os.path.join(data_folder, 'y_test.csv.gz'),
        compression='gzip'
    ).squeeze()   # Convert single column DataFrame to Series

    print(f" Cleaned data loaded.")
    print(f"   Full dataset : {df.shape}")
    print(f"   X_train      : {X_train.shape}")
    print(f"   X_test       : {X_test.shape}")
    print(f"   y_train      : {y_train.shape}")
    print(f"   y_test       : {y_test.shape}")

    return {
        'df'     : df,
        'X_train': X_train,
        'X_test' : X_test,
        'y_train': y_train,
        'y_test' : y_test
    }