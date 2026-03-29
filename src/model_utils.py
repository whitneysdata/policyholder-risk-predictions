import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_all_models(models_folder: str = '../models'):
    """
    Load all trained models and scaler saved by Chapter 3.

    Parameters
    ----------
    models_folder : str
        Path to models folder. Default '../models' works
        when calling from the notebooks/ folder.

    Returns
    -------
    dict with keys:
        lr_model  — trained Logistic Regression
        rf_model  — trained Random Forest
        xgb_model — trained XGBoost
        scaler    — fitted StandardScaler
    """
    files = {
        'lr_model' : 'logistic_regression.pkl',
        'rf_model' : 'random_forest.pkl',
        'xgb_model': 'xgboost.pkl',
        'scaler'   : 'scaler.pkl'
    }

    # Verify all files exist before loading
    for key, filename in files.items():
        path = os.path.join(models_folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f" Model file not found: {path}\n"
                f"   Run Chapter 3 first to train and save all models."
            )

    # Load all models
    models = {
        key: joblib.load(os.path.join(models_folder, filename))
        for key, filename in files.items()
    }

    print(" All models loaded from models/ folder.")
    for key, filename in files.items():
        print(f"   {key:<12} ← {filename}")

    return models


def get_predictions(models: dict,
                    X_test: pd.DataFrame,
                    X_test_scaled: np.ndarray):
    """
    Generate class predictions and probabilities from all models.

    Logistic Regression uses scaled features.
    Random Forest and XGBoost use unscaled features.

    Parameters
    ----------
    models        : dict       — output of load_all_models()
    X_test        : DataFrame  — unscaled test features
    X_test_scaled : np.ndarray — scaled test features

    Returns
    -------
    dict with prediction arrays for each model
    """
    return {
        # Logistic Regression — scaled features
        'lr_pred'  : models['lr_model'].predict(X_test_scaled),
        'lr_proba' : models['lr_model'].predict_proba(X_test_scaled)[:, 1],

        # Random Forest — unscaled features
        'rf_pred'  : models['rf_model'].predict(X_test),
        'rf_proba' : models['rf_model'].predict_proba(X_test)[:, 1],

        # XGBoost — unscaled features
        'xgb_pred' : models['xgb_model'].predict(X_test),
        'xgb_proba': models['xgb_model'].predict_proba(X_test)[:, 1]
    }


def get_model_params_summary(models_folder: str = '../models'):
    """
    Load and display the saved model parameters CSV.

    Parameters
    ----------
    models_folder : str — path to models folder

    Returns
    -------
    pd.DataFrame — model parameters and CV scores
    """
    path = os.path.join(models_folder, 'model_parameters.csv')

    if not os.path.exists(path):
        print("  model_parameters.csv not found. Run Chapter 3 first.")
        return None

    params_df = pd.read_csv(path)
    print(" Model Parameters Summary:")
    print(params_df.to_string(index=False))
    return params_df

def build_metrics_table(y_true, predictions: dict) -> pd.DataFrame:
    """
    Build a full metrics comparison table for all three models.

    Parameters
    ----------
    y_true      : array-like — true test labels
    predictions : dict — output of get_predictions()

    Returns
    -------
    pd.DataFrame — metrics table with one row per model
    """
    from sklearn.metrics import (
        accuracy_score, precision_score,
        recall_score, f1_score, roc_auc_score
    )

    rows = []
    for name, pred_key, proba_key in [
        ('Logistic Regression', 'lr_pred',  'lr_proba'),
        ('Random Forest',       'rf_pred',  'rf_proba'),
        ('XGBoost',             'xgb_pred', 'xgb_proba')
    ]:
        rows.append({
            'Model'    : name,
            'Accuracy' : round(accuracy_score(y_true, predictions[pred_key]), 4),
            'Precision': round(precision_score(y_true, predictions[pred_key],
                                               zero_division=0), 4),
            'Recall'   : round(recall_score(y_true, predictions[pred_key],
                                            zero_division=0), 4),
            'F1-Score' : round(f1_score(y_true, predictions[pred_key],
                                        zero_division=0), 4),
            'AUC-ROC'  : round(roc_auc_score(y_true, predictions[proba_key]), 4)
        })

    return pd.DataFrame(rows)
    