!\[Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square\&logo=python)









!\[Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square\&logo=jupyter)









!\[XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red?style=flat-square)









!\[SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=flat-square)









!\[License](https://img.shields.io/badge/License-MIT-green?style=flat-square)









!\[Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)







\---



\# Policyholder Risk Stratification and Claim Probability Prediction

\## A Comparative Machine Learning Study | Research Portfolio Project



> \*\*Author:\*\* Whitney Kemuma

> \*\*Degree:\*\* BSc Actuarial Science

> \*\*Applying For:\*\* Research Master's in \[Data Science / Machine Leaning / Risk Analysis]

> \*\*Dataset:\*\* Porto Seguro Safe Driver Prediction — Kaggle (595,212 records)

> \*\*Tools:\*\* Python, Scikit-learn, XGBoost, SHAP, imbalanced-learn



\---



\## Abstract



Predicting whether a policyholder will file an insurance claim is one of the

most commercially significant problems in actuarial analytics. This project

builds and evaluates a machine learning pipeline for binary claim prediction

using the Porto Seguro Safe Driver dataset, a real world insurance dataset

with over 595,000 records.



Three models are trained and compared: Logistic Regression as an interpretable

statistical baseline, Random Forest as a tree-based ensemble, and XGBoost as

an advanced gradient boosting algorithm. Class imbalance (26:1 ratio) is

addressed using class\_weight='balanced' and scale\_pos\_weight. SHAP values

provide post-hoc explainability. Predicted probabilities are converted into

a structured 0–100 risk scoring system categorising policyholders as Low,

Medium, or High risk.



\---



\## Research Questions



| | Question |

|--|----------|

| \*\*RQ1\*\* | Which model achieves the highest discriminative performance for claim prediction on an imbalanced dataset? |

| \*\*RQ2\*\* | Which policyholder features are the strongest predictors of claim probability? |

| \*\*RQ3\*\* | How effectively can predicted probabilities be converted into an interpretable risk scoring system? |

| \*\*RQ4\*\* | Does class\_weight='balanced' meaningfully improve recall for the minority claim class? |



\---



\## Project Structure

policyholder-risk-prediction/

│

├── data/

│   ├── train.csv.zip                    <- Raw dataset (Porto Seguro, Kaggle)

│   ├── train\_cleaned.csv.gz             <- Cleaned dataset (Chapter 1 output)

│   ├── X\_train.csv.gz                   <- Training features

│   ├── X\_test.csv.gz                    <- Test features

│   ├── y\_train.csv.gz                   <- Training labels

│   ├── y\_test.csv.gz                    <- Test labels

│   ├── fig2\_1\_class\_imbalance.png       <- EDA outputs

│   ├── fig2\_2\_feature\_distributions.png

│   ├── fig2\_3\_correlation\_heatmap.png

│   ├── fig2\_4\_outlier\_boxplots.png

│   ├── fig4\_1\_confusion\_matrices.png    <- Evaluation outputs

│   ├── fig4\_2\_roc\_curves.png

│   ├── fig4\_3\_cv\_comparison.png

│   ├── fig4\_4\_feature\_importance.png

│   ├── fig5\_1\_risk\_distribution.png     <- Risk scoring outputs

│   ├── fig5\_2\_risk\_validation.png

│   ├── fig5\_3\_shap\_importance.png

│   ├── fig5\_4\_shap\_beeswarm.png

│   ├── fig5\_5\_shap\_waterfall.png

│   ├── model\_metrics.csv                <- Final metrics table

│   └── policyholder\_risk\_scores.csv     <- Final risk scores output

│

├── models/

│   ├── logistic\_regression.pkl          <- Trained Logistic Regression

│   ├── random\_forest.pkl                <- Trained Random Forest

│   ├── xgboost.pkl                      <- Trained XGBoost (final model)

│   ├── scaler.pkl                       <- Fitted StandardScaler

│   └── model\_parameters.csv            <- Parameters and CV scores

│

├── notebooks/

│   ├── Chapter\_0\_Research\_Foundation.ipynb  <- Abstract, Intro, Lit Review

│   ├── Chapter\_1\_Data.ipynb                 <- Data cleaning and preparation

│   ├── Chapter\_2\_EDA.ipynb                  <- Exploratory data analysis

│   ├── Chapter\_3\_Modeling.ipynb             <- Model training

│   ├── Chapter\_4\_Evaluation.ipynb           <- Model evaluation

│   └── Chapter\_5\_Risk\_Scoring.ipynb         <- Risk scoring and SHAP

│

├── src/

│   ├── data\_loader.py                   <- Data loading and pipeline functions

│   ├── eda\_utils.py                     <- EDA helper functions

│   ├── model\_utils.py                   <- Model loading and prediction functions

│   └── scoring\_utils.py                 <- Risk scoring and validation functions

│

├── README.md

└── requirements.txt

\---



\## Methodology



\### 1. Data Preparation

\- Raw data loaded from Porto Seguro Kaggle dataset (595,212 records)

\- Missing values encoded as -1 replaced with NaN and imputed using median/mode

\- Categorical features one-hot encoded

\- Engineered feature: `risk\_flag\_sum` — count of active binary risk indicators

\- Stratified 80/20 train-test split preserving 3.6% claim rate



\### 2. Class Imbalance Strategy

The dataset has a 26:1 class imbalance (no claim vs claim).

Rather than oversampling on a dataset of this scale, `class\_weight='balanced'`

is applied to Logistic Regression and Random Forest, and `scale\_pos\_weight`

is applied to XGBoost, adjusting each model's loss function to penalise

missed claims more heavily.



\### 3. Models Trained



| Model | Type | Imbalance Fix |

|-------|------|---------------|

| Logistic Regression | Statistical baseline (GLM equivalent) | class\_weight='balanced' |

| Random Forest | Bagging ensemble | class\_weight='balanced' |

| XGBoost | Gradient boosting | scale\_pos\_weight |



\### 4. Evaluation

Models evaluated on held out test set using AUC-ROC (primary), F1-score,

Precision, Recall and Accuracy. 5-fold stratified cross validation confirms

generalisation.



\### 5. Risk Scoring System

Risk Score = Predicted Probability x 100

0  - 30  →  Low Risk

30 - 60  →  Medium Risk

60 - 100 →  High Risk

\### 6. Explainability

SHAP TreeExplainer applied to XGBoost predictions:

\- Global importance — which features matter most overall

\- Beeswarm — direction and magnitude of feature impacts

\- Waterfall — individual High Risk policyholder explanation



\---



\## Results



> Full results available in `notebooks/Chapter\_4\_Evaluation.ipynb`

> and `data/model\_metrics.csv`



| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |

|-------|----------|-----------|--------|----------|---------|

| Logistic Regression | 0.6214 | 0.0530 | 0.5568 | 0.0968 | 0.6272 |

| Random Forest | 0.8780 | 0.0642 | 0.1729 | 0.0936 | 0.06076 |

| XGBoost | 0.6657 | 0.0565 | 0.5204 | 0.1019 | 0.6343 |






\---



\## How to Run



\### 1. Clone the repository

```bash

git clone https://github.com/whitneysdata/policyholder-risk-prediction.git

cd policyholder-risk-prediction

2\. Install dependencies

pip install -r requirements.txt

3\. Download the dataset

Download train.csv from

Porto Seguro Safe Driver Prediction

and place the zip file in the data/ folder as train.csv.zip

4\. Run notebooks in order

Chapter\_0\_Research\_Foundation.ipynb  <- Read only, no code

Chapter\_1\_Data.ipynb                 <- Run first

Chapter\_2\_EDA.ipynb                  <- Run second

Chapter\_3\_Modeling.ipynb             <- Run third (takes 10-20 min)

Chapter\_4\_Evaluation.ipynb           <- Run fourth

Chapter\_5\_Risk\_Scoring.ipynb         <- Run fifth

Each notebook is fully self-contained and can also be run independently.

Key Dependencies

| Library | Version | Purpose |

|---------|---------|---------|

| scikit-learn | >=1.3.0 | Models, metrics, preprocessing |

| xgboost | >=1.7.0 | Gradient boosting model |

| shap | >=0.42.0 | Model explainability |

| pandas | >=2.0.0 | Data manipulation |

| numpy | >=1.24.0 | Numerical operations |

| matplotlib | >=3.7.0 | Visualisation |

| seaborn | >=0.12.0 | Statistical plots |

| joblib | >=1.3.0 | Model persistence |

| imbalanced-learn | >=0.11.0 | SMOTE and imbalance tools |

References

Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

Chawla, N. V., et al. (2002). SMOTE. JAIR, 16, 321-357.

Chen, T., \& Guestrin, C. (2016). XGBoost. Proceedings of KDD 2016.

Frees, E. W., et al. (2014). Predictive Modeling in Actuarial Science. Cambridge.

Lundberg, S. M., \& Lee, S. I. (2017). A Unified Approach to Interpreting

Model Predictions. NeurIPS 2017.

Wüthrich, M. V. (2018). Machine Learning in Individual Claims Reserving.

Scandinavian Actuarial Journal.

License

This project is licensed under the MIT License.

See LICENSE for details.

This project was developed as part of a Research Master's scholarship

portfolio application in Actuarial Data Science.

\---



