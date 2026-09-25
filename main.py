"""
Churn Prediction Pipeline — single-file version.
Запуск: python main.py
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix, PrecisionRecallDisplay
)
import lightgbm as lgb
import optuna
import shap
from joblib import dump

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = 'data/Telco-Customer-Churn.csv'
MODEL_PATH = 'models/lgbm_churn.joblib'


def load_data(path: str):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    df = df.drop('customerID', axis=1)
    
    df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['IsNewCustomer'] = (df['tenure'] < 6).astype(int)
    
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                    'ChargesPerMonth', 'IsNewCustomer']
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f'[data] {len(df)} rows, {X.shape[1]} features, churn={y.mean():.3f}')
    return X_train, X_test, y_train, y_test, list(X.columns)


def evaluate_model(y_true, y_proba, name: str):
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        'pr_auc': average_precision_score(y_true, y_proba),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'f1': f1_score(y_true, y_pred),
    }
    print(f'\n[{name}] PR-AUC={metrics["pr_auc"]:.4f} ROC-AUC={metrics["roc_auc"]:.4f} F1={metrics["f1"]:.4f}')
    print(classification_report(y_true, y_pred, target_names=['Stay', 'Churn'], digits=3))
    return metrics


def train_baselines(X_train, X_test, y_train, y_test):
    results = {}
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results['LogReg'] = lr.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, results['LogReg'], 'Logistic Regression')
    
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results['RandomForest'] = rf.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, results['RandomForest'], 'Random Forest')
    return results


def train_lgbm_optuna(X_train, X_test, y_train, y_test, n_trials: int = 50):
    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'average_precision', 'boosting_type': 'gbdt',
            'n_estimators': 2000, 'learning_rate': trial.suggest_float('lr', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 200),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'is_unbalance': True, 'random_state': 42, 'verbose': -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        return average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print(f'\n[optuna] best PR-AUC: {study.best_value:.4f}')
    
    best_params = study.best_params
    best_params.update({'objective': 'binary', 'metric': 'average_precision', 'boosting_type': 'gbdt',
                        'n_estimators': 2000, 'is_unbalance': True, 'random_state': 42, 'verbose': -1})
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    y_proba = model.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_proba, 'LightGBM + Optuna')
    
    os.makedirs('models', exist_ok=True)
    dump(model, MODEL_PATH)
    print(f'[saved] {MODEL_PATH}')
    return model, y_proba


def plot_results(results: dict, y_test, feature_names, lgbm_model, X_test):
    # PR-кривые
    plt.figure(figsize=(9, 6))
    for name, y_proba in results.items():
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, name=name)
    plt.title('Precision-Recall Curves')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('pr_curves.png', dpi=150)
    plt.close()
    print('[saved] pr_curves.png')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, (results['LightGBM'] >= 0.5).astype(int))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Stay', 'Pred Churn'],
                yticklabels=['True Stay', 'True Churn'])
    plt.title('Confusion Matrix (LightGBM)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print('[saved] confusion_matrix.png')
    
    # SHAP (новый API)
    explainer = shap.TreeExplainer(lgbm_model)
    shap_explanation = explainer(X_test)
    
    if len(shap_explanation.shape) == 3:
        shap_values_for_plot = shap_explanation.values[:, :, 1]
    else:
        shap_values_for_plot = shap_explanation.values
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_for_plot, X_test, feature_names=feature_names, 
                      show=False, max_display=15)
    plt.title('SHAP Feature Importance (Churn)')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150)
    plt.close()
    print('[saved] shap_summary.png')


if __name__ == '__main__':
    print('='*60)
    print('Churn Prediction Pipeline')
    print('='*60)
    
    X_train, X_test, y_train, y_test, feature_names = load_data(DATA_PATH)
    baseline_results = train_baselines(X_train, X_test, y_train, y_test)
    lgbm_model, lgbm_proba = train_lgbm_optuna(X_train, X_test, y_train, y_test, n_trials=50)
    
    all_results = {'LogReg': baseline_results['LogReg'], 
                   'RandomForest': baseline_results['RandomForest'], 
                   'LightGBM': lgbm_proba}
    plot_results(all_results, y_test, feature_names, lgbm_model, X_test)
    
    print('\n' + '='*60)
    print('DONE. Files: pr_curves.png, confusion_matrix.png, shap_summary.png')
    print('='*60)