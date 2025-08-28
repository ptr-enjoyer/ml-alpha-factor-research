import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
import seaborn as sns
from xgboost import XGBClassifier
import itertools
from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings('ignore')

#------------------------------------------------------------------------------

def make_classification_target(close: pd.Series, horizon: int) -> pd.DataFrame:
    """
    Create forward return and binary classification target from a price series.

    Parameters
    ----------
    close : pd.Series
        Series of closing prices, indexed by date.
    horizon : int, default 5
        Forecast horizon in days (e.g., 1 = next day, 5 = next week).

    Returns
    -------
    target_df : pd.DataFrame
        DataFrame with:
            'futret_<horizon>' : forward return over the horizon
            'target'           : binary label (1 if up, 0 if down or flat)
    """
    # Forward return
    futret = close.shift(-horizon) / close - 1
    futret_name = f"futret_{horizon}"

    # Binary target: 1 if forward return > 0, else 0
    target = (futret > 0).astype(int)

    # Combine into DataFrame
    target_df = pd.DataFrame({
        futret_name: futret,
        "target": target
    }, index=close.index)

    # Drop last horizon rows with NaNs in forward return
    target_df = target_df.dropna(subset=["target"])

    return target_df


def train_test_split_time_series(X: pd.DataFrame, y: pd.Series, train_size=0.6, val_size=0.2) -> tuple:
    """
    Split features and target into training, validation, and test sets based on time.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame.
    y : pd.Series
        Target Series.
    train_size : float, default 0.6
        Proportion of data to use for training.
    val_size : float, default 0.2
        Proportion of data to use for validation.

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
        Split feature DataFrames.
    y_train, y_val, y_test : pd.Series
        Split target Series.
    """
    n = len(X)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

def importance_df(pipe, X_val, y_val, n_repeats, random_state=17) -> pd.DataFrame:
    """ Calculate and plot permutation feature importance.
    Parameters:
        pipe: Trained sklearn Pipeline with model.
        X_val (pd.DataFrame): Validation feature set.
        y_val (pd.Series): Validation target.
        n_repeats (int): Number of permutation repeats.
        random_state (int): Random seed for reproducibility.
    Returns:
        pd.DataFrame: DataFrame with feature importances.
    """
    result = permutation_importance(pipe, X_val, y_val, n_repeats) # Calculates permutation importance and stores in result
    perm_df =  pd.DataFrame({
        'feature': X_val.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(by='importance_mean', ascending=False)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(perm_df['feature'], perm_df['importance_mean'], xerr=perm_df['importance_std'])
    ax.set_xlabel('Mean Importance')
    ax.set_title('Permutation Feature Importance')
    plt.show()
    
    return perm_df


def correlation_heatmap(X_train: pd.DataFrame):
    """ Plot correlation heatmap of features.
    Parameters:
        X_train (pd.DataFrame): Training feature set.
    """
    corr = X_train.corr() # Compute correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Feature Correlation Heatmap')
    plt.show()


def feat_select(corr_mat: pd.DataFrame, importances: pd.DataFrame, threshold: float = 0.85) -> list:
    """
    Select features based on permutation importance.
    Parameters
    ----------
    importances : pd.DataFrame
        DataFrame with 'feature', 'importance', and 'std' columns.
    threshold : float, default 0.85
    Returns
    -------
    selected_features : list
        List of selected feature names and dropout features.
    """
    features = importances.iloc[:,0]
    feat_set = set(features)
    drop_features = []

    while feat_set:
        feat = feat_set.pop()
        sieve = corr_mat[feat] > threshold # Booleans that correspond to features correlated with current feature 
        high_corr = corr_mat.index[sieve] # Highly correlated features
        high_corr_set = set(high_corr) # Set of highly correlated features
        high_corr_set.add(feat)
        sub_features  = importances[importances['feature'].isin(high_corr_set)] # Subset of features that are highly correlated with the current feature
        best_feature = sub_features.sort_values(by='importance', ascending=False).iloc[0] # Best feature based on importance
        high_corr_set.remove(best_feature['feature'])
        drop_features.extend(list(high_corr_set))
        feat_set -= high_corr_set

    selected_features = [feat for feat in features if feat not in drop_features]
    return selected_features, drop_features


def optimise_params(hyper_params, X_train_corr, y_train, X_val_corr, y_val,
                        max_depth=3, subsample=0.8, cosample_bytree=0.8,
                        use_label_encoder=False, eval_metric='logloss'):
    results = []
    
    for params in tqdm_notebook(hyper_params):
        lr, n_est = params
        pipe = Pipeline([
            ('xgb', XGBClassifier(n_estimators=n_est,
                                  learning_rate=lr,
                                  max_depth=max_depth,
                                  subsample=subsample,
                                  colsample_bytree=cosample_bytree,
                                  use_label_encoder=use_label_encoder,
                                  eval_metric=eval_metric))
        ])
        
        pipe.fit(X_train_corr, y_train)
        y_pred_val_prob = pipe.predict_proba(X_val_corr)[:,1]
        val_acc = accuracy_score(y_val, (y_pred_val_prob > 0.55).astype(int))
        roc_auc = roc_auc_score(y_val, y_pred_val_prob)
        results.append((params, val_acc, roc_auc))
    
    results_df = pd.DataFrame(results, columns=['Params(lr, n_est)', 'val_acc', 'roc_auc'])
    results_df = results_df.sort_values(by='roc_auc', ascending=False).reset_index(drop=True)
    return results_df


