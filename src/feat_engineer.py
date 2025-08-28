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
