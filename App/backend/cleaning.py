import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC

# Column name of the label
LABEL_COL = 'koi_disposition'
# Columns to be discarded from the dataset
DISCARD_COLS = []
# Random seed for reproducibility
RANDOM_SEED = 42
# Suffixes indicating error columns
ERROR_SUFFIXES = ['_err', '_err1', '_err2']
# Noise level to be added to numerical features (as a fraction of the std deviation)
NOISE_LEVEL = 0.01
# Max number of distinct values for a column to be considered categorical
MAX_CAT_VALUES = 10
# Percentage of data to be missing to discard a column
MAX_MISSING_VALUES = 0.5
# Max colinearity threshold
MAX_COLINEARITY = 0.80
# Flag to apply scaling (StandardScaler; mean=0, std=1)
APPLY_SCALER = False
# Flag to apply Gaussian noise
APPLY_NOISE = False
# Flag to apply SMOTENC (oversampling for categorical+numerical data)
APPLY_SMOTENC = False
# Flag to discard error columns at the end
DISCARD_ERROR_COLS = True
# Numerical replacement for string categorical values
STRING_CAT_REPLACEMENT = {'CONFIRMED': 1, 'CANDIDATE': 0, 'FALSE POSITIVE': -1}

def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the DataFrame into features and labels.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        features (pd.DataFrame): The feature columns.
        labels (pd.DataFrame): The label column."""
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in DataFrame.")
    labels = df[LABEL_COL].to_frame(name=LABEL_COL)
    features = df.drop(columns=[LABEL_COL])
    return features, labels
def discard_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Discards specified columns from the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to discard.
    Returns:
        pd.DataFrame: The DataFrame with specified columns discarded."""
    return df.drop(columns=columns, errors='ignore')
def replace_string_values(df: pd.DataFrame, replacements: dict) -> pd.DataFrame:
    """Replaces string values in the DataFrame based on a mapping dictionary.
    Args:
        df (pd.DataFrame): The input DataFrame.
        replacements (dict): A dictionary mapping old values to new values.
    Returns:
        pd.DataFrame: The DataFrame with string values replaced."""
    df = df.replace(replacements)
    return df
def discard_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Discards columns with all missing values from the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with empty columns discarded."""
    return df.dropna(axis=1, how='all')
def discard_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Discards columns with more than MAX_MISSING_VALUES proportion of missing values.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with columns having excessive missing values discarded."""
    return df.loc[:, df.isnull().mean() < MAX_MISSING_VALUES]
def discard_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Discards columns with constant values from the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with constant columns discarded."""
    return df.loc[:, df.nunique() > 1]
def get_categorical_columns(df: pd.DataFrame) -> list:
    """Identifies categorical columns based on the number of distinct values.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        list: List of categorical column names."""
    return [col for col in df.columns if df[col].nunique() <= MAX_CAT_VALUES]
def discard_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Discards columns with string data types from the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with string columns discarded."""
    string_cols = df.select_dtypes(include=['object']).columns
    return df.drop(columns=string_cols, errors='ignore')
def discard_colinear_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Discards columns that are highly colinear with others based on MAX_COLINEARITY threshold.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with colinear columns discarded."""
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > MAX_COLINEARITY)]
    return df.drop(columns=to_drop, errors='ignore')
def standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes the feature DataFrame using StandardScaler. Not applied to categorical features.
    Args:
        df (pd.DataFrame): The feature DataFrame.
    Returns:
        pd.DataFrame: The standardized feature DataFrame."""
    categorical_cols = get_categorical_columns(df)
    numerical_cols = [col for col in df.columns if col not in categorical_cols]
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df
def add_gaussian_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Gaussian noise to numerical features in the DataFrame.
    Args:
        df (pd.DataFrame): The feature DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with added Gaussian noise."""
    categorical_cols = get_categorical_columns(df)
    numerical_cols = [col for col in df.columns if col not in categorical_cols]
    for col in numerical_cols:
        std_dev = df[col].std()
        np.random.seed(RANDOM_SEED)
        noise = np.random.normal(0, NOISE_LEVEL * std_dev, size=df.shape[0])
        df[col] += noise
    return df
def augment_data(features: pd.DataFrame) -> pd.DataFrame:
    """Applies data augmentation techniques like scaling and noise addition.
    Args:
        df (pd.DataFrame): The feature DataFrame.
    Returns:
        pd.DataFrame: The augmented feature DataFrame."""
    features = standardize_features(features)
    df = add_gaussian_noise(df)
    return df

class SkewAwareImputer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.imputers = {}

    def fit(self, df, y=None):
        df = pd.DataFrame(df)  # ensure DataFrame for column-wise ops
        self.imputers = {}
        for col in df.columns:
            skew = df[col].dropna().skew()
            if abs(skew) < self.threshold:
                strategy = "mean"
            else:
                strategy = "median"
            imputer = SimpleImputer(strategy=strategy)
            imputer.fit(df[[col]])
            self.imputers[col] = imputer
        return self

    def transform(self, df):
        df = pd.DataFrame(df).copy()
        for col, imputer in self.imputers.items():
            df[col] = imputer.transform(df[[col]])
        return df.values
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values in the feature DataFrame using SkewAwareImputer.
    Args:
        df (pd.DataFrame): The feature DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with missing values filled."""
    imputer = SkewAwareImputer()
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def pipeline(df: pd.DataFrame, label_col: str = LABEL_COL, discard_cols: list = DISCARD_COLS, random_seed: int = RANDOM_SEED,
             error_suffixes: list = ERROR_SUFFIXES, noise_level: float = NOISE_LEVEL, max_cat_values: int = MAX_CAT_VALUES,
             max_missing_values: float = MAX_MISSING_VALUES, max_collinearity: float = MAX_COLINEARITY,
             apply_scaler: bool = APPLY_SCALER, apply_noise: bool = APPLY_NOISE,
             apply_smotenc: bool = APPLY_SMOTENC, discard_error_cols: bool = DISCARD_ERROR_COLS,
             replacements: dict = STRING_CAT_REPLACEMENT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Comprehensive data preprocessing pipeline.
    Args:
        df (pd.DataFrame): The input DataFrame.
        label_col (str): The name of the label column.
        discard_cols (list): List of columns to discard.
        random_seed (int): Random seed for reproducibility.
        error_suffixes (list): List of suffixes indicating error columns.
        noise_level (float): Noise level to be added to numerical features.
        max_cat_values (int): Max number of distinct values for a column to be considered categorical.
        max_missing_values (float): Percentage of data to be missing to discard a column.
        max_collinearity (float): Max collinearity threshold.
        apply_scaler (bool): Flag to apply scaling.
        apply_noise (bool): Flag to apply Gaussian noise.
        apply_smotenc (bool): Flag to apply SMOTENC oversampling.
    Returns:
        features (pd.DataFrame): The preprocessed feature DataFrame.
        labels (pd.Series): The label Series."""
    df = replace_string_values(df, replacements)
    df = discard_columns(df, discard_cols)
    df = discard_empty_columns(df)
    df = discard_missing_values(df)
    df = discard_constant_columns(df)
    df = discard_string_columns(df)
    df = discard_colinear_columns(df)
    features, labels = split_features_labels(df)
    features = fill_missing_values(features)
    if apply_scaler and not apply_smotenc:
        features = standardize_features(features)
    if apply_noise:
        features = add_gaussian_noise(features)
    if discard_error_cols:
        error_cols = [col for col in features.columns if any(col.endswith(suffix) for suffix in error_suffixes)]
        features = discard_columns(features, error_cols)
    if apply_smotenc:
        categorical_cols = get_categorical_columns(features)
        categorical_indices = [features.columns.get_loc(col) for col in categorical_cols]
        smote = SMOTENC(categorical_features=categorical_indices, random_state=random_seed)
        features_res, labels_res = smote.fit_resample(features, labels)
        features = pd.DataFrame(features_res, columns=features.columns)
        labels = pd.DataFrame(labels_res, columns=labels.columns)
        if apply_scaler:
            features = standardize_features(features)
    return features, labels