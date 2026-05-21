import pandas as pd

# This two things are based on https://pmc.ncbi.nlm.nih.gov/articles/PMC7349550/#:~:text=Notwithstanding%2C%20not,process


def variance_treshold_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Variance Threshold is a feature selection method that removes features with low variance.
    The idea is that if a feature has low variance, it may not be informative for the model.
    Each group of features has its own variance treshold.
    """
    # TODO, USE: from sklearn.feature_selection import VarianceThreshold

    return df


def correlation_based_selection(
    df: pd.DataFrame, threshold: float = 0.90
) -> pd.DataFrame:
    """
    Correlation-based feature selection identifies and removes features that are highly correlated with each other.
    The idea is that if two features are highly correlated, they may provide redundant information to the model.
    This method helps to reduce multicollinearity and can improve model performance.

    Parameters:
    - df: Input DataFrame containing the features.
    - threshold: Correlation threshold above which features will be considered redundant and removed.

    Returns:
    - A DataFrame with redundant features removed based on the specified correlation threshold.
    """
    # TODO
    return df
