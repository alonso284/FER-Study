import pandas as pd

# This two things are based on https://pmc.ncbi.nlm.nih.gov/articles/PMC7349550/#:~:text=Notwithstanding%2C%20not,process


def variance_treshold_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Variance Threshold is a feature selection method that removes features with low variance.
    The idea is that if a feature has low variance, it may not be informative for the model.
    Each group of features has its own variance treshold.
    """
    # TODO, USE: from sklearn.feature_selection import VarianceThreshold
    # @SergioGzzBrz already checked this, nothing needs to change from this so I will add that to the final doc
    # Don't worry about this

    return df
