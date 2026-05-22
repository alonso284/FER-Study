import pandas as pd

PERCEIVED_TIREDNESS = {
    "Not tired": 0,
    "Slightly tired": 1,
    "Moderately tired": 2,
}

PERCEIVED_STRESS = {
    "Not stressed": 0,
    "Slightly stressed": 1,
    "Moderately stressed": 2,
    "Very stressed": 3,
}

WEARING_GLASSES = {
    "No": 0,
    "Yes": 1,
}

GENDER = {
    "Male": 0,
    "Female": 1,
}


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Created the prediction labels that will be used for model training.
    Cleans the data by handling missing values, NaN, etc.
    In addition, removes rows with neutral predictions (50~ % of the data) to avoid inflated confidence on trivial predictions.
    Changes string values to numerical values.
    """
    filtered_df = df[df["Face_Detection"] == 1]
    filtered_df = df[df["HeadBandOn"] == 1]
    df_clean = filtered_df.copy()

    # Get the prediction value
    emotion_cols = [
        "resmasknet_anger",
        "resmasknet_disgust",
        "resmasknet_fear",
        "resmasknet_happiness",
        "resmasknet_sadness",
        "resmasknet_surprise",
        "resmasknet_neutral",
    ]

    df_clean = df_clean.dropna(subset=emotion_cols, how="all")

    df_clean["resmasknet_max_emotion_value"] = df_clean[emotion_cols].max(axis=1)
    df_clean["resmasknet_dominant_emotion"] = df_clean[emotion_cols].idxmax(axis=1)

    # Clean neutral predictions
    df_clean = df_clean[df_clean["resmasknet_dominant_emotion"] != "resmasknet_neutral"]

    df_clean["Perceived_Tiredness"] = (
        df_clean["Perceived_Tiredness"].map(PERCEIVED_TIREDNESS).astype("Int64")
    )
    df_clean["Perceived_Stress"] = (
        df_clean["Perceived_Stress"].map(PERCEIVED_STRESS).astype("Int64")
    )
    df_clean["Gender"] = df_clean["Gender"].map(GENDER).astype("Int64")
    df_clean["Wearing_Glasses"] = (
        df_clean["Wearing_Glasses"].map(WEARING_GLASSES).astype("Int64")
    )
    return df_clean


def align_lag_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns the body signals in the dataframe based correlation analysis
    """
    # @luis-amado
    return df


def normalize_signals_with_mediation_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the relevant body signals in the dataframe based on analysis per person done using the meditation as a baseline
    """
    # @AndresDlg562
    return df
