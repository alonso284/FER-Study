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
    required_cols = {"Subject_ID", "Task_Num"}
    if not required_cols.issubset(df.columns):
        return df

    signal_candidates = [
        "EDA",
        "BVP",
    ]
    signal_cols = [col for col in signal_candidates if col in df.columns]
    if not signal_cols:
        return df

    baseline_task = 2.1 #Initial Meditation 
    baseline_df = df[df["Task_Num"] == baseline_task]
    if baseline_df.empty:
        return df

    baseline_means = (
        baseline_df.groupby("Subject_ID")[signal_cols]
        .mean()
        .add_prefix("baseline_")
    )

    normalized = df.join(baseline_means, on="Subject_ID")
    for col in signal_cols:
        baseline_col = f"baseline_{col}"
        normalized[col] = normalized[col].where(
            normalized[baseline_col].isna(),
            normalized[col] - normalized[baseline_col],
        )

    normalized = normalized.drop(columns=baseline_means.columns)
    return normalized
