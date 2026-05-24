import pandas as pd
import numpy as np
from scipy.stats import pearsonr
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
    Cleans the data by handling missing values, NaN, etc.
    In addition, removes rows with neutral predictions (50~ % of the data) to avoid inflated confidence on trivial predictions.
    Also changes string values to numerical values
    """
    df = df[df["Face_Detection"] == 1]
    df = df[df["HeadBandOn"] == 1]

    # Get the certainty o prediction
    emotion_cols = [
        "resmasknet_anger",
        "resmasknet_disgust",
        "resmasknet_fear",
        "resmasknet_happiness",
        "resmasknet_sadness",
        "resmasknet_surprise",
        "resmasknet_neutral",
    ]

    df["resmasknet_max_emotion_value"] = df[emotion_cols].max(axis=1)
    df["resmasknet_dominant_emotion"] = df[emotion_cols].idxmax(axis=1)

    # Clean neutral predictions
    df = df[df["resmasknet_dominant_emotion"] != "resmasknet_neutral"]

    df["Perceived_Tiredness"] = (
        df["Perceived_Tiredness"].map(PERCEIVED_TIREDNESS).astype("Int64")
    )
    df["Perceived_Stress"] = (
        df["Perceived_Stress"].map(PERCEIVED_STRESS).astype("Int64")
    )
    df["Gender"] = df["Gender"].map(GENDER).astype("Int64")
    df["Wearing_Glasses"] = df["Wearing_Glasses"].map(WEARING_GLASSES).astype("Int64")
    return df

def generate_brainwave_means(df: pd.DataFrame) -> pd.DataFrame:
    df["Alpha_Mean"] = df[['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']].mean(axis=1)
    df["Beta_Mean"] = df[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']].mean(axis=1)
    df["Gamma_Mean"] = df[['Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10']].mean(axis=1)
    df["Delta_Mean"] = df[['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10']].mean(axis=1)
    df["Theta_Mean"] = df[['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10']].mean(axis=1)
    return df

def align_specific_signal(df: pd.DataFrame, base_signals: list[str], lagging_signal: str, max_lag: int = 8):
    lags = np.arange(-max_lag, max_lag + 1)
    p_threshold = 0.05

    # Initialize the new aligned column with the original data
    new_col = f"{lagging_signal}_aligned"
    df[new_col] = df[lagging_signal]

    # For each task, find the best correlation between all base_signals and the lagging signal
    # Then use the best lag for the entire subject's timeline
    # Reasoning for this: some signals may affect different brainwaves and therefore the best correlation lag might not apply to the other brainwaves

    optimal_lag = 0
    optimal_p_value = np.nan

    task_ids = df["Task_Num"].unique()
    for task_id in task_ids:
        # Find the task window
        task_start = df[df['Task_Num'] == task_id].index[0]
        task_end = df[df['Task_Num'] == task_id].index[-1]
        window = df.loc[task_start : task_end].copy()

        s2 = window[lagging_signal]

        for base_signal in base_signals:
            s1 = window[base_signal]

            for lag in lags:
                s2_shifted = s2.shift(lag)
                temp_df = pd.concat([s1, s2_shifted], axis=1).dropna()
                
                if len(temp_df) > 2:
                    x = temp_df.iloc[:, 0].values
                    y = temp_df.iloc[:, 1].values
                    
                    # Check if either array is exactly constant (causes error otherwise)
                    if np.all(x == x[0]) or np.all(y == y[0]):
                        r, p = np.nan, np.nan
                    else:
                        # Only calculate if there is variation in both signals
                        r, p = pearsonr(x, y)
                else:
                    r, p = np.nan, np.nan
                    
                if not np.isnan(p) and (np.isnan(optimal_p_value) or p < optimal_p_value):
                    optimal_p_value = p
                    optimal_lag = lag

    if np.isnan(optimal_p_value) or optimal_p_value > p_threshold:
        # Couldn't find an optimal lag, so leave as is
        return np.nan
    
    # Update only the specific window in the new column with the optimal shift
    df[new_col] = df[new_col].shift(optimal_lag)
    return optimal_lag

def align_lag_signals_subject(subject_df: pd.DataFrame) -> pd.DataFrame:
    base_signals = ["Alpha_Mean", "Delta_Mean", "Theta_Mean", "Beta_Mean", "Gamma_Mean"]
    lagging_signals = ["EDA", "BVP", "Temperature"]
    resmasknet_signals = ["resmasknet_happiness", "resmasknet_anger", "resmasknet_disgust", "resmasknet_fear", "resmasknet_sadness", "resmasknet_surprise", "resmasknet_neutral"]

    for lagging_signal in lagging_signals:
        align_specific_signal(subject_df, base_signals, lagging_signal)
    
    # All resmasknet signals are assumed to have the same lag so we just need to find the lag for one
    resmasknet_lag = align_specific_signal(subject_df, base_signals, resmasknet_signals[0])
    if not np.isnan(resmasknet_lag):
        for resmasknet_signal in resmasknet_signals[1:]:
            col_name = f"{resmasknet_signal}_aligned"
            subject_df[col_name] = subject_df[resmasknet_signal].shift(resmasknet_lag)

    print(f"Aligned: {subject_df["Subject_ID"].iloc[0]}")

    return subject_df

def align_lag_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns the body signals in the dataframe based correlation analysis
    """

    df = generate_brainwave_means(df)

    # We found that aligning the body signals using per person and per signals is the best choice
    # Since the fastest signal are the brainwaves, those are assumed to be the baseline and other signals are moved
    aligned_df = pd.DataFrame()

    subject_ids = df["Subject_ID"].unique()
    for subject_id in subject_ids:
        subject_df = df[df["Subject_ID"] == subject_id]
        aligned_subject_df = align_lag_signals_subject(subject_df)
        aligned_df = pd.concat([aligned_df, aligned_subject_df])

    return aligned_df


def normalize_signals_with_mediation_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the relevant body signals in the dataframe based on analysis per person done using the meditation as a baseline
    """
    # @AndresDlg562
    return df
