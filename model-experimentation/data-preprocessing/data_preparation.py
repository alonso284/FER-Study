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
        "resmasknet_anger_aligned",
        "resmasknet_disgust_aligned",
        "resmasknet_fear_aligned",
        "resmasknet_happiness_aligned",
        "resmasknet_sadness_aligned",
        "resmasknet_surprise_aligned",
        "resmasknet_neutral_aligned",
    ]

    # Due to lag shifting, some rows are all NA and need to be handled
    valid_mask = df[emotion_cols].notna().any(axis=1)
    df["resmasknet_max_emotion_value"] = np.nan
    df["resmasknet_dominant_emotion"] = ""

    df.loc[valid_mask, "resmasknet_max_emotion_value"] = df.loc[valid_mask, emotion_cols].max(axis=1)
    df.loc[valid_mask, "resmasknet_dominant_emotion"] = df.loc[valid_mask, emotion_cols].idxmax(axis=1)

    # Clean neutral predictions
    df = df[df["resmasknet_dominant_emotion"] != "resmasknet_neutral_aligned"]

    df["Perceived_Tiredness"] = (
        df["Perceived_Tiredness"].map(PERCEIVED_TIREDNESS).astype("Int64")
    )
    df["Perceived_Stress"] = (
        df["Perceived_Stress"].map(PERCEIVED_STRESS).astype("Int64")
    )
    df["Gender"] = df["Gender"].map(GENDER).astype("Int64")
    df["Wearing_Glasses"] = df["Wearing_Glasses"].map(WEARING_GLASSES).astype("Int64")
    return df

def _generate_brainwave_means(df: pd.DataFrame) -> pd.DataFrame:
    df["Alpha_Mean"] = df[['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']].mean(axis=1)
    df["Beta_Mean"] = df[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']].mean(axis=1)
    df["Gamma_Mean"] = df[['Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10']].mean(axis=1)
    df["Delta_Mean"] = df[['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10']].mean(axis=1)
    df["Theta_Mean"] = df[['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10']].mean(axis=1)
    return df

def _align_specific_signal(df: pd.DataFrame, base_signals: list[str], lagging_signal: str, max_lag: int = 8):
    lags = np.arange(-max_lag, max_lag + 1)
    p_threshold = 0.05

    new_col = f"{lagging_signal}_aligned"
    df[new_col] = df[lagging_signal]

    optimal_lag = 0
    optimal_p_value = np.nan

    task_ids = df["Task_Num"].unique()
    
    for task_id in task_ids:
        # Get the boolean mask for the window
        task_mask = df['Task_Num'] == task_id
        
        # Extract the numpy array once for the lagging signal
        s2_np = df.loc[task_mask, lagging_signal].to_numpy()

        for base_signal in base_signals:
            # Extract the numpy array once for the base signal
            s1_np = df.loc[task_mask, base_signal].to_numpy()

            for lag in lags:
                # Shift the data in the lagging column
                if lag > 0:
                    x_raw = s1_np[lag:]
                    y_raw = s2_np[:-lag]
                elif lag < 0:
                    x_raw = s1_np[:lag]
                    y_raw = s2_np[-lag:]
                else:
                    x_raw = s1_np
                    y_raw = s2_np
                
                # Replicate dropna() behavior for preexisting NaNs in the data
                valid_mask = ~(np.isnan(x_raw) | np.isnan(y_raw))
                x = x_raw[valid_mask]
                y = y_raw[valid_mask]
                
                if len(x) > 2:
                    # Check if either array is exactly constant
                    if np.all(x == x[0]) or np.all(y == y[0]):
                        continue
                    
                    # Compute correlation
                    r, p = pearsonr(x, y)
                    
                    if not np.isnan(p) and (np.isnan(optimal_p_value) or p < optimal_p_value):
                        optimal_p_value = p
                        optimal_lag = lag

    if np.isnan(optimal_p_value) or optimal_p_value > p_threshold:
        return np.nan
    
    df[new_col] = df[new_col].shift(optimal_lag)
    return optimal_lag

def _align_lag_signals_subject(subject_df: pd.DataFrame) -> pd.DataFrame:
    subject_df = subject_df.copy()
    
    base_signals = ["Alpha_Mean", "Delta_Mean", "Theta_Mean", "Beta_Mean", "Gamma_Mean"]
    lagging_signals = ["EDA", "BVP", "Temperature"]
    resmasknet_signals = ["resmasknet_happiness", "resmasknet_anger", "resmasknet_disgust", "resmasknet_fear", "resmasknet_sadness", "resmasknet_surprise", "resmasknet_neutral"]

    for lagging_signal in lagging_signals:
        _align_specific_signal(subject_df, base_signals, lagging_signal)
    
    # Resmasknet signals all should have the same lag, so only calculate it once and apply to all
    
    resmasknet_lag = _align_specific_signal(subject_df, base_signals, resmasknet_signals[0])
    if not np.isnan(resmasknet_lag):
        for resmasknet_signal in resmasknet_signals[1:]:
            col_name = f"{resmasknet_signal}_aligned"
            subject_df[col_name] = subject_df[resmasknet_signal].shift(resmasknet_lag)

    return subject_df

def align_lag_signals(df: pd.DataFrame, progress: bool = False) -> pd.DataFrame:
    """
    Aligns the body signals in the dataframe based correlation analysis
    """
    df = _generate_brainwave_means(df)

    subject_ids = df["Subject_ID"].unique()
    
    # Pre-allocate a list to hold subject dataframes
    aligned_subjects = []
    
    for subject_id in subject_ids:
        subject_df = df[df["Subject_ID"] == subject_id]
        aligned_subject_df = _align_lag_signals_subject(subject_df)
        aligned_subjects.append(aligned_subject_df)
        if progress: print(f"Aligned subject {subject_id}")

    # Concat once at the very end
    aligned_df = pd.concat(aligned_subjects)

    return aligned_df


def normalize_signals_with_mediation_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the relevant body signals in the dataframe based on analysis per person done using the meditation as a baseline
    """
    # @AndresDlg562
    return df
