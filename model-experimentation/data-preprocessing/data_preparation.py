import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data by handling missing values, NaN, etc.
    In addition, removes rows with neutral predictions (50~ % of the data) to avoid inflated confidence on trivial predictions.
    """
    # @SergioGzzBrz
    return df

def align_specific_signal(df: pd.DataFrame, col1: str, col2: str, task_ids: list[float], max_lag: int = 10):
    lags = np.arange(-max_lag, max_lag + 1)

    # Initialize the new aligned column with the original data
    new_col = f"{col2}_aligned"
    df[new_col] = df[col2]

    for task_id in task_ids:

        # Find the task window
        task_start = df[df['Task_Num'] == task_id].index[0]
        task_end = df[df['Task_Num'] == task_id].index[-1]

        window = df.loc[task_start : task_end].copy()
        s1 = window[col1]
        s2 = window[col2]

        corrs = []
        p_values = []
        for lag in lags:
            s2_shifted = s2.shift(-lag)
            temp_df = pd.concat([s1, s2_shifted], axis=1).dropna()
            
            if len(temp_df) > 2:
                x = temp_df.iloc[:, 0].values
                y = temp_df.iloc[:, 1].values
                
                r, p = pearsonr(x, y)
            else:
                r, p = np.nan, np.nan
                
            corrs.append(r)
            p_values.append(p)

        optimal_idx = np.argmin(p_values)

        # Safely find the optimal lag by ignoring NaNs
        valid_indices = [i for i, p in enumerate(p_values) if not np.isnan(p)]
        
        if valid_indices:
            # Optimal lag based on lowest p-value
            optimal_idx = valid_indices[np.argmin([p_values[i] for i in valid_indices])]
            optimal_lag = lags[optimal_idx]
            
            # Update only the specific window in the new column with the optimal shift
            df.loc[task_start : task_end, new_col] = s2.shift(-optimal_lag)

def align_lag_signals_subject(subject_df: pd.DataFrame, task_ids: list[float]) -> pd.DataFrame:
    # should be more related to all brainwave signals
    base_signal = "Alpha_TP9"
    lagging_signals = ["EDA", "resmasknet_happiness"]

    for lagging_signal in lagging_signals:
        lag, corr, p_val = align_specific_signal(subject_df, base_signal, lagging_signal, task_ids)
        print(lag, corr, p_val, lagging_signal)

    return subject_df

def align_lag_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns the body signals in the dataframe based correlation analysis
    """
    # We found that aligning the body signals using per person, per task and per signals is the best choice
    # Since the fastest signal are the brainwaves, those are assumed to be the baseline and other signals are moved

    task_ids = df["Task_Num"]

    aligned_df = pd.DataFrame()

    subject_ids = df["Subject_ID"].unique()
    for subject_id in subject_ids:
        subject_df = df[df["Subject_ID"] == subject_id]
        aligned_subject_df = align_lag_signals_subject(subject_df, task_ids)
        aligned_df = pd.concat([aligned_df, aligned_subject_df])

    return aligned_df


def normalize_signals_with_mediation_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the relevant body signals in the dataframe based on analysis per person done using the meditation as a baseline
    """
    # @AndresDlg562
    return df
