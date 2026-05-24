import pandas as pd
import data_preparation
import time

def generate_prepared_file(file_name: str = "../../Clean_NPFC-TEST_Database_V2.csv"):
  df = pd.read_csv('../../NPFC-Test_Database_V2.csv')

  print("Preparing data file (this can take a few minutes)...")

  start_time = time.perf_counter()

  df = data_preparation.align_lag_signals(df)
  df = data_preparation.prepare_data(df)
  df = data_preparation.normalize_signals_with_mediation_baseline(df)

  end_time = time.perf_counter()  

  df.to_csv(file_name)
  print(f"File created: {file_name} ({end_time - start_time:0.4f}s)")