import pandas as pd
from logger import log_info, log_warning

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna()
    return df

def clean_data(df):
    log_info("Data cleaning started")
    
    df = df.dropna()
    
    log_info("Missing values removed")
    log_info("Data cleaning finished")
    
    return df