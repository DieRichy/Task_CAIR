# src/evaluator/scoring.py
# This file is to implement scoring functions for pred-vs-gt evaluations
# src/scoring.py
import pandas as pd
import numpy as np

def calculate_rejection_metrics(df, metric_name,):
    """
    calculate rejection metrics.
    metric_name: 'logprob' or 'logtoku'
    """
    # Rank from most certain to least certain
    # logprob and logtoku: the greater number means the more confident llm is about its prediction closer to 0

    df_sorted = df.sort_values(by=metric_name, ascending=False).reset_index(drop=True)  # 从小到大，最不自信的先
    total_samples = len(df_sorted)
    results = []

    # compute accuracy at different rejection percentiles from 0% (no rejection) to 95% rejection
    for rej_percentile in range(0, 100, 5): #(0-95%)
        keep_ratio = (100 - rej_percentile) / 100
        num_keep = max(1, int(total_samples * keep_ratio))
        
        subset = df_sorted.head(num_keep)
        accuracy = subset['is_correct'].mean()
        
        results.append({
            "Keep Percentile": 100 - rej_percentile,  # keep ratio
            "accuracy": accuracy
        })
    
    return pd.DataFrame(results)