from evaluator.plotting import plot_uncertainty_heatmap
from evaluator.scoring import calculate_rejection_metrics
import pandas as pd

df = pd.read_csv("results/boolq_results.csv")   

logprob_df = calculate_rejection_metrics(df, metric_name='logprob',) # the greater logprob means more confident
logtoku_df = calculate_rejection_metrics(df, metric_name='logtoku',) #  the smaller logtoku means more uncertain

plot_uncertainty_heatmap(logprob_df, logtoku_df,save_path="results/rejection_heatmap.png")

