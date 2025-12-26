#src/evaluator/plotting.py
####################     
#This file is to implement plotting functions for evaluation results

# src/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_uncertainty_heatmap(prob_results, toku_results, save_path="results/rejection_heatmap.png"):
    """
    plot uncertainty heatmap for rejection metrics.
    
    参数:
        prob_results: dict, includes 'percentile' and 'accuracy' key
        toku_results: dict, includes 'percentile' and 'accuracy' key
        save_path: str, save path for the heatmap image
    """
    # data preparation
    keep_percentiles = prob_results['Keep Percentile']
    prob_accuracy = prob_results['accuracy']
    toku_accuracy = toku_results['accuracy']
    
    # combine data for heatmap (columns: prob_accuracy, toku_accuracy)
    data = np.column_stack([prob_accuracy, toku_accuracy])
    
    # plotting
    plt.figure(figsize=(10, 12))
    
    # heatmap plotting
    ax = sns.heatmap(data[::-1],  # reverse the order to have highest keep rate on top
                     annot=True,  # show values
                     fmt='.3f',   # value format: 3 decimal places
                     cmap='YlGnBu',  # colormap (blue green yellow)
                     yticklabels=keep_percentiles[::-1],  # y-axis labels from 100 to 5%
                     xticklabels=['acc_w_rejection_prob', 'acc_w_rejection_uncertainty_2'],  # x-axis labels
                     cbar_kws={'label': ''},  # color bar
                     vmin=np.min(data), 
                     vmax=1.0,  # color range
                     linewidths=0.5,  # grid line width
                     linecolor='white')  # grid line color
    
    # labels
    plt.ylabel('Keep Percentile', fontsize=12)
    plt.xlabel('')
    
    # titles
    plt.title('Accuracy vs. Keep Rate', fontsize=14, pad=20)
    
    # xticks
    plt.xticks(rotation=0, ha='center')
    
    # layout adjustment
    plt.tight_layout()
    
    # save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {save_path}")
    
    # show
    plt.show()


