# src/metrics.py 
# this file is to define evaluation metrics
# Part 1: Uncertainty Metrics
# 1. LogTokU uncertainty (-EU * AU) based on token logits (our main metric).
import numpy as np 
from scipy.special import softmax, digamma


# top k 
def topk(arr, k):
    """
    return the top k values and theire indices from logists
    """
    indices = np.argpartition(arr, -k)[-k:]
    values = arr[indices]
    return values, indices

def calc_au(logits, k):
    """
    Calculate Aleatoric Uncertainty (AU) from logits.
    """
    top_values, _ = topk(logits, k)
    alpha = np.array([top_values])
    alpha_0 = alpha.sum(axis=1, keepdims=True) 
    psi_alpha_k_plus_1 = digamma(alpha + 1)
    psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
    result = - (alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
    au = result.sum(axis=1)[0]
    ###debug###
    print(f"Aleatoric Uncertainty (AU): {au}")
    ###debug###
    return au


def calc_eu(logits, k):
    """
    Calculate Epistemic Uncertainty (EU) from logits.
    """
    top_values, _ = topk(logits, k)
    eu = k / (np.sum(np.maximum(0, top_values)) + k)
    ###debug###
    print(f"Epistemic Uncertainty (EU): {eu}")
    ###debug###
    return eu


def calc_logtoku(logits, k):
    """
    Calculate LogTokU uncertainty (-AU * EU) from logits.
    """
    au = calc_au(logits, k)
    eu = calc_eu(logits, k)
    return - (eu * au)

def calc_logprob(logits):
    """
    Calculate Log-Probability from logits.
    """
    probs = softmax(logits)
    top_value = np.max(probs)
    return np.log(top_value + 1e-10) # log of probility

