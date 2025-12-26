# Task_CAIR

This repository reproduces the core uncertainty estimation experiments from **"Estimating LLM Uncertainty with Evidence" (Ma et al., 2025)**.

## 📖 Project Overview

Accurate indication of an LLM's generation confidence is critical, especially in high-stakes scenarios like medical applications. This project implements a **training-free** pipeline to estimate the uncertainty of LLM/VLM generations using internal model signals.

## 🔬 Core Metrics Implemented

### 1. **Logprob**
The logarithmic probability of the first generated token, used as a baseline for uncertainty.

### 2. **Logtoku**
The advanced uncertainty metric derived from the reference paper, which incorporates evidence-based estimation.

## 📊 Application: Reliability Estimation

Based on **Section 5: APPLICATION II: RELIABILITY ESTIMATION** of the reference paper:

The reliability of a token is represented as:

\[
R(a_t) = -\text{AU}(a_t) \cdot \text{EU}(a_t)
\]

Where:
- \(\text{AU}(a_t)\) represents Aleatoric Uncertainty
- \(\text{EU}(a_t)\) represents Epistemic Uncertainty
---

## 📂 Document Structure

* `src/metrics.py`: Metric calculation functions based on the original [Logtoku implementation](https://github.com/MaHuanAAA/logtoku/blob/main/SenU/metrics.py).
---

## 🛠️ Environment Setup

This project uses [`uv`](https://www.google.com/search?q=%5Bhttps://github.com/astral-sh/uv%5D(https://github.com/astral-sh/uv)) for high-performance environment management.


### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Task_CAIR.git
cd Task_CAIR

# Create and sync environment using uv
uv sync

# Or using standard pip
# pip install -r requirements.txt

```

---

## 🚀 Usage

### 1. Model Preparation

The pipeline supports various LLMs/VLMs. By default, it is configured for:

*`Qwen2.5-3B-Instruct`


### 2. Running Inference & Evaluation

Execute the main script to run the uncertainty estimation on datasets like `BoolQ`.

```bash
python src/main.py --model_path <path_to_model> --dataset boolq --num_samples 100

```

Note: You can trim datasets (e.g., first 100 rows) for faster inference.

---

## 📊 Results & Visualization

The core evaluation method is **Selective Generation (Accuracy vs. Keep Rate)**. By rejecting samples with high uncertainty, the model's precision on the remaining samples should increase.

My Example:
Figure 1 2000 examples:
<img width="936" height="1076" alt="2000samples_test" src="https://github.com/user-attachments/assets/18fc55dd-31e8-4c55-b4f8-76912b4a81dc" />
Figure 2 500 examples:
<img width="2813" height="3569" alt="500samples_test" src="https://github.com/user-attachments/assets/e32c092e-b79d-40fa-b7f6-7baeb78434fc" />

Key Findings:
* **No Rejection**: Accuracy are highest when no rejection 
* **95% Rejection**: Accuracy reaches lowest which closes to randomly guessing (the accuracy is only 50%)
* **Trend stability**: Although the Accuracy is not increasing as the expected, But indeed the "-eu2 * au2" (logtoku) is a better indicator than simple logprobs which flacuates throughout the experiment.




