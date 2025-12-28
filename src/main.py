from define_llm import load_model, get_uncertainty
from metrics import calc_logtoku, calc_logprob
from prompt import apply_chat_template
from dataset_loader import BoolQLoader
from evaluator.plotting import plot_uncertainty_heatmap
from evaluator.scoring import calculate_rejection_metrics
import pandas as pd
def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    model_name="Qwen/Qwen2.5-3B-Instruct"
    # Load samples from BoolQ / CQA validation set
    # loader = BoolQLoader(model_name=model_name, split="validation", num_samples=2000)
    # or #
    loader = CommonsenseQALoader(model_name=model_name, split="validation", num_samples=1200) # commonsenseQA validation has 1.22k rows
    data = loader.get_data()
    #initialize result list
    all_result = []
    # 1. inferecne and collect results
    for i, item in enumerate(data):
        prompt = item["prompt"]
        gt = item["gt"]
        
        # Get uncertainty metrics
        result = get_uncertainty(model, tokenizer, prompt, k=2) # default k=2
 
        # #Debug: print some
        # print(f"Question {i+1}: {item['prompt'][:300]}...")  # Truncate for display
        # print(f"Predicted Token: {result['token']}")
        print(f"LogProb: {result['logprob']:.4f}")
        print(f"LogTokU: {result['logtoku']:.4f}")
        # print(f"Ground Truth: {gt}")
        # print("-" * 50)
        
        # check correct ratio
        is_correct = (result['token'].strip().lower() == gt.strip().lower())
        all_result.append({
            "is_correct": is_correct,
            "logprob": result['logprob'],
            "logtoku": result['logtoku']
        })
    #2. save data to dataframe
    df = pd.DataFrame(all_result)
    df.to_csv("results/commonsense_results.csv", index=False)

    #3. calculate rejection metrics
    #3.1 logprob df
    logprob_df = calculate_rejection_metrics(df, metric_name='logprob',) # the greater logprob means more confident
    #3.2 logtoku df
    logtoku_df = calculate_rejection_metrics(df, metric_name='logtoku',) #  the smaller logtoku means more uncertain

    #4. Plotting
    plot_uncertainty_heatmap(logprob_df, logtoku_df)

if __name__ == "__main__":
    main()
