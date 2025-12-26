# /src/define_llm.py
# this file is responsible for extracting logits
# from src.metrics import calc_logtoku, calc_logprob
import numpy as np 
from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics import calc_logtoku, calc_logprob
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
# downloaded model path
"""this is downloaded model path """
model_path = "/Users/frankdzzz/models/Qwen2.5-3B-Instruct"

def load_model(model_path = model_path):
    print(f"HF Loading model: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained( 
        pretrained_model_name_or_path=model_path, 
        dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval() # inference mode
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"Model vocab_size: {model.config.vocab_size}")  # Debug
    
    return model,tokenizer

def get_uncertainty(model,tokenizer,prompt,k=2,debug=False):
    """
    Calculate uncertainty metrics (LogToku and LogProb) for a given prompt.
    args:
        prompt: str, input prompt
        k: int, top-k tokens to consider for LogToku calculation
    returns:
        logtoku: float, LogToku metric
        logprob: float, LogProb metric
    """
    #1. response generation setup
    inputs = tokenizer(prompt, return_tensors="pt") 
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    print(f"Inputs shape: {inputs['input_ids'].shape}")  # Debug
    
    input_ids = inputs['input_ids'].to(device)
    
    # attention mask?
    attention_mask = inputs["attention_mask"]

    #2. generate outputs
    with torch.inference_mode():
        response = model.generate(
            input_ids,
            attention_mask=attention_mask,
            output_logits=True,
            return_dict_in_generate=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=10, # ensure the output includes answer
        )

    
    # 3. Extract logits for the first generated token
    
    # I use the first token to calculate uncertainty metrics
    logits = response.logits[0][0].squeeze(0).cpu().numpy()  # shape -> [vocab_size]
    
    # get the generated token IDs
    # 1) get the length of input
    input_len  = input_ids.shape[1]
    generated_ids = response.sequences[0]
    # 2) get the predicted token 
    model_answer_token_id = generated_ids[input_len]
    model_answer = tokenizer.decode(model_answer_token_id)
 

    if debug:
        ######## Debug #######
        first_generated_token_id = generated_ids[input_len] 
        print(f"First generated token: '{tokenizer.decode(first_generated_token_id)}' (ID: {first_generated_token_id})")
        model_output = tokenizer.decode(generated_ids)
        print(f"Model output: {model_output}")
        ######################

    #4. Calculate metrics
    logprob = calc_logprob(logits)
    logtoku = calc_logtoku(logits, k)


    return {
        "token" : model_answer,
        "first_token_logits" :logits,
        "logprob": logprob,
        "logtoku": logtoku   
    }