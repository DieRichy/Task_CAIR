# src/dataset_loader.py
# This file is to load datasets from Hugging Face Datasets library

from datasets import load_dataset
from datasets import load_dataset
from prompt import apply_chat_template

class BoolQLoader:
    def __init__(self, model_name, split="validation", num_samples=500):
        self.model_name = model_name
        self.dataset = load_dataset("google/boolq", split=split).select(range(num_samples))

    def get_data(self):
        formatted_data = []
        system_instruction = "Answer the question concisely with only 'True' or 'False'."
        
        for item in self.dataset:
            user_question = f"Passage: {item['passage']}\nQuestion: {item['question']}"
            
            # import from src.utils.prompt_utils import apply_chat_template
            full_prompt = apply_chat_template(
                self.model_name, 
                system_instruction, 
                user_question
            )
            
            formatted_data.append({
                "prompt": full_prompt,
                "gt": str([item['answer'])
            })
        return formatted_data

class CommonsenseQALoader:
    def __init__(self, model_name, split="validation", num_samples=1200): # size of the validation subset of CommonsenseQA is about 1.2K
        self.model_name = model_name
        self.dataset = load_dataset("commonsense_qa", split=split).select(range(num_samples))

    def get_data(self):
        formatted_data = []
        system_instruction = "Answer the question with the correct option letter (A, B, C, D, or E)."
        
        for item in self.dataset:
            question = item['question']
            choices = item['choices']
            options = f"A) {choices['text'][0]} B) {choices['text'][1]} C) {choices['text'][2]} D) {choices['text'][3]} E) {choices['text'][4]}"
            user_question = f"Question: {question}\nOptions: {options}"
            
            full_prompt = apply_chat_template(
                self.model_name, 
                system_instruction, 
                user_question
            )
            
            formatted_data.append({
                "prompt": full_prompt,
                "gt": item['answerKey']  # like "A"
            })
        return formatted_data
