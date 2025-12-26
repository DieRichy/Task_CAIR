# src/dataset_loader.py
# This file is to load datasets from Hugging Face Datasets library

from datasets import load_dataset
from datasets import load_dataset
from prompt import apply_chat_template

class BoolQLoader:
    def __init__(self, model_name, split="validation", num_samples=100):
        self.model_name = model_name
        self.dataset = load_dataset("google/boolq", split=split).select(range(num_samples))
        # BoolQ answers are boolean, map them to "Yes"/"No"
        self.label_map = {True: "Yes", False: "No"}

    def get_data(self):
        formatted_data = []
        system_instruction = "Answer the question concisely with only 'Yes' or 'No'."
        
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
                "gt": self.label_map[item['answer']]
            })
        return formatted_data