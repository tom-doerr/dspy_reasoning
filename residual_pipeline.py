#!/usr/bin/env python3

import dspy
import json
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

class SearchReplaceModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.ChainOfThought('input -> search, replace')
        
    def forward(self, input_text: str) -> str:
        result = self.process(input=input_text)
        if not hasattr(result, 'search') or not hasattr(result, 'replace'):
            return input_text
        return input_text.replace(result.search, result.replace)

class SearchReplacePipeline(dspy.Module):
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.layers = [SearchReplaceModule() for _ in range(num_layers)]

    def forward(self, task: str) -> str:
        current = task
        for layer in self.layers:
            current = layer(current)
        try:
            # Try to evaluate the final expression
            return str(eval(current))
        except:
            return current

def evaluate_pipeline(dataset_path: str = "math_dataset.json", num_threads: int = 10, num_layers: int = 3) -> float:
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
    dspy.settings.configure(lm=lm)
    pipeline = SearchReplacePipeline(num_layers=num_layers)
    
    correct = 0
    
    def evaluate_task(task_data):
        try:
            predicted = pipeline(task_data['task'])
            expected = float(task_data['solution'])
            predicted_num = float(predicted)
            return abs(predicted_num - expected) < 0.01
        except (ValueError, TypeError):
            return False
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(evaluate_task, task_data)
            for task_data in dataset[:100]
        ]
        
        for future in as_completed(futures):
            correct += future.result()
    
    accuracy = correct / len(futures)
    print(f"Pipeline accuracy: {accuracy:.1%}")
    return accuracy

if __name__ == "__main__":
    evaluate_pipeline(num_layers=3)
