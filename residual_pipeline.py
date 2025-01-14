#!/usr/bin/env python3

import dspy
import json
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class SearchReplaceModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.ChainOfThought('input -> search_block, replace_block')
        
    def forward(self, input_text: str) -> str:
        result = self.process(input=input_text)
        if not hasattr(result, 'search_block') or not hasattr(result, 'replace_block'):
            return input_text
        return input_text.replace(result.search_block, result.replace_block)

class SearchReplacePipeline(dspy.Module):
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.layers = [SearchReplaceModule() for _ in range(num_layers)]

    def forward(self, task: str) -> str:
        current = task
        for layer in self.layers:
            current = layer(current)
        return current



class SearchReplaceIterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.ChainOfThought('input, iteration -> search_block, replace_block')
        
    def forward(self, input_text: str, iteration: int) -> str:
        result = self.process(input=input_text, iteration=iteration)
        if not hasattr(result, 'search_block') or not hasattr(result, 'replace_block'):
            return input_text
        return input_text.replace(result.search_block, result.replace_block)

class SearchReplaceIterPipeline(dspy.Module):
    def __init__(self, num_iters: int = 10):
        super().__init__()
        self.layers = [SearchReplaceIterModule() for _ in range(num_iters)]

    def forward(self, task: str) -> str:
        current = task
        # for layer in self.layers:
        for iteration, layer in enumerate(self.layers):
            # current = layer(current)
            current = layer(current, iteration)
        return current






        # try:
            # # Try to evaluate the final expression
            # return str(eval(current))
        # except:
            # return current

def evaluate_pipeline(
    dataset_path: str = "math_dataset.json", 
    num_threads: int = 10, 
    num_layers: int = 10,
    model: str = "deepseek/deepseek-chat",
    temperature: float = 0.3
) -> float:
    print(f"\nEvaluating SearchReplace Pipeline with {num_layers} layers using {model}...")
    start_time = time.time()
    
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    lm = dspy.LM(model=model, temperature=temperature, cache=False)
    dspy.settings.configure(lm=lm)
    pipeline = SearchReplacePipeline(num_layers=num_layers)
    
    correct = 0
    total_tasks = min(len(dataset), 100)
    results = []
    
    def evaluate_task(task_data):
        try:
            task = task_data['task']
            expected = float(task_data['solution'])
            
            predicted = pipeline(task)
            predicted_num = float(predicted)
            
            is_correct = abs(predicted_num - expected) < 0.01
            return {
                'task': task,
                'predicted': predicted,
                'expected': expected,
                'correct': is_correct
            }
        except (ValueError, TypeError) as e:
            return {
                'task': task_data['task'],
                'error': str(e),
                'correct': False
            }
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(evaluate_task, task_data)
            for task_data in dataset[:total_tasks]
        ]
        
        with tqdm(total=total_tasks, desc="Evaluating") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.get('correct', False):
                    correct += 1
                pbar.update(1)
                
                # Display running accuracy
                current_accuracy = correct / len(results)
                pbar.set_postfix({'accuracy': f'{current_accuracy:.1%}'})
    
    accuracy = correct / total_tasks
    elapsed = time.time() - start_time
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Time taken: {elapsed:.1f}s")
    print(f"Tasks evaluated: {total_tasks}")
    
    # Display some example predictions
    print("\nExample predictions:")
    for i, result in enumerate(results[:5]):
        print(f"\nTask {i+1}:")
        print(f"Input: {result['task']}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Predicted: {result['predicted']}")
            print(f"Expected: {result['expected']}")
            print(f"Correct: {result['correct']}")
    
    return accuracy

if __name__ == "__main__":
    evaluate_pipeline(num_layers=3)
