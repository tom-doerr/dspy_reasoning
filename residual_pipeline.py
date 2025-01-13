#!/usr/bin/env python3

import dspy
import json
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

class ResidualMathPipeline(dspy.Module):
    def __init__(self, max_iterations: int = 3):
        super().__init__()
        self.max_iterations = max_iterations
        self.calculate = dspy.ChainOfThought('task -> solution')
        self.refine = dspy.ChainOfThought('task, previous_solution -> refined_solution')

    def forward(self, task: str) -> str:
        """Forward pass with residual connections for iterative refinement"""
        result = self.calculate(task=task)
        
        # Iterative refinement with residual connections
        for _ in range(self.max_iterations - 1):
            refined = self.refine(task=task, previous_solution=result.solution)
            result = refined
            
        return result.refined_solution or result.solution

def evaluate_pipeline(dataset_path: str = "math_dataset.json", num_threads: int = 10) -> float:
    """Evaluate the pipeline on a math dataset"""
    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    # Initialize pipeline
    lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
    dspy.settings.configure(lm=lm)
    pipeline = ResidualMathPipeline()
    
    correct = 0
    total = len(dataset)
    
    def evaluate_task(task_data):
        try:
            predicted = pipeline(task_data['task'])
            expected = float(task_data['solution'])
            predicted_num = float(predicted)
            return abs(predicted_num - expected) < 0.01
        except (ValueError, TypeError):
            return False
    
    # Evaluate in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(evaluate_task, task_data)
            for task_data in dataset[:100]  # Evaluate on first 100 samples
        ]
        
        for future in as_completed(futures):
            correct += future.result()
    
    accuracy = correct / len(futures)
    print(f"Pipeline accuracy: {accuracy:.1%}")
    return accuracy

if __name__ == "__main__":
    evaluate_pipeline()
