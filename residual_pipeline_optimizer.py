#!/usr/bin/env python3

import dspy
import json
import time
import numpy as np
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from residual_pipeline import SearchReplacePipeline, evaluate_pipeline

class PipelineOptimizer:
    def __init__(self):
        self.best_config = None
        self.best_accuracy = 0.0
        self.results_history = []
        
    def bootstrap_dataset(self, dataset: List[Dict], num_bootstrap: int = 5) -> List[Dict]:
        indices = np.random.choice(len(dataset), size=num_bootstrap, replace=True)
        return [dataset[i] for i in indices]
    
    def optimize(self, 
                dataset_path: str = "math_dataset.json",
                num_threads: int = 10,
                num_iterations: int = 10,
                bootstrap_size: int = 5) -> Dict:
        
        print("\nStarting Pipeline Bootstrap Optimization...")
        start_time = time.time()
        
        # Fixed configuration
        config = {
            'num_layers': 3,
            'temperature': 1.0
        }
        
        # Load dataset
        with open(dataset_path) as f:
            full_dataset = json.load(f)
        
        # Configure model
        lm = dspy.LM(model="deepseek/deepseek-chat", 
                     temperature=config['temperature'],
                     cache=False)
        dspy.settings.configure(lm=lm)
        
        with tqdm(total=num_iterations, desc="Bootstrap Iterations") as pbar:
            for iteration in range(num_iterations):
                # Create bootstrap sample
                bootstrap_data = self.bootstrap_dataset(full_dataset, bootstrap_size)
                
                # Train on bootstrap sample
                pipeline = SearchReplacePipeline(num_layers=config['num_layers'])
                for sample in bootstrap_data:
                    pipeline(sample['task'])  # Fine-tune on sample
                
                # Evaluate on remaining data
                accuracy = evaluate_pipeline(
                    dataset_path=dataset_path,
                    num_layers=config['num_layers'],
                    num_threads=num_threads
                )
                
                result = {
                    **config,
                    'accuracy': accuracy,
                    'iteration': iteration,
                    'timestamp': time.time()
                }
                self.results_history.append(result)
                
                # Update best accuracy
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_config = config
                
                pbar.update(1)
                pbar.set_postfix({
                    'best_acc': f'{self.best_accuracy:.1%}',
                    'iter': iteration
                })
        
        elapsed = time.time() - start_time
        
        # Print optimization results
        print("\nOptimization Results:")
        print(f"Time taken: {elapsed:.1f}s")
        print(f"Configurations tested: {len(configs)}")
        print(f"\nBest Configuration:")
        print(f"Number of layers: {self.best_config['num_layers']}")
        print(f"Temperature: {self.best_config['temperature']}")
        print(f"Accuracy: {self.best_accuracy:.1%}")
        
        # Print performance progression
        print("\nPerformance History:")
        for result in sorted(self.results_history, 
                           key=lambda x: x['accuracy'], 
                           reverse=True)[:5]:
            print(f"\nLayers: {result['num_layers']}, "
                  f"Temp: {result['temperature']:.1f}, "
                  f"Accuracy: {result['accuracy']:.1%}")
        
        return self.best_config

def main():
    optimizer = PipelineOptimizer()
    best_config = optimizer.optimize(num_iterations=10, bootstrap_size=5)
    
    print("\nRunning final evaluation...")
    final_accuracy = evaluate_pipeline(num_layers=3)  # Fixed 3 layers
    print(f"\nFinal accuracy: {final_accuracy:.1%}")

if __name__ == "__main__":
    main()
