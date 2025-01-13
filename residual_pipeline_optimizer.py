#!/usr/bin/env python3

import dspy
import json
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from residual_pipeline import SearchReplacePipeline, evaluate_pipeline

class PipelineOptimizer:
    def __init__(self):
        self.best_config = None
        self.best_accuracy = 0.0
        self.results_history = []
        
    def optimize(self, 
                dataset_path: str = "math_dataset.json",
                min_layers: int = 2,
                max_layers: int = 5,
                temperatures: List[float] = [0.1, 0.3, 0.5, 0.7],
                num_threads: int = 10) -> Dict:
        
        print("\nStarting Pipeline Optimization...")
        start_time = time.time()
        
        configs = []
        for num_layers in range(min_layers, max_layers + 1):
            for temp in temperatures:
                configs.append({
                    'num_layers': num_layers,
                    'temperature': temp
                })
        
        with tqdm(total=len(configs), desc="Testing Configurations") as pbar:
            for config in configs:
                # Configure model
                lm = dspy.LM(model="deepseek/deepseek-chat", 
                            temperature=config['temperature'],
                            cache=False)
                dspy.settings.configure(lm=lm)
                
                # Evaluate configuration
                accuracy = evaluate_pipeline(
                    dataset_path=dataset_path,
                    num_layers=config['num_layers'],
                    num_threads=num_threads
                )
                
                result = {
                    **config,
                    'accuracy': accuracy,
                    'timestamp': time.time()
                }
                self.results_history.append(result)
                
                # Update best config
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_config = config
                
                pbar.update(1)
                pbar.set_postfix({
                    'best_acc': f'{self.best_accuracy:.1%}',
                    'layers': config['num_layers'],
                    'temp': config['temperature']
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
    best_config = optimizer.optimize()
    
    print("\nRunning final evaluation with best configuration...")
    lm = dspy.LM(model="deepseek/deepseek-chat",
                 temperature=best_config['temperature'],
                 cache=False)
    dspy.settings.configure(lm=lm)
    
    final_accuracy = evaluate_pipeline(num_layers=best_config['num_layers'])
    print(f"\nFinal accuracy with optimized configuration: {final_accuracy:.1%}")

if __name__ == "__main__":
    main()
