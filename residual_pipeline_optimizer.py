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
                use_mipro: bool = False) -> Dict:
        
        print("\nStarting Pipeline Optimization...")
        start_time = time.time()
        
        # Fixed configuration
        config = {
            'num_layers': 3,
            'temperature': 1.0,
            'model': "deepseek/deepseek-chat"
        }
        
        # Load dataset
        with open(dataset_path) as f:
            full_dataset = json.load(f)
            
        # Create trainset for MIPROv2
        trainset = []
        for item in full_dataset[:100]:  # Use first 100 examples for training
            trainset.append(dspy.Example(
                task=item['task'],
                solution=item['solution']
            ).with_inputs('task'))
            
        if use_mipro:
            print("\nUsing MIPROv2 optimizer...")
            # Configure model
            lm = dspy.LM(model=config['model'],
                        temperature=config['temperature'],
                        cache=False)
            dspy.settings.configure(lm=lm)
            
            # Define metric function
            def metric(example, prediction, trace=None):
                try:
                    pred = float(prediction.solution)
                    exp = float(example.solution)
                    return int(abs(pred - exp) < 0.01)
                except:
                    return 0
                    
            # Configure MIPROv2
            teleprompter = dspy.teleprompt.MIPROv2(
                metric=metric,
                num_candidates=3,
                num_threads=num_threads,
                temperature=config['temperature'],
                max_bootstrapped_demos=3,
                max_labeled_demos=4
            )
            
            # Create and optimize pipeline
            pipeline = SearchReplacePipeline(num_layers=config['num_layers'])
            optimized_pipeline = teleprompter.compile(
                student=pipeline,
                trainset=trainset,
                num_trials=5
            )
            
            # Evaluate optimized pipeline
            accuracy = evaluate_pipeline(
                dataset_path=dataset_path,
                num_layers=config['num_layers'],
                num_threads=num_threads,
                model=config['model'],
                temperature=config['temperature']
            )
            
            result = {
                **config,
                'accuracy': accuracy,
                'timestamp': time.time()
            }
            self.results_history.append(result)
            
            # Update best accuracy
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_config = config
        else:
            # Just evaluate baseline pipeline
            accuracy = evaluate_pipeline(
                dataset_path=dataset_path,
                num_layers=config['num_layers'],
                num_threads=num_threads,
                model=config['model'],
                temperature=config['temperature']
            )
            
            result = {
                **config,
                'accuracy': accuracy,
                'timestamp': time.time()
            }
            self.results_history.append(result)
            
            # Update best accuracy
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_config = config
        
        elapsed = time.time() - start_time
        
        # Print optimization results
        print("\nOptimization Results:")
        print(f"Time taken: {elapsed:.1f}s")
        print(f"Bootstrap iterations completed: {len(self.results_history)}")
        print(f"\nBest Configuration:")
        print(f"Number of layers: {self.best_config['num_layers']}")
        print(f"Temperature: {self.best_config['temperature']}")
        print(f"Accuracy: {self.best_accuracy:.1%}")
        
        # Print performance progression if we have results
        if self.results_history:
            print("\nPerformance History:")
            for result in sorted(self.results_history, 
                               key=lambda x: x['accuracy'], 
                               reverse=True)[:5]:
                print(f"\nLayers: {result['num_layers']}, "
                      f"Temp: {result['temperature']:.1f}, "
                      f"Accuracy: {result['accuracy']:.1%}")
        
        return self.best_config

def main():
    # Test with and without MIPROv2
    optimizer = PipelineOptimizer()
    use_mipro = True
    
    baseline_config = optimizer.optimize(use_mipro=use_mipro)
    

if __name__ == "__main__":
    main()
