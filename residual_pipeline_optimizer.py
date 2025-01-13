#!/usr/bin/env python3

import dspy
import json
import time
import numpy as np
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from residual_pipeline import SearchReplacePipeline, evaluate_pipeline
from residual_pipeline import SearchReplaceIterPipeline

PIPELINE_TYPE_STANDARD = "standard"
PIPELINE_TYPE_ITER = "iter"

class PipelineOptimizer:
    def __init__(self, pipeline_type: str = PIPELINE_TYPE_STANDARD):
        self.best_config = None
        self.best_accuracy = 0.0
        self.results_history = []
        self.pipeline_type = pipeline_type
        
    def _create_teleprompter(self, metric, num_threads):
        """Create and configure MIPROv2 teleprompter"""
        return dspy.teleprompt.MIPROv2(
            metric=metric,
            num_candidates=3,
            num_threads=num_threads,
            max_bootstrapped_demos=3,
            max_labeled_demos=4,
            auto='light',
            bootstrap_fewshot=True
        )

    def bootstrap_dataset(self, dataset: List[Dict], num_bootstrap: int = 5) -> List[Dict]:
        indices = np.random.choice(len(dataset), size=num_bootstrap, replace=True)
        return [dataset[i] for i in indices]
    
    def _create_pipeline(self, config):
        """Create appropriate pipeline based on configured type"""
        if self.pipeline_type == PIPELINE_TYPE_ITER:
            return SearchReplaceIterPipeline(num_iters=config['num_layers'])
        return SearchReplacePipeline(num_layers=config['num_layers'])

    def _evaluate_pipeline(self, config, dataset_path, num_threads):
        """Evaluate pipeline with given configuration"""
        return evaluate_pipeline(
            dataset_path=dataset_path,
            num_layers=config['num_layers'],
            num_threads=num_threads,
            model=config['model'],
            temperature=config['temperature']
        )

    def _get_default_config(self) -> Dict:
        """Get default configuration for optimization"""
        return {
            'num_layers': 10,
            'temperature': 1.0,
            'model': "deepseek/deepseek-chat"
        }

    def _load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load dataset from JSON file"""
        with open(dataset_path) as f:
            return json.load(f)

    def _create_trainset(self, dataset: List[Dict]) -> List[dspy.Example]:
        """Create training set from dataset"""
        trainset = []
        for item in dataset[:100]:  # Use first 100 examples for training
            trainset.append(dspy.Example(
                task=item['task'],
                solution=item['solution']
            ).with_inputs('task'))
        return trainset

    def _configure_model(self, config: Dict) -> None:
        """Configure DSPy language model"""
        lm = dspy.LM(
            model=config['model'],
            temperature=config['temperature'],
            cache=False
        )
        dspy.settings.configure(lm=lm)

    def _create_fewshot_examples(self, trainset: List[dspy.Example]) -> List[dspy.Example]:
        """Create few-shot examples from training set"""
        fewshot_examples = []
        for example in trainset[:5]:  # Use first 5 examples for few-shot
            fewshot_examples.append(dspy.Example(
                task=example.task,
                solution=example.solution
            ).with_inputs('task'))
        return fewshot_examples

    def optimize(self, 
                dataset_path: str = "math_dataset.json",
                num_threads: int = 100,
                use_mipro: bool = False) -> Dict:
        
        print("\nStarting Pipeline Optimization...")
        start_time = time.time()
        
        config = self._get_default_config()
        
        full_dataset = self._load_dataset(dataset_path)
        trainset = self._create_trainset(full_dataset)
            
        if use_mipro:
            print("\nUsing MIPROv2 optimizer...")
            self._configure_model(config)
            
            # Define metric function
            def metric(example, prediction, trace=None):
                try:
                    pred = float(prediction.solution)
                    exp = float(example.solution)
                    return int(abs(pred - exp) < 0.01)
                except:
                    return 0
                    
            teleprompter = self._create_teleprompter(metric, num_threads)
            
            # Create few-shot examples
            fewshot_examples = self._create_fewshot_examples(trainset)
            
            pipeline = self._create_pipeline(config)
            optimized_pipeline = teleprompter.compile(
                student=pipeline,
                trainset=trainset,
                requires_permission_to_run=False,
                num_trials=5,
                fewshot_examples=fewshot_examples
            )
            
            accuracy = self._evaluate_pipeline(config, dataset_path, num_threads)
            
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
    # pipeline_type = PIPELINE_TYPE_STANDARD  # Change to PIPELINE_TYPE_ITER to use iterative pipeline
    pipeline_type = PIPELINE_TYPE_ITER
    optimizer = PipelineOptimizer(pipeline_type=pipeline_type)
    # use_mipro = True
    use_mipro = False
    
    baseline_config = optimizer.optimize(use_mipro=use_mipro)
    

if __name__ == "__main__":
    main()
