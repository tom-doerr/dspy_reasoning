#!/usr/bin/env python3

import dspy
import json
from dspy.teleprompt import MIPROv2
from math_calculator import MathCalculator, MathCalculationSignature
import tqdm

class MathOptimizer:
    def __init__(self):
        self.lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
        dspy.settings.configure(lm=self.lm)
        self.calculator = MathCalculator()
        
    def load_dataset(self, dataset_path="math_dataset.json"):
        with open(dataset_path) as f:
            dataset = json.load(f)
        # First 100 samples for validation, rest for training
        return dataset[100:], dataset[:100]

    def create_trainset(self, dataset):
        trainset = []
        for item in dataset:
        # for item in tqdm.tqdm(dataset):
            trainset.append(dspy.Example(
                task=item['task'],
                solution=item['solution']
            ).with_inputs('task'))
        return trainset

    def optimize(self, trainset, num_candidates=10, base_model=None):
        # Define the metric function
        def metric(example, prediction, trace=None):
            try:
                return int(abs(float(prediction.solution) - float(example.solution)) < 0.01)
            except:
                return 0

        # Configure MIPRO optimizer
        teleprompter = MIPROv2(
            metric=metric,
            num_candidates=num_candidates,
            init_temperature=1.0,
            prompt_model=self.lm,
            task_model=self.lm,
            num_threads=10,
            auto='light',
            # requires_permission_to_run=False,
        )

        # Run optimization with required parameters
        optimized_calculator = teleprompter.compile(
            base_model if base_model else self.calculator,
            trainset=trainset,
            num_trials=10,  # Number of optimization trials
            max_bootstrapped_demos=3,  # Max bootstrapped examples
            max_labeled_demos=4,  # Max labeled examples
            requires_permission_to_run=False,
            # eval_kwargs={"num_threads": 1}  # Evaluation settings
        )

        return optimized_calculator

    def save_optimized_model(self, optimized_calculator, path="optimized_math_calculator.json"):
        optimized_calculator.save(path)
        print(f"Optimized model saved to {path}")

from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_single_task(calculator, item):
    try:
        result = calculator(task=item['task'])
        return int(abs(float(result.solution) - float(item['solution'])) < 0.01)
    except:
        return 0

def evaluate_model(calculator, dataset, num_threads=10):
    correct = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(evaluate_single_task, calculator, item)
            for item in dataset[:100]
        ]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            correct += future.result()
    return correct / 100  # Return accuracy

if __name__ == "__main__":
    optimizer = MathOptimizer()
    
    # Load and split dataset
    train_data, val_data = optimizer.load_dataset()
    trainset = optimizer.create_trainset(train_data)
    
    # Initialize results tracking
    results = []
    current_calculator = optimizer.calculator
    
    # Evaluate initial model on validation set
    print("Evaluating initial model...")
    initial_accuracy = evaluate_model(current_calculator, val_data, num_threads=10)
    results.append(("Initial", initial_accuracy))
    print(f"Initial accuracy: {initial_accuracy:.1%}")
    
    # Run multiple optimization iterations
    num_iterations = 3
    for i in range(num_iterations):
        print(f"\nStarting optimization iteration {i+1}/{num_iterations}...")
        
        # Run optimization on current calculator
        optimized_calculator = optimizer.optimize(trainset, num_candidates=5, base_model=current_calculator)
        
        # Evaluate optimized model on validation set
        accuracy = evaluate_model(optimized_calculator, val_data, num_threads=10)
        results.append((f"Iteration {i+1}", accuracy))
        print(f"Optimization iteration {i+1} accuracy: {accuracy:.1%}")
        
        # Save optimized model
        model_path = f"optimized_math_calculator_iter{i+1}.json"
        optimizer.save_optimized_model(optimized_calculator, model_path)
        
        # Set as current calculator for next iteration
        current_calculator = optimized_calculator
    
    # Print all results
    print("\nFinal Results:")
    for stage, accuracy in results:
        print(f"{stage}: {accuracy:.1%}")
    
    print("Optimization complete!")
