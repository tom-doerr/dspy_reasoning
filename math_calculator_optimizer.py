#!/usr/bin/env python3

import dspy
import json
from dspy.teleprompt import MIPROv2
from math_calculator import MathCalculator, MathCalculationSignature

class MathOptimizer:
    def __init__(self):
        self.lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
        dspy.settings.configure(lm=self.lm)
        self.calculator = MathCalculator()
        
    def load_dataset(self, dataset_path="math_dataset.json"):
        with open(dataset_path) as f:
            dataset = json.load(f)
        return dataset[:100]  # Use subset for optimization

    def create_trainset(self, dataset):
        trainset = []
        for item in dataset:
            trainset.append(dspy.Example(
                task=item['task'],
                solution=item['solution']
            ).with_inputs('task'))
        return trainset

    def optimize(self, trainset, num_candidates=10):
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
            auto='medium',
            # requires_permission_to_run=False,
        )

        # Run optimization with required parameters
        optimized_calculator = teleprompter.compile(
            self.calculator,
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

def evaluate_model(calculator, dataset):
    correct = 0
    for item in dataset[:100]:  # Evaluate on first 100 samples
        try:
            result = calculator(task=item['task'])
            if abs(float(result.solution) - float(item['solution'])) < 0.01:
                correct += 1
        except:
            continue
    return correct / 100  # Return accuracy

if __name__ == "__main__":
    optimizer = MathOptimizer()
    
    # Load and prepare dataset
    dataset = optimizer.load_dataset()
    trainset = optimizer.create_trainset(dataset)
    
    # Initialize results tracking
    results = []
    current_calculator = optimizer.calculator
    
    # Evaluate initial model
    initial_accuracy = evaluate_model(current_calculator, dataset)
    results.append(("Initial", initial_accuracy))
    print(f"Initial accuracy: {initial_accuracy:.1%}")
    
    # Run multiple optimization iterations
    num_iterations = 3
    for i in range(num_iterations):
        print(f"\nStarting optimization iteration {i+1}/{num_iterations}...")
        
        # Run optimization
        optimized_calculator = optimizer.optimize(trainset, num_candidates=5)
        
        # Evaluate optimized model
        accuracy = evaluate_model(optimized_calculator, dataset)
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
