#!/usr/bin/env python3

import dspy
import json
from dspy.teleprompt import MIPROv2, BootstrapFewShotWithRandomSearch, BootstrapFewShot
from math_calculator import MathCalculator, MathCalculationSignature
import tqdm
import logging

logging.basicConfig(filename='optimization_log.txt', level=logging.INFO)

# Set global tqdm configuration
tqdm.tqdm.pandas()
tqdm.tqdm.get_lock().locks = []
tqdm.tqdm.ncols = 60

class MathOptimizer:
    def __init__(self):
        self.lm = dspy.LM(model="deepseek/deepseek-chat", temperature=1.5, cache=False)
        dspy.settings.configure(lm=self.lm)
        self.calculator = MathCalculator()
        self.student = None
        self.teacher = None
        
    def set_student(self, student):
        """Set the student model for optimization"""
        self.student = student
        
    def set_teacher(self, teacher):
        """Set the teacher model for optimization"""
        self.teacher = teacher
        
    def load_dataset(self, dataset_path="math_dataset.json"):
        with open(dataset_path) as f:
            dataset = json.load(f)
        # First 100 samples for validation, rest for training
        return dataset[100:], dataset[:100]

    def create_trainset(self, dataset):
        trainset = []
        # for item in tqdm.tqdm(dataset[:100], ncols=60):
        for item in dataset:
            trainset.append(dspy.Example(
                task=item['task'],
                solution=item['solution']
            ).with_inputs('task'))
        return trainset

    def optimize(self, trainset, num_candidates=10, base_model=None):
        # Define the metric function with subtask reasoning evaluation
        def metric(example, prediction, trace=None):
            try:
                # Handle both string and numeric solutions
                pred_solution = float(prediction.solution) if isinstance(prediction.solution, str) else prediction.solution
                exp_solution = float(example.solution) if isinstance(example.solution, str) else example.solution
                
                # Compare with tolerance for floating point numbers
                accuracy = int(abs(pred_solution - exp_solution) < 0.01)
                
                # Evaluate subtask reasoning quality
                if hasattr(prediction, 'reasoning'):
                    reasoning = prediction.reasoning.lower()
                    # Check for subtask indicators
                    if 'subtask' in reasoning or 'step' in reasoning or 'part' in reasoning:
                        # Additional points for using subtask reasoning
                        accuracy += 1
                        # Check for proper combination of subtasks
                        if 'combine' in reasoning or 'final result' in reasoning:
                            accuracy += 1
                
                return min(accuracy, 1)  # Cap at 1 to maintain binary metric
            except (ValueError, TypeError, AttributeError) as e:
                print(f"Metric error: {e}")
                return 0

        # Configure MIPRO optimizer with subtask reasoning focus
        teleprompter = MIPROv2(
            metric=metric,
            num_candidates=num_candidates,
            init_temperature=1.0,
            prompt_model=self.lm,
            task_model=self.lm,
            num_threads=100,
            auto='light',
            track_stats=True,
            additional_constraints=[
                "Encourage splitting complex problems into subtasks",
                "Ensure proper combination of subtask results",
                "Maintain clear reasoning between subtasks"
            ]
        )

        # Set student and teacher if not already set
        if self.student is None:
            self.set_student(base_model)
        if self.teacher is None:
            self.set_teacher(base_model)

        # Run optimization with subtask reasoning focus
        optimized_calculator = teleprompter.compile(
            student=self.student,
            teacher=self.teacher,
            trainset=trainset,
            num_trials=7,
            max_bootstrapped_demos=3,
            max_labeled_demos=4,
            requires_permission_to_run=False,
            minibatch=True,
            optimization_focus=[
                "Task splitting accuracy", 
                "Subtask reasoning quality",
                "Result combination correctness"
            ]
        )


        return optimized_calculator

    def save_optimized_model(self, optimized_calculator, path="optimized_models/optimized_math_calculator.json"):
        optimized_calculator.save(path)
        print(f"Optimized model saved to {path}")

from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_single_task(calculator, item):
    try:
        result = calculator.forward(item['task'])
        # Handle both string and numeric solutions
        pred_solution = float(result.solution) if isinstance(result.solution, str) else result.solution
        exp_solution = float(item['solution']) if isinstance(item['solution'], str) else item['solution']
        
        # Compare with tolerance for floating point numbers
        return int(abs(pred_solution - exp_solution) < 0.01)
    except (ValueError, TypeError, AttributeError) as e:
        print(f"Evaluation error for task {item['task']}: {e}")
        return 0

def evaluate_model(calculator, dataset, num_threads=10):
    correct = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(evaluate_single_task, calculator, item)
            for item in dataset[:100]
        ]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), 
                              ncols=60):
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
    initial_accuracy = evaluate_model(current_calculator, val_data, num_threads=20)
    results.append(("Initial", initial_accuracy))
    print(f"Initial accuracy: {initial_accuracy:.1%}")

    current_calculator_student = current_calculator.deepcopy()
    
    # Run multiple optimization iterations with memory cleanup
    num_iterations = 10
    for i in range(num_iterations):
        print(f"\nStarting optimization iteration {i+1}/{num_iterations}...")
        current_calculator = current_calculator.deepcopy()
        current_calculator = current_calculator.reset_copy()
        
        # Set student and teacher for this iteration
        optimizer.set_student(current_calculator_student)
        optimizer.set_teacher(current_calculator)
        
        # Run optimization on current calculator
        optimized_calculator = optimizer.optimize(trainset, num_candidates=3, base_model=current_calculator)
        
        # Evaluate optimized model on validation set
        accuracy = evaluate_model(optimized_calculator, val_data, num_threads=20)
        
        # Explicit memory cleanup
        del current_calculator
        import gc
        gc.collect()
        current_calculator = optimized_calculator
        results.append((f"Iteration {i+1}", accuracy))
        print(f"Optimization iteration {i+1} accuracy: {accuracy:.1%}")
        
        # Save optimized model to optimized_models directory
        import os
        os.makedirs("optimized_models", exist_ok=True)
        model_path = f"optimized_models/optimized_math_calculator_iter{i+1}.json"
        optimizer.save_optimized_model(optimized_calculator, model_path)
        
        # Set as current calculator for next iteration
        current_calculator = optimized_calculator
    
    # Print all results
    print("\nFinal Results:")
    for stage, accuracy in results:
        print(f"{stage}: {accuracy:.1%}")
    
    print("Optimization complete!")
