#!/usr/bin/env python3

import dspy
import json
import time
from tqdm import tqdm

class MathCalculationSignature(dspy.Signature):
    """Solve math calculation tasks using chain-of-thought reasoning"""
    task = dspy.InputField(desc="The math calculation task to solve")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning to solve the task")
    solution = dspy.OutputField(desc="The numerical solution to the task")

class MathCalculator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.calculate = dspy.ChainOfThought(MathCalculationSignature)

    def forward(self, task):
        """Forward pass for the math calculator"""
        result = self.calculate(task=task)
        return dspy.Prediction(
            reasoning=result.reasoning,
            solution=result.solution
        )

    def evaluate_on_dataset(self, dataset_path="math_dataset.json"):
        start_time = time.time()
        
        with open(dataset_path) as f:
            dataset = json.load(f)
        
        correct = 0
        total = len(dataset)
        dataset = dataset[:100]  # Evaluate on a subset of the dataset
        
        for i, item in enumerate(dataset, 1):
            task = item['task']
            expected_solution = item['solution']
            
            print(f"\n--- Evaluating Task {i}/{len(dataset)} ---")
            print(f"Task: {task}")
            print(f"Expected Solution: {expected_solution}")
            
            result = self.calculate(task=task)
            
            print(f"\nModel Reasoning:")
            print(result.reasoning)
            print(f"\nModel Solution: {result.solution}")
            
            try:
                # Compare solutions with some tolerance for floating point
                if abs(float(result.solution) - float(expected_solution)) < 0.01:
                    correct += 1
                    print("✅ Correct")
                else:
                    print("❌ Incorrect")
            except Exception as e:
                print(f"⚠️ Error evaluating solution: {str(e)}")
                continue
            
            print(f"Current Accuracy: {correct}/{i} ({correct/i:.1%})")
                
        pipeline_metrics = {
            "total_tasks": len(dataset),  # Use actual number of evaluated tasks
            "correct_answers": correct,
            "accuracy": correct / len(dataset),
            "time_seconds": time.time() - start_time
        }
        
        print(f"\nMath Calculator Evaluation Results:")
        print(f"Total Tasks: {pipeline_metrics['total_tasks']}")
        print(f"Correct Answers: {pipeline_metrics['correct_answers']}")
        print(f"Accuracy: {pipeline_metrics['accuracy']:.1%}")
        print(f"Time: {pipeline_metrics['time_seconds']:.2f} seconds")
        
        # Save results
        with open("math_calculator_benchmark.json", "w") as f:
            json.dump(pipeline_metrics, f, indent=2)
            
        print("\nBenchmark results saved to math_calculator_benchmark.json")
        return pipeline_metrics

if __name__ == "__main__":
    # Configure DSPy
    lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
    dspy.settings.configure(lm=lm)
    
    # Train and evaluate calculator
    calculator = MathCalculator()
    calculator.evaluate_on_dataset()
