#!/usr/bin/env python3

import dspy
import json
import time
import tqdm
from pprint import pprint

class MathCalculationSignature(dspy.Signature):
    """Solve math calculation tasks using chain-of-thought reasoning"""
    task = dspy.InputField(desc="The math calculation task to solve")
    context = dspy.InputField(desc="Context from previous iterations", default="")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning to solve the task")
    solution = dspy.OutputField(desc="The numerical solution to the task. Must be a number.")
    notes_output = dspy.OutputField(desc="Notes for next iteration", default="")
    iteration_control = dspy.OutputField(
        desc="Must be either 'continue' or 'terminate'. Use 'terminate' only when absolutely certain the solution is correct.",
        default="continue"
    )

class MathCalculator(dspy.Module):
    def __init__(self, max_iterations=5):
        super().__init__()
        self.calculate = dspy.ChainOfThought(MathCalculationSignature)
        self.max_iterations = max_iterations

    def forward(self, task):
        """Forward pass for the math calculator with iterative reasoning"""
        context = ""  # Accumulates all reasoning, solutions and notes
        final_reasoning = ""
        final_solution = ""
        
        for iteration in range(self.max_iterations):
            try:
                result = self.calculate(task=task, context=context)
                
                # Validate required fields
                if not all(hasattr(result, field) for field in ['reasoning', 'solution', 'notes_output', 'iteration_control']):
                    raise ValueError("Missing required fields in model output")
                    
                print("Iteration result:")
                print(f"Reasoning: {result.reasoning}")
                print(f"Solution: {result.solution}")
                print(f"Notes: {result.notes_output}")
                print(f"Control: {result.iteration_control}")
                print("-" * 40)
                
                # Accumulate reasoning
                final_reasoning += f"\nIteration {iteration + 1} Reasoning:\n{result.reasoning}"
                
                # Build context for next iteration
                iteration_context = (
                    f"Iteration {iteration + 1}:\n"
                    f"Reasoning: {result.reasoning}\n"
                    f"Solution: {result.solution}\n"
                    f"Notes: {result.notes_output}\n"
                )
                context += "\n" + iteration_context
                
                # Store the latest solution
                final_solution = result.solution
                
                # Check if we should terminate
                if result.iteration_control.lower().strip() == "terminate":
                    break
                    
            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {str(e)}")
                continue
                
        return dspy.Prediction(
            reasoning=final_reasoning,
            solution=final_solution,
            notes_output=context
        )

    def evaluate_on_dataset(self, dataset_path="math_dataset.json", max_iter=None):
        start_time = time.time()
        
        with open(dataset_path) as f:
            dataset = json.load(f)
        
        # Evaluate on first 1000 samples
        dataset = dataset[:1000]
        
        # Use provided max_iter or default to instance value
        eval_iter = max_iter if max_iter is not None else self.max_iterations
        
        # Results storage
        results = {
            "correct": 0,
            "time": 0
        }
        
        # Evaluate all samples
        for i, item in enumerate(tqdm.tqdm(dataset, ncols=60), 1):
            task = item['task']
            expected_solution = item['solution']
            
            # Evaluate with specified max iterations
            iter_start = time.time()
            result = self._forward_with_max_iter(task, max_iter=eval_iter)
            results["time"] += time.time() - iter_start
            
            if self._is_correct(result.solution, expected_solution):
                results["correct"] += 1
                
            # Print progress
            if i % 100 == 0:
                print(f"\nProgress after {i} samples:")
                print(f"Correct: {results['correct']}/{i} ({results['correct']/i:.1%})")
                print(f"Time: {results['time']:.2f}s")
                
        # Calculate final metrics
        total_time = time.time() - start_time
        results["accuracy"] = results["correct"] / len(dataset)
        results["total_time"] = total_time
        results["max_iter"] = eval_iter
        
        # Print final results
        print("\nEvaluation Results:")
        print(f"Max Iterations: {eval_iter}")
        print(f"Correct Answers: {results['correct']}/{len(dataset)} ({results['accuracy']:.1%})")
        print(f"Total Time: {results['total_time']:.2f}s")
        
        # Save results
        with open("math_calculator_benchmark.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print("\nBenchmark results saved to math_calculator_benchmark.json")
        return results
        
    def _forward_with_max_iter(self, task, max_iter):
        """Modified forward pass with configurable max iterations"""
        context = ""
        final_reasoning = ""
        final_solution = ""
        
        for iteration in range(max_iter):
            try:
                result = self.calculate(task=task, context=context)
                
                # Validate required fields
                if not all(hasattr(result, field) for field in ['reasoning', 'solution', 'notes_output', 'iteration_control']):
                    raise ValueError("Missing required fields in model output")
                    
                # Accumulate reasoning
                final_reasoning += f"\nIteration {iteration + 1} Reasoning:\n{result.reasoning}"
                
                # Build context for next iteration
                iteration_context = (
                    f"Iteration {iteration + 1}:\n"
                    f"Reasoning: {result.reasoning}\n"
                    f"Solution: {result.solution}\n"
                    f"Notes: {result.notes_output}\n"
                )
                context += "\n" + iteration_context
                
                # Store the latest solution
                final_solution = result.solution
                
                # Check if we should terminate
                if result.iteration_control.lower().strip() == "terminate":
                    break
                    
            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {str(e)}")
                continue
                
        return dspy.Prediction(
            reasoning=final_reasoning,
            solution=final_solution,
            notes_output=context
        )
        
    def _is_correct(self, predicted, expected):
        """Compare solutions with tolerance for floating point"""
        try:
            return abs(float(predicted) - float(expected)) < 0.01
        except Exception as e:
            print(f"⚠️ Error evaluating solution: {str(e)}")
            return False
            

if __name__ == "__main__":
    # Configure DSPy
    lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
    dspy.settings.configure(lm=lm)
    
    # Create calculator instances
    calculator_iter1 = MathCalculator(max_iterations=1)
    calculator_iter5 = MathCalculator(max_iterations=5)
    
    # Evaluate both configurations
    print("\nEvaluating with max_iter=1...")
    results_iter1 = calculator_iter1.evaluate_on_dataset()
    
    print("\nEvaluating with max_iter=5...")
    results_iter5 = calculator_iter5.evaluate_on_dataset()
    
    # Print comparison
    print("\nComparison Results:")
    print(f"Max Iter=1: Accuracy={results_iter1['accuracy']:.1%}, Time={results_iter1['total_time']:.2f}s")
    print(f"Max Iter=5: Accuracy={results_iter5['accuracy']:.1%}, Time={results_iter5['total_time']:.2f}s")
    
    # Save comparison results
    comparison = {
        "max_iter_1": results_iter1,
        "max_iter_5": results_iter5
    }
    with open("math_calculator_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("\nComparison results saved to math_calculator_comparison.json")
