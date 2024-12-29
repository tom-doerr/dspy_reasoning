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
    def __init__(self):
        super().__init__()
        self.calculate = dspy.ChainOfThought(MathCalculationSignature)

    def forward(self, task):
        """Forward pass for the math calculator with iterative reasoning"""
        max_iterations = 5
        context = ""  # Accumulates all reasoning, solutions and notes
        final_reasoning = ""
        final_solution = ""
        
        for iteration in range(max_iterations):
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

    def evaluate_on_dataset(self, dataset_path="math_dataset.json"):
        start_time = time.time()
        
        with open(dataset_path) as f:
            dataset = json.load(f)
        
        # Evaluate on first 1000 samples
        dataset = dataset[:1000]
        
        # Create a copy with max_iter=1
        single_iter_calculator = self.deepcopy()
        single_iter_calculator.forward = lambda task: self._forward_with_max_iter(task, max_iter=1)
        
        # Results storage
        results = {
            "max_iter_5": {"correct": 0, "time": 0},
            "max_iter_1": {"correct": 0, "time"}
        }
        
        # Evaluate both versions
        for i, item in enumerate(tqdm.tqdm(dataset, ncols=60), 1):
            task = item['task']
            expected_solution = item['solution']
            
            # Evaluate max_iter=5 version
            iter5_start = time.time()
            result5 = self.forward(task)
            results["max_iter_5"]["time"] += time.time() - iter5_start
            if self._is_correct(result5.solution, expected_solution):
                results["max_iter_5"]["correct"] += 1
                
            # Evaluate max_iter=1 version
            iter1_start = time.time()
            result1 = single_iter_calculator.forward(task)
            results["max_iter_1"]["time"] += time.time() - iter1_start
            if self._is_correct(result1.solution, expected_solution):
                results["max_iter_1"]["correct"] += 1
                
            # Print progress
            if i % 100 == 0:
                print(f"\nProgress after {i} samples:")
                self._print_results(results, i)
                
        # Calculate final metrics
        total_time = time.time() - start_time
        results["max_iter_5"]["accuracy"] = results["max_iter_5"]["correct"] / len(dataset)
        results["max_iter_1"]["accuracy"] = results["max_iter_1"]["correct"] / len(dataset)
        results["total_time"] = total_time
        
        # Print final results
        print("\nFinal Results:")
        self._print_results(results, len(dataset))
        
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
            
    def _print_results(self, results, total):
        """Print formatted results"""
        print("\nMax Iterations Comparison:")
        print(f"Max Iter=5: {results['max_iter_5']['correct']}/{total} ({results['max_iter_5']['correct']/total:.1%})")
        print(f"Max Iter=1: {results['max_iter_1']['correct']}/{total} ({results['max_iter_1']['correct']/total:.1%})")
        print(f"\nTime Comparison:")
        print(f"Max Iter=5: {results['max_iter_5']['time']:.2f}s")
        print(f"Max Iter=1: {results['max_iter_1']['time']:.2f}s")

if __name__ == "__main__":
    # Configure DSPy
    lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
    dspy.settings.configure(lm=lm)
    
    # Train and evaluate calculator
    calculator = MathCalculator()
    calculator.evaluate_on_dataset()
