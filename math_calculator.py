#!/usr/bin/env python3

import dspy
import json
import time
import tqdm
from pprint import pprint
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from signatures import SolutionSelectorSignature, MathCalculationSignature
from math_evaluator import MathEvaluator

class MathCalculator(dspy.Module):
    def __init__(self, max_iterations=5, num_attempts=3):
        super().__init__()
        self.calculate = dspy.ChainOfThought(MathCalculationSignature)
        self.select_solution = dspy.ChainOfThought(SolutionSelectorSignature)
        self.max_iterations = max_iterations
        self.num_attempts = num_attempts

    def _split_task(self, task):
        """Split a complex task into subtasks"""
        # Split on operators while preserving them
        operators = ['+', '-', '*', '/', '^', '√', '%']
        subtasks = []
        current_task = ""
        
        for char in task:
            if char in operators:
                if current_task:
                    subtasks.append(current_task.strip())
                subtasks.append(char)
                current_task = ""
            else:
                current_task += char
                
        if current_task:
            subtasks.append(current_task.strip())
            
        return subtasks

    def _combine_subtask_results(self, subtask_results):
        """Combine results from subtasks into a final solution"""
        # Build and evaluate the combined expression
        combined_expression = ""
        for result in subtask_results:
            if result.solution in ['+', '-', '*', '/', '^', '√', '%']:
                combined_expression += f" {result.solution} "
            else:
                combined_expression += f" {result.solution} "
                
        try:
            # Use safe evaluation
            import operator
            import math
            allowed_operators = {
                '+': operator.add,
                '-': operator.sub,
                '*': operator.mul,
                '/': operator.truediv,
                '^': operator.pow,
                '%': operator.mod,
                '√': math.sqrt
            }
            
            # Parse and evaluate the expression safely
            stack = []
            for token in combined_expression.split():
                if token in allowed_operators:
                    if token == '√':
                        operand = stack.pop()
                        stack.append(allowed_operators[token](operand))
                    else:
                        right = stack.pop()
                        left = stack.pop()
                        stack.append(allowed_operators[token](left, right))
                else:
                    try:
                        stack.append(float(token))
                    except ValueError:
                        stack.append(0)  # Default to 0 for invalid tokens
            return str(stack[0]) if stack else "0"
        except Exception as e:
            print(f"Error combining subtasks: {e}")
            return "0"

    def forward(self, task):
        """Forward pass for the math calculator with task splitting"""
        # First try to split the task into subtasks
        subtasks = self._split_task(task)
        
        if len(subtasks) > 1:
            # Process each subtask independently
            subtask_results = []
            for subtask in subtasks:
                if subtask in ['+', '-', '*', '/', '^', '√', '%']:
                    # Keep operators as-is
                    subtask_results.append(dspy.Prediction(
                        reasoning="Operator",
                        solution=subtask,
                        notes_output=""
                    ))
                else:
                    # Process numerical subtasks
                    result = self._forward_with_max_iter(subtask, self.max_iterations)
                    subtask_results.append(result)
            
            # Combine subtask results
            final_solution = self._combine_subtask_results(subtask_results)
            final_reasoning = "\n".join(
                f"Subtask {i+1} ({subtask}):\n{r.reasoning}\nSolution: {r.solution}\n" 
                for i, (subtask, r) in enumerate(zip(subtasks, subtask_results))
            )
            
            return dspy.Prediction(
                reasoning=f"Task split into {len(subtasks)} subtasks:\n{final_reasoning}",
                solution=final_solution,
                notes_output="Task split into subtasks"
            )
        else:
            # Fall back to original processing if no subtasks found
            attempts = []
            
            # Run multiple attempts
            for attempt in range(self.num_attempts):
                context = ""
                final_reasoning = ""
                final_solution = ""
                
                for iteration in range(self.max_iterations):
                    try:
                        result = self.calculate(task=task, context=context)
                        
                        # Validate required fields
                        if not all(hasattr(result, field) for field in ['reasoning', 'solution', 'notes_output', 'iteration_control']):
                            raise ValueError("Missing required fields in model output")
                            
                        # Accumulate reasoning
                    final_reasoning += f"\nAttempt {attempt + 1}, Iteration {iteration + 1} Reasoning:\n{result.reasoning}"
                    
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
                    print(f"Error in attempt {attempt + 1}, iteration {iteration + 1}: {str(e)}")
                    continue
                    
            attempts.append({
                'reasoning': final_reasoning,
                'solution': final_solution,
                'notes_output': context
            })
        
        # Select the best solution
        selection_result = self.select_solution(
            task=task,
            solutions=[f"Attempt {i+1}:\nReasoning: {a['reasoning']}\nSolution: {a['solution']}" 
                      for i, a in enumerate(attempts)]
        )
        
        # Find the selected solution
        selected_solution = selection_result.selected_solution
        selection_reasoning = selection_result.selection_reasoning
        
        # Try to match the selected solution
        for attempt in attempts:
            if attempt['solution'] == selected_solution:
                # Add selection reasoning to the final output
                final_reasoning = (
                    f"Selected Solution Reasoning:\n{selection_reasoning}\n\n"
                    f"Solution Details:\n{attempt['reasoning']}"
                )
                return dspy.Prediction(
                    reasoning=final_reasoning,
                    solution=attempt['solution'],
                    notes_output=attempt['notes_output']
                )
                
        # If no solution was selected, choose the most consistent one
        if len(attempts) > 1:
            # Find the most common solution
            solution_counts = Counter(a['solution'] for a in attempts)
            most_common_solution = solution_counts.most_common(1)[0][0]
            
            # Return the first attempt with the most common solution
            for attempt in attempts:
                if attempt['solution'] == most_common_solution:
                    final_reasoning = (
                        "No clear selection - using most consistent solution:\n"
                        f"Solution appeared {solution_counts[most_common_solution]} times\n\n"
                        f"Solution Details:\n{attempt['reasoning']}"
                    )
                    return dspy.Prediction(
                        reasoning=final_reasoning,
                        solution=attempt['solution'],
                        notes_output=attempt['notes_output']
                    )
                    
        # Fall back to the first attempt
        final_reasoning = (
            "Using first attempt as fallback solution\n\n"
            f"Solution Details:\n{attempts[0]['reasoning']}"
        )
        return dspy.Prediction(
            reasoning=final_reasoning,
            solution=attempts[0]['solution'],
            notes_output=attempts[0]['notes_output']
        )

    def evaluate_on_dataset(self, dataset_path="math_dataset.json", max_iter=None, num_threads=10):
        evaluator = MathEvaluator(self, num_threads)
        return evaluator.evaluate_on_dataset(dataset_path)
        
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
            predicted_num = float(predicted)
            expected_num = float(expected)
            return abs(predicted_num - expected_num) < 0.01
        except (ValueError, TypeError) as e:
            print(f"⚠️ Error evaluating solution - invalid number format: {str(e)}")
            return False
        except Exception as e:
            print(f"⚠️ Unexpected error evaluating solution: {str(e)}")
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
    results_iter1 = calculator_iter1.evaluate_on_dataset(num_threads=100)
    
    print("\nEvaluating with max_iter=5...")
    results_iter5 = calculator_iter5.evaluate_on_dataset(num_threads=100)
    
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
