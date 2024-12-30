#!/usr/bin/env python3

import dspy
import math
import json
import time
import tqdm
from pprint import pprint
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from signatures import (
    SolutionSelectorSignature, 
    MathCalculationSignature,
    TaskSplitterSignature,
    SubtaskResultSelectorSignature
)
from math_evaluator import MathEvaluator

class MathCalculator(dspy.Module):
    """Base math calculator module that ProblemSolver extends"""
    def __init__(self):
        super().__init__()
        self.calculate = dspy.ChainOfThought(MathCalculationSignature)

    def forward(self, task):
        """Basic forward pass without advanced reasoning"""
        result = self.calculate(task=task)
        return dspy.Prediction(
            reasoning=result.reasoning,
            solution=result.solution,
            notes_output=result.notes_output
        )

class ProblemSolver(dspy.Module):
    def __init__(self, max_iterations=5, num_attempts=3, subtask_attempts=3):
        """Initialize the ProblemSolver with DSPy modules and configuration.
        
        Args:
            max_iterations: Maximum number of reasoning iterations per attempt
            num_attempts: Number of attempts to solve each task
            subtask_attempts: Number of attempts to solve each subtask
        """
        super().__init__()
        # Initialize instance variables first
        self.max_iterations = max_iterations
        self.num_attempts = num_attempts
        self.subtask_attempts = subtask_attempts
        
        self.reasoning_tree = {
            'root': None,
            'nodes': {},
            'metadata': {
                'start_time': time.time(),
                'config': {
                    'max_iterations': self.max_iterations,
                    'num_attempts': self.num_attempts,
                    'subtask_attempts': self.subtask_attempts
                }
            }
        }
        self.current_node_id = 0
        
        # Initialize DSPy modules
        self.calculate = dspy.ChainOfThought(MathCalculationSignature)
        self.select_solution = dspy.ChainOfThought(SolutionSelectorSignature)
        self.split_task = dspy.ChainOfThought(TaskSplitterSignature)
        self.select_subtask_result = dspy.ChainOfThought(SubtaskResultSelectorSignature)

    def _create_node(self, task, parent_id=None, node_type='task', input_data=None, output_data=None):
        """Create a new node in the reasoning tree with input/output tracking"""
        node_id = f"node_{self.current_node_id}"
        self.current_node_id += 1
        
        node = {
            'id': node_id,
            'type': node_type,
            'task': task,
            'parent': parent_id,
            'children': [],
            'attempts': [],
            'input': input_data if input_data else {},
            'output': output_data if output_data else {},
            'timestamp': time.time()
        }
        
        self.reasoning_tree['nodes'][node_id] = node
        
        if parent_id:
            self.reasoning_tree['nodes'][parent_id]['children'].append(node_id)
            
        if not self.reasoning_tree['root']:
            self.reasoning_tree['root'] = node_id
            
        return node_id

    def _split_task(self, task, depth=0, max_depth=3):
        """Split a general problem into subtasks using DSPy reasoning"""
        if depth >= max_depth:
            print(f"Max recursion depth {max_depth} reached for task: {task}")
            return [task]
            
        try:
            # Log task splitting attempt
            print(f"Attempting to split task (Depth {depth}): {task}")
            
            result = self.split_task(task=task, context="")
            if not hasattr(result, 'subtasks'):
                print(f"Failed to split task - no subtasks returned: {task}")
                return [task]
                
            # Parse subtasks from the output
            subtasks = []
            if isinstance(result.subtasks, str):
                subtasks = [s.strip() for s in result.subtasks.split('\n') if s.strip()]
            elif isinstance(result.subtasks, list):
                subtasks = [str(s).strip() for s in result.subtasks if str(s).strip()]
                
            # Log the split reasoning and results
            print(f"Task Split Reasoning (Depth {depth}):\n{result.split_reasoning}")
            print(f"Generated Subtasks: {subtasks}")
            
            # Recursively split subtasks if needed
            final_subtasks = []
            for subtask in subtasks:
                try:
                    # Only split further if the subtask is complex enough
                    if len(subtask.split()) > 5:  # Simple heuristic based on length
                        final_subtasks.extend(self._split_task(subtask, depth+1, max_depth))
                    else:
                        final_subtasks.append(subtask)
                except Exception as e:
                    print(f"Error recursively splitting subtask {subtask}: {e}")
                    final_subtasks.append(subtask)
                    
            return final_subtasks if final_subtasks else [task]
        except Exception as e:
            print(f"Error splitting task {task}: {e}")
            return [task]

    def _combine_subtask_results(self, subtask_results: List[dspy.Prediction]) -> Dict[str, Any]:
        """Combine results from DSPy-generated subtasks"""
        if not subtask_results:
            return dspy.Prediction(
                reasoning="No subtask results to combine",
                solution=None,
                notes_output=""
            )
            
        # Build combined reasoning
        combined_reasoning = []
        combined_solution = []
        
        for i, result in enumerate(subtask_results, 1):
            combined_reasoning.append(
                f"Subtask {i}:\n"
                f"Reasoning: {result.reasoning}\n"
                f"Solution: {result.solution}\n"
            )
            if result.solution:
                combined_solution.append(str(result.solution))
                
        # Combine solutions in a meaningful way
        final_solution = "\n".join(combined_solution) if combined_solution else "No solution found"
            
        return dspy.Prediction(
            reasoning="Combined subtask results:\n" + "\n".join(combined_reasoning),
            solution=final_solution,
            notes_output="Combined results from subtasks"
        )

    def forward(self, task):
        """Forward pass for the math calculator with recursive task splitting"""
        # First try to split the task into subtasks recursively
        subtasks = self._split_task(task, max_depth=3)  # Set max recursion depth
        
        if len(subtasks) > 1:
            # Process each subtask independently with multiple attempts
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
                    # Process numerical subtasks with multiple attempts
                    result = self._process_subtask(subtask)
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
                        
                except ValueError as e:
                    print(f"Validation error in attempt {attempt + 1}, iteration {iteration + 1}: {str(e)}")
                    continue
                except RuntimeError as e:
                    print(f"Runtime error in attempt {attempt + 1}, iteration {iteration + 1}: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Unexpected error in attempt {attempt + 1}, iteration {iteration + 1}: {str(e)}")
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
                      for i, a in enumerate(attempts)],
            selection_criteria="Select the solution that is mathematically correct, logically consistent, "
                             "has clear reasoning, and provides a complete solution to the task"
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
        
    def _process_subtask(self, subtask, parent_id=None):
        """Process a subtask with multiple attempts and select the best result"""
        # Create node for this subtask
        subtask_node_id = self._create_node(
            task=subtask,
            parent_id=parent_id,
            node_type='subtask',
            input_data={
                'subtask': subtask,
                'parent_id': parent_id,
                'timestamp': time.time()
            }
        )
        
        attempts = []
        
        for attempt in range(self.subtask_attempts):
            # Create node for this attempt
            attempt_node_id = self._create_node(
                task=subtask,
                parent_id=subtask_node_id,
                node_type='attempt',
                input_data={
                    'attempt_number': attempt + 1,
                    'subtask': subtask,
                    'timestamp': time.time()
                }
            )
            try:
                result = self._forward_with_max_iter(subtask, self.max_iterations)
                
                # Update attempt node with output
                self.reasoning_tree['nodes'][attempt_node_id]['output'] = {
                    'reasoning': result.reasoning,
                    'solution': result.solution,
                    'notes': result.notes_output,
                    'timestamp': time.time()
                }
                
                attempts.append({
                    'reasoning': result.reasoning,
                    'solution': result.solution,
                    'notes': result.notes_output
                })
            except Exception as e:
                print(f"Error in subtask attempt {attempt + 1}: {e}")
                continue
                
        # Select the best result using DSPy
        if len(attempts) > 1:
            selection_result = self.select_subtask_result(
                subtask=subtask,
                attempts=[f"Attempt {i+1}:\nReasoning: {a['reasoning']}\nSolution: {a['solution']}" 
                         for i, a in enumerate(attempts)]
            )
            
            # Find the selected solution
            for attempt in attempts:
                if attempt['solution'] == selection_result.selected_solution:
                    return dspy.Prediction(
                        reasoning=f"Selected Solution Reasoning:\n{selection_result.selection_reasoning}\n\n"
                                f"Solution Details:\n{attempt['reasoning']}",
                        solution=attempt['solution'],
                        notes_output=attempt['notes']
                    )
                    
        # If no selection or only one attempt, return the first result
        if attempts:
            return dspy.Prediction(
                reasoning=attempts[0]['reasoning'],
                solution=attempts[0]['solution'],
                notes_output=attempts[0]['notes']
            )
            
        # Fallback if all attempts failed
        return dspy.Prediction(
            reasoning="All attempts failed to solve the subtask",
            solution="0",
            notes_output=""
        )

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
                    
            except ValueError as e:
                print(f"Validation error in iteration {iteration + 1}: {str(e)}")
                continue
            except RuntimeError as e:
                print(f"Runtime error in iteration {iteration + 1}: {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error in iteration {iteration + 1}: {str(e)}")
                continue
                
        return dspy.Prediction(
            reasoning=final_reasoning,
            solution=final_solution,
            notes_output=context
        )
        
    def save_reasoning_tree(self, path="reasoning_tree.json"):
        """Save the full reasoning tree to a JSON file with enhanced details"""
        # Add final metadata
        self.reasoning_tree['metadata']['end_time'] = time.time()
        self.reasoning_tree['metadata']['duration'] = (
            self.reasoning_tree['metadata']['end_time'] - 
            self.reasoning_tree['metadata']['start_time']
        )
        
        # Save with pretty printing
        with open(path, "w") as f:
            json.dump(self.reasoning_tree, f, indent=2, sort_keys=True)
            
        print(f"Reasoning tree saved to {path}")
        print(f"Total nodes: {len(self.reasoning_tree['nodes'])}")
        print(f"Duration: {self.reasoning_tree['metadata']['duration']:.2f}s")

    def _is_correct(self, predicted, expected):
        """Compare solutions with tolerance for floating point"""
        try:
            # Handle both string and numeric inputs
            predicted_num = float(predicted) if isinstance(predicted, str) else float(predicted)
            expected_num = float(expected) if isinstance(expected, str) else float(expected)
            
            # Handle NaN and infinity
            if math.isnan(predicted_num) or math.isnan(expected_num):
                return False
            if math.isinf(predicted_num) or math.isinf(expected_num):
                return False
                
            # Compare with tolerance
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
    
    # Create calculator instance with subtask processing
    calculator = ProblemSolver(max_iterations=3, num_attempts=2, subtask_attempts=2)
    
    # Test complex task that should be split into subtasks
    complex_task = "Calculate (3 + 4) * (5 - 2) / (6 + 3)"
    
    print(f"\nProcessing complex task: {complex_task}")
    result = calculator.forward(complex_task)
    
    print("\nFinal Result:")
    print(f"Reasoning:\n{result.reasoning}")
    print(f"Solution: {result.solution}")
    
    # Save result
    with open("subtask_result.json", "w") as f:
        json.dump({
            "task": complex_task,
            "reasoning": result.reasoning,
            "solution": result.solution
        }, f, indent=2)
    print("\nResult saved to subtask_result.json")
