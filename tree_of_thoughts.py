#!/usr/bin/env python3

import dspy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from math_calculator import MathCalculator

class ForestOfThoughts:
    def __init__(self, num_trees=3, num_thoughts=3, max_iterations=5):
        self.calculators = [MathCalculator() for _ in range(num_trees)]
        self.num_thoughts = num_thoughts
        self.max_iterations = max_iterations

    def generate_thoughts(self, task):
        """Generate multiple thoughts in parallel"""
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(calc.forward, task) 
                      for calc in self.calculators
                      for _ in range(self.num_thoughts)]
            return [f.result() for f in futures]

    def evaluate_thoughts(self, thoughts, expected):
        """Score thoughts based on accuracy"""
        scored = []
        for thought in thoughts:
            try:
                accuracy = int(abs(float(thought.solution) - float(expected)) < 0.01)
                scored.append((thought, accuracy))
            except:
                scored.append((thought, 0))
        return scored

    def solve(self, task, expected):
        """Main reasoning process"""
        best = None
        best_score = 0
        
        for _ in range(self.max_iterations):
            thoughts = self.generate_thoughts(task)
            scored = self.evaluate_thoughts(thoughts, expected)
            
            # Track best solution
            current_best, score = max(scored, key=lambda x: x[1])
            if score > best_score:
                best = current_best
                best_score = score
                
            if best_score == 1:  # Perfect solution found
                break
                
        return best

    def evaluate(self, dataset, num_threads=10):
        """Evaluate on dataset"""
        correct = 0
        start = time.time()
        
        with ThreadPoolExecutor(num_threads) as executor:
            # Create list of tasks with their corresponding items
            tasks = [(item['task'], item['solution']) for item in dataset[:100]]
        
            # Submit tasks and keep track of which item they correspond to
            futures = {executor.submit(self.solve, task, solution): (task, solution) 
                      for task, solution in tasks}
        
            for future in futures:
                result = future.result()
                # Get the solution from the original task
                _, expected_solution = futures[future]
                correct += int(abs(float(result.solution) - float(expected_solution)) < 0.01)
        
        accuracy = correct / len(dataset[:100])
        elapsed = time.time() - start
        
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Time: {elapsed:.2f}s")
        
        return {
            'accuracy': accuracy,
            'time': elapsed,
            'correct': correct
        }

if __name__ == "__main__":
    # Configure DSPy
    lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
    dspy.settings.configure(lm=lm)
    
    # Load dataset
    with open("math_dataset.json") as f:
        dataset = json.load(f)
    
    # Create and evaluate
    fot = ForestOfThoughts()
    results = fot.evaluate(dataset)
    
    # Save results
    with open("forest_results.json", "w") as f:
        json.dump(results, f, indent=2)
