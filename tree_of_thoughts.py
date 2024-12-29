#!/usr/bin/env python3

import dspy
import json
import time
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from math_calculator import MathCalculator, MathCalculationSignature

class TreeOfThoughts(dspy.Module):
    def __init__(self, num_thoughts=3, max_depth=3, max_iterations=5):
        super().__init__()
        self.num_thoughts = num_thoughts
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.calculator = MathCalculator(max_iterations=1)
        
    def generate_thoughts(self, task, context):
        """Generate multiple parallel thoughts for a given task"""
        thoughts = []
        for _ in range(self.num_thoughts):
            result = self.calculator.forward(task)
            thoughts.append({
                'reasoning': result.reasoning,
                'solution': result.solution,
                'notes': result.notes_output
            })
        return thoughts
        
    def evaluate_thoughts(self, thoughts, expected_solution):
        """Evaluate and score each thought"""
        scored_thoughts = []
        for thought in thoughts:
            try:
                accuracy = int(abs(float(thought['solution']) - float(expected_solution)) < 0.01)
                scored_thoughts.append({
                    **thought,
                    'score': accuracy,
                    'error': None
                })
            except Exception as e:
                scored_thoughts.append({
                    **thought,
                    'score': 0,
                    'error': str(e)
                })
        return scored_thoughts
        
    def forward(self, task, expected_solution):
        """Tree of Thoughts reasoning process"""
        best_solution = None
        best_score = 0
        context = ""
        
        for iteration in range(self.max_iterations):
            # Generate initial thoughts
            thoughts = self.generate_thoughts(task, context)
            
            # Evaluate thoughts
            scored_thoughts = self.evaluate_thoughts(thoughts, expected_solution)
            
            # Select best thought
            current_best = max(scored_thoughts, key=lambda x: x['score'])
            if current_best['score'] > best_score:
                best_solution = current_best['solution']
                best_score = current_best['score']
                
            # Update context with best thought
            context += f"\nIteration {iteration + 1} Best Thought:\n"
            context += f"Reasoning: {current_best['reasoning']}\n"
            context += f"Solution: {current_best['solution']}\n"
            context += f"Score: {current_best['score']}\n"
            
            # Early termination if perfect score
            if best_score == 1:
                break
                
        return dspy.Prediction(
            best_solution=best_solution,
            best_score=best_score,
            final_context=context
        )
        
    def evaluate_on_dataset(self, dataset_path="math_dataset.json", num_threads=10):
        """Evaluate ToT approach on dataset"""
        start_time = time.time()
        
        with open(dataset_path) as f:
            dataset = json.load(f)
        
        # Evaluate on first 100 samples
        dataset = dataset[:100]
        
        # Results storage
        results = {
            "correct": 0,
            "time": 0
        }
        
        # Function to evaluate single task
        def evaluate_single(item):
            task = item['task']
            expected_solution = item['solution']
            
            # Evaluate with ToT
            iter_start = time.time()
            result = self.forward(task, expected_solution)
            elapsed = time.time() - iter_start
            
            correct = result.best_score
            return correct, elapsed
        
        # Evaluate all samples in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(evaluate_single, item)
                for item in dataset
            ]
            
            for i, future in enumerate(tqdm.tqdm(as_completed(futures), total=len(futures), ncols=60), 1):
                correct, elapsed = future.result()
                results["correct"] += int(correct)
                results["time"] += elapsed
                
                # Print progress
                if i % 100 == 0:
                    print(f"\nProgress after {i} samples:")
                    print(f"Correct: {results['correct']}/{i} ({results['correct']/i:.1%})")
                    print(f"Time: {results['time']:.2f}s")
                
        # Calculate final metrics
        total_time = time.time() - start_time
        results["accuracy"] = results["correct"] / len(dataset)
        results["total_time"] = total_time
        
        # Print final results
        print("\nEvaluation Results:")
        print(f"Correct Answers: {results['correct']}/{len(dataset)} ({results['accuracy']:.1%})")
        print(f"Total Time: {results['total_time']:.2f}s")
        
        # Save results
        with open("tree_of_thoughts_benchmark.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print("\nBenchmark results saved to tree_of_thoughts_benchmark.json")
        return results

if __name__ == "__main__":
    # Configure DSPy
    lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
    dspy.settings.configure(lm=lm)
    
    # Create and evaluate Tree of Thoughts
    tot = TreeOfThoughts(num_thoughts=3, max_depth=3, max_iterations=5)
    results = tot.evaluate_on_dataset(num_threads=100)
    
    print("\nTree of Thoughts Evaluation Complete!")
