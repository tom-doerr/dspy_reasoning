#!/usr/bin/env python3

import dspy
import json
import time
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from math_calculator import MathCalculator, MathCalculationSignature
from collections import defaultdict

class ForestOfThoughts(dspy.Module):
    def __init__(self, num_trees=3, num_thoughts=3, max_depth=3, max_iterations=5):
        super().__init__()
        self.num_trees = num_trees
        self.num_thoughts = num_thoughts
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.calculators = [MathCalculator(max_iterations=1) for _ in range(num_trees)]
        
    def generate_thoughts(self, task, context):
        """Generate multiple parallel thoughts across all trees"""
        all_thoughts = []
        with ThreadPoolExecutor(max_workers=self.num_trees) as executor:
            futures = [
                executor.submit(self._generate_tree_thoughts, calculator, task, context)
                for calculator in self.calculators
            ]
            for future in futures:
                all_thoughts.extend(future.result())
        return all_thoughts
        
    def _generate_tree_thoughts(self, calculator, task, context):
        """Generate thoughts for a single tree"""
        thoughts = []
        for _ in range(self.num_thoughts):
            result = calculator.forward(task)
            thoughts.append({
                'reasoning': result.reasoning,
                'solution': result.solution,
                'notes': result.notes_output,
                'tree_id': id(calculator)  # Track which tree generated this thought
            })
        return thoughts
        
    def evaluate_thoughts(self, thoughts, expected_solution):
        """Evaluate and score each thought with dynamic self-correction"""
        scored_thoughts = []
        for thought in thoughts:
            try:
                # Calculate accuracy score
                accuracy = int(abs(float(thought['solution']) - float(expected_solution)) < 0.01)
                
                # Calculate reasoning quality score
                reasoning_score = self._evaluate_reasoning_quality(thought['reasoning'])
                
                # Combined score with weights
                score = 0.7 * accuracy + 0.3 * reasoning_score
                
                scored_thoughts.append({
                    **thought,
                    'score': score,
                    'accuracy': accuracy,
                    'reasoning_score': reasoning_score,
                    'error': None
                })
            except Exception as e:
                scored_thoughts.append({
                    **thought,
                    'score': 0,
                    'accuracy': 0,
                    'reasoning_score': 0,
                    'error': str(e)
                })
        return scored_thoughts
        
    def _evaluate_reasoning_quality(self, reasoning):
        """Evaluate the quality of reasoning steps"""
        # Simple heuristic - count number of valid reasoning steps
        steps = [step.strip() for step in reasoning.split('\n') if step.strip()]
        return min(len(steps) / 10, 1.0)  # Normalize to 0-1 range
        
    def forward(self, task, expected_solution):
        """Forest of Thoughts reasoning process"""
        best_solution = None
        best_score = 0
        context = ""
        tree_votes = defaultdict(int)
        
        for iteration in range(self.max_iterations):
            # Generate thoughts across all trees
            thoughts = self.generate_thoughts(task, context)
            
            # Evaluate and score thoughts
            scored_thoughts = self.evaluate_thoughts(thoughts, expected_solution)
            
            # Dynamic self-correction
            corrected_thoughts = self._self_correct(scored_thoughts)
            
            # Update context with best thoughts from each tree
            context = self._update_context(context, corrected_thoughts, iteration)
            
            # Collect votes from each tree
            current_best = max(corrected_thoughts, key=lambda x: x['score'])
            tree_votes[current_best['tree_id']] += 1
            
            # Update overall best solution
            if current_best['score'] > best_score:
                best_solution = current_best['solution']
                best_score = current_best['score']
                
            # Early termination if perfect score
            if best_score == 1:
                break
                
        # Final voting mechanism
        final_solution = self._final_vote(tree_votes, corrected_thoughts)
        
        return dspy.Prediction(
            best_solution=final_solution,
            best_score=best_score,
            final_context=context,
            tree_votes=dict(tree_votes)
        )
        
    def _self_correct(self, scored_thoughts):
        """Perform dynamic self-correction based on scores"""
        # Filter out thoughts with low scores
        threshold = 0.5
        corrected = [thought for thought in scored_thoughts if thought['score'] >= threshold]
        
        # If all thoughts are bad, keep the best one
        if not corrected:
            corrected = [max(scored_thoughts, key=lambda x: x['score'])]
            
        return corrected
        
    def _update_context(self, context, thoughts, iteration):
        """Update context with best thoughts from each tree"""
        context += f"\nIteration {iteration + 1} Summary:\n"
        
        # Group thoughts by tree
        tree_thoughts = defaultdict(list)
        for thought in thoughts:
            tree_thoughts[thought['tree_id']].append(thought)
            
        # Add best thought from each tree to context
        for tree_id, thoughts in tree_thoughts.items():
            best = max(thoughts, key=lambda x: x['score'])
            context += f"Tree {tree_id} Best Thought:\n"
            context += f"Reasoning: {best['reasoning']}\n"
            context += f"Solution: {best['solution']}\n"
            context += f"Score: {best['score']:.2f}\n\n"
            
        return context
        
    def _final_vote(self, tree_votes, thoughts):
        """Final voting mechanism to determine solution"""
        # Get the tree with most votes
        winning_tree = max(tree_votes.keys(), key=lambda x: tree_votes[x])
        
        # Get best thought from winning tree
        winning_thoughts = [t for t in thoughts if t['tree_id'] == winning_tree]
        if winning_thoughts:
            return max(winning_thoughts, key=lambda x: x['score'])['solution']
        return None
        
    def evaluate_on_dataset(self, dataset_path="math_dataset.json", num_threads=10):
        """Evaluate Forest of Thoughts approach on dataset"""
        start_time = time.time()
        
        with open(dataset_path) as f:
            dataset = json.load(f)
        
        # Evaluate on first 100 samples
        dataset = dataset[:100]
        
        # Results storage
        results = {
            "correct": 0,
            "time": 0,
            "tree_votes": defaultdict(int),
            "accuracy_by_tree": defaultdict(lambda: {"correct": 0, "total": 0})
        }
        
        # Function to evaluate single task
        def evaluate_single(item):
            task = item['task']
            expected_solution = item['solution']
            
            # Evaluate with Forest of Thoughts
            iter_start = time.time()
            result = self.forward(task, expected_solution)
            elapsed = time.time() - iter_start
            
            correct = result.best_score
            return correct, elapsed, result.tree_votes
        
        # Evaluate all samples in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(evaluate_single, item)
                for item in dataset
            ]
            
            for i, future in enumerate(tqdm.tqdm(as_completed(futures), total=len(futures), ncols=60), 1):
                correct, elapsed, votes = future.result()
                results["correct"] += int(correct)
                results["time"] += elapsed
                
                # Track tree votes and accuracy
                for tree_id, count in votes.items():
                    results["tree_votes"][tree_id] += count
                    results["accuracy_by_tree"][tree_id]["total"] += 1
                    if correct:
                        results["accuracy_by_tree"][tree_id]["correct"] += 1
                
                # Print progress
                if i % 100 == 0:
                    print(f"\nProgress after {i} samples:")
                    print(f"Correct: {results['correct']}/{i} ({results['correct']/i:.1%})")
                    print(f"Time: {results['time']:.2f}s")
                    print("Tree Votes:", dict(results["tree_votes"]))
                
        # Calculate final metrics
        total_time = time.time() - start_time
        results["accuracy"] = results["correct"] / len(dataset)
        results["total_time"] = total_time
        
        # Calculate tree accuracies
        for tree_id, stats in results["accuracy_by_tree"].items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        # Print final results
        print("\nEvaluation Results:")
        print(f"Correct Answers: {results['correct']}/{len(dataset)} ({results['accuracy']:.1%})")
        print(f"Total Time: {results['total_time']:.2f}s")
        print("\nTree Performance:")
        for tree_id, stats in results["accuracy_by_tree"].items():
            print(f"Tree {tree_id}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")
        print("\nFinal Tree Votes:", dict(results["tree_votes"]))
        
        # Save results
        with open("forest_of_thoughts_benchmark.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print("\nBenchmark results saved to forest_of_thoughts_benchmark.json")
        return results

if __name__ == "__main__":
    # Configure DSPy
    lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
    dspy.settings.configure(lm=lm)
    
    # Create and evaluate Forest of Thoughts
    fot = ForestOfThoughts(num_trees=3, num_thoughts=3, max_depth=3, max_iterations=5)
    results = fot.evaluate_on_dataset(num_threads=100)
    
    print("\nForest of Thoughts Evaluation Complete!")
