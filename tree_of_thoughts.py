#!/usr/bin/env python3

import dspy
import json
import time
import logging
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, Future
from math_calculator import MathCalculator, MathCalculationSignature

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReasoningTree:
    def __init__(self, calculator: MathCalculator):
        self.calculator = calculator
        self.thoughts = []
        self.score = 0
        
    def generate_thought(self, task: str) -> dspy.Prediction:
        """Generate a new thought using this tree's calculator"""
        thought = self.calculator.forward(task)
        self.thoughts.append(thought)
        return thought
        
    def evaluate_thought(self, thought: dspy.Prediction, expected: float) -> float:
        """Evaluate a thought and update tree score"""
        try:
            pred_solution = float(thought.solution)
            accuracy = int(abs(pred_solution - expected) < 0.01)
            self.score += accuracy
            return accuracy
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Error evaluating thought: {e}")
            return 0

class ForestOfThoughts:
    def __init__(self, num_trees: int = 3, num_thoughts: int = 3, max_iterations: int = 5):
        """Initialize Forest of Thoughts with multiple reasoning trees.
        
        Args:
            num_trees: Number of parallel reasoning trees to use
            num_thoughts: Number of thoughts to generate per tree
            max_iterations: Maximum iterations for solving each task
        """
        self.trees = [ReasoningTree(MathCalculator()) for _ in range(num_trees)]
        self.num_thoughts = num_thoughts
        self.max_iterations = max_iterations
        self.consensus_threshold = 0.7  # Percentage of trees that must agree
        logger.info(f"Initialized ForestOfThoughts with {num_trees} trees, {num_thoughts} thoughts per tree")

    def generate_thoughts(self, task: str) -> List[dspy.Prediction]:
        """Generate multiple thoughts in parallel using all trees.
        
        Args:
            task: The math task to solve
            
        Returns:
            List of predictions from all trees
        """
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(tree.generate_thought, task)
                for tree in self.trees
                for _ in range(self.num_thoughts)
            ]
            return [f.result() for f in futures]

    def evaluate_thoughts(
        self, 
        thoughts: List[dspy.Prediction], 
        expected: float
    ) -> List[Tuple[dspy.Prediction, int]]:
        """Score thoughts and update tree scores.
        
        Args:
            thoughts: List of predictions to evaluate
            expected: Expected solution value
            
        Returns:
            List of tuples containing (prediction, accuracy_score)
        """
        scored = []
        for thought in thoughts:
            # Find which tree generated this thought
            for tree in self.trees:
                if thought in tree.thoughts:
                    accuracy = tree.evaluate_thought(thought, expected)
                    scored.append((thought, accuracy))
                    break
            else:
                logger.warning("Thought not found in any tree")
                scored.append((thought, 0))
        return scored

    def solve(self, task: str, expected: float) -> Optional[dspy.Prediction]:
        """Main reasoning process to find consensus solution.
        
        Args:
            task: The math task to solve
            expected: Expected solution value
            
        Returns:
            Consensus prediction found, or None if no valid solution
        """
        best = None
        best_score = 0
        
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate thoughts across all trees
            thoughts = self.generate_thoughts(task)
            
            # Evaluate thoughts and update tree scores
            scored = self.evaluate_thoughts(thoughts, expected)
            
            # Find consensus solution
            consensus = self._find_consensus(scored)
            if consensus:
                best = consensus
                best_score = 1
                logger.info("Consensus solution found")
                break
                
            # Track best individual solution
            current_best, score = max(scored, key=lambda x: x[1])
            if score > best_score:
                best = current_best
                best_score = score
                logger.info(f"New best solution found with score {best_score}")
                
            if best_score == 1:  # Perfect solution found
                logger.info("Perfect solution found, terminating early")
                break
                
        return best
        
    def _find_consensus(self, scored_thoughts: List[Tuple[dspy.Prediction, int]]) -> Optional[dspy.Prediction]:
        """Find a consensus solution across trees.
        
        Args:
            scored_thoughts: List of scored predictions
            
        Returns:
            Consensus prediction if found, else None
        """
        # Group thoughts by solution
        solution_counts = {}
        for thought, score in scored_thoughts:
            if score == 1:  # Only consider correct solutions
                solution_counts[thought.solution] = solution_counts.get(thought.solution, 0) + 1
                
        # Find solution with highest agreement
        if solution_counts:
            best_solution = max(solution_counts.keys(), key=lambda x: solution_counts[x])
            if solution_counts[best_solution] >= self.consensus_threshold * len(self.trees):
                # Return first thought with this solution
                for thought, _ in scored_thoughts:
                    if thought.solution == best_solution:
                        return thought
        return None

    def evaluate(
        self, 
        dataset: List[Dict[str, str]], 
        num_threads: int = 10
    ) -> Dict[str, float]:
        """Evaluate model on dataset.
        
        Args:
            dataset: List of math tasks with solutions
            num_threads: Number of parallel threads to use
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Starting evaluation on {len(dataset[:100])} tasks")
        correct = 0
        start = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Create list of tasks with their corresponding items
            tasks = [(item['task'], float(item['solution'])) for item in dataset[:100]]
        
            # Submit tasks and keep track of which item they correspond to
            futures = {
                executor.submit(self.solve, task, solution): (task, solution)
                for task, solution in tasks
            }
        
            for future in futures:
                result = future.result()
                if result is None:
                    continue
                    
                # Get the solution from the original task
                _, expected_solution = futures[future]
                correct += int(abs(float(result.solution) - expected_solution) < 0.01)
        
        accuracy = correct / len(dataset[:100])
        elapsed = time.time() - start
        
        logger.info(f"Evaluation complete - Accuracy: {accuracy:.1%}, Time: {elapsed:.2f}s")
        
        return {
            'accuracy': accuracy,
            'time': elapsed,
            'correct': correct
        }

def configure_dspy() -> None:
    """Configure DSPy settings."""
    lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
    dspy.settings.configure(lm=lm)
    logger.info("DSPy configured with DeepSeek model")

def main() -> None:
    """Main execution function."""
    configure_dspy()
    
    # Load dataset
    try:
        with open("math_dataset.json") as f:
            dataset = json.load(f)
        logger.info(f"Loaded dataset with {len(dataset)} items")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Create and evaluate
    fot = ForestOfThoughts()
    results = fot.evaluate(dataset)
    
    # Save results
    try:
        with open("forest_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to forest_results.json")
    except IOError as e:
        logger.error(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
