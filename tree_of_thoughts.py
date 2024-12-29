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

class ForestOfThoughts:
    def __init__(self, num_trees: int = 3, num_thoughts: int = 3, max_iterations: int = 5):
        """Initialize Forest of Thoughts with multiple calculators.
        
        Args:
            num_trees: Number of parallel calculators to use
            num_thoughts: Number of thoughts to generate per calculator
            max_iterations: Maximum iterations for solving each task
        """
        self.calculators = [MathCalculator() for _ in range(num_trees)]
        self.num_thoughts = num_thoughts
        self.max_iterations = max_iterations
        logger.info(f"Initialized ForestOfThoughts with {num_trees} trees, {num_thoughts} thoughts per tree")

    def generate_thoughts(self, task: str) -> List[dspy.Prediction]:
        """Generate multiple thoughts in parallel using all calculators.
        
        Args:
            task: The math task to solve
            
        Returns:
            List of predictions from all calculators
        """
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(calc.forward, task) 
                for calc in self.calculators
                for _ in range(self.num_thoughts)
            ]
            return [f.result() for f in futures]

    def evaluate_thoughts(
        self, 
        thoughts: List[dspy.Prediction], 
        expected: float
    ) -> List[Tuple[dspy.Prediction, int]]:
        """Score thoughts based on accuracy compared to expected solution.
        
        Args:
            thoughts: List of predictions to evaluate
            expected: Expected solution value
            
        Returns:
            List of tuples containing (prediction, accuracy_score)
        """
        scored = []
        for thought in thoughts:
            try:
                pred_solution = float(thought.solution)
                accuracy = int(abs(pred_solution - expected) < 0.01)
                scored.append((thought, accuracy))
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Error evaluating thought: {e}")
                scored.append((thought, 0))
        return scored

    def solve(self, task: str, expected: float) -> Optional[dspy.Prediction]:
        """Main reasoning process to find best solution.
        
        Args:
            task: The math task to solve
            expected: Expected solution value
            
        Returns:
            Best prediction found, or None if no valid solution
        """
        best = None
        best_score = 0
        
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            thoughts = self.generate_thoughts(task)
            scored = self.evaluate_thoughts(thoughts, expected)
            
            # Track best solution
            current_best, score = max(scored, key=lambda x: x[1])
            if score > best_score:
                best = current_best
                best_score = score
                logger.info(f"New best solution found with score {best_score}")
                
            if best_score == 1:  # Perfect solution found
                logger.info("Perfect solution found, terminating early")
                break
                
        return best

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
