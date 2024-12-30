import json
import time
import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

class MathEvaluator:
    def __init__(self, calculator, num_threads=10, max_samples=100):
        self.calculator = calculator
        self.num_threads = num_threads
        self.max_samples = max_samples

    def evaluate_single_task(self, item):
        task = item['task']
        expected_solution = item['solution']
        
        iter_start = time.time()
        result = self.calculator._forward_with_max_iter(task, max_iter=self.calculator.max_iterations)
        elapsed = time.time() - iter_start
        
        correct = self.calculator._is_correct(result.solution, expected_solution)
        return correct, elapsed

    def evaluate_on_dataset(self, dataset_path="math_dataset.json"):
        start_time = time.time()
        
        with open(dataset_path) as f:
            dataset = json.load(f)
        
        dataset = dataset[:self.max_samples] if hasattr(self, 'max_samples') else dataset[:100]
        
        results = {
            "correct": 0,
            "time": 0
        }
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(self.evaluate_single_task, item)
                for item in dataset
            ]
            
            for i, future in enumerate(tqdm.tqdm(as_completed(futures), total=len(futures), ncols=60), 1):
                correct, elapsed = future.result()
                results["correct"] += int(correct)
                results["time"] += elapsed
                
                if i % 100 == 0:
                    print(f"\nProgress after {i} samples:")
                    print(f"Correct: {results['correct']}/{i} ({results['correct']/i:.1%})")
                    print(f"Time: {results['time']:.2f}s")
                
        total_time = time.time() - start_time
        results["accuracy"] = results["correct"] / len(dataset)
        results["total_time"] = total_time
        results["max_iter"] = self.calculator.max_iterations
        
        print("\nEvaluation Results:")
        print(f"Max Iterations: {self.calculator.max_iterations}")
        print(f"Correct Answers: {results['correct']}/{len(dataset)} ({results['accuracy']:.1%})")
        print(f"Total Time: {results['total_time']:.2f}s")
        
        with open("math_calculator_benchmark.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print("\nBenchmark results saved to math_calculator_benchmark.json")
        return results
