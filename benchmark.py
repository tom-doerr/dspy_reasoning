#!/usr/bin/env python3
import time
import json
from jeopardy_dataset import JeopardyDatasetGenerator
from reasoning_pipeline import ActionReasoning

def benchmark_jeopardy():
    print("Benchmarking Jeopardy question generation...")
    generator = JeopardyDatasetGenerator()
    
    # Test categories
    categories = ["History", "Science", "Literature"]
    
    # Warm up
    generator.generate_dataset(categories, num_questions_per_category=1)
    
    # Benchmark
    start_time = time.time()
    dataset = generator.generate_dataset(categories, num_questions_per_category=5)
    elapsed_time = time.time() - start_time
    
    # Save results
    with open("jeopardy_benchmark.json", "w") as f:
        json.dump({
            "total_questions": len(dataset),
            "time_seconds": elapsed_time,
            "questions_per_second": len(dataset) / elapsed_time,
            "sample_questions": dataset[:3]  # Include sample questions
        }, f, indent=2)
    
    print(f"Generated {len(dataset)} questions in {elapsed_time:.2f} seconds")
    print(f"Performance: {len(dataset)/elapsed_time:.2f} questions/second")

def benchmark_reasoning():
    print("\nBenchmarking reasoning pipeline...")
    pipeline = ActionReasoning()
    
    # Test context and objective
    context = "The Eiffel Tower is located in Paris, France. It was completed in 1889."
    objective = "Determine when the Eiffel Tower was completed and if we should continue investigating."
    
    # Warm up
    pipeline(context=context, objective=objective)
    
    # Benchmark
    num_runs = 10
    start_time = time.time()
    for _ in range(num_runs):
        result = pipeline(context=context, objective=objective)
    elapsed_time = time.time() - start_time
    
    # Save results
    with open("reasoning_benchmark.json", "w") as f:
        json.dump({
            "total_runs": num_runs,
            "time_seconds": elapsed_time,
            "runs_per_second": num_runs / elapsed_time,
            "last_result": {
                "reasoning_output": result.reasoning_output,
                "action": result.action
            }
        }, f, indent=2)
    
    print(f"Completed {num_runs} reasoning runs in {elapsed_time:.2f} seconds")
    print(f"Performance: {num_runs/elapsed_time:.2f} runs/second")

if __name__ == "__main__":
    benchmark_jeopardy()
    benchmark_reasoning()
    print("\nBenchmark results saved to jeopardy_benchmark.json and reasoning_benchmark.json")
