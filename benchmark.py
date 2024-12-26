#!/usr/bin/env python3
import time
import json
import dspy
from jeopardy_dataset import JeopardyDatasetGenerator
from reasoning_pipeline import run_reasoning_pipeline

def assess_reasoning_pipeline(question, answer, hint):
    # Set up test context and objective
    context = f"""
    Final Question: {question}
    Hint: {hint}
    Answer: {answer}
    """
    objective = """
    1. Verify if the question correctly leads to the answer
    2. Determine if the question requires reasoning beyond the hint
    3. Assess if the hint provides meaningful guidance without giving away the answer
    """
    
    # Track pipeline performance metrics
    metrics = {
        "iterations": 0,
        "correct_termination": False,
        "reasoning_depth": 0,
        "decision_accuracy": 0,
        "reasoning_quality": 0,
        "outputs": []
    }
    
    def capture_reasoning(iteration, context, objective, result):
        metrics["iterations"] += 1
        metrics["outputs"].append({
            "iteration": iteration,
            "context": context,
            "objective": objective,
            "result": result
        })
        
        # Analyze reasoning quality
        reasoning_text = result.reasoning_output.lower()
        if "correct" in reasoning_text:
            metrics["reasoning_quality"] += 1
        if "requires reasoning" in reasoning_text:
            metrics["reasoning_depth"] += 1
            
        # Check decision accuracy
        action = result.action.lower().strip()
        if iteration > 1 and "terminate" in action:
            metrics["correct_termination"] = True
            metrics["decision_accuracy"] += 1
    
    # Run the pipeline
    run_reasoning_pipeline(context, objective, callback=capture_reasoning)
    
    # Calculate final metrics
    if metrics["iterations"] > 0:
        metrics["reasoning_quality"] /= metrics["iterations"]
        metrics["reasoning_depth"] /= metrics["iterations"]
        metrics["decision_accuracy"] /= metrics["iterations"]
    
    return metrics

def benchmark_reasoning_pipeline():
    print("Benchmarking Reasoning Pipeline Performance...")
    
    # Load generated dataset
    with open("jeopardy_dataset.json") as f:
        dataset = json.load(f)
    
    # Track pipeline performance metrics
    pipeline_metrics = {
        "total_questions": len(dataset),
        "total_iterations": 0,
        "correct_terminations": 0,
        "average_reasoning_depth": 0,
        "average_decision_accuracy": 0,
        "average_reasoning_quality": 0,
        "time_seconds": 0
    }
    
    start_time = time.time()
    for i, item in enumerate(dataset, 1):
        print(f"\nTesting Pipeline on Question {i}/{len(dataset)}")
        result = assess_reasoning_pipeline(
            item["question"],
            item["answer"],
            item["hint"]
        )
        
        # Update metrics
        pipeline_metrics["total_iterations"] += result["iterations"]
        pipeline_metrics["correct_terminations"] += int(result["correct_termination"])
        pipeline_metrics["average_reasoning_depth"] += result["reasoning_depth"]
        pipeline_metrics["average_decision_accuracy"] += result["decision_accuracy"]
        pipeline_metrics["average_reasoning_quality"] += result["reasoning_quality"]
        
        # Print progress
        print(f"\nCurrent Progress: {i}/{pipeline_metrics['total_questions']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Reasoning Quality: {result['reasoning_quality']:.1%}")
        print(f"Decision Accuracy: {result['decision_accuracy']:.1%}")
    
    elapsed_time = time.time() - start_time
    print()  # New line after progress
    
    # Calculate final averages
    pipeline_metrics["time_seconds"] = elapsed_time
    pipeline_metrics["average_iterations"] = pipeline_metrics["total_iterations"] / pipeline_metrics["total_questions"]
    pipeline_metrics["correct_termination_rate"] = pipeline_metrics["correct_terminations"] / pipeline_metrics["total_questions"]
    pipeline_metrics["average_reasoning_depth"] /= pipeline_metrics["total_questions"]
    pipeline_metrics["average_decision_accuracy"] /= pipeline_metrics["total_questions"]
    pipeline_metrics["average_reasoning_quality"] /= pipeline_metrics["total_questions"]
    
    # Save results
    with open("reasoning_benchmark.json", "w") as f:
        json.dump(pipeline_metrics, f, indent=2)
    
    print(f"Tested pipeline on {pipeline_metrics['total_questions']} questions in {elapsed_time:.2f} seconds")
    print(f"Average Iterations: {pipeline_metrics['average_iterations']:.1f}")
    print(f"Correct Termination Rate: {pipeline_metrics['correct_termination_rate']:.1%}")
    print(f"Average Reasoning Depth: {pipeline_metrics['average_reasoning_depth']:.1%}")
    print(f"Average Decision Accuracy: {pipeline_metrics['average_decision_accuracy']:.1%}")
    print(f"Average Reasoning Quality: {pipeline_metrics['average_reasoning_quality']:.1%}")

if __name__ == "__main__":
    benchmark_reasoning_pipeline()
    print("\nBenchmark results saved to reasoning_benchmark.json")
