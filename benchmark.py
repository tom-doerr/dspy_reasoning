#!/usr/bin/env python3
import time
import json
import dspy
from jeopardy_dataset import JeopardyDatasetGenerator
from reasoning_pipeline import run_reasoning_pipeline

def assess_jeopardy_quality(question, answer, initial_question, hint):
    # Use the reasoning pipeline to assess the question
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
    
    # Run the reasoning pipeline and capture its output
    reasoning_output = []
    def capture_reasoning(iteration, context, objective, result):
        reasoning_output.append({
            "iteration": iteration,
            "context": context,
            "objective": objective,
            "result": result
        })
    
    run_reasoning_pipeline(context, objective, callback=capture_reasoning)
    
    # Analyze the reasoning output
    correct = False
    requires_reasoning = False
    iterations = len(reasoning_output)
    
    # Check final reasoning output for assessment
    if reasoning_output:
        final_result = reasoning_output[-1]["result"]
        if "correct" in final_result.reasoning_output.lower():
            correct = True
        if "requires reasoning" in final_result.reasoning_output.lower():
            requires_reasoning = True
    
    return {
        "correct": correct,
        "requires_reasoning": requires_reasoning,
        "iterations": iterations,
        "reasoning_output": reasoning_output
    }

def benchmark_jeopardy():
    print("Benchmarking Jeopardy question quality assessment using iterative reasoning...")
    
    # Load generated dataset
    with open("jeopardy_dataset.json") as f:
        dataset = json.load(f)
    
    # Assess quality
    quality_metrics = {
        "total_questions": len(dataset),
        "correct_answers": 0,
        "requires_reasoning": 0,
        "time_seconds": 0,
        "reasoning_iterations": []
    }
    
    start_time = time.time()
    for i, item in enumerate(dataset, 1):
        print(f"\nAssessing Question {i}/{len(dataset)}")
        result = assess_jeopardy_quality(
            item["question"],
            item["answer"],
            item["initial_question"],
            item["hint"]
        )
        
        # Update metrics based on reasoning pipeline output
        if result["correct"]:
            quality_metrics["correct_answers"] += 1
        if result["requires_reasoning"]:
            quality_metrics["requires_reasoning"] += 1
        
        # Calculate current stats
        current_success = quality_metrics["correct_answers"]
        current_total = i
        success_rate = (current_success / current_total) * 100 if current_total > 0 else 0
        
        # Print progress
        print(f"\nCurrent Progress: {i}/{quality_metrics['total_questions']}")
        print(f"Current Success Rate: {success_rate:.1f}%")
    
    elapsed_time = time.time() - start_time
    print()  # New line after progress
    
    quality_metrics["time_seconds"] = elapsed_time
    quality_metrics["accuracy"] = quality_metrics["correct_answers"] / quality_metrics["total_questions"]
    quality_metrics["reasoning_quality"] = quality_metrics["requires_reasoning"] / quality_metrics["total_questions"]
    
    # Save results
    with open("jeopardy_benchmark.json", "w") as f:
        json.dump(quality_metrics, f, indent=2)
    
    print(f"Assessed {quality_metrics['total_questions']} questions in {elapsed_time:.2f} seconds")
    print(f"Answer Accuracy: {quality_metrics['accuracy']:.2%}")
    print(f"Reasoning Quality: {quality_metrics['reasoning_quality']:.2%} of questions require reasoning")
    print(f"Average Reasoning Iterations: {sum(quality_metrics['reasoning_iterations'])/len(quality_metrics['reasoning_iterations']):.1f}")

if __name__ == "__main__":
    benchmark_jeopardy()
    print("\nBenchmark results saved to jeopardy_benchmark.json")
