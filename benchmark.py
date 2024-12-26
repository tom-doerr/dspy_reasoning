#!/usr/bin/env python3
import time
import json
import dspy
from jeopardy_dataset import JeopardyDatasetGenerator
from reasoning_pipeline import ActionReasoning

def assess_jeopardy_quality(question, answer):
    # Define a signature for answer verification
    class AnswerVerificationSignature(dspy.Signature):
        question = dspy.InputField(desc="The Jeopardy-style question")
        generated_answer = dspy.InputField(desc="The generated answer to verify")
        is_correct = dspy.OutputField(desc="True if the answer is correct for the question, False otherwise")
    
    # Create a verification module
    verifier = dspy.ChainOfThought(AnswerVerificationSignature)
    
    # Verify the answer
    result = verifier(question=question, generated_answer=answer)
    return result.is_correct.lower() in ["true", "yes"]

def benchmark_jeopardy():
    print("Benchmarking Jeopardy question quality assessment...")
    
    # Load generated dataset
    with open("jeopardy_dataset.json") as f:
        dataset = json.load(f)
    
    # Assess quality
    quality_metrics = {
        "total_questions": len(dataset),
        "correct_answers": 0,
        "time_seconds": 0
    }
    
    start_time = time.time()
    for item in dataset:
        if assess_jeopardy_quality(item["question"], item["answer"]):
            quality_metrics["correct_answers"] += 1
    elapsed_time = time.time() - start_time
    
    quality_metrics["time_seconds"] = elapsed_time
    quality_metrics["accuracy"] = quality_metrics["correct_answers"] / quality_metrics["total_questions"]
    
    # Save results
    with open("jeopardy_benchmark.json", "w") as f:
        json.dump(quality_metrics, f, indent=2)
    
    print(f"Assessed {quality_metrics['total_questions']} questions in {elapsed_time:.2f} seconds")
    print(f"Answer Accuracy: {quality_metrics['accuracy']:.2%}")

def assess_reasoning_quality(result, context, objective):
    # Check if the reasoning output contains the correct year
    year_correct = "1889" in result.reasoning_output
    
    # Check if the reasoning output addresses the objective
    objective_met = all(
        keyword.lower() in result.reasoning_output.lower()
        for keyword in ["completed", "investigating"]
    )
    
    # Check if action is correct based on context
    action_correct = (
        result.action.lower() == "terminate" 
        if "1889" in context 
        else result.action.lower() == "reasoning"
    )
    
    return {
        "year_correct": year_correct,
        "objective_met": objective_met,
        "action_correct": action_correct
    }

def benchmark_reasoning():
    print("\nBenchmarking reasoning pipeline...")
    pipeline = ActionReasoning()
    
    # Test cases
    test_cases = [
        {
            "context": "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
            "objective": "Determine when the Eiffel Tower was completed and if we should continue investigating.",
            "expected_action": "terminate"
        },
        {
            "context": "The Statue of Liberty was a gift from France to the United States.",
            "objective": "Determine when the Statue of Liberty was completed and if we should continue investigating.",
            "expected_action": "reasoning"
        }
    ]
    
    quality_metrics = {
        "total_runs": 0,
        "year_correct": 0,
        "objective_met": 0,
        "action_correct": 0,
        "time_seconds": 0
    }
    
    # Benchmark
    start_time = time.time()
    for test_case in test_cases:
        for _ in range(5):  # 5 runs per test case
            result = pipeline(context=test_case["context"], objective=test_case["objective"])
            assessment = assess_reasoning_quality(result, test_case["context"], test_case["objective"])
            
            quality_metrics["total_runs"] += 1
            quality_metrics["year_correct"] += int(assessment["year_correct"])
            quality_metrics["objective_met"] += int(assessment["objective_met"])
            quality_metrics["action_correct"] += int(assessment["action_correct"])
    
    elapsed_time = time.time() - start_time
    quality_metrics["time_seconds"] = elapsed_time
    
    # Calculate percentages
    quality_metrics["year_accuracy"] = quality_metrics["year_correct"] / quality_metrics["total_runs"]
    quality_metrics["objective_accuracy"] = quality_metrics["objective_met"] / quality_metrics["total_runs"]
    quality_metrics["action_accuracy"] = quality_metrics["action_correct"] / quality_metrics["total_runs"]
    
    # Save results
    with open("reasoning_benchmark.json", "w") as f:
        json.dump(quality_metrics, f, indent=2)
    
    print(f"Completed {quality_metrics['total_runs']} reasoning runs in {elapsed_time:.2f} seconds")
    print(f"Year Accuracy: {quality_metrics['year_accuracy']:.2%}")
    print(f"Objective Accuracy: {quality_metrics['objective_accuracy']:.2%}")
    print(f"Action Accuracy: {quality_metrics['action_accuracy']:.2%}")

if __name__ == "__main__":
    benchmark_jeopardy()
    benchmark_reasoning()
    print("\nBenchmark results saved to jeopardy_benchmark.json and reasoning_benchmark.json")
