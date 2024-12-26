#!/usr/bin/env python3
import time
import json
import dspy
from jeopardy_dataset import JeopardyDatasetGenerator

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
    for i, item in enumerate(dataset, 1):
        if assess_jeopardy_quality(item["question"], item["answer"]):
            quality_metrics["correct_answers"] += 1
        
        # Calculate current stats
        current_success = quality_metrics["correct_answers"]
        total = quality_metrics["total_questions"]
        percentage = (current_success / total) * 100
        
        # Print progress
        print(f"\rProgress: {i}/{total} | Success: {current_success} | Rate: {percentage:.1f}%", end="", flush=True)
    
    elapsed_time = time.time() - start_time
    print()  # New line after progress
    
    quality_metrics["time_seconds"] = elapsed_time
    quality_metrics["accuracy"] = quality_metrics["correct_answers"] / quality_metrics["total_questions"]
    
    # Save results
    with open("jeopardy_benchmark.json", "w") as f:
        json.dump(quality_metrics, f, indent=2)
    
    print(f"Assessed {quality_metrics['total_questions']} questions in {elapsed_time:.2f} seconds")
    print(f"Answer Accuracy: {quality_metrics['accuracy']:.2%}")

if __name__ == "__main__":
    benchmark_jeopardy()
    print("\nBenchmark results saved to jeopardy_benchmark.json")
