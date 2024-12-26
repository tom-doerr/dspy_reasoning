#!/usr/bin/env python3
import time
import json
import dspy
from jeopardy_dataset import JeopardyDatasetGenerator

def assess_jeopardy_quality(question, answer, initial_question, hint):
    # Define a signature for answer verification
    class AnswerVerificationSignature(dspy.Signature):
        question = dspy.InputField(desc="The Jeopardy-style question")
        initial_question = dspy.InputField(desc="The initial direct question")
        hint = dspy.InputField(desc="The hint used to create the question")
        generated_answer = dspy.InputField(desc="The generated answer to verify")
        is_correct = dspy.OutputField(desc="True if the answer is correct for the question, False otherwise")
        requires_reasoning = dspy.OutputField(desc="True if the question requires reasoning beyond the initial question")
    
    # Create a verification module
    verifier = dspy.ChainOfThought(AnswerVerificationSignature)
    
    # Verify the answer and reasoning quality
    result = verifier(
        question=question,
        initial_question=initial_question,
        hint=hint,
        generated_answer=answer
    )
    
    # Return both correctness and reasoning quality
    return {
        "correct": result.is_correct.lower() in ["true", "yes"],
        "requires_reasoning": result.requires_reasoning.lower() in ["true", "yes"]
    }

def benchmark_jeopardy():
    print("Benchmarking Jeopardy question quality assessment...")
    
    # Load generated dataset
    with open("jeopardy_dataset.json") as f:
        dataset = json.load(f)
    
    # Assess quality
    quality_metrics = {
        "total_questions": len(dataset),
        "correct_answers": 0,
        "requires_reasoning": 0,
        "time_seconds": 0
    }
    
    start_time = time.time()
    for i, item in enumerate(dataset, 1):
        result = assess_jeopardy_quality(
            item["question"],
            item["answer"],
            item["initial_question"],
            item["hint"]
        )
        
        if result["correct"]:
            quality_metrics["correct_answers"] += 1
        if result["requires_reasoning"]:
            quality_metrics["requires_reasoning"] += 1
        
        # Calculate current stats
        current_success = quality_metrics["correct_answers"]
        current_total = i
        success_rate = (current_success / current_total) * 100 if current_total > 0 else 0
        
        # Print progress
        print(f"\rSuccess: {current_success} | Success Rate: {success_rate:.1f}% | Progress: {i}/{quality_metrics['total_questions']}", end="", flush=True)
    
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

if __name__ == "__main__":
    benchmark_jeopardy()
    print("\nBenchmark results saved to jeopardy_benchmark.json")
