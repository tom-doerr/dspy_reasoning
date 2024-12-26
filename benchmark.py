#!/usr/bin/env python3
import time
import json
import dspy
from jeopardy_dataset import JeopardyDatasetGenerator
from reasoning_pipeline import run_reasoning_pipeline

class VerifyAnswerSignature(dspy.Signature):
    predicted_answer = dspy.InputField(desc="The answer predicted by the model")
    correct_answer = dspy.InputField(desc="The known correct answer")
    verification = dspy.OutputField(desc="True if answers match semantically, False otherwise")

class AnswerVerifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.verify = dspy.ChainOfThought(VerifyAnswerSignature)

    def forward(self, predicted_answer, correct_answer):
        return self.verify(predicted_answer=predicted_answer, correct_answer=correct_answer)

def verify_answer_match(predicted_answer, correct_answer):
    """Check if the predicted answer matches the correct answer using semantic verification"""
    verifier = AnswerVerifier()
    result = verifier(predicted_answer, correct_answer)
    return result.verification.lower().strip() in ["true", "yes", "correct"]

def benchmark_reasoning_pipeline():
    print("Benchmarking Reasoning Pipeline Performance...")
    
    # Load generated dataset
    with open("jeopardy_dataset.json") as f:
        dataset = json.load(f)
    
    # Track pipeline performance metrics
    pipeline_metrics = {
        "total_questions": len(dataset),
        "total_iterations": 0,
        "correct_answers": 0,
        "time_seconds": 0
    }
    
    start_time = time.time()
    for i, item in enumerate(dataset, 1):
        print(f"\nTesting Pipeline on Question {i}/{len(dataset)}")
        # Run reasoning pipeline directly
        context = f"""
        Final Question: {item["question"]}
        Hint: {item["hint"]}
        Answer: {item["answer"]}
        """
        objective = "Determine the correct answer to the question using the provided hint"
        
        reasoning_output = []
        def capture_reasoning(iteration, context, objective, result):
            reasoning_output.append({
                "iteration": iteration,
                "context": context,
                "objective": objective,
                "result": result
            })
        
        run_reasoning_pipeline(context, objective, callback=capture_reasoning)
        
        # Check if final reasoning output matches correct answer
        if reasoning_output:
            final_result = reasoning_output[-1]["result"]
            is_correct = verify_answer_match(final_result.reasoning_output, item["answer"])
            pipeline_metrics["correct_answers"] += int(is_correct)
            pipeline_metrics["total_iterations"] += len(reasoning_output)
        
        # Print progress
        print(f"\nCurrent Progress: {i}/{pipeline_metrics['total_questions']}")
        print(f"Iterations: {len(reasoning_output)}")
        current_accuracy = pipeline_metrics["correct_answers"] / i
        print(f"Current Accuracy: {current_accuracy:.1%}")
    
    elapsed_time = time.time() - start_time
    print()  # New line after progress
    
    # Calculate final averages
    pipeline_metrics["time_seconds"] = elapsed_time
    pipeline_metrics["accuracy"] = pipeline_metrics["correct_answers"] / pipeline_metrics["total_questions"]
    pipeline_metrics["average_iterations"] = pipeline_metrics["total_iterations"] / pipeline_metrics["total_questions"]
    
    # Save results
    with open("reasoning_benchmark.json", "w") as f:
        json.dump(pipeline_metrics, f, indent=2)
    
    print(f"Tested pipeline on {pipeline_metrics['total_questions']} questions in {elapsed_time:.2f} seconds")
    print(f"Average Iterations: {pipeline_metrics['average_iterations']:.1f}")
    print(f"Answer Accuracy: {pipeline_metrics['accuracy']:.1%}")

if __name__ == "__main__":
    benchmark_reasoning_pipeline()
    print("\nBenchmark results saved to reasoning_benchmark.json")
