#!/usr/bin/env python3

from deepeval.benchmarks import Aider
from deepeval.benchmarks.tasks import AiderTask
from deepeval.models import DeepSeekModel
import dspy
import json
from tqdm import tqdm

class DeepSeekEvaluator:
    def __init__(self):
        # Configure DeepSeek model
        self.lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
        dspy.settings.configure(lm=self.lm)
        
        # Initialize DeepEval Aider benchmark
        self.benchmark = Aider(
            tasks=[
                AiderTask.CODE_EDITING,
                AiderTask.CODE_REFACTORING
            ],
            n=100  # Number of code generation samples
        )

    def evaluate(self):
        print("Starting DeepSeek evaluation on Aider benchmark...")
        
        # Create DeepSeek model wrapper for DeepEval
        class DeepSeekWrapper(DeepSeekModel):
            def generate_samples(self, prompt: str, n: int, temperature: float) -> tuple[str, float]:
                # Use DSPy's DeepSeek model for generation
                result = self.lm(prompt)
                return result, 1.0  # Return generated text and confidence score

        # Evaluate the model
        self.benchmark.evaluate(model=DeepSeekWrapper(), k=10)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Overall Score: {self.benchmark.overall_score:.1%}")
        print("\nTask-wise Scores:")
        for task, score in self.benchmark.task_scores.items():
            print(f"{task}: {score:.1%}")

        # Save results
        results = {
            "overall_score": self.benchmark.overall_score,
            "task_scores": self.benchmark.task_scores,
            "model": "deepseek/deepseek-chat",
            "temperature": 0.3
        }
        
        with open("deepseek_aider_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print("\nResults saved to deepseek_aider_results.json")

if __name__ == "__main__":
    evaluator = DeepSeekEvaluator()
    evaluator.evaluate()
