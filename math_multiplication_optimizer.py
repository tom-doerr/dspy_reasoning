#!/usr/bin/env python3

import dspy
import random
from typing import List
from dspy.teleprompt import MIPROv2

class MultiplicationSignature(dspy.Signature):
    """Solve multiplication problems step by step."""
    task = dspy.InputField(desc="multiplication task as a string")
    solution = dspy.OutputField(desc="final solution as a number")

class MultiplicationSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(MultiplicationSignature)

    def forward(self, task):
        return self.generate_answer(task=task)

def generate_multiplication_dataset(num_samples=1000) -> List[dspy.Example]:
    """Generate multiplication problems with solutions."""
    dataset = []
    for _ in range(num_samples):
        a = random.randint(1, 10000)
        b = random.randint(1, 10000)
        task = f"{a} * {b}"
        solution = a * b
        dataset.append(dspy.Example(task=task, solution=solution).with_inputs('task'))
    return dataset

def evaluate_multiplication(example, prediction, trace=None):
    """Evaluate if predicted solution matches expected."""
    try:
        pred = float(prediction.solution)
        exp = float(example.solution)
        return int(abs(pred - exp) < 0.01)
    except:
        return 0

def optimize_multiplication_solver():
    # Configure language model
    lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
    dspy.settings.configure(lm=lm)

    # Generate dataset
    dataset = generate_multiplication_dataset(1000)
    trainset = dataset[:800]  # 80% training
    devset = dataset[800:]    # 20% validation

    # Initialize MIPROv2 optimizer
    teleprompter = MIPROv2(
        metric=evaluate_multiplication,
        num_candidates=3,
        num_threads=10,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
        auto='light'
    )

    # Create and optimize solver
    student = MultiplicationSolver()
    optimized_solver = teleprompter.compile(
        student, 
        trainset=trainset, 
        valset=devset,
        requires_permission_to_run=False
    )

    # Evaluate on validation set
    correct = 0
    for example in devset:
        prediction = optimized_solver(example.task)
        correct += evaluate_multiplication(example, prediction)

    accuracy = correct / len(devset)
    print(f"Validation accuracy: {accuracy:.1%}")

    return optimized_solver

if __name__ == "__main__":
    solver = optimize_multiplication_solver()
