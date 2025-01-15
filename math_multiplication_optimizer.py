#!/usr/bin/env python3

import dspy
import random
import numpy as np
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
        # a = random.randint(1, 10000)
        # b = random.randint(1, 10000)
        max_num = int(1e5)
        a = random.randint(1, max_num)
        b = random.randint(1, max_num)
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
        # auto='light'
        auto='medium'
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
    # vs unoptimized solver
    student = MultiplicationSolver()
    for example in devset:
        prediction = student(example.task)
        correct += evaluate_multiplication(example, prediction)

    accuracy = correct / len(devset)
    print(f"Unoptimized accuracy: {accuracy:.1%}")

    return optimized_solver

def quick_optimize():
    dspy.settings.configure(lm=dspy.LM(model="deepseek/deepseek-chat"))
    dataset = [dspy.Example(task=f"{a}*{b}", solution=a*b).with_inputs('task') 
              for a,b in zip(np.random.randint(1e4,1e5,1000), 
                           np.random.randint(1e4,1e5,1000))]
    train, val = dataset[:800], dataset[800:]  # 80/20 split
    metric = lambda e,p,trace=None: int(abs(float(p.solution)-float(e.solution))<0.01)
    llm_program = dspy.ChainOfThought('task -> solution')
    llm_program_compiled = MIPROv2(metric=metric,
                    auto='medium').compile(
        llm_program, trainset=train, valset=val)
        # MultiplicationSolver(), trainset=train, valset=val)
    accuracy = sum(metric(e, llm_program_compiled(e.task)) for e in val) / len(val)
    print(f"Optimized accuracy: {accuracy:.1%}")



    student = MultiplicationSolver()
    for example in devset:
        prediction = student(example.task)
        correct += metric(example, prediction)

    accuracy = correct / len(devset)
    print(f"Unoptimized accuracy: {accuracy:.1%}")



if __name__ == "__main__":
    # solver = optimize_multiplication_solver()
    quick_optimize()
