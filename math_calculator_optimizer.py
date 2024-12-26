#!/usr/bin/env python3

import dspy
import json
from dspy.teleprompt import MIPRO
from math_calculator import MathCalculator, MathCalculationSignature

class MathOptimizer:
    def __init__(self):
        self.lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
        dspy.settings.configure(lm=self.lm)
        self.calculator = MathCalculator()
        
    def load_dataset(self, dataset_path="math_dataset.json"):
        with open(dataset_path) as f:
            dataset = json.load(f)
        return dataset[:100]  # Use subset for optimization

    def create_trainset(self, dataset):
        trainset = []
        for item in dataset:
            trainset.append(dspy.Example(
                task=item['task'],
                solution=item['solution']
            ).with_inputs('task'))
        return trainset

    def optimize(self, trainset, num_candidates=10):
        # Define the metric function
        def metric(example, prediction, trace=None):
            try:
                return int(abs(float(prediction.solution) - float(example.solution)) < 0.01)
            except:
                return 0

        # Configure MIPRO optimizer
        teleprompter = MIPRO(
            metric=metric,
            num_candidates=num_candidates,
            init_temperature=1.0,
            prompt_model=self.lm,
            task_model=self.lm
        )

        # Run optimization
        optimized_calculator = teleprompter.compile(
            self.calculator,
            trainset=trainset
        )

        return optimized_calculator

    def save_optimized_model(self, optimized_calculator, path="optimized_math_calculator.json"):
        optimized_calculator.save(path)
        print(f"Optimized model saved to {path}")

if __name__ == "__main__":
    optimizer = MathOptimizer()
    
    # Load and prepare dataset
    dataset = optimizer.load_dataset()
    trainset = optimizer.create_trainset(dataset)
    
    # Run optimization
    print("Starting optimization process...")
    optimized_calculator = optimizer.optimize(trainset, num_candidates=5)
    
    # Save optimized model
    optimizer.save_optimized_model(optimized_calculator)
    
    print("Optimization complete!")
